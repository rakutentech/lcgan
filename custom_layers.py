import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import warnings
import scipy.signal
import scipy.optimize

from utils.style_ops import conv2d_resample
from utils.style_ops import upfirdn2d
from utils.style_ops import bias_act
from utils.style_ops import fma
import utils.style_misc as misc


class suppress_tracer_warnings(warnings.catch_warnings):
    def __enter__(self):
        super().__enter__()
        warnings.simplefilter('ignore', category=torch.jit.TracerWarning)
        return self


class Conv2dLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, activation='linear', up=1, down=1, resample_filter=[1,3,3,1]):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain

        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        bias = torch.zeros([out_channels]) if bias else None
        self.weight = torch.nn.Parameter(weight)
        self.bias = torch.nn.Parameter(bias) if bias is not None else None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = self.bias.to(x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down,
                                            padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain)
        return x


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, activation="linear", lr_multiplier=1, bias_init=0):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == "linear" and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


def modulated_conv2d(x, weight, latent, noise=None, up=1, down=1, padding=0, resample_filter=None,
                     demodulate=True, flip_weight=True, fused_modconv=True):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight.norm(float("inf"), dim=[1, 2, 3], keepdim=True))
        latent = latent / latent.norm(float("inf"), dim=1, keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * latent.reshape(batch_size, 1, -1, 1, 1)  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        x = x * latent.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(x=x,
                                            w=weight.to(x.dtype),
                                            f=resample_filter,
                                            up=up,
                                            down=down,
                                            padding=padding,
                                            flip_weight=flip_weight)
        if demodulate and noise is not None:
            x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # Execute as one fused op using grouped convolution.
    with misc.suppress_tracer_warnings():  # this value will be treated as a constant
        batch_size = int(batch_size)
    misc.assert_shape(x, [batch_size, in_channels, None, None])
    x = x.reshape(1, -1, *x.shape[2:])
    w = w.reshape(-1, in_channels, kh, kw)
    x = conv2d_resample.conv2d_resample(x=x,
                                        w=w.to(x.dtype),
                                        f=resample_filter,
                                        up=up,
                                        down=down,
                                        padding=padding,
                                        groups=batch_size,
                                        flip_weight=flip_weight)
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class SynthesisLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, resolution, 
                 kernel_size=3, up=1, activation="relu", resample_filter=[1,3,3,1], use_noise=False):
        super().__init__()
        self.up = up
        self.activation = activation
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.use_noise = use_noise
        
        self.affine = FullyConnectedLayer(self.latent_dim, in_channels, bias_init=1, activation='linear')
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))\
            .to(memory_format=torch.contiguous_format)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        if self.use_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            self.register_buffer("noise_const", torch.randn([self.resolution, self.resolution]))
        
    def forward(self, x, latent, gain=1):
        noise = None
        if self.use_noise:
            # noise = torch.randn([x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
            noise = self.noise_const * self.noise_strength
            
        flip_weight = (self.up == 1)  # slightly faster
        latent = self.affine(latent)
        x = modulated_conv2d(x=x,
                             weight=self.weight,
                             latent=latent,
                             noise=noise,
                             up=self.up,
                             padding=self.padding,
                             resample_filter=self.resample_filter,
                             flip_weight=flip_weight,
                             demodulate=True,
                             fused_modconv=True)
        
        act_gain = self.act_gain * gain
        x = bias_act.bias_act(x, self.bias.to(x.dtype), act=self.activation, gain=act_gain)
        return x


class SynthesisBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, geo_latent_dim, app_latent_dim, resolution, max_flow_scale,
                 activation="relu", resample_filter=[1, 3, 3, 1], use_noise=False, use_fp16=False, use_flow=False, skip=False):
        super().__init__()
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.use_noise = use_noise
        self.use_flow = use_flow
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.skip = skip
        
        self.conv0 = SynthesisLayer(in_channels=in_channels,
                                    out_channels=out_channels,
                                    latent_dim=app_latent_dim,
                                    resolution=resolution,
                                    up=2,
                                    activation=activation,
                                    resample_filter=resample_filter,
                                    use_noise=self.use_noise)

        self.conv1 = SynthesisLayer(in_channels=out_channels,
                                    out_channels=out_channels,
                                    latent_dim=app_latent_dim,
                                    resolution=resolution,
                                    activation=activation,
                                    use_noise=self.use_noise)
        
        if self.skip:
            self.skip_layer = Conv2dLayer(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          bias=False,
                                          up=2,
                                          resample_filter=resample_filter)
        if self.use_flow:
            self.max_flow_scale = max_flow_scale
            self.flow_layer = SynthesisLayer(in_channels=in_channels,
                                             out_channels=2,
                                             latent_dim=geo_latent_dim,
                                             resolution=resolution,
                                             kernel_size=3,
                                             up=2,
                                             activation='tanh',
                                             resample_filter=resample_filter,
                                             use_noise=False)
    
    def get_coordinates(self,b,h,w,device):
        grid_y, grid_x = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                                        torch.arange(w, dtype=torch.float32, device=device), 
                                        indexing='ij')
        norm_grid_y = (2 * grid_y / (h - 1)) - 1
        norm_grid_x = (2 * grid_x / (w - 1)) - 1
        coordinates = torch.stack((norm_grid_x, norm_grid_y)).unsqueeze(0).repeat([b, 1, 1, 1])        
        return coordinates
    
    def forward(self, x, latent1, latent2, force_fp32=False):
        iter1 = iter(latent1.unbind(dim=1))
        iter2 = iter(latent2.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        x = x.to(dtype=dtype)
        
        if self.skip:
            if self.use_flow:
                flowfield = self.flow_layer(x, next(iter1))
            y = self.skip_layer(x, gain=np.sqrt(0.5))
            x = self.conv0(x, next(iter2))
            x = self.conv1(x, next(iter2), gain=np.sqrt(0.5))
            x = y.add_(x)
            if self.use_flow:
                b, c, h, w = x.size()
                coordinates = self.get_coordinates(b,h,w,x.device).to(dtype=torch.float32, device=x.device)
                correspondence_map = coordinates + flowfield.to(dtype=torch.float32, device=x.device) * self.max_flow_scale
                x = x.to(dtype=torch.float32, device=x.device)
                x = F.grid_sample(x, correspondence_map.permute(0, 2, 3, 1), align_corners=True, mode='bicubic')
                x = x.to(dtype=dtype)            
        else:
            if self.use_flow:
                flowfield = self.flow_layer(x, next(iter1))
            x = self.conv0(x, next(iter2))
            if self.use_flow:
                b, c, h, w = x.size()
                coordinates = self.get_coordinates(b,h,w,x.device).to(dtype=torch.float32, device=x.device)
                correspondence_map = coordinates + flowfield.to(dtype=torch.float32, device=x.device) * self.max_flow_scale
                x = x.to(dtype=torch.float32, device=x.device)
                x = F.grid_sample(x, correspondence_map.permute(0, 2, 3, 1), align_corners=True, mode='bicubic')
                x = x.to(dtype=dtype)            
            x = self.conv1(x, next(iter2))

        if self.use_flow==False:
            flowfield = None
        assert x.dtype == dtype
        return x, flowfield


class DiscriminatorBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 activation="relu", resample_filter=[1, 3, 3, 1], use_fp16=False, skip=False):
        super().__init__()
        self.use_fp16 = use_fp16
        self.register_buffer("resample_filter", upfirdn2d.setup_filter(resample_filter))
        self.skip = skip

        self.conv0 = Conv2dLayer(in_channels,
                                 in_channels,
                                 kernel_size=3,
                                 activation=activation)

        self.conv1 = Conv2dLayer(in_channels,
                                 out_channels,
                                 kernel_size=3,
                                 activation=activation,
                                 down=2,
                                 resample_filter=resample_filter)

        if self.skip:
            self.skip_layer = Conv2dLayer(in_channels,
                                          out_channels,
                                          kernel_size=1,
                                          bias=False,
                                          down=2,
                                          resample_filter=resample_filter)

    def forward(self, x, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        x = x.to(dtype=dtype)
        if self.skip:
            y = self.skip_layer(x, gain=np.sqrt(0.5))
            x = self.conv0(x)
            x = self.conv1(x, gain=np.sqrt(0.5))
            x = y.add_(x)
        else:
            x = self.conv0(x)
            x = self.conv1(x)

        assert x.dtype == dtype
        return x


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self, in_channels, resolution, mbstd_group_size=4, activation="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.resolution = resolution
        self.mb_std = MinibatchStdLayer(group_size=mbstd_group_size)
        self.conv = Conv2dLayer(in_channels+1, in_channels, kernel_size=3, activation=activation)
        self.fc = FullyConnectedLayer(in_channels * (resolution**2), in_channels, activation=activation, lr_multiplier=0.01)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        # Main layers.
        x = self.mb_std(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        return x


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        with suppress_tracer_warnings(): # as_tensor results are registered as constants
            G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2,3,4])             # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


class MappingNetwork(torch.nn.Module):
    def __init__(self, channels_list, activation="relu", lr_multiplier=0.01):
        super().__init__()
        self.eps = 1e-6
        self.matrix_size = channels_list[0]
        self.diagonal_params = torch.nn.Parameter(torch.ones([self.matrix_size]))    # diagonal elements
        self.basis_params = torch.nn.Parameter(torch.randn([self.matrix_size,self.matrix_size]))   # off-diagonal elements
        self.num_layers = len(channels_list)-1
        mlp = []
        for idx in range(self.num_layers):
            dim_in = channels_list[idx]
            dim_out = channels_list[idx+1]
            mlp += [FullyConnectedLayer(dim_in, dim_out, activation=activation, lr_multiplier=lr_multiplier)]
        self.mlp = nn.Sequential(*mlp)

    def orthogonalize(self, matrix):
        Q, _ = torch.qr(matrix)
        return Q

    def forward(self, z):
        batch_size = z.size(0)
        D_sqrt = torch.diag(torch.abs(self.diagonal_params) + self.eps)
        B = self.orthogonalize(torch.tanh(self.basis_params))
        L = torch.matmul(B, D_sqrt)
        L_ = L.unsqueeze(0).repeat([batch_size, 1, 1])  # [b, m, m]
        z_ = z.unsqueeze(2) # [b, m, 1]
        x = torch.bmm(L_, z_).squeeze() # [b, m]
        x = self.mlp(x)
        return x


class MultiLayerPerceptron(torch.nn.Module):
    def __init__(self, channels_list, activation="relu", lr_multiplier=0.01):
        super().__init__()
        self.num_layers = len(channels_list)-1
        if self.num_layers > 0:
            mlp = []
            for idx in range(self.num_layers):
                dim_in = channels_list[idx]
                dim_out = channels_list[idx+1]
                if idx < self.num_layers-1:
                    mlp += [FullyConnectedLayer(dim_in, dim_out, activation=activation, lr_multiplier=lr_multiplier)]
                else:
                    mlp += [FullyConnectedLayer(dim_in, dim_out, activation='linear', lr_multiplier=lr_multiplier)]
            self.mlp = nn.Sequential(*mlp)
            
    def forward(self, z):
        x = self.mlp(z)
        return x 


class ToRGBBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, latent_dim, resolution,
                 activation="relu", use_noise=False, use_fp16=False):
        super().__init__()
        self.resolution = resolution
        self.use_fp16 = use_fp16
        self.use_noise = use_noise

        self.conv0 = SynthesisLayer(in_channels=in_channels,
                                    out_channels=in_channels,
                                    latent_dim=latent_dim,
                                    resolution=resolution,
                                    activation=activation,
                                    use_noise=True)

        self.affine = FullyConnectedLayer(latent_dim, in_channels, bias_init=1)
        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, 1, 1]))\
            .to(memory_format=torch.contiguous_format)
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (1**2))
    
    def forward(self, x, latent1, force_fp32=False):
        iter1 = iter(latent1.unbind(dim=1))
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        x = x.to(dtype=dtype)
        x = self.conv0(x, next(iter1))        
        latent_ = self.affine(next(iter1)) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, latent=latent_, demodulate=False, fused_modconv=True)
        x = bias_act.bias_act(x, self.bias.to(x.dtype))
        return x
    