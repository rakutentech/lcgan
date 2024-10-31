import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Optional for StyleGAN-v1
class AdaIN(nn.Module):
    def __init__(self, latent_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.linear = EqualizedLinear(latent_dim, num_features*2, bias=1.0, lr_mul=0.01)

    def forward(self, x, s):
        h = self.linear(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

    
class EqualizedWeight(nn.Module):
    def __init__(self, shape, lr_mul=1.0):
        super().__init__()
        self.c = 1 / np.sqrt(np.prod(shape[1:])) * lr_mul
        self.weight = nn.Parameter(torch.randn(shape).div_(lr_mul))

    def forward(self):
        return self.weight * self.c


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=0.0, lr_mul=1.0):
        super().__init__()
        self.weight = EqualizedWeight([out_features, in_features], lr_mul)
        self.bias = nn.Parameter(torch.ones(out_features) * bias)
        self.lr_mul = lr_mul

    def forward(self, x):
        return F.linear(x, self.weight(), bias=self.bias * self.lr_mul)

    
class EqualizedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride=1, no_bias=False, lr_mul=1.0):
        super().__init__()
        self.padding = kernel_size // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size], lr_mul)
        self.no_bias = no_bias
        if not self.no_bias:
            self.bias = nn.Parameter(torch.zeros([out_features]))
        self.stride = stride
        self.lr_mul = lr_mul

    def forward(self, x):
        if self.no_bias:
            x = F.conv2d(x, self.weight(), stride=self.stride, padding=self.padding)
        else:
            x = F.conv2d(x, self.weight(), stride=self.stride, bias=self.bias * self.lr_mul, padding=self.padding)
        return x


class ModulatedConv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, eps=1e-8, lr_mul=1.0):
        super().__init__()
        self.out_features = out_features
        self.padding = (kernel_size - 1) // 2
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size], lr_mul)
        self.bias = nn.Parameter(torch.zeros([out_features]))
        self.lr_mul = lr_mul
        self.eps = eps

    def forward(self, x, s):
        b, c, h, w = x.shape
        s = s[:, None, :, None, None]
        weight = self.weight()[None, :, :, :, :]
        weight = weight * s
        sigma_inv = torch.rsqrt((weight ** 2).sum(dim=(2, 3, 4), keepdim=True) + self.eps)
        weight = weight * sigma_inv
        x = x.reshape(1, -1, h, w)
        _, _, *ws = weight.shape
        weight = weight.reshape(b * self.out_features, *ws)
        x = F.conv2d(x, weight, padding=self.padding, groups=b)
        x = x.reshape(b, self.out_features, h, w)
        x = x + self.bias.view(1, -1, 1, 1) * self.lr_mul
        return x


class SynthesisLayer(nn.Module):
    def __init__(self, in_features, out_features, latent_dim, resolution, kernel_size=3, lr_mul=1.0, use_noise=False, legacy_norm=False):
        super().__init__()
        self.latent_dim = latent_dim
        self.resolution = resolution
        self.use_noise = use_noise
        self.legacy_norm = legacy_norm
        if self.legacy_norm:
            self.adain = AdaIN(self.latent_dim, in_features)
            self.conv = EqualizedConv2d(in_features, out_features, kernel_size, lr_mul=1.0)
        else:
            self.linear = EqualizedLinear(self.latent_dim, in_features, bias=1.0, lr_mul=1.0)
            self.modulated_conv = ModulatedConv2d(in_features, out_features, kernel_size, lr_mul=1.0)
            
        if self.use_noise:
            self.noise_strength = nn.Parameter(torch.zeros([]))
            self.register_buffer("noise_const", torch.randn([self.resolution, self.resolution]))

    def forward(self, x, latent):
        # Convolution with demodulation
        if self.legacy_norm:
            x = self.adain(x, latent)
            x = self.conv(x)
        else:
            s = self.linear(latent)
            x = self.modulated_conv(x, s)
        # Noise addition
        if self.use_noise:
            noise = self.noise_const * self.noise_strength
            x = x + noise
        return x


class SynthesisBlock(nn.Module):
    def __init__(self, in_features, out_features, g_latent_dim, a_latent_dim, resolution, max_flow_scale, use_noise=False):
        super().__init__()
        self.resolution = resolution
        self.use_noise = use_noise
        self.max_flow_scale = max_flow_scale
        self.modulated_conv0 = SynthesisLayer(in_features, out_features, a_latent_dim, resolution, use_noise=self.use_noise)
        self.modulated_conv1 = SynthesisLayer(out_features, out_features, a_latent_dim, resolution, use_noise=self.use_noise)
        self.skip_layer = EqualizedConv2d(in_features, out_features, kernel_size=1, no_bias=True, lr_mul=1.0)
        self.flow_layer = SynthesisLayer(in_features, 2, g_latent_dim, resolution, use_noise=False)
        self.gain = np.sqrt(2)
        self.skip_gain = np.sqrt(0.5)
        
    def get_coordinates(self, b, h, w, device):
        grid_y, grid_x = torch.meshgrid(torch.arange(h, dtype=torch.float32, device=device),
                                        torch.arange(w, dtype=torch.float32, device=device), 
                                        indexing='ij')
        norm_grid_y = (2 * grid_y / (h - 1)) - 1
        norm_grid_x = (2 * grid_x / (w - 1)) - 1
        coordinates = torch.stack((norm_grid_x, norm_grid_y)).unsqueeze(0).repeat([b, 1, 1, 1])        
        return coordinates
    
    def forward(self, x, g_latent, a_latent):
        g_iter = iter(g_latent.unbind(dim=1))
        a_iter = iter(a_latent.unbind(dim=1))

        # convolution operations
        skip = self.skip_layer(x) * self.skip_gain
        skip = F.interpolate(skip, scale_factor=2, mode='bilinear')
                
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        flowfield = self.flow_layer(x, next(g_iter))
        flowfield = torch.tanh(flowfield)

        x = self.modulated_conv0(x, next(a_iter))
        x = F.leaky_relu(x, 0.2) * self.gain
        x = self.modulated_conv1(x, next(a_iter))
        x = F.leaky_relu(x, 0.2)
        x = skip.add_(x)
        
        # feature warping
        b, c, h, w = x.size()
        coordinates = self.get_coordinates(b, h, w, x.device).to(dtype=torch.float32, device=x.device)
        correspondence_map = coordinates + flowfield.to(dtype=torch.float32, device=x.device) * self.max_flow_scale
        x = F.grid_sample(x, correspondence_map.permute(0, 2, 3, 1), mode='bilinear')
        return x


class ToRGBBlock(nn.Module):
    def __init__(self, in_features, out_features, a_latent_dim, resolution, use_noise=False):
        super().__init__()
        self.resolution = resolution
        self.use_noise = use_noise
        self.modulated_conv0 = SynthesisLayer(in_features, in_features, a_latent_dim, resolution, use_noise=self.use_noise)
        self.modulated_conv1 = SynthesisLayer(in_features, out_features, a_latent_dim, resolution, kernel_size=1, use_noise=False)
    
    def forward(self, x, a_latent):
        a_iter = iter(a_latent.unbind(dim=1))
        x = self.modulated_conv0(x, next(a_iter))
        x = F.leaky_relu(x, 0.2)
        x = self.modulated_conv1(x, next(a_iter))
        return x
    

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv0 = EqualizedConv2d(in_features, in_features, kernel_size=3, lr_mul=1.0)
        self.conv1 = EqualizedConv2d(in_features, out_features, kernel_size=3, stride=2, lr_mul=1.0)
        self.skip_layer = EqualizedConv2d(in_features, out_features, kernel_size=1, no_bias=True, lr_mul=1.0)
        self.gain = np.sqrt(2)
        self.skip_gain = np.sqrt(0.5)

    def forward(self, x):
        skip = F.avg_pool2d(x, kernel_size=3, stride=2, padding=1)
        skip = self.skip_layer(skip) * self.skip_gain
        x = self.conv0(x)
        x = F.leaky_relu(x, 0.2) * self.gain
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.2)
        x = skip.add_(x)
        return x


class DiscriminatorEpilogue(nn.Module):
    def __init__(self, in_features, resolution, mbstd_group_size=4):
        super().__init__()
        self.resolution = resolution
        self.mb_std = MinibatchStdLayer(group_size=mbstd_group_size)
        self.conv = EqualizedConv2d(in_features + 1, in_features, kernel_size=3, lr_mul=1.0)
        self.linear = EqualizedLinear(in_features * (resolution ** 2), in_features, lr_mul=0.01)

    def forward(self, x):
        x = self.mb_std(x)
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.linear(x.flatten(1))
        x = F.leaky_relu(x, 0.2)
        return x


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F
        y = x.reshape(G, -1, F, c, H, W)    # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)               # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)          # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()               # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])           # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)          # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)            # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)        # [NCHW]   Append to input as new channels.
        return x


class MappingNetwork(nn.Module):
    def __init__(self, channels_list, lr_mul=0.01):
        super().__init__()
        self.eps = 1e-8
        self.matrix_size = channels_list[0]
        self.diagonal_params = nn.Parameter(torch.randn([self.matrix_size]))    # diagonal elements
        self.basis_params = nn.Parameter(torch.randn([self.matrix_size, self.matrix_size]))   # off-diagonal elements
        self.num_layers = len(channels_list) - 1
        mlp = []
        for idx in range(self.num_layers):
            in_features = channels_list[idx]
            out_features = channels_list[idx + 1]
            mlp += [EqualizedLinear(in_features, out_features, lr_mul=lr_mul)]
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
        z_ = z.unsqueeze(2)  # [b, m, 1]
        x = torch.bmm(L_, z_).squeeze(2)  # [b, m]
        x = self.mlp(x)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, channels_list, lr_mul=0.01):
        super().__init__()
        self.num_layers = len(channels_list) - 1
        if self.num_layers > 0:
            mlp = []
            for idx in range(self.num_layers):
                in_features = channels_list[idx]
                out_features = channels_list[idx + 1]
                mlp += [EqualizedLinear(in_features, out_features, lr_mul=lr_mul)]
                if idx < self.num_layers - 1:
                    mlp += [nn.LeakyReLU(0.2)]
            self.mlp = nn.Sequential(*mlp)
            
    def forward(self, z):
        x = self.mlp(z)
        return x