import torch
import torch.nn as nn
import numpy as np
from custom_layers import *


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_w = args.img_w
        self.img_h = args.img_h
        self.nf = args.nf
        self.max_nf = args.max_nf
        self.last_block_resolution = 4
        self.log_last_block_resolution = int(np.log2(self.last_block_resolution))
        self.num_blocks = int(np.log2(self.img_w)) - self.log_last_block_resolution
        self.MP = args.MP
        self.geo_projection_dim = args.geo_projection_dim
        self.app_projection_dim = args.app_projection_dim
        
        blocks = []
        blocks += [Conv2dLayer(3, self.nf, kernel_size=1, activation='lrelu')]
        for k in range(self.num_blocks):
            in_channels = min(self.nf * (2 ** k), self.max_nf)
            out_channels = min(self.nf * (2 ** (k + 1)), self.max_nf)
            print(k, in_channels, out_channels)
            blocks += [DiscriminatorBlock(in_channels, 
                                          out_channels, 
                                          activation='lrelu', 
                                          skip=False, 
                                          use_fp16=self.MP)]

        self.shared_model = nn.Sequential(*blocks)
        self.discriminator_epilogue = DiscriminatorEpilogue(out_channels, resolution=self.last_block_resolution, mbstd_group_size=8, activation='lrelu')
        self.logit_mapper = MultiLayerPerceptron([out_channels, 1], activation='linear')
        self.projection_header1 = MultiLayerPerceptron([out_channels * 16, out_channels * 4, out_channels, self.geo_projection_dim], activation='lrelu')
        self.projection_header2 = MultiLayerPerceptron([out_channels * 16, out_channels * 4, out_channels, self.app_projection_dim], activation='lrelu')
        
    def forward(self, image, get_embedding_features=False):
        h = self.shared_model(image)
        logit = self.logit_mapper(self.discriminator_epilogue(h))
        geometry_embedding = None
        appearance_embedding = None
        if get_embedding_features:
            x = h.flatten(1)
            geometry_embedding = torch.nn.functional.normalize(self.projection_header1(x))
            appearance_embedding = torch.nn.functional.normalize(self.projection_header2(x))
            
        return logit, geometry_embedding, appearance_embedding


class Generator(torch.nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.img_w = args.img_w
        self.img_h = args.img_h
        self.nf = args.nf
        self.max_nf = args.max_nf 
        self.first_block_resolution = 4
        self.log_first_block_resolution = int(np.log2(self.first_block_resolution))
        self.num_blocks = int(np.log2(self.img_w)) - self.log_first_block_resolution
        self.num_fine_block = 2
        self.num_coarse_block = self.num_blocks - self.num_fine_block        
         
        self.MP = args.MP
        self.geo_latent_dim = args.geo_latent_dim
        self.app_latent_dim = args.app_latent_dim
        self.geo_noise_dim = args.geo_noise_dim
        self.app_noise_dim = args.app_noise_dim

        self.w_avg_beta = 0.9999
        self.register_buffer("avg_latent1", torch.zeros([self.geo_latent_dim]))
        self.register_buffer("avg_latent2", torch.zeros([self.app_latent_dim]))

        geometry_channels = [self.geo_noise_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim,
                             self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, 
                             self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim]
        
        appearance_channels = [self.app_noise_dim, self.app_latent_dim//4, self.app_latent_dim//2, self.app_latent_dim, self.app_latent_dim,
                               self.app_latent_dim, self.app_latent_dim, self.app_latent_dim, self.app_latent_dim, self.app_latent_dim,
                               self.app_latent_dim, self.app_latent_dim, self.app_latent_dim]
        
        self.geometry_mapping = MappingNetwork(geometry_channels, activation='linear')
        self.appearance_mapping = MappingNetwork(appearance_channels, activation='linear')
        self.const = torch.nn.Parameter(torch.randn([self.max_nf,
                                                     self.first_block_resolution,
                                                     self.first_block_resolution]))
        blocks = []
        dim_in = self.max_nf
        for i in range(self.num_blocks):
            dim_out = self.nf * 2 ** (self.num_blocks - i - 1)
            dim_out = min(dim_out, self.max_nf)
            out_resolution = 2 ** (self.log_first_block_resolution + 1 + i)
            print(i, dim_in, dim_out, out_resolution)
            blocks += [SynthesisBlock(dim_in,
                                      dim_out,
                                      self.geo_latent_dim,
                                      self.app_latent_dim,
                                      out_resolution,
                                      max_flow_scale=0.1,
                                      activation='lrelu',
                                      skip=True,
                                      use_noise=True,
                                      use_flow=True,
                                      use_fp16=self.MP)]
            dim_in = dim_out
            
        self.model = nn.Sequential(*blocks)
        self.rgb_layer = ToRGBBlock(dim_out, 3, self.app_latent_dim, out_resolution, activation='lrelu')
                
    def forward(self, rand_noise1, rand_noise2, w_psi=-1.0):
        batch_size = rand_noise1.size(0)
        geometry_code = self.geometry_mapping(rand_noise1)
        appearance_code = self.appearance_mapping(rand_noise2)
        
        # truncation trick
        if w_psi <= 0:
            self.avg_latent1.copy_(geometry_code.detach().mean(0).lerp(self.avg_latent1, self.w_avg_beta))
            self.avg_latent2.copy_(appearance_code.detach().mean(0).lerp(self.avg_latent2, self.w_avg_beta))
            
        if w_psi > 0.0:
            geometry_code = self.avg_latent1.lerp(geometry_code, w_psi)
            appearance_code = self.avg_latent2.lerp(appearance_code, w_psi)

        geometry_code = geometry_code.unsqueeze(1).repeat([1, self.num_blocks, 1])
        appearance_code = appearance_code.unsqueeze(1).repeat([1, (self.num_blocks+1)*2, 1])

        feature_maps = []
        flow_maps = []
        geometry_index, appearance_index = 0, 0
        x = self.const.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        for i in range(self.num_blocks):
            x, flowfield = self.model[i](x, 
                                         geometry_code.narrow(1, geometry_index, 1), 
                                         appearance_code.narrow(1, appearance_index, 2))
            geometry_index += 1
            appearance_index += 2
            feature_maps.append(x)
            flow_maps.append(flowfield)
                         
        x = x.to(dtype=torch.float32)
        out = self.rgb_layer(x, appearance_code.narrow(1, appearance_index, 2))
        return out, feature_maps, flow_maps