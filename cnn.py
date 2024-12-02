import torch
import torch.nn as nn
import numpy as np
from custom_layers import *


class Discriminator(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.img_resolution = args.img_resolution
        self.last_block_resolution = 4
        self.log_last_block_resolution = int(np.log2(self.last_block_resolution))
        self.num_blocks = int(np.log2(self.img_resolution)) - self.log_last_block_resolution
        self.geo_projection_dim = args.geo_projection_dim
        self.app_projection_dim = args.app_projection_dim
        self.max_nf = 512
        self.base_nf = 32 if self.img_resolution == 1024 else 64 if self.img_resolution == 512 else 128
        
        blocks = []
        blocks += [EqualizedConv2d(3, self.base_nf, kernel_size=1)]
        blocks += [nn.LeakyReLU(0.2)]
        for i in range(self.num_blocks):
            in_features = min(self.base_nf * (2 ** i), self.max_nf)
            out_features = min(self.base_nf * (2 ** (i + 1)), self.max_nf)
            blocks += [DiscriminatorBlock(in_features, out_features, skip=True)]

        self.shared_model = nn.Sequential(*blocks)
        self.discriminator_epilogue = DiscriminatorEpilogue(out_features, resolution=self.last_block_resolution, mbstd_group_size=8)
        self.logit_mapper = ProjectionHead([out_features, 1])
        self.projection_header1 = ProjectionHead([out_features * 16, out_features * 4, out_features, self.geo_projection_dim])
        self.projection_header2 = ProjectionHead([out_features * 16, out_features * 4, out_features, self.app_projection_dim])
        
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
        self.img_resolution = args.img_resolution
        self.first_block_resolution = 4
        self.log_first_block_resolution = int(np.log2(self.first_block_resolution))
        self.num_blocks = int(np.log2(self.img_resolution)) - self.log_first_block_resolution
        self.max_nf = 512
        self.base_nf = 32 if self.img_resolution == 1024 else 64 if self.img_resolution == 512 else 128

        self.geo_latent_dim = args.geo_latent_dim
        self.app_latent_dim = args.app_latent_dim
        self.geo_noise_dim = args.geo_noise_dim
        self.app_noise_dim = args.app_noise_dim
        self.max_flow_scale = args.max_flow_scale

        self.w_avg_beta = 0.998
        self.register_buffer("avg_latent1", torch.zeros([self.geo_latent_dim]))
        self.register_buffer("avg_latent2", torch.zeros([self.app_latent_dim]))

        geometry_channels = [self.geo_noise_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim,
                             self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim, 
                             self.geo_latent_dim, self.geo_latent_dim, self.geo_latent_dim]
        
        appearance_channels = [self.app_noise_dim, self.app_latent_dim//4, self.app_latent_dim//2, self.app_latent_dim, self.app_latent_dim,
                               self.app_latent_dim, self.app_latent_dim, self.app_latent_dim, self.app_latent_dim, self.app_latent_dim,
                               self.app_latent_dim, self.app_latent_dim, self.app_latent_dim]
        
        self.geometry_mapping = MappingNetwork(geometry_channels)
        self.appearance_mapping = MappingNetwork(appearance_channels)
        self.const = torch.nn.Parameter(torch.randn([self.max_nf, self.first_block_resolution, self.first_block_resolution]))
        blocks = []
        in_features = self.max_nf
        for i in range(self.num_blocks):
            out_features = self.base_nf * 2 ** (self.num_blocks - i - 1)
            out_features = min(out_features, self.max_nf)
            out_resolution = 2 ** (self.log_first_block_resolution + 1 + i)
            blocks += [SynthesisBlock(in_features, out_features, self.geo_latent_dim, self.app_latent_dim, out_resolution, self.max_flow_scale, use_noise=True)]
            in_features = out_features
            
        self.model = nn.Sequential(*blocks)
        self.rgb_layer = ToRGBBlock(out_features, 3, self.app_latent_dim, out_resolution, use_noise=True)
                
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
        geometry_index, appearance_index = 0, 0
        x = self.const.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        for i in range(self.num_blocks):
                x = self.model[i](x, 
                                  geometry_code.narrow(1, geometry_index, 1), 
                                  appearance_code.narrow(1, appearance_index, 2))
                geometry_index += 1
                appearance_index += 2
                
        out = self.rgb_layer(x, appearance_code.narrow(1, appearance_index, 2))
        return out