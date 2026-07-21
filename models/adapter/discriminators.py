# Copyright (c) 2024 NVIDIA CORPORATION.
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

"""Discriminators for WavLM embeddings"""
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv2d, Conv1d
from torch.nn.utils import weight_norm, spectral_norm
from typing import Optional, List, Union, Dict, Tuple, Any



class Discriminator1D(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        embed_dim,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(in_dim, embed_dim, kernel_size=1)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=7, stride=2, padding=3)),
                norm_f(Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)),
            ]
        )
        self.conv_post = norm_f(Conv1d(embed_dim, 1, 3, 1, padding=1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """x: (B, T, D)"""
        x = x.transpose(1,2)
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(
        self,
        in_dim = 1024,
        sub_dims = [32, 64, 128, 256, 512, 1024],
        use_spectral_norm=False):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                Discriminator1D(in_dim, embed_dim, use_spectral_norm=use_spectral_norm)
                for embed_dim in sub_dims
            ]
        )

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
    ]:
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


if __name__ == "__main__":
    
    discriminator = MultiScaleDiscriminator()

    
    x = torch.randn(1, 1024, 100)
    y = torch.randn(1, 1024, 100)
    z = discriminator(x, y)

    
    