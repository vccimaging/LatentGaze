"""IR-SE-50 encoders for StyleGAN inversion.

Adapted from pixel2style2pixel by Richardson et al. (https://github.com/eladrich/pixel2style2pixel,
MIT License, see LICENSE-pixel2style2pixel in this folder).
"""

import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """Description of one residual unit."""


def _block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def ir50_blocks():
    return [
        _block(64, 64, 3),
        _block(64, 128, 4),
        _block(128, 256, 14),
        _block(256, 512, 3),
    ]


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * w


class BottleneckIRSE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class EqualLinear(nn.Module):
    """Linear layer with equalized learning rate (StyleGAN style)."""

    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x):
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias=bias)


def _ir_se50_trunk(in_channels):
    input_layer = nn.Sequential(nn.Conv2d(in_channels, 64, (3, 3), 1, 1, bias=False),
                                nn.BatchNorm2d(64), nn.PReLU(64))
    body = nn.Sequential(*[BottleneckIRSE(b.in_channel, b.depth, b.stride)
                           for block in ir50_blocks() for b in block])
    return input_layer, body


class PooledEncoder(nn.Module):
    """IR-SE-50 trunk -> global average pooling -> EqualLinear(512, latent_dim).

    This is the compact encoder of the inversion teacher P_T (Supp. F.2): a single latent
    vector that is broadcast to all StyleGAN layers.
    """

    def __init__(self, in_channels=1, latent_dim=256):
        super().__init__()
        self.input_layer, self.body = _ir_se50_trunk(in_channels)
        self.output_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, latent_dim)

    def forward(self, x):
        x = self.output_pool(self.body(self.input_layer(x)))
        return self.linear(x.flatten(1))


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c = out_c
        num_pools = int(np.log2(spatial))
        modules = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for _ in range(num_pools - 1):
            modules += [nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        if x.shape[2] > 1 or x.shape[3] > 1:
            x = F.adaptive_avg_pool2d(x, (1, 1))
        return self.linear(x.view(-1, self.out_c))


class GradualStyleEncoder(nn.Module):
    """pSp feature-pyramid encoder producing one style vector per StyleGAN layer (W+)."""

    coarse_ind, middle_ind = 3, 7

    def __init__(self, in_channels=3, latent_dim=512, num_ws=16):
        super().__init__()
        self.input_layer, self.body = _ir_se50_trunk(in_channels)
        self.style_count = num_ws
        self.styles = nn.ModuleList()
        for i in range(num_ws):
            spatial = 16 if i < self.coarse_ind else (32 if i < self.middle_ind else 64)
            self.styles.append(GradualStyleBlock(512, latent_dim, spatial))
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _upsample_add(x, y):
        return F.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        latents = [self.styles[j](c3) for j in range(self.coarse_ind)]
        p2 = self._upsample_add(c3, self.latlayer1(c2))
        latents += [self.styles[j](p2) for j in range(self.coarse_ind, self.middle_ind)]
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        latents += [self.styles[j](p1) for j in range(self.middle_ind, self.style_count)]
        return torch.stack(latents, dim=1)
