"""LPIPS perceptual distance.

Vendored from lpips-pytorch by Sou Uchida (https://github.com/S-aiueo32/lpips-pytorch,
BSD 2-Clause License, see LICENSE in this folder).
"""

from collections import OrderedDict
from itertools import chain

import torch
from torch import nn
from torchvision import models


def _normalize_activation(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)
    return x / (norm_factor + eps)


def _linear_weights(net_type: str = "alex", version: str = "0.1"):
    url = ("https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/"
           f"master/lpips/weights/v{version}/{net_type}.pth")
    old = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
    new = OrderedDict()
    for key, val in old.items():
        new[key.replace("lin", "").replace("model.", "")] = val
    return new


class _AlexFeatures(nn.Module):
    target_layers = [2, 5, 8, 10, 12]
    n_channels_list = [64, 192, 384, 256, 256]

    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("std", torch.Tensor([.458, .448, .450])[None, :, None, None])
        self.layers = models.alexnet(weights="IMAGENET1K_V1").features
        for p in chain(self.parameters(), self.buffers()):
            p.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        out = []
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i in self.target_layers:
                out.append(_normalize_activation(x))
            if len(out) == len(self.target_layers):
                break
        return out


class _LinLayers(nn.ModuleList):
    def __init__(self, n_channels_list):
        super().__init__([nn.Sequential(nn.Identity(), nn.Conv2d(nc, 1, 1, 1, 0, bias=False))
                          for nc in n_channels_list])
        for p in self.parameters():
            p.requires_grad = False


class LPIPS(nn.Module):
    def __init__(self, net_type: str = "alex", version: str = "0.1"):
        super().__init__()
        if net_type != "alex" or version != "0.1":
            raise NotImplementedError("Only LPIPS-alex v0.1 is used in this project")
        self.net = _AlexFeatures()
        self.lin = _LinLayers(self.net.n_channels_list)
        self.lin.load_state_dict(_linear_weights(net_type, version))
        self.eval()

    def forward(self, x, y):
        feats_x, feats_y = self.net(x), self.net(y)
        res = [lin((fx - fy) ** 2).mean((2, 3), True) for fx, fy, lin in zip(feats_x, feats_y, self.lin)]
        return torch.sum(torch.cat(res, 0)) / x.shape[0]
