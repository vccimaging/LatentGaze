"""Conversions between the [-1, 1] ("gan") and ImageNet image normalizations."""

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
GRAY_IMAGENET_MEAN = 0.449
GRAY_IMAGENET_STD = 0.226


def _stats(channels: int, device):
    if channels == 3:
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    elif channels == 1:
        mean = torch.tensor([GRAY_IMAGENET_MEAN], device=device).view(1, 1, 1, 1)
        std = torch.tensor([GRAY_IMAGENET_STD], device=device).view(1, 1, 1, 1)
    else:
        raise ValueError(f"Unsupported channel count: {channels}")
    return mean, std


def gan_to_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean, std = _stats(x.shape[1], x.device)
    return (x * 0.5 + 0.5 - mean) / std
