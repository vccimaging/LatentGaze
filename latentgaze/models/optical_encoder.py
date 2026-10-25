"""Differentiable optical encoder: 16 binary masks, one measurement each (Sec. 3.1)."""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def disk(size: int = 256, center: float = 127.0, radius: float = 127.0) -> torch.Tensor:
    """Circular aperture of the microlens array, (size, size) in {0, 1}."""
    yy, xx = np.ogrid[:size, :size]
    return torch.from_numpy((np.sqrt((xx - center) ** 2 + (yy - center) ** 2) <= radius).astype(np.float32))


class OpticalEncoder(nn.Module):
    def __init__(self, num_masks: int = 16, param_size: int = 64, image_size: int = 256):
        super().__init__()
        self.num_masks = num_masks
        self.param_size = param_size
        self.image_size = image_size
        self.weight = nn.Parameter(torch.rand(num_masks, param_size * param_size))
        self.register_buffer("aperture", disk(image_size).flatten())

    def continuous_masks(self) -> torch.Tensor:
        """Upsampled continuous masks, shape (H*W, num_masks)."""
        w = self.weight.view(1, self.num_masks, self.param_size, self.param_size)
        if self.param_size != self.image_size:
            w = F.interpolate(w, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return w.view(self.num_masks, -1).t()

    def binary_masks(self) -> torch.Tensor:
        """Binary masks (H*W, num_masks) with straight-through gradients, aperture applied."""
        w = self.continuous_masks()
        binary = torch.clamp(w, 0, 1).round()
        return ((binary - w).detach() + w) * self.aperture.view(-1, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, C, 256, 256) in [-1, 1]. Returns measurements y (B, num_masks)."""
        x = image * 0.5 + 0.5
        masks = self.binary_masks()
        return sum(x[:, c].flatten(1) @ masks for c in range(x.shape[1]))

    @torch.no_grad()
    def export(self) -> np.ndarray:
        """Fabrication layout: (num_masks, 256, 256) uint8 in {0, 1}, row-major."""
        return self.binary_masks().t().reshape(self.num_masks, self.image_size, self.image_size) \
            .round().to(torch.uint8).cpu().numpy()
