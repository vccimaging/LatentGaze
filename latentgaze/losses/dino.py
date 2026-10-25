"""DINOv2 feature loss (colour steering)."""

import torch
import torch.nn.functional as F
from torch import nn

from ..normalization import IMAGENET_MEAN, IMAGENET_STD


class DINOv2Loss(nn.Module):
    def __init__(self, layers=(5, 9), resize_to: int = 252):
        super().__init__()
        import timm
        self.layers = sorted(layers)
        self.resize_to = resize_to
        self.backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                                          dynamic_img_size=True).eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _prep(self, x):
        x = ((x + 1) * 0.5).clamp(0, 1)
        x = F.interpolate(x, (self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        return (x - self.mean) / self.std

    def _features(self, x):
        return self.backbone.forward_intermediates(x, indices=self.layers, output_fmt="NCHW", norm=True,
                                                   stop_early=True, intermediates_only=True)

    def forward(self, pred, target):
        with torch.no_grad():
            target_feats = self._features(self._prep(target))
        pred_feats = self._features(self._prep(pred))
        loss = 0.0
        for p, t in zip(pred_feats, target_feats):
            p, t = F.normalize(p, dim=1, eps=1e-6), F.normalize(t, dim=1, eps=1e-6)
            loss = loss + (1 - (p * t).sum(dim=1)).mean()
        return loss / len(pred_feats)
