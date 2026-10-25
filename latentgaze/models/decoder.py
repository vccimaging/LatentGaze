"""Measurement decoder P: 16 measurements to latent h, gaze and validity (Sec. 3.1)."""

import torch
from torch import nn

from .optical_encoder import OpticalEncoder


class MeasurementDecoder(nn.Module):
    def __init__(self, num_measurements: int = 16, width: int = 256, depth: int = 2, dropout: float = 0.2):
        super().__init__()
        self.norm = nn.LayerNorm(num_measurements)
        self.lift = nn.Sequential(nn.Linear(num_measurements, width), nn.SiLU(), nn.Dropout(dropout),
                                  nn.Linear(width, width))
        layers = []
        for _ in range(depth):
            layers += [nn.LayerNorm(width), nn.SiLU(), nn.Dropout(dropout), nn.Linear(width, width)]
        self.bottleneck = nn.Sequential(*layers)
        self.gaze_head = nn.Sequential(nn.Linear(width, width // 2), nn.SiLU(), nn.Linear(width // 2, 3))
        self.valid_head = nn.Sequential(nn.Linear(width, width // 8), nn.SiLU(), nn.Linear(width // 8, 1))

    def forward(self, y):
        """Returns (h (B, width), gaze (B, 3), validity logit (B, 1))."""
        h = self.lift(self.norm(y))
        h = h + self.bottleneck(h)
        return h, self.gaze_head(h), self.valid_head(h)


def load_decoder(path: str, map_location="cpu") -> MeasurementDecoder:
    """Measurement decoder of a ``train_masks.py`` or ``calibrate.py`` checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    decoder = MeasurementDecoder(**ckpt["config"]["decoder"])
    decoder.load_state_dict(ckpt["decoder"])
    return decoder


def load_masks_decoder(path: str, map_location="cpu"):
    """(OpticalEncoder, MeasurementDecoder) of a ``train_masks.py`` checkpoint."""
    ckpt = torch.load(path, map_location=map_location)
    optical = OpticalEncoder(**ckpt["config"]["optical_encoder"])
    optical.load_state_dict(ckpt["optical_encoder"])
    decoder = MeasurementDecoder(**ckpt["config"]["decoder"])
    decoder.load_state_dict(ckpt["decoder"])
    return optical, decoder
