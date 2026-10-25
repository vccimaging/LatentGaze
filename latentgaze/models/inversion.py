"""StyleGAN inversion encoders: the grayscale teacher P_T and the colour W+ encoder."""

import torch
from torch import nn

from .encoders import GradualStyleEncoder, PooledEncoder


def _mlp_head(dim_in, dim_hidden, dim_out):
    return nn.Sequential(nn.Linear(dim_in, dim_hidden), nn.SiLU(), nn.Linear(dim_hidden, dim_out))


class InversionTeacher(nn.Module):
    def __init__(self, in_channels: int = 1, latent_dim: int = 256, num_ws: int = 14):
        super().__init__()
        self.num_ws = num_ws
        self.encoder = PooledEncoder(in_channels, latent_dim)
        self.gaze_head = _mlp_head(latent_dim, latent_dim // 2, 3)
        self.valid_head = _mlp_head(latent_dim, latent_dim // 8, 1)

    def encode(self, x):
        """Compact latent h (B, latent_dim)."""
        return self.encoder(x)

    def forward(self, x):
        h = self.encoder(x)
        w_plus = h.unsqueeze(1).repeat(1, self.num_ws, 1)
        return w_plus, self.gaze_head(h), self.valid_head(h)


class InversionWPlus(nn.Module):
    def __init__(self, in_channels: int = 3, latent_dim: int = 512, num_ws: int = 16):
        super().__init__()
        self.num_ws = num_ws
        self.encoder = GradualStyleEncoder(in_channels, latent_dim, num_ws)
        self.gaze_head = _mlp_head(latent_dim, latent_dim // 2, 3)
        self.valid_head = _mlp_head(latent_dim, latent_dim // 8, 1)

    def forward(self, x):
        w_plus = self.encoder(x)
        h = w_plus[:, 0]
        return w_plus, self.gaze_head(h), self.valid_head(h)


def build_inversion(cfg) -> nn.Module:
    """cfg: {type: teacher|wplus, in_channels, latent_dim, num_ws}."""
    kind = cfg.get("type", "teacher")
    kwargs = dict(in_channels=cfg.get("in_channels", 1 if kind == "teacher" else 3),
                  latent_dim=cfg.get("latent_dim", 256 if kind == "teacher" else 512),
                  num_ws=cfg.get("num_ws", 14 if kind == "teacher" else 16))
    if kind == "teacher":
        return InversionTeacher(**kwargs)
    if kind == "wplus":
        return InversionWPlus(**kwargs)
    raise ValueError(f"Unknown inversion type '{kind}'")


def load_inversion(path: str, map_location="cpu") -> nn.Module:
    """Build an inversion encoder from a release checkpoint (stores its own config)."""
    ckpt = torch.load(path, map_location=map_location)
    model = build_inversion(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model
