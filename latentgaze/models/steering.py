"""Gaze-steering controllers (Sec. 3.2)."""

import torch
import torch.nn.functional as F
from torch import nn


class Mish(nn.Module):
    """x * tanh(softplus(x)), written out explicitly to match the released weights bit for bit."""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SteeringCompact(nn.Module):
    def __init__(self, latent_dim: int = 256, num_ws: int = 14, hidden: int = 256):
        super().__init__()
        self.num_ws = num_ws
        self.main_mlp = nn.Sequential(nn.Linear(6, hidden), Mish(), nn.Linear(hidden, hidden), Mish())
        self.scaling_network = nn.Sequential(nn.Linear(1, hidden // 4), Mish(), nn.Linear(hidden // 4, 1),
                                             nn.Sigmoid())
        self.output_layer = nn.Linear(hidden, latent_dim)

    def forward(self, g_src, g_tgt, w_plus=None):
        scale = self.scaling_network(torch.norm(g_tgt - g_src, dim=1, keepdim=True))
        delta = scale * self.output_layer(self.main_mlp(torch.cat([g_src, g_tgt], dim=1)))
        return delta.unsqueeze(1).repeat(1, self.num_ws, 1)


class SteeringWPlus(nn.Module):
    def __init__(self, latent_dim: int = 512, num_ws: int = 16, hidden: int = 256, extra_in: int = 0):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(6, hidden), nn.LayerNorm(hidden), Mish(),
                                   nn.Linear(hidden, hidden), Mish())
        self.layer_heads = nn.ModuleList([nn.Linear(hidden + extra_in, latent_dim) for _ in range(num_ws)])
        self.gate_linear = nn.Linear(1, 1, bias=False)

    def _features(self, g_src, g_tgt, w_plus):
        return self.trunk(torch.cat([g_src, g_tgt], dim=1))

    def forward(self, g_src, g_tgt, w_plus=None):
        magnitude = torch.norm(g_tgt - g_src, dim=1, keepdim=True)
        gate = torch.tanh(self.gate_linear(magnitude)) * magnitude
        feats = self._features(g_src, g_tgt, w_plus)
        deltas = torch.stack([head(feats) for head in self.layer_heads], dim=1)
        return deltas * gate.unsqueeze(-1)


class SteeringWPlusIdentity(SteeringWPlus):
    def __init__(self, latent_dim: int = 512, num_ws: int = 16, hidden: int = 256, identity_dim: int = 64):
        super().__init__(latent_dim, num_ws, hidden, extra_in=identity_dim)
        self.identity_encoder = nn.Sequential(nn.Linear(latent_dim, 128), nn.LayerNorm(128), Mish(),
                                              nn.Linear(128, identity_dim), nn.LayerNorm(identity_dim))

    def _features(self, g_src, g_tgt, w_plus):
        trunk = self.trunk(torch.cat([g_src, g_tgt], dim=1))
        identity = self.identity_encoder(w_plus.detach().mean(dim=1))
        return torch.cat([trunk, identity], dim=1)


STEERING_MODELS = {"compact": SteeringCompact, "wplus": SteeringWPlus, "wplus_identity": SteeringWPlusIdentity}


def build_steering(cfg) -> nn.Module:
    kind = cfg.get("type", "compact")
    if kind not in STEERING_MODELS:
        raise ValueError(f"Unknown steering type '{kind}'")
    kwargs = dict(latent_dim=cfg.get("latent_dim", 256 if kind == "compact" else 512),
                  num_ws=cfg.get("num_ws", 14 if kind == "compact" else 16))
    if kind == "wplus_identity":
        kwargs["identity_dim"] = cfg.get("identity_dim", 64)
    return STEERING_MODELS[kind](**kwargs)


def load_steering(path: str, map_location="cpu") -> nn.Module:
    ckpt = torch.load(path, map_location=map_location)
    model = build_steering(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    return model
