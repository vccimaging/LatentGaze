"""Gaze vectors, angles and gaze losses."""

import math

import torch
import torch.nn.functional as F


def pitchyaw_to_vector(pitchyaw: torch.Tensor) -> torch.Tensor:
    """(B, 2) pitch/yaw in radians -> (B, 3) unit vectors (front hemisphere, z > 0)."""
    pitch, yaw = pitchyaw[:, 0], pitchyaw[:, 1]
    return torch.stack([torch.cos(pitch) * torch.sin(yaw),
                        torch.sin(pitch),
                        torch.cos(pitch) * torch.cos(yaw)], dim=1)


def vector_to_pitchyaw(vectors: torch.Tensor) -> torch.Tensor:
    """(B, 3) dataset gaze vectors (z < 0) -> (B, 2) pitch/yaw in radians."""
    v = -F.normalize(vectors, dim=1)
    return torch.stack([torch.arcsin(v[:, 1]), torch.atan2(v[:, 0], v[:, 2])], dim=1)


def angular_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-sample angle between two sets of 3D vectors, in degrees. Shape (B,)."""
    cos = (a * b).sum(dim=1) / (a.norm(dim=1).clamp(min=1e-7) * b.norm(dim=1).clamp(min=1e-7))
    return torch.acos(cos.clamp(-1.0, 1.0)) * 180.0 / math.pi


def angular_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean angle (radians) between predicted and target directions."""
    cos = F.cosine_similarity(F.normalize(pred, dim=1), F.normalize(target, dim=1), dim=1)
    return torch.acos(cos.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).mean()
