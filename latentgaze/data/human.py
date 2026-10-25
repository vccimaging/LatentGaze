"""Prototype recordings: ``human_data_XXX.npz`` with ``measures`` (N, 16) and ``vectors`` (N, 3)."""

import glob
import os
import re
from typing import Dict, Tuple

import numpy as np
import torch

_PATTERN = re.compile(r"human_data_(\d+)\.npz$")


def subject_file(root: str, subject: int) -> str:
    return os.path.join(root, f"human_data_{subject:03d}.npz")


def find_subjects(root: str) -> Dict[int, str]:
    """Map subject id -> npz path for every ``human_data_XXX.npz`` in ``root``."""
    found = {}
    for path in glob.glob(os.path.join(root, "human_data_*.npz")):
        m = _PATTERN.search(os.path.basename(path))
        if m:
            found[int(m.group(1))] = path
    return dict(sorted(found.items()))


def load_subject(path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (measures (N, 16), vectors (N, 3)) as float32 tensors, as stored (no normalization)."""
    data = np.load(path)
    return (torch.tensor(data["measures"], dtype=torch.float32),
            torch.tensor(data["vectors"], dtype=torch.float32))
