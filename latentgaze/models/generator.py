"""Frozen StyleGAN2 generator; images are ``synthesis(w_plus + w_mean)``."""

import os
import sys
import warnings

import torch
from torch import nn
from torch.utils import cpp_extension

_STYLEGAN_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party",
                                             "stylegan2_ada_pytorch"))


def _import_stylegan():
    if _STYLEGAN_DIR not in sys.path:
        sys.path.insert(0, _STYLEGAN_DIR)
    import dnnlib
    import legacy
    from torch_utils.ops import bias_act, upfirdn2d
    if torch.cuda.is_available() and cpp_extension.CUDA_HOME is None and not upfirdn2d._inited:
        # the custom CUDA kernels cannot be compiled without a CUDA toolkit
        warnings.warn("CUDA toolkit not found (CUDA_HOME): StyleGAN2 uses its slower reference ops")
        bias_act._inited = upfirdn2d._inited = True
    return legacy


def load_stylegan_pickle(path: str, key: str = "G_ema") -> nn.Module:
    legacy = _import_stylegan()
    with open(path, "rb") as f:
        return legacy.load_network_pkl(f)[key]


class Generator(nn.Module):
    def __init__(self, network_pkl: str, mean_samples: int = 10000, mean_seed: int = 0, device="cpu"):
        super().__init__()
        G = load_stylegan_pickle(network_pkl).eval().requires_grad_(False).to(device)
        self.synthesis = G.synthesis
        self.num_ws, self.w_dim, self.img_channels, self.img_resolution = \
            G.num_ws, G.w_dim, G.img_channels, G.img_resolution
        rng = torch.Generator().manual_seed(mean_seed)
        z = torch.randn(mean_samples, G.z_dim, generator=rng).to(device)
        with torch.no_grad():
            w_mean = G.mapping(z, None, truncation_psi=1.0).mean(0, keepdim=True)
        self.register_buffer("w_mean", w_mean)  # (1, num_ws, w_dim)

    def forward(self, w_plus: torch.Tensor) -> torch.Tensor:
        """w_plus: (B, num_ws, w_dim) offset from w_mean. Returns images in [-1, 1].

        The highest-resolution blocks run in fp16 on GPU (as during training); on CPU, where
        half-precision convolutions are unavailable, everything runs in fp32.
        """
        return self.synthesis(w_plus + self.w_mean, noise_mode="const",
                              force_fp32=w_plus.device.type != "cuda")

    def train(self, mode: bool = True):
        # the generator is always frozen and in eval mode
        return super().train(False)
