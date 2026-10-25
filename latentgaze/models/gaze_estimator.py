"""Single-eye gaze estimators E."""

import torch
from torch import nn


def _backbone(arch: str, pretrained: bool, image_size: int):
    if arch == "efficientnet_b0":
        from efficientnet_pytorch import EfficientNet
        if pretrained:
            net = EfficientNet.from_pretrained("efficientnet-b0", image_size=image_size)
        else:
            net = EfficientNet.from_name("efficientnet-b0", image_size=image_size)
        net._fc = nn.Identity()
        return net, 1280
    if arch == "resnet18":
        from torchvision.models import resnet18
        net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Identity()
        return net, 512
    raise ValueError(f"Unsupported gaze estimator architecture '{arch}'")


def _to_grayscale_stem(net: nn.Module, arch: str) -> None:
    """Replace the first convolution by a 1-channel one initialised with the mean RGB filter.

    For EfficientNet the replacement is a plain ``Conv2d(1, 32, 3, stride=2)`` without the
    "same" padding of the original stem (this is how the released grayscale E was built).
    """
    if arch == "efficientnet_b0":
        old = net._conv_stem
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        net._conv_stem = new
    elif arch == "resnet18":
        old = net.conv1
        new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        net.conv1 = new


class GazeEstimator(nn.Module):
    def __init__(self, arch: str = "efficientnet_b0", in_channels: int = 1, pretrained: bool = False,
                 dropout: float = 0.0, image_size: int = 256):
        super().__init__()
        self.backbone, feat = _backbone(arch, pretrained, image_size)
        if in_channels == 1:
            _to_grayscale_stem(self.backbone, arch)
        self.gaze_head = nn.Sequential(
            nn.Linear(feat, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256),
            nn.Linear(256, 3),
        )
        self.valid_head = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """x: ImageNet-normalized image. Returns (gaze (B, 3), validity logit (B, 1))."""
        f = self.backbone(x)
        return self.gaze_head(f), self.valid_head(f)


def build_gaze_estimator(cfg, pretrained: bool = None) -> GazeEstimator:
    return GazeEstimator(arch=cfg.get("arch", "efficientnet_b0"),
                         in_channels=cfg.get("in_channels", 1),
                         pretrained=cfg.get("pretrained", False) if pretrained is None else pretrained,
                         dropout=cfg.get("dropout", 0.0),
                         image_size=cfg.get("image_size", 256))


def load_gaze_estimator(path: str, map_location="cpu") -> GazeEstimator:
    ckpt = torch.load(path, map_location=map_location)
    model = build_gaze_estimator(ckpt["config"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model
