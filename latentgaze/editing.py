"""Gaze editing: invert, displace the latent, synthesize."""

import csv
import math
import os

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .geometry import pitchyaw_to_vector
from .models import load_gaze_estimator, load_inversion, load_steering
from .models.generator import Generator
from .normalization import gan_to_imagenet


class GazeEditor:
    def __init__(self, generator, inversion, controller, estimator, device="cpu", size=256):
        self.device, self.size = device, size
        self.generator = Generator(generator, device=device)
        self.inversion = load_inversion(inversion).to(device).eval()
        self.controller = load_steering(controller).to(device).eval()
        self.estimator = load_gaze_estimator(estimator).to(device).eval()
        self.grayscale = self.generator.img_channels == 1

    @torch.no_grad()
    def invert(self, image):
        return self.inversion(image)[0]

    @torch.no_grad()
    def render(self, w_plus):
        image = self.generator(w_plus)
        if image.shape[-1] != self.size:
            image = F.interpolate(image, size=(self.size, self.size), mode="bilinear", align_corners=False)
        return image

    @torch.no_grad()
    def gaze(self, image):
        return self.estimator(gan_to_imagenet(image))[0]

    @torch.no_grad()
    def steer(self, w_plus, g_src, g_tgt):
        return self.render(w_plus + self.controller(g_src, g_tgt, w_plus))


def target_from_degrees(pitch_deg: float, yaw_deg: float) -> torch.Tensor:
    """Gaze vector in the dataset convention (z < 0) for a pitch/yaw target in degrees."""
    py = torch.tensor([[math.radians(pitch_deg), math.radians(yaw_deg)]])
    return -pitchyaw_to_vector(py)


def load_subject_crops(subject_dir: str, csv_path: str, grayscale: bool, size: int = 256):
    """Valid right-eye crops of one subject, read from the per-eye JPG crops + CSV labels.

    CSV rows: frame, camXX.JPG, gr_x, gr_y, gr_z, gl_x, gl_y, gl_z, valid_r, valid_l.
    The full 45 mm crop is resized to ``size`` (no centre crop, no aperture). Returns
    (images in [-1, 1] (N, C, size, size), gaze vectors (N, 3)).
    """
    tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    images, gazes = [], []
    with open(csv_path) as f:
        for row in csv.reader(f):
            if not row or int(row[8]) == 0:
                continue
            cam = os.path.splitext(row[1])[0]
            img = tf(Image.open(os.path.join(subject_dir, row[0], f"{cam}_r.JPG")).convert("RGB"))
            images.append(img.mean(dim=0, keepdim=True) if grayscale else img)
            gazes.append(torch.tensor([float(x) for x in row[2:5]]))
    return torch.stack(images), torch.stack(gazes)
