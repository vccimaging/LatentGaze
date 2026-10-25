# Added for "Low Latency Gaze Tracking via Latent Optical Sensing" (arXiv 2605.17990);
# not part of the original NVIDIA stylegan2-ada-pytorch release. Distributed with this
# derivative work under the NVIDIA Source Code License (LICENSE.txt). See MODIFICATIONS.md.

"""Gaze-preserving augmentation and FID protocol (`train.py --gaze_augment=1`).

Training images are stored at 288x288 (= CROP_SIZE + 2 * MAX_SHIFT) and the
networks operate at CROP_SIZE = 256:

* Training reals (`augment_reals`): a 256x256 window whose offset from the
  centred crop is drawn uniformly from {-16, ..., 16} px per axis and per
  sample, a circular aperture (radius 128 px, centred at 127.5) with the
  outside set to -1, then per-sample brightness (+U[-0.2, 0.2]) and contrast
  (x U[0.8, 1.2] around the per-channel in-aperture mean) jitter, a clamp to
  [-1, 1] and the aperture again. No flips or rotations, so the gaze direction
  of the image is preserved.
* Metrics (`metrics/metric_utils.py`): reals are centre-cropped 288 -> 256 and
  multiplied by the aperture (outside = 0 in uint8); fakes get the same
  aperture (outside = -1, i.e. 0 in uint8). No brightness/contrast jitter.
"""

import torch

#----------------------------------------------------------------------------

CROP_SIZE       = 256   # Network resolution.
MAX_SHIFT       = 16    # Max random translation (px) of the crop window.
APERTURE_RADIUS = 128   # Circular aperture radius (px), centred at (CROP_SIZE - 1) / 2.
BRIGHTNESS      = 0.2   # Additive brightness jitter, U[-BRIGHTNESS, BRIGHTNESS] (images in [-1, 1]).
CONTRAST        = 0.2   # Multiplicative contrast jitter, U[1 - CONTRAST, 1 + CONTRAST].

#----------------------------------------------------------------------------

def make_aperture_mask(device, size=CROP_SIZE, radius=APERTURE_RADIUS):
    """Binary circular mask, float32 tensor of shape [1, 1, size, size]."""
    y, x = torch.meshgrid(torch.arange(size, device=device, dtype=torch.float32),
                          torch.arange(size, device=device, dtype=torch.float32), indexing='ij')
    c = (size - 1) / 2
    return ((x - c) ** 2 + (y - c) ** 2 <= radius ** 2).float()[None, None]

def apply_aperture(img, mask):
    """Keep `img` inside the aperture and set it to -1 (black) outside. `img` in [-1, 1]."""
    return img * mask + (-1) * (1 - mask)

def center_crop(img, size=CROP_SIZE):
    """Centre crop of an NCHW tensor or array to size x size."""
    H, W = img.shape[2], img.shape[3]
    y0, x0 = (H - size) // 2, (W - size) // 2
    return img[:, :, y0:y0+size, x0:x0+size]

#----------------------------------------------------------------------------

def augment_reals(img, mask):
    """Gaze-preserving augmentation of a batch of training reals.

    img:  float tensor [B, C, H, W] in [-1, 1], H = W >= CROP_SIZE + 2 * MAX_SHIFT (288).
    mask: aperture from `make_aperture_mask` on the same device.
    Returns a float tensor [B, C, CROP_SIZE, CROP_SIZE] in [-1, 1].
    """
    B, _C, H, W = img.shape
    c0 = (H - CROP_SIZE) // 2 # 16 for 288 px images
    assert H == W and c0 >= MAX_SHIFT, f'--gaze_augment expects square images of at least {CROP_SIZE + 2 * MAX_SHIFT} px, got {H}x{W}'

    # Random translation: per-sample crop offset in [c0 - MAX_SHIFT, c0 + MAX_SHIFT].
    crops = []
    for i in range(B):
        dy = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (1,)).item()
        dx = torch.randint(-MAX_SHIFT, MAX_SHIFT + 1, (1,)).item()
        y0 = c0 + dy
        x0 = c0 + dx
        crops.append(img[i:i+1, :, y0:y0+CROP_SIZE, x0:x0+CROP_SIZE])
    img = torch.cat(crops, dim=0)
    img = apply_aperture(img, mask)

    # Per-sample brightness and contrast; the mean is taken over in-aperture pixels only.
    brightness = (torch.rand(B, 1, 1, 1, device=img.device) - 0.5) * (2 * BRIGHTNESS)
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=img.device) - 0.5) * (2 * CONTRAST)
    mean = (img * mask).sum(dim=[2, 3], keepdim=True) / mask.sum()
    img = (img - mean) * contrast + mean + brightness
    img = img.clamp(-1, 1)

    # Re-apply the aperture so that the background is exactly -1.
    return apply_aperture(img, mask)

#----------------------------------------------------------------------------
