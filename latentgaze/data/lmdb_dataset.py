"""Eye-patch datasets from an LMDB or a folder of PNG files; items are ``(input, clean, gaze, valid)``."""

import csv
import os
import pickle

import cv2
import lmdb
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from ..normalization import IMAGENET_MEAN, IMAGENET_STD

GAN_MEAN, GAN_STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


class RandomGaussianNoise:
    def __init__(self, min_std=0.01, max_std=0.10):
        self.min_std, self.max_std = min_std, max_std

    def __call__(self, x):
        std = torch.empty(1).uniform_(self.min_std, self.max_std).item()
        return x + torch.randn(x.size()) * std


def circular_aperture(size: int = 256) -> torch.Tensor:
    """Binary disk of radius size/2 centred at (size/2, size/2), shape (1, size, size)."""
    yy, xx = np.ogrid[:size, :size]
    c = size / 2.0
    return torch.from_numpy(((xx - c) ** 2 + (yy - c) ** 2 <= c ** 2).astype(np.float32)).unsqueeze(0)


class EyePatchLMDB(Dataset):
    def __init__(self, path: str, train: bool, grayscale: bool = True, normalization: str = "gan",
                 augment: bool = True, aug_prob: float = 0.5, translate: int = 16,
                 aperture: bool = True, centre_crop: bool = True, crop_size: int = 256, base_size: int = 288):
        self.path = path
        self.train = train
        self.grayscale = grayscale
        self.augment = augment and train
        self.translate = translate if train else 0
        self.centre_crop = centre_crop
        if self.translate and not centre_crop:
            raise ValueError("translate needs centre_crop=True (the whole-patch resize has no margin to shift into)")
        self.aperture = aperture
        self.crop_size = crop_size
        self.base_size = base_size
        self.env = None
        self.length = self._count()

        mean, std = (GAN_MEAN, GAN_STD) if normalization == "gan" else (IMAGENET_MEAN, IMAGENET_STD)
        normalize = transforms.Normalize(mean=mean, std=std)
        self.to_tensor = transforms.Compose([transforms.ToTensor(), normalize])
        if self.augment:
            self.augment_tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=11, sigma=(0.1, 10.0))], p=aug_prob),
                transforms.RandomApply([RandomGaussianNoise(0.01, 0.10)], p=aug_prob),
                transforms.Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                normalize,
            ])
        black = torch.tensor([-m / s for m, s in zip(mean, std)])
        self.black = black.mean().view(1, 1, 1) if grayscale else black.view(3, 1, 1)
        self.aperture_mask = circular_aperture(crop_size)

    def __len__(self):
        return self.length

    def _count(self):
        with lmdb.open(self.path, readonly=True, lock=False, readahead=False) as env:
            return env.stat()["entries"]

    def test_index(self, idx):
        """Position of sample ``idx`` in the full test set (the LMDB key)."""
        return idx

    def subject(self, idx):
        """Subject of sample ``idx`` if known without reading the record ("" otherwise)."""
        return ""

    def _read(self, idx):
        if self.env is None:
            self.env = lmdb.open(self.path, readonly=True, lock=False, readahead=False)
        with self.env.begin() as txn:
            raw = txn.get(str(idx).encode())
        if raw is None:
            raise KeyError(f"Sample {idx} not found in {self.path}")
        return pickle.loads(raw)

    def _geometry(self, image):
        """Resize to the 288 px base size, translate and centre-crop (or area-resize the whole patch to
        the crop size with ``centre_crop=False``). Returns (image, resized or cropped?)."""
        h = image.shape[0]
        if h == self.crop_size:
            return image, False
        if not self.centre_crop:
            return cv2.resize(image, (self.crop_size, self.crop_size), interpolation=cv2.INTER_AREA), True
        if h != self.base_size:
            interp = cv2.INTER_AREA if h > self.base_size else cv2.INTER_LINEAR
            image = cv2.resize(image, (self.base_size, self.base_size), interpolation=interp)
        if self.translate > 0:
            tx, ty = np.random.randint(-self.translate, self.translate + 1, size=2)
            shift = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, shift, (self.base_size, self.base_size),
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        s = (self.base_size - self.crop_size) // 2
        return image[s:s + self.crop_size, s:s + self.crop_size], True

    def __getitem__(self, idx):
        sample = self._read(idx)
        image = np.asarray(sample["image"], dtype=np.float32) / 255.0
        image, cropped = self._geometry(image)

        clean = self.to_tensor(image)
        if self.augment:
            inp = self.augment_tf(Image.fromarray((image * 255).astype(np.uint8)))
        else:
            inp = clean.clone()

        if self.grayscale:
            inp = inp.mean(dim=0, keepdim=True)
            clean = clean.mean(dim=0, keepdim=True)
        if self.aperture and cropped:
            m = self.aperture_mask
            inp = inp * m + self.black * (1.0 - m)
            clean = clean * m + self.black * (1.0 - m)

        gaze = torch.as_tensor(np.asarray(sample["label"], dtype=np.float32))
        valid = torch.as_tensor(float(sample["valid"]), dtype=torch.float32)
        return inp, clean, gaze, valid

    def __del__(self):
        if getattr(self, "env", None) is not None:  # also when __init__ raised before opening it
            self.env.close()


class EyePatchFolder(EyePatchLMDB):
    """Eye patches stored as PNG files, listed in ``<root>/labels.csv`` with the columns

        image, test_index, subject, gaze_x, gaze_y, gaze_z, valid

    (``image`` relative to ``root``, ``test_index`` = key of the patch in the full test LMDB; other
    columns, e.g. frame / camera / eye, are ignored). PNG is lossless, so a patch gives exactly the
    tensors of the LMDB record it was exported from.
    """

    def _count(self):
        with open(os.path.join(self.path, "labels.csv"), newline="") as f:
            self.rows = list(csv.DictReader(f))
        return len(self.rows)

    def test_index(self, idx):
        return int(self.rows[idx]["test_index"])

    def subject(self, idx):
        return self.rows[idx].get("subject", "")

    def _read(self, idx):
        row = self.rows[idx]
        image = np.asarray(Image.open(os.path.join(self.path, row["image"])).convert("RGB"))
        label = np.array([float(row[k]) for k in ("gaze_x", "gaze_y", "gaze_z")], dtype=np.float32)
        return {"image": image, "label": label, "valid": int(row["valid"])}


def open_eye_patches(path: str, **kwargs) -> EyePatchLMDB:
    """``EyePatchFolder`` if ``path`` holds a ``labels.csv``, otherwise ``EyePatchLMDB``."""
    if os.path.isfile(os.path.join(path, "labels.csv")):
        return EyePatchFolder(path, **kwargs)
    return EyePatchLMDB(path, **kwargs)


def worker_init_fn(worker_id: int) -> None:
    """Give every DataLoader worker its own numpy stream (used for the random translation)."""
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed + worker_id)


def build_dataset(data_cfg, train: bool) -> EyePatchLMDB:
    return open_eye_patches(
        data_cfg.train_lmdb if train else data_cfg.test_lmdb,
        train=train,
        grayscale=data_cfg.get("grayscale", True),
        normalization=data_cfg.get("normalization", "gan"),
        augment=data_cfg.get("augment", True),
        aug_prob=data_cfg.get("aug_prob", 0.5),
        translate=data_cfg.get("translate", 16),
        aperture=data_cfg.get("aperture", True),
        centre_crop=data_cfg.get("centre_crop", True),
    )


def build_loader(data_cfg, train: bool, batch_size: int, num_workers: int = 8, shuffle=None):
    dataset = build_dataset(data_cfg, train)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train if shuffle is None else shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
