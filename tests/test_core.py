"""CPU unit tests: python -m pytest tests"""

import math
import os
import pickle
import sys

import cv2
import lmdb
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.config import load_config
from latentgaze.data import EyePatchLMDB, open_eye_patches
from latentgaze.geometry import angular_error, angular_loss, pitchyaw_to_vector, vector_to_pitchyaw
from latentgaze.models import (InversionTeacher, MeasurementDecoder, OpticalEncoder,
                               SteeringCompact, SteeringWPlus, SteeringWPlusIdentity)
from latentgaze.utils import count_parameters

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_parameter_counts_match_paper():
    assert count_parameters(MeasurementDecoder()) == 244_324           # Supp. D / F.3
    assert count_parameters(SteeringCompact()) == 133_569
    assert count_parameters(SteeringWPlus()) == 2_173_441
    assert count_parameters(SteeringWPlusIdentity()) == 2_772_033


def test_optical_encoder_is_binary_and_differentiable():
    enc = OpticalEncoder()
    masks = enc.binary_masks()
    assert masks.shape == (256 * 256, 16)
    assert set(masks.detach().unique().tolist()) <= {0.0, 1.0}
    assert masks[enc.aperture == 0].abs().sum() == 0                  # nothing outside the aperture
    y = enc(torch.rand(2, 1, 256, 256) * 2 - 1)
    assert y.shape == (2, 16)
    y.sum().backward()
    assert enc.weight.grad is not None and enc.weight.grad.abs().sum() > 0   # straight-through gradient
    assert enc.export().shape == (16, 256, 256)


def test_steering_identity_edit_is_zero_for_gated_models():
    g = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
    for model in (SteeringWPlus(), SteeringWPlusIdentity()):
        delta = model(g, g, torch.randn(4, 16, 512))
        assert delta.abs().max() == 0


def test_decoder_and_teacher_shapes():
    h, gaze, valid = MeasurementDecoder()(torch.rand(5, 16))
    assert h.shape == (5, 256) and gaze.shape == (5, 3) and valid.shape == (5, 1)
    w_plus, gaze, valid = InversionTeacher()(torch.rand(1, 1, 256, 256))
    assert w_plus.shape == (1, 14, 256) and torch.equal(w_plus[:, 0], w_plus[:, 13])


def test_geometry():
    py = torch.tensor([[0.1, -0.3], [-0.2, 0.5]])
    v = -pitchyaw_to_vector(py)                     # dataset convention: z < 0
    assert torch.allclose(vector_to_pitchyaw(v), py, atol=1e-6)
    assert angular_error(v, v).abs().max() < 1e-3
    a, b = torch.tensor([[1.0, 0, 0]]), torch.tensor([[0.0, 1, 0]])
    assert math.isclose(angular_error(a, b).item(), 90.0, abs_tol=1e-4)
    assert math.isclose(angular_loss(a, b).item(), math.pi / 2, abs_tol=1e-4)


def test_configs_load_and_inherit():
    cfg = load_config(os.path.join(ROOT, "configs/steering_rgb.yaml"), ["train.max_iters=7"])
    assert cfg.train.max_iters == 7
    assert cfg.model.latent_dim == 512
    assert cfg.train.scheduler == "warmup_plateau_cos"   # inherited from steering_gray.yaml


def test_config_overrides_are_checked():
    path = os.path.join(ROOT, "configs/masks_decoder.yaml")
    assert load_config(path, ["train.lr_masks=1e-4"]).train.lr_masks == pytest.approx(1e-4)   # not the string '1e-4'
    with pytest.raises(KeyError, match="max_iters"):
        load_config(path, ["train.max_iter=5"])     # typo: rejected with a suggestion
    assert load_config(path, ["+train.new_key=5"]).train.new_key == 5


@pytest.fixture()
def tiny_lmdb(tmp_path):
    path = str(tmp_path / "tiny_lmdb")
    env = lmdb.open(path, map_size=1 << 26)
    rng = np.random.default_rng(0)
    with env.begin(write=True) as txn:
        for i, size in enumerate([576, 288, 256]):
            rec = {"image": rng.integers(0, 256, (size, size, 3), dtype=np.uint8),
                   "label": np.array([0.1, -0.2, -0.97], np.float32), "valid": 1}
            txn.put(str(i).encode(), pickle.dumps(rec, protocol=4))
    env.close()
    return path


def test_dataset_preprocessing(tiny_lmdb):
    ds = EyePatchLMDB(tiny_lmdb, train=True)
    for i in range(len(ds)):
        inp, clean, gaze, valid = ds[i]
        assert inp.shape == clean.shape == (1, 256, 256)
        assert gaze.shape == (3,) and valid == 1
    inp, clean, _, _ = ds[1]
    assert clean[0, 0, 0] == -1.0                   # outside the aperture = normalized black
    rgb = EyePatchLMDB(tiny_lmdb, train=False, grayscale=False, normalization="imagenet", aperture=False)
    assert rgb[0][0].shape == (3, 256, 256)
    whole = EyePatchLMDB(tiny_lmdb, train=False, grayscale=False, normalization="gan", aperture=False,
                         centre_crop=False)
    with lmdb.open(tiny_lmdb, readonly=True, lock=False) as env, env.begin() as txn:
        rec = pickle.loads(txn.get(b"1"))
    expected = cv2.resize(rec["image"].astype(np.float32) / 255.0, (256, 256), interpolation=cv2.INTER_AREA)
    assert torch.allclose(whole[1][0], torch.from_numpy(expected).permute(2, 0, 1) * 2 - 1, atol=1e-6)
    with pytest.raises(ValueError, match="centre_crop"):
        EyePatchLMDB(tiny_lmdb, train=True, translate=16, centre_crop=False)


def test_folder_reader_matches_lmdb(tiny_lmdb, tmp_path):
    folder = tmp_path / "examples"
    (folder / "images").mkdir(parents=True)
    rows = ["image,test_index,subject,gaze_x,gaze_y,gaze_z,valid"]
    with lmdb.open(tiny_lmdb, readonly=True, lock=False) as env, env.begin() as txn:
        for i in (2, 0):
            rec = pickle.loads(txn.get(str(i).encode()))
            Image.fromarray(rec["image"]).save(folder / "images" / f"{i}.png")
            rows.append(f"images/{i}.png,{i},s,{','.join(repr(float(x)) for x in rec['label'])},{rec['valid']}")
    (folder / "labels.csv").write_text("\n".join(rows) + "\n")
    ref = EyePatchLMDB(tiny_lmdb, train=False)
    ds = open_eye_patches(str(folder), train=False)
    assert len(ds) == 2 and [ds.test_index(k) for k in range(2)] == [2, 0]
    for k in range(2):
        for a, b in zip(ds[k], ref[ds.test_index(k)]):
            assert torch.equal(torch.as_tensor(a), torch.as_tensor(b))
