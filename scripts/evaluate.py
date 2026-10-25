"""Gaze error of P, P_T or E on the test subjects (Table 1).

    python scripts/evaluate.py masks_decoder weights/masks_decoder.pt --test-data data/lmdb/test_lmdb
    python scripts/evaluate.py inversion weights/inversion_teacher.pt --test-data data/lmdb/test_lmdb
    python scripts/evaluate.py gaze_estimator weights/gaze_estimator_gray.pt --test-data data/lmdb/test_lmdb
"""

import argparse
import csv
import math
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.data import open_eye_patches
from latentgaze.geometry import angular_error, vector_to_pitchyaw
from latentgaze.models import load_gaze_estimator, load_inversion, load_masks_decoder


def in_channels(model):
    """Input channels of the first convolution (1 for the grayscale models)."""
    return next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d)).in_channels


def build(kind, path, device):
    """Returns (predict(images) -> gaze, dataset kwargs, decoder or None)."""
    if kind == "masks_decoder":
        optical, decoder = load_masks_decoder(path)
        optical, decoder = optical.to(device).eval(), decoder.to(device).eval()
        return (lambda x: decoder(optical(x))[1]), dict(grayscale=True, normalization="gan"), decoder
    if kind == "inversion":
        model = load_inversion(path).to(device).eval()
        return (lambda x: model(x)[1]), dict(grayscale=in_channels(model) == 1, normalization="gan"), None
    if kind == "gaze_estimator":
        model = load_gaze_estimator(path).to(device).eval()
        return (lambda x: model(x)[0]), dict(grayscale=in_channels(model) == 1, normalization="imagenet"), None
    raise ValueError(kind)


def profile(decoder, device):
    params = sum(p.numel() for p in decoder.parameters())
    msg = f"decoder parameters: {params:,}"
    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(decoder, torch.zeros(1, decoder.norm.normalized_shape[0], device=device))
        flops.unsupported_ops_warnings(False)
        msg += f", {2 * flops.total() / 1e6:.3f} MFLOPs (2 x MACs of linear and normalization layers)"
    except ImportError:
        msg += " (install fvcore to count FLOPs)"
    print(msg)


COLUMNS = ["test_index", "subject", "gaze_x", "gaze_y", "gaze_z", "pitch_deg", "yaw_deg",
           "pred_x", "pred_y", "pred_z", "error_deg"]


def write_predictions(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        for row in rows:
            writer.writerow(row[:2] + [f"{v:.6f}" for v in row[2:]])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["masks_decoder", "inversion", "gaze_estimator"])
    ap.add_argument("checkpoint")
    ap.add_argument("--test-data", required=True,
                    help="test LMDB or folder with labels.csv")
    ap.add_argument("--aperture", type=int, default=None,
                    help="apply the circular aperture to 288 px patches (default: 1 for grayscale models, "
                         "0 for colour models, as in their training)")
    ap.add_argument("--centre-crop", type=int, default=None,
                    help="centre-crop 288 px patches to 256 px instead of resizing the whole patch (default: 1 for "
                         "grayscale models, 0 for colour models, as in their training)")
    ap.add_argument("--max-angle", type=float, nargs="*", default=[])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--profile", action="store_true", help="print decoder parameter / FLOP count")
    ap.add_argument("--save-predictions", default=None, help="write per-sample predictions to this CSV")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predict, data_kw, decoder = build(args.kind, args.checkpoint, device)
    if args.profile and decoder is not None:
        profile(decoder, device)
    aperture = data_kw["grayscale"] if args.aperture is None else bool(args.aperture)
    centre_crop = data_kw["grayscale"] if args.centre_crop is None else bool(args.centre_crop)
    dataset = open_eye_patches(args.test_data, train=False, augment=False, aperture=aperture,
                               centre_crop=centre_crop, **data_kw)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    test_index, subjects, gazes, preds, errors, pitchyaw = [], [], [], [], [], []
    start = 0
    for image, _, gaze, valid in loader:
        mask = valid > 0.5
        kept = [start + i for i in torch.nonzero(mask).flatten().tolist()]
        test_index += [dataset.test_index(i) for i in kept]
        subjects += [dataset.subject(i) for i in kept]
        start += len(valid)
        if not mask.any():  # an empty batch breaks some CUDA kernels
            continue
        gaze = gaze[mask].to(device)
        pred = predict(image[mask].to(device))
        gazes.append(gaze.cpu())
        preds.append(pred.cpu())
        errors.append(angular_error(pred, gaze).cpu())
        pitchyaw.append(vector_to_pitchyaw(gaze).cpu() * 180.0 / math.pi)
    if not errors:
        sys.exit(f"no valid samples in {args.test_data}")
    gazes, preds, errors, pitchyaw = torch.cat(gazes), torch.cat(preds), torch.cat(errors), torch.cat(pitchyaw)

    print(f"valid samples: {errors.mean():.4f} deg  (n={len(errors)})")
    for limit in args.max_angle:
        sel = (pitchyaw.abs() <= limit + 1e-4).all(dim=1)
        print(f"|pitch|,|yaw| <= {limit:g} deg: {errors[sel].mean():.4f} deg  (n={int(sel.sum())})")
    if args.save_predictions:
        write_predictions(args.save_predictions, [[i, s, *g, *py, *p, e] for i, s, g, py, p, e in zip(
            test_index, subjects, gazes.tolist(), pitchyaw.tolist(), preds.tolist(), errors.tolist())])
        print(f"saved {len(errors)} predictions to {args.save_predictions}")


if __name__ == "__main__":
    main()
