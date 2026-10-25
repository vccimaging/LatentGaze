r"""Round-trip evaluation of a gaze-steering controller (Sec. 5.2).

    python scripts/eval_steering.py --crops data/crops --generator weights/stylegan2_rgb512.pkl \
        --inversion weights/inversion_wplus_rgb.pt --controller weights/steering_wplus_identity_rgb.pt \
        --gaze-estimator weights/gaze_estimator_rgb_resnet18.pt --out results/steering_rgb.csv
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.editing import GazeEditor, load_subject_crops
from latentgaze.geometry import angular_error


def psnr(a, b):
    mse = (((a + 1) / 2 - (b + 1) / 2) ** 2).mean(dim=[1, 2, 3])
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def valid_counts(csv_paths):
    """Number of valid right-eye crops (CSV rows with valid_r = 1) of every subject."""
    counts = []
    for path in csv_paths:
        with open(path) as f:
            counts.append(sum(1 for row in csv.reader(f) if row and int(row[8]) == 1))
    return counts


def paper_draw(counts, n_src, n_tgt):
    """(source, target) indices of every subject as drawn in the paper's evaluation.

    The paper's evaluation seeded torch with 42, permuted the first subject's samples once (a
    visualisation cell) and then, for every subject, drew one int64 (the base seed of the DataLoader
    that loaded the subject) before the permutation that picks the sources and targets.
    """
    g = torch.Generator().manual_seed(42)
    torch.randperm(counts[0], generator=g)
    draws = []
    for n in counts:
        torch.empty((), dtype=torch.int64).random_(generator=g)
        perm = torch.randperm(n, generator=g)
        draws.append((perm[:n_src], perm[n_src:n_src + n_tgt]))
    return draws


def ssim(a, b):
    from skimage.metrics import structural_similarity
    a = ((a + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    b = ((b + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    return torch.tensor([structural_similarity(x, y, channel_axis=2, data_range=1.0) for x, y in zip(a, b)])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", required=True, help="folder with subjectXXXX_ethz/ crops and subjectXXXX_ethz.csv")
    ap.add_argument("--split", default="data/splits/test.csv", help="split list whose subjects are evaluated")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--inversion", required=True)
    ap.add_argument("--controller", required=True)
    ap.add_argument("--gaze-estimator", required=True)
    ap.add_argument("--sources", type=int, default=20)
    ap.add_argument("--targets", type=int, default=20)
    ap.add_argument("--draw", choices=["paper", "seeded"], default="paper",
                    help="paper: the source/target pairs of the paper; seeded: a draw from --seed")
    ap.add_argument("--seed", type=int, default=0, help="seed of --draw seeded")
    ap.add_argument("--subjects", nargs="*", default=None, help="restrict to these subject names")
    ap.add_argument("--out", default=None, help="optional per-subject CSV")
    args = ap.parse_args()

    import lpips
    device = "cuda" if torch.cuda.is_available() else "cpu"
    editor = GazeEditor(args.generator, args.inversion, args.controller, args.gaze_estimator, device)
    lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    rgb = (lambda x: x.repeat(1, 3, 1, 1)) if editor.grayscale else (lambda x: x)

    with open(args.split, newline="") as f:
        split = {r["subject"] for r in csv.DictReader(f)}
    csv_paths = sorted(p for p in glob.glob(os.path.join(args.crops, "subject*.csv"))
                       if os.path.basename(p)[:-4].split("_")[0] in split)
    if args.draw == "paper":
        draws = paper_draw(valid_counts(csv_paths), args.sources, args.targets)
    rows = []
    for k, csv_path in enumerate(csv_paths):
        subject = os.path.splitext(os.path.basename(csv_path))[0]
        if args.subjects and subject not in args.subjects:
            continue
        images, gazes = load_subject_crops(csv_path[:-4], csv_path, editor.grayscale)
        if args.draw == "paper":
            src_idx, tgt_idx = draws[k]
        else:
            perm = torch.randperm(len(images), generator=torch.Generator().manual_seed(args.seed + k))
            src_idx, tgt_idx = perm[:args.sources], perm[args.sources:args.sources + args.targets]
        g_tgt = gazes[tgt_idx].to(device)
        m = {key: [] for key in ("forward_deg", "backward_deg", "psnr", "ssim", "lpips")}
        for i in src_idx:
            w_src = editor.invert(images[i:i + 1].to(device))
            src = editor.render(w_src)
            g_src = editor.gaze(src).expand(len(g_tgt), -1)
            w_src = w_src.expand(len(g_tgt), -1, -1)

            forward = editor.steer(w_src, g_src, g_tgt)
            g_fwd = editor.gaze(forward)
            back = editor.steer(editor.invert(forward), g_fwd, g_src)

            src = src.expand(len(g_tgt), -1, -1, -1)
            m["forward_deg"].append(angular_error(g_fwd, g_tgt).cpu())
            m["backward_deg"].append(angular_error(editor.gaze(back), g_src).cpu())
            m["psnr"].append(psnr(back, src).cpu())
            m["ssim"].append(ssim(back, src))
            m["lpips"].append(lpips_fn(rgb(back), rgb(src)).flatten().cpu())
        row = {"subject": subject, **{key: torch.cat(v).float().mean().item() for key, v in m.items()}}
        rows.append(row)
        print(" ".join(f"{key} {val:.4f}" if isinstance(val, float) else val for key, val in row.items()))

    keys = [k for k in rows[0] if k != "subject"]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    print("mean over subjects: " + " ".join(f"{k} {v:.4f}" for k, v in mean.items()))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["subject"] + keys)
            writer.writeheader()
            writer.writerows(rows + [{"subject": "mean", **mean}])


if __name__ == "__main__":
    main()
