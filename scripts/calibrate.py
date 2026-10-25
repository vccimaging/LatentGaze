"""Per-subject calibration on the prototype recordings (Sec. 5.3, Table 2).

    python scripts/calibrate.py --config configs/calibration.yaml
"""

import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.calibration import calibrate_subject, run_seed
from latentgaze.config import parse_args, save_config
from latentgaze.data.human import find_subjects, load_subject
from latentgaze.models import load_decoder
from latentgaze.utils import get_logger

METRICS = ("base_eval", "best_eval", "final_eval", "calib_fit")


def as_list(value):
    return list(value) if isinstance(value, (list, tuple)) else [value]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    summary = []
    for k in sorted({r["K"] for r in rows}):
        group = [r for r in rows if r["K"] == k]
        entry = {"K": k, "subjects": len(group)}
        for key in METRICS:
            values = np.array([r[key] for r in group])
            entry[f"{key}_mean"] = round(float(values.mean()), 4)
            entry[f"{key}_std"] = round(float(values.std(ddof=1)), 4) if len(values) > 1 else float("nan")
        summary.append(entry)
    return summary


def main(cfg):
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_config(cfg, os.path.join(cfg.output_dir, "config.yaml"))
    log = get_logger(log_file=os.path.join(cfg.output_dir, "log.txt"))

    decoder_config = torch.load(cfg.checkpoint, map_location="cpu")["config"]["decoder"]
    decoder = load_decoder(cfg.checkpoint)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    decoder.to(device)

    files = find_subjects(cfg.data.root)
    if not files:
        raise FileNotFoundError(f"no human_data_XXX.npz files in {cfg.data.root}")
    subjects = list(files) if cfg.data.subjects == "all" else [int(s) for s in as_list(cfg.data.subjects)]
    c = cfg.calibration
    train_kwargs = dict(epochs=int(c.epochs), lr=float(c.lr), weight_decay=float(c.weight_decay),
                        grad_clip=float(c.grad_clip), batch_size=int(c.batch_size),
                        eval_every=int(c.eval_every), keep_best=bool(c.keep_best))
    if cfg.save_models:
        os.makedirs(os.path.join(cfg.output_dir, "models"), exist_ok=True)

    rows, curves = [], []
    for subject in subjects:
        measures, vectors = load_subject(files[subject])
        for k in as_list(cfg.budgets):
            k = min(int(k), len(measures) - 1)
            seed = run_seed(cfg.seed, subject, k)
            res = calibrate_subject(decoder, measures, vectors, k, seed, **train_kwargs)
            rows.append({"subject": subject, "K": k, "n_eval": len(res["eval_idx"]), "best_epoch": res["best_epoch"],
                         **{key: round(res[key], 4) for key in METRICS}})
            h = res["history"]
            curves += [{"subject": subject, "K": k, "epoch": e, "train_loss": round(l, 6),
                        "eval_error": round(ev, 4), "calib_error": round(ce, 4)}
                       for e, l, ev, ce in zip(h["epoch"], h["train_loss"], h["eval_error"], h["calib_error"])]
            log.info(f"S{subject:03d} K={k:2d}: {res['base_eval']:.2f} -> best {res['best_eval']:.2f} "
                     f"@ epoch {res['best_epoch']} (final {res['final_eval']:.2f}) deg "
                     f"on {len(res['eval_idx'])} held-out")
            if cfg.save_models:
                torch.save({"config": {"decoder": decoder_config},
                            "decoder": {n: v.cpu() for n, v in res["model"].state_dict().items()},
                            "meta": {"subject": subject, "K": k, "seed": seed, "calib_idx": res["calib_idx"],
                                     "eval_idx": res["eval_idx"], "best_epoch": res["best_epoch"],
                                     "best_eval": res["best_eval"], "final_eval": res["final_eval"],
                                     "keep_best": train_kwargs["keep_best"], "source": cfg.checkpoint}},
                           os.path.join(cfg.output_dir, "models", f"S{subject:03d}_K{k}.pt"))
            write_csv(os.path.join(cfg.output_dir, "results.csv"), rows)
            write_csv(os.path.join(cfg.output_dir, "curves.csv"), curves)

    summary = summarize(rows)
    write_csv(os.path.join(cfg.output_dir, "summary.csv"), summary)
    for s in summary:
        log.info(f"K={s['K']:2d} ({s['subjects']} subjects): "
                 f"best {s['best_eval_mean']:.2f} +- {s['best_eval_std']:.2f}, "
                 f"final {s['final_eval_mean']:.2f} +- {s['final_eval_std']:.2f}, "
                 f"uncalibrated {s['base_eval_mean']:.2f} deg")


if __name__ == "__main__":
    main(parse_args(__doc__))
