"""Train a single-eye gaze estimator E.

    python scripts/train_gaze_estimator.py --config configs/gaze_estimator_gray.yaml
    python scripts/train_gaze_estimator.py --config configs/gaze_estimator_rgb.yaml
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.config import parse_args
from latentgaze.engine import Run, evaluate_gaze, validity_bce
from latentgaze.models import build_gaze_estimator
from latentgaze.optim import build_scheduler
from latentgaze.utils import count_parameters, infinite_loader


def main(cfg):
    run = Run(cfg)
    t = cfg.train
    train_loader, test_loader = run.loaders(cfg.data, t.batch_size, t.get("test_batch_size", 64), t.num_workers)

    model = build_gaze_estimator(cfg.model)
    run.info(f"Gaze estimator {cfg.model.arch}: {count_parameters(model):,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = build_scheduler(optimizer, t.scheduler, run.total_steps(t.max_iters),
                                max_lr=t.lr, pct_start=t.pct_start)
    model, optimizer, scheduler = run.accelerator.prepare(model, optimizer, scheduler)

    best = float("inf")
    batches = infinite_loader(train_loader)
    model.train()
    for step in range(1, t.max_iters + 1):
        image, _, gaze, valid = next(batches)
        pred, logit = model(image)
        mask = valid > 0.5
        loss_gaze = F.l1_loss(pred[mask], gaze[mask]) if mask.any() else pred.sum() * 0
        loss = loss_gaze + t.alpha_valid * validity_bce(logit, valid)

        optimizer.zero_grad(set_to_none=True)
        run.accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        if step % t.log_every == 0:
            run.info(f"step {step} loss {loss.item():.4f} gaze_l1 {loss_gaze.item():.4f} "
                     f"lr {scheduler.get_last_lr()[0]:.2e}")
        err = None
        if step % t.eval_every == 0 or step == t.max_iters:
            model.eval()
            err = evaluate_gaze(lambda x: model(x)[0], test_loader, run.accelerator)
            model.train()
            run.info(f"[eval] step {step} test angular error {err:.3f} deg")
        save_step = step % t.save_every == 0 or step == t.max_iters
        new_best = err is not None and err < best
        if new_best or save_step:
            state = dict(config=cfg.model.to_dict() | {"pretrained": False},
                         model=run.accelerator.unwrap_model(model).state_dict(),
                         meta={"step": step, "test_error_deg": err})
            if new_best:
                best = err
                run.save("best.pt", **state)
            if save_step:
                run.save(f"step_{step:06d}.pt", **state)
    run.info(f"done, best test error {best:.3f} deg")


if __name__ == "__main__":
    main(parse_args(__doc__))
