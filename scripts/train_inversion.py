"""Train a StyleGAN inversion encoder.

    python scripts/train_inversion.py --config configs/inversion_gray.yaml
    accelerate launch --num_processes 2 scripts/train_inversion.py --config configs/inversion_rgb.yaml
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.config import parse_args
from latentgaze.engine import Run, evaluate_gaze, shape_filtered_load, validity_bce
from latentgaze.losses import LPIPS
from latentgaze.models import build_inversion, load_gaze_estimator
from latentgaze.models.generator import Generator
from latentgaze.normalization import gan_to_imagenet
from latentgaze.optim import build_scheduler
from latentgaze.utils import count_parameters, infinite_loader, set_requires_grad


def render(generator, w_plus, size):
    """Synthesize G(w_plus) and resize it to the training resolution."""
    image = generator(w_plus)
    if image.shape[-1] != size:
        image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False)
    return image


def inversion_losses(encoder, generator, estimator, lpips, image, clean, gaze, valid):
    """Unweighted loss terms of Eq. 10 (+ LPIPS) for one batch; image terms are masked by v."""
    v, v4 = valid.view(-1, 1), valid.view(-1, 1, 1, 1)
    w_plus, pred, logit = encoder(image)
    recon = render(generator, w_plus, clean.shape[-1])
    losses = {
        "gaze": F.l1_loss(pred * v, gaze * v),
        "valid": validity_bce(logit, valid),
        "mse": F.mse_loss(recon * v4, clean * v4),
    }
    if lpips is not None:
        rgb = (lambda x: x.repeat(1, 3, 1, 1)) if recon.shape[1] == 1 else (lambda x: x)
        losses["lpips"] = lpips(rgb(recon) * v4, rgb(clean) * v4)
    if estimator is not None:
        with torch.no_grad():
            target_gaze = estimator(gan_to_imagenet(clean))[0]
        losses["gaze_consistency"] = F.l1_loss(estimator(gan_to_imagenet(recon))[0] * v, target_gaze * v)
    return losses


def main(cfg):
    run = Run(cfg)
    t, w = cfg.train, cfg.loss
    train_loader, test_loader = run.loaders(cfg.data, t.batch_size, t.get("test_batch_size", 32), t.num_workers)

    encoder = build_inversion(cfg.model)
    if cfg.get("init_checkpoint"):
        shape_filtered_load(encoder, torch.load(cfg.init_checkpoint, map_location="cpu")["model"], run.info)
    run.info(f"Inversion encoder ({cfg.model.type}): {count_parameters(encoder):,} parameters")

    generator = Generator(cfg.generator, device=run.device)
    estimator = None
    if w.gaze_consistency > 0:
        estimator = load_gaze_estimator(cfg.gaze_estimator).to(run.device).eval()
        set_requires_grad(estimator, False)
    lpips = LPIPS().to(run.device) if w.lpips > 0 else None

    optimizer = torch.optim.AdamW(encoder.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = build_scheduler(optimizer, t.scheduler, run.total_steps(t.max_iters),
                                max_lr=t.lr, pct_start=t.pct_start)
    encoder, optimizer, scheduler = run.accelerator.prepare(encoder, optimizer, scheduler)

    best = float("inf")
    batches = infinite_loader(train_loader)
    encoder.train()
    for step in range(1, t.max_iters + 1):
        image, clean, gaze, valid = next(batches)
        losses = inversion_losses(encoder, generator, estimator, lpips, image, clean, gaze, valid)
        loss = sum(w[k] * l for k, l in losses.items())

        optimizer.zero_grad(set_to_none=True)
        run.accelerator.backward(loss)
        optimizer.step()
        scheduler.step()

        if step % t.log_every == 0:
            run.info(f"step {step} loss {loss.item():.4f} " +
                     " ".join(f"{k} {l.item():.4f}" for k, l in losses.items()) +
                     f" lr {scheduler.get_last_lr()[0]:.2e}")
        err = None
        if step % t.eval_every == 0 or step == t.max_iters:
            encoder.eval()
            err = evaluate_gaze(lambda x: encoder(x)[1], test_loader, run.accelerator)
            encoder.train()
            run.info(f"[eval] step {step} test angular error {err:.3f} deg")
        save_step = step % t.save_every == 0 or step == t.max_iters
        new_best = err is not None and err < best
        if new_best or save_step:
            state = dict(config=cfg.model.to_dict(), model=run.accelerator.unwrap_model(encoder).state_dict(),
                         meta={"step": step, "test_error_deg": err})
            if new_best:
                best = err
                run.save("best.pt", **state)
            if save_step:
                run.save(f"step_{step:06d}.pt", **state)
    run.info(f"done, best test error {best:.3f} deg")


if __name__ == "__main__":
    main(parse_args(__doc__))
