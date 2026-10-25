"""Train the 16 binary optical masks and the measurement decoder P (Sec. 3.1).

    python scripts/train_masks.py --config configs/masks_decoder.yaml
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.config import parse_args
from latentgaze.engine import Run, evaluate_gaze, validity_bce
from latentgaze.geometry import angular_loss
from latentgaze.models import MeasurementDecoder, OpticalEncoder, load_inversion
from latentgaze.optim import build_scheduler
from latentgaze.utils import count_parameters, infinite_loader, set_requires_grad


def distillation_losses(optical, decoder, teacher, image, clean, gaze, valid):
    """Unweighted loss terms of Eq. 6 for one batch."""
    h, pred, logit = decoder(optical(image))
    losses = {"valid": validity_bce(logit, valid)}
    mask = valid > 0.5
    if mask.any():
        with torch.no_grad():
            w_t, gaze_t, _ = teacher(clean)
        losses["gaze"] = angular_loss(pred[mask], gaze[mask])
        losses["distill_gaze"] = angular_loss(pred[mask], gaze_t[mask])
        losses["distill_feat"] = F.mse_loss(h[mask], w_t[mask, 0])
    return losses


def main(cfg):
    run = Run(cfg)
    t, w = cfg.train, cfg.loss
    train_loader, test_loader = run.loaders(cfg.data, t.batch_size, t.get("test_batch_size", 64), t.num_workers)

    teacher = load_inversion(cfg.teacher).to(run.device).eval()
    set_requires_grad(teacher, False)

    optical = OpticalEncoder(**cfg.optical_encoder)
    decoder = MeasurementDecoder(**cfg.decoder)
    heads = "random"
    if cfg.get("init_heads_from_teacher", True):
        decoder.gaze_head.load_state_dict(teacher.gaze_head.state_dict())
        decoder.valid_head.load_state_dict(teacher.valid_head.state_dict())
        heads = "initialized from the teacher"
    run.info(f"Optical encoder: {count_parameters(optical):,} mask values, "
             f"decoder: {count_parameters(decoder):,} parameters, gaze and validity heads {heads}")

    total = run.total_steps(t.max_iters)
    sched_args = t.get("scheduler_args") or {}
    sched_args = sched_args.to_dict() if hasattr(sched_args, "to_dict") else dict(sched_args)

    def scheduler(optimizer):  # onecycle peaks at the group's learning rate
        return build_scheduler(optimizer, t.scheduler, total, max_lr=optimizer.param_groups[0]["lr"], **sched_args)

    opt_dec = torch.optim.AdamW(decoder.parameters(), lr=t.lr_decoder, weight_decay=t.wd_decoder)
    opt_mask = torch.optim.AdamW(optical.parameters(), lr=t.lr_masks, weight_decay=t.wd_masks)
    optimizers = [opt_dec, opt_mask]
    schedulers = [scheduler(opt_dec), scheduler(opt_mask)]

    acc = run.accelerator
    decoder = acc.prepare(decoder)
    optical = acc.prepare(optical)
    optimizers = [acc.prepare_optimizer(o) for o in optimizers]
    schedulers = [acc.prepare_scheduler(s) for s in schedulers]
    params = [p for m in (optical, decoder) for p in m.parameters()]

    def predict(x):
        return decoder(optical(x))[1]

    best = float("inf")
    batches = infinite_loader(train_loader)
    optical.train()
    decoder.train()
    for step in range(1, t.max_iters + 1):
        image, clean, gaze, valid = next(batches)
        if t.get("input_noise", 0.0) > 0:  # Gaussian noise on the encoder input, std in [0, 1] intensity units
            image = image + 2.0 * t.input_noise * torch.randn_like(image)
        losses = distillation_losses(optical, decoder, teacher, image, clean, gaze, valid)
        loss = sum(w[k] * l for k, l in losses.items())

        for o in optimizers:
            o.zero_grad(set_to_none=True)
        acc.backward(loss)
        if t.grad_clip > 0:
            acc.clip_grad_norm_(params, t.grad_clip)
        for o, s in zip(optimizers, schedulers):
            o.step()
            s.step()

        if step % t.log_every == 0:
            run.info(f"step {step} loss {loss.item():.4f} " +
                     " ".join(f"{k} {l.item():.4f}" for k, l in losses.items()))
        err = None
        if step % t.eval_every == 0 or step == t.max_iters:
            optical.eval()
            decoder.eval()
            err = evaluate_gaze(predict, test_loader, acc)
            optical.train()
            decoder.train()
            run.info(f"[eval] step {step} test angular error {err:.3f} deg")
        save_step = step % t.save_every == 0 or step == t.max_iters
        new_best = err is not None and err < best
        if new_best or save_step:
            state = dict(config={"optical_encoder": {k: cfg.optical_encoder[k] for k in
                                                     ("num_masks", "param_size", "image_size")},
                                 "decoder": cfg.decoder.to_dict()},
                         optical_encoder=acc.unwrap_model(optical).state_dict(),
                         decoder=acc.unwrap_model(decoder).state_dict(),
                         meta={"step": step, "test_error_deg": err})
            if new_best:
                best = err
                run.save("best.pt", **state)
            if save_step:
                run.save(f"step_{step:06d}.pt", **state)
    run.info(f"done, best test error {best:.3f} deg")


if __name__ == "__main__":
    main(parse_args(__doc__))
