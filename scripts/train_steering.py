"""Train a gaze-steering controller (Sec. 3.2).

    python scripts/train_steering.py --config configs/steering_gray.yaml
    python scripts/train_steering.py --config configs/steering_rgb.yaml
"""

import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from latentgaze.config import parse_args
from latentgaze.engine import Run
from latentgaze.geometry import angular_error
from latentgaze.losses import LPIPS
from latentgaze.models import build_steering, load_gaze_estimator, load_inversion
from latentgaze.models.generator import Generator
from latentgaze.normalization import gan_to_imagenet
from latentgaze.optim import build_scheduler
from latentgaze.utils import count_parameters, infinite_loader, set_requires_grad


def three_channels(x):
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


class SteeringObjective:
    """Frozen networks + loss terms of the three steering steps."""

    def __init__(self, generator, inversion, estimator, lpips, weights, dino=None):
        self.generator, self.inversion, self.estimator = generator, inversion, estimator
        self.lpips, self.dino, self.w = lpips, dino, weights

    def edit(self, controller, image, g_src, g_tgt):
        """Invert ``image``, displace its latent by C(g_src, g_tgt) and render at the input size."""
        with torch.no_grad():
            w_plus = self.inversion(image)[0]
        edited = self.generator(w_plus + controller(g_src, g_tgt, w_plus))
        if edited.shape[-1] != image.shape[-1]:
            edited = F.interpolate(edited, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return edited, w_plus

    def gaze(self, image):
        return self.estimator(gan_to_imagenet(image))[0]

    def forward_step(self, controller, image, clean, gaze, valid, perm):
        v, v4 = valid.view(-1, 1), valid.view(-1, 1, 1, 1)
        g_tgt = gaze[perm]
        edited, w_src = self.edit(controller, image, gaze, g_tgt)
        e3 = three_channels(edited) * v4
        losses = {
            "gaze_fwd": F.l1_loss(self.gaze(edited) * v, g_tgt * v),
            "percep": self.w.perceptual * (self.lpips(e3, three_channels(clean[perm]) * v4)
                                           + 0.5 * self.lpips(e3, three_channels(clean) * v4)),
        }
        if self.dino is not None:
            losses["dino_fwd"] = self.w.dino * self.dino(three_channels(edited), three_channels(clean))
        return losses, edited, w_src

    def roundtrip_step(self, controller, edited, clean, gaze, valid, perm):
        v, v4 = valid.view(-1, 1), valid.view(-1, 1, 1, 1)
        back, _ = self.edit(controller, edited.detach(), gaze[perm], gaze)
        back_in = gan_to_imagenet(back)
        losses = {
            "gaze_back": F.l1_loss(self.estimator(back_in)[0] * v, gaze * v),
            "mse": F.mse_loss(back_in * v4, gan_to_imagenet(clean) * v4),
        }
        if self.dino is not None:
            losses["dino_back"] = self.w.dino * self.dino(three_channels(back_in), three_channels(clean))
        return losses

    def identity_step(self, controller, gaze, w_src):
        return {"identity": self.w.identity * controller(gaze, gaze, w_src).pow(2).mean()}


def main(cfg):
    run = Run(cfg)
    t, w = cfg.train, cfg.loss
    acc = run.accelerator
    train_loader, test_loader = run.loaders(cfg.data, t.batch_size, t.batch_size, t.num_workers)

    inversion = load_inversion(cfg.inversion).to(run.device).eval()
    estimator = load_gaze_estimator(cfg.gaze_estimator).to(run.device).eval()
    set_requires_grad(inversion, False)
    set_requires_grad(estimator, False)
    dino = None
    if w.get("dino", 0) > 0:
        from latentgaze.losses.dino import DINOv2Loss
        dino = DINOv2Loss().to(run.device)
    objective = SteeringObjective(Generator(cfg.generator, device=run.device), inversion, estimator,
                                  LPIPS().to(run.device), w, dino)

    controller = build_steering(cfg.model)
    run.info(f"Steering controller ({cfg.model.type}): {count_parameters(controller):,} parameters")

    optimizer = torch.optim.AdamW(controller.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    scheduler = build_scheduler(optimizer, t.scheduler, t.max_iters)
    controller, optimizer = acc.prepare(controller, optimizer)

    def optimize(losses):
        acc.backward(sum(losses.values()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return losses

    batches = infinite_loader(train_loader)
    controller.train()
    for step in range(1, t.max_iters + 1):
        image, clean, gaze, valid = next(batches)
        perm = torch.randperm(image.shape[0], device=image.device)

        losses, edited, w_src = objective.forward_step(controller, image, clean, gaze, valid, perm)
        optimize(losses)
        losses.update(optimize(objective.roundtrip_step(controller, edited, clean, gaze, valid, perm)))
        if w.identity > 0:
            losses.update(optimize(objective.identity_step(controller, gaze, w_src)))
        scheduler.step()

        if step % t.log_every == 0:
            run.info(f"step {step} " + " ".join(f"{k} {l.item():.4f}" for k, l in losses.items()) +
                     f" lr {scheduler.get_last_lr()[0]:.2e}")
        if step % t.save_every == 0 or step == t.max_iters:
            fwd, bwd = evaluate(objective, controller, test_loader, acc, t.get("eval_batches", 20))
            run.info(f"[eval] step {step} forward {fwd:.2f} deg, round trip {bwd:.2f} deg (E-measured)")
            run.save(f"step_{step:06d}.pt", config=cfg.model.to_dict(),
                     model=acc.unwrap_model(controller).state_dict(),
                     meta={"step": step, "eval_forward_deg": fwd, "eval_back_deg": bwd})
    run.info("done")


@torch.no_grad()
def evaluate(objective, controller, loader, acc, max_batches):
    """Quick check on test patches: angular error of E on forward and round-trip edits."""
    errs = torch.zeros(2, device=acc.device)
    count = torch.zeros((), device=acc.device)
    for i, (image, _, gaze, valid) in enumerate(loader):
        if i == max_batches:
            break
        mask = valid > 0.5
        g_tgt = gaze[torch.randperm(gaze.shape[0], device=gaze.device)]
        edited, _ = objective.edit(controller, image, gaze, g_tgt)
        back, _ = objective.edit(controller, edited, g_tgt, gaze)
        errs[0] += angular_error(objective.gaze(edited)[mask], g_tgt[mask]).sum()
        errs[1] += angular_error(objective.gaze(back)[mask], gaze[mask]).sum()
        count += mask.sum()
    errs, count = acc.reduce(errs, "sum"), acc.reduce(count, "sum")
    return (errs / count.clamp(min=1)).tolist()


if __name__ == "__main__":
    main(parse_args(__doc__))
