"""Learning-rate schedules."""

import math

from torch.optim import lr_scheduler


def build_scheduler(optimizer, name: str, total_steps: int, **kw):
    """
    onecycle            OneCycleLR (cosine), peak = ``max_lr``, warm-up fraction ``pct_start``.
                        Used by the gaze estimator and the inversion encoders.
    plateau_linear      Constant for the first ``plateau`` (0.8) of training, then linear decay to
                        ``end_factor`` (0.1) x, after an optional linear warm-up from 0.01x over the
                        first ``warmup`` (0.0) fraction. Used by the optical encoder + decoder.
    warmup_plateau_cos  2 % linear warm-up, constant until 65 %, cosine decay to 1 %.
                        Used by the gaze-steering controllers.
    constant            No schedule.
    """
    if name == "onecycle":
        return lr_scheduler.OneCycleLR(optimizer, max_lr=kw["max_lr"], total_steps=total_steps,
                                       pct_start=kw.get("pct_start", 0.01), anneal_strategy="cos")
    if name == "plateau_linear":
        plateau_end = int(total_steps * kw.get("plateau", 0.8))
        warmup = int(total_steps * kw.get("warmup", 0.0))  # optional linear warm-up from 0.01x
        decay = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=kw.get("end_factor", 0.1),
                                      total_iters=total_steps - plateau_end)
        if warmup == 0:
            return lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=plateau_end), decay],
                milestones=[plateau_end],
            )
        return lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup),
                        lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=plateau_end - warmup), decay],
            milestones=[warmup, plateau_end],
        )
    if name == "warmup_plateau_cos":
        warmup = int(total_steps * 0.02)
        decay_start = int(total_steps * 0.65)

        def factor(step):
            if step < warmup:
                return step / max(warmup, 1)
            if step < decay_start:
                return 1.0
            progress = (step - decay_start) / max(total_steps - decay_start, 1)
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return lr_scheduler.LambdaLR(optimizer, factor)
    if name == "constant":
        return lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    raise ValueError(f"Unknown scheduler '{name}'")
