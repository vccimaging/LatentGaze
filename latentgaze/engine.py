"""Helpers shared by the training scripts: run set-up, evaluation, checkpoints."""

import os

import torch
import torch.nn.functional as F
from accelerate import Accelerator

from .config import save_config
from .data import build_loader
from .geometry import angular_error
from .utils import get_logger, seed_everything


class Run:
    """Accelerator + output directory + logger for one training run."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.accelerator = Accelerator()
        seed_everything(cfg.seed + self.accelerator.process_index)
        self.dir = os.path.join(cfg.get("output_dir", "runs"), cfg.name)
        if self.is_main:
            os.makedirs(self.dir, exist_ok=True)
            save_config(cfg, os.path.join(self.dir, "config.yaml"))
        self.accelerator.wait_for_everyone()
        self.log = get_logger(cfg.name, os.path.join(self.dir, "train.log") if self.is_main else None)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def device(self):
        return self.accelerator.device

    def info(self, msg):
        if self.is_main:
            self.log.info(msg)

    def loaders(self, data_cfg, batch_size, test_batch_size=64, num_workers=8):
        train = build_loader(data_cfg, True, batch_size, num_workers)
        test = build_loader(data_cfg, False, test_batch_size, num_workers)
        return self.accelerator.prepare_data_loader(train), self.accelerator.prepare_data_loader(test)

    def total_steps(self, max_iters):
        """Scheduler length: schedulers prepared by accelerate step once per process."""
        return max_iters * self.accelerator.num_processes

    def save(self, filename, **obj):
        if self.is_main:
            torch.save(obj, os.path.join(self.dir, filename))


@torch.no_grad()
def evaluate_gaze(predict, loader, accelerator, input_index=0):
    """Mean angular error (degrees) over valid test samples.

    ``predict(batch_images) -> gaze (B, 3)``; ``input_index`` selects the image fed to the model
    (0 = network input, 1 = clean image; identical on the test split).
    """
    err_sum = torch.zeros((), device=accelerator.device)
    count = torch.zeros((), device=accelerator.device)
    for batch in loader:
        images, gaze, valid = batch[input_index], batch[2], batch[3]
        # every batch goes through the model, even without valid samples: under multi-GPU DDP all
        # processes must run the same forward passes, or they wait on each other indefinitely
        pred = predict(images)
        # gather_for_metrics drops the samples the distributed sampler repeats to even out the batches
        err, mask = accelerator.gather_for_metrics(
            torch.stack([angular_error(pred, gaze), (valid > 0.5).float()], dim=1)).unbind(1)
        mask = mask > 0.5
        err_sum += err[mask].sum()
        count += mask.sum()
    return (err_sum / count.clamp(min=1)).item()


def validity_bce(logit, valid):
    """BCE on validity logits, applied to all samples (logits clamped for stability)."""
    return F.binary_cross_entropy_with_logits(logit.clamp(-50.0, 50.0), valid.view(-1, 1))


def shape_filtered_load(model, state, logger=None):
    """Load the entries of ``state`` whose name and shape match ``model`` (warm starts)."""
    own = model.state_dict()
    kept = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(kept))
    model.load_state_dict(kept, strict=False)
    if logger is not None:
        logger(f"Warm start: loaded {len(kept)} tensors, skipped {len(skipped)}: {skipped}")
    return model
