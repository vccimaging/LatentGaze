"""Per-subject calibration of the measurement decoder (Sec. 5.3)."""

import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .geometry import angular_error, angular_loss, vector_to_pitchyaw
from .utils import seed_everything


def run_seed(seed: int, subject: int, k: int) -> int:
    """Seed of one (subject, K) run."""
    return seed + 1000 * subject + k


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))


def grid_select(pitchyaw_deg: np.ndarray, k: int) -> List[int]:
    """Indices of ``k`` samples approximating a regular grid over the pitch/yaw bounding box.

    A rows x cols grid (aspect ratio of the box, fewest surplus points) is laid out with a 5 %
    inset. While there are surplus targets, the closest pair is found and the target nearer the
    centroid is dropped. Each remaining target, in row-major order, is then snapped to its nearest
    unused sample. Deterministic.
    """
    py = np.asarray(pitchyaw_deg, dtype=np.float32)
    n = len(py)
    if k >= n:
        return list(range(n))
    if k == 1:
        centroid = py.mean(axis=0, keepdims=True).astype(np.float64)
        return [int(np.sqrt(((py - centroid) ** 2).sum(1)).argmin())]

    lo, hi = py.min(axis=0), py.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    aspect = float(span[1] / span[0])

    best, best_excess = (1, k), k
    for r in range(1, k + 1):
        c = max(1, round(r * aspect))
        for cols in (c, max(1, c - 1)):
            if r * cols >= k and r * cols - k < best_excess:
                best, best_excess = (r, cols), r * cols - k
        if r > math.isqrt(k) + 2:
            break
    rows, cols = best

    lo, hi, span = lo.astype(np.float64), hi.astype(np.float64), span.astype(np.float64)
    pitch = np.linspace(lo[0] + 0.05 * span[0], hi[0] - 0.05 * span[0], rows)
    yaw = np.linspace(lo[1] + 0.05 * span[1], hi[1] - 0.05 * span[1], cols)
    targets = np.array([[p, y] for p in pitch for y in yaw])

    while len(targets) > k:
        d = _pairwise_distances(targets)
        np.fill_diagonal(d, np.inf)
        i, j = np.unravel_index(d.argmin(), d.shape)
        centroid = targets.mean(axis=0)
        drop = i if np.linalg.norm(targets[i] - centroid) < np.linalg.norm(targets[j] - centroid) else j
        targets = np.delete(targets, drop, axis=0)

    selected, available = [], set(range(n))
    for t in targets:
        for idx in np.argsort(np.linalg.norm(py - t, axis=1)):
            if int(idx) in available:
                selected.append(int(idx))
                available.discard(int(idx))
                break
    return selected


def split_calibration(vectors: torch.Tensor, k: int) -> Tuple[List[int], List[int]]:
    """(calibration indices in grid order, held-out indices in data order) for a budget of ``k``."""
    py = np.degrees(vector_to_pitchyaw(vectors.float()).numpy())
    calib = grid_select(py, min(k, len(vectors) - 1))
    chosen = set(calib)
    return calib, [i for i in range(len(vectors)) if i not in chosen]


@torch.no_grad()
def angular_errors(model: torch.nn.Module, measures: torch.Tensor, labels: torch.Tensor,
                   batch_size: int = 64) -> np.ndarray:
    """Per-sample angle (degrees) between predicted and reference gaze. Switches ``model`` to eval."""
    model.eval()
    device = next(model.parameters()).device
    errors = [angular_error(model(y.to(device))[1], g.to(device)).cpu()
              for y, g in DataLoader(TensorDataset(measures, labels), batch_size=batch_size)]
    return torch.cat(errors).double().numpy()


def evaluate(model: torch.nn.Module, measures: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean angular error in degrees."""
    return float(np.mean(angular_errors(model, measures, labels)))


def calibrate(decoder: torch.nn.Module, calib_measures: torch.Tensor, calib_labels: torch.Tensor,
              eval_measures: torch.Tensor = None, eval_labels: torch.Tensor = None, *,
              epochs: int = 3500, lr: float = 1e-4, weight_decay: float = 0.5, grad_clip: float = 0.05,
              batch_size: int = 16, eval_every: int = 50, keep_best: bool = True, seed: int = 0):
    """Fine-tune a copy of ``decoder`` on K (measurement, unit gaze vector) pairs.

    Returns ``(model, history)``. ``history`` holds, for every evaluation, ``epoch``,
    ``train_loss``, ``eval_error`` (held-out set) and ``calib_error`` (calibration set), plus
    ``best_epoch``, ``best_eval`` and ``final_eval``. With ``keep_best`` the returned model has
    the weights of the lowest held-out error, otherwise those of the last epoch.
    """
    seed_everything(seed)
    device = next(decoder.parameters()).device
    model = copy.deepcopy(decoder)
    params = list(model.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(TensorDataset(calib_measures.float(), calib_labels.float()),
                        batch_size=min(batch_size, len(calib_measures)), shuffle=True, generator=generator)

    monitor = eval_measures is not None and eval_every > 0
    history = {"epoch": [], "train_loss": [], "eval_error": [], "calib_error": []}
    best_eval, best_epoch, best_state, final_eval = float("inf"), 0, None, float("nan")
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for y, g in loader:
            y, g = y.to(device), g.to(device)
            opt.zero_grad(set_to_none=True)
            loss = angular_loss(model(y)[1], g)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if monitor and (epoch == 1 or epoch == epochs or epoch % eval_every == 0):
            # every evaluation pass draws one number from the global RNG (DataLoader base seed),
            # so the evaluation schedule is part of the dropout random stream on CPU
            final_eval = evaluate(model, eval_measures, eval_labels)
            history["epoch"].append(epoch)
            history["train_loss"].append(float(np.mean(losses)))
            history["eval_error"].append(final_eval)
            history["calib_error"].append(evaluate(model, calib_measures, calib_labels))
            if final_eval < best_eval:
                best_eval, best_epoch = final_eval, epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.eval()
    if keep_best and best_state is not None:
        model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_eval=best_eval, final_eval=final_eval)
    return model, history


def calibrate_subject(decoder: torch.nn.Module, measures: torch.Tensor, vectors: torch.Tensor,
                      k: int, seed: int, **train_kwargs) -> Dict:
    """Grid split, calibration and evaluation of one subject for a budget of ``k`` points.

    ``measures`` (N, 16) and ``vectors`` (N, 3) as returned by ``latentgaze.data.human.load_subject``.
    """
    calib_idx, eval_idx = split_calibration(vectors, k)
    labels = F.normalize(vectors.float(), dim=1)
    calib = measures[calib_idx], labels[calib_idx]
    held_out = measures[eval_idx], labels[eval_idx]
    base_eval = evaluate(decoder, *held_out)
    model, history = calibrate(decoder, *calib, *held_out, seed=seed, **train_kwargs)
    return {"model": model, "history": history, "calib_idx": calib_idx, "eval_idx": eval_idx,
            "base_eval": base_eval, "best_eval": history["best_eval"], "best_epoch": history["best_epoch"],
            "final_eval": history["final_eval"], "calib_fit": evaluate(model, *calib)}
