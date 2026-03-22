from __future__ import annotations

import numpy as np


def participation_ratio(x: np.ndarray) -> float:
    """Effective dimension from eigenvalue concentration."""
    if x.size == 0 or x.shape[0] < 2:
        return float("nan")
    xc = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(xc, rowvar=False)
    evals = np.linalg.eigvalsh(cov)
    evals = np.clip(evals, 0.0, None)
    denom = np.square(evals).sum()
    if denom <= 0:
        return float("nan")
    return float(np.square(evals.sum()) / denom)


def mean_cosine_to_reference(x: np.ndarray, ref: np.ndarray) -> float:
    if x.size == 0 or ref.size == 0:
        return float("nan")
    x_norm = np.linalg.norm(x, axis=1) + 1e-12
    r_norm = np.linalg.norm(ref, axis=1) + 1e-12
    dots = np.sum(x * ref, axis=1)
    return float(np.mean(dots / (x_norm * r_norm)))


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0 or x.shape[0] != y.shape[0]:
        return float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    hsic_xy = np.square(x.T @ y).sum()
    hsic_xx = np.square(x.T @ x).sum()
    hsic_yy = np.square(y.T @ y).sum()
    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    return float(hsic_xy / denom)


def summarise_checkpoint_shift(
    baseline: dict[int, np.ndarray],
    ablated: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    summary: dict[int, dict[str, float]] = {}
    for layer_i, base_arr in baseline.items():
        abl_arr = ablated.get(layer_i)
        if abl_arr is None:
            continue
        n = min(len(base_arr), len(abl_arr))
        if n == 0:
            continue
        base_use = base_arr[:n]
        abl_use = abl_arr[:n]
        summary[layer_i] = {
            "cka_to_baseline": linear_cka(base_use, abl_use),
            "cosine_to_baseline": mean_cosine_to_reference(abl_use, base_use),
            "participation_ratio": participation_ratio(abl_use),
            "hidden_norm": float(np.linalg.norm(abl_use, axis=1).mean()),
        }
    return summary

