"""
Compute covariance-based late-layer rank metrics from saved L8 residuals.

This script complements `collect_L8.py`, which saves merged residual matrices
as:

  {prompt_id: [n_layers, d_model]}

For each layer it computes:
  - participation ratio
  - effective rank
  - top-k PCA variance explained

These metrics are useful as a robustness check for the TwoNN-based intrinsic
dimensionality claim, because they operate directly on the same residual
matrices but use covariance-spectrum estimators instead of local neighbor
geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_residuals(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path) as data:
        residuals = {k: data[k] for k in data.files}
    prompt_ids = list(residuals.keys())
    if not prompt_ids:
        raise ValueError(f"No residuals found in {path}")
    return prompt_ids, residuals


def _subsample_prompt_ids(
    prompt_ids: list[str],
    max_prompts: int | None,
    seed: int,
) -> list[str]:
    if max_prompts is None or len(prompt_ids) <= max_prompts:
        return prompt_ids
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(prompt_ids), size=max_prompts, replace=False)
    return [prompt_ids[i] for i in sorted(idx.tolist())]


def _spectrum_metrics(X: np.ndarray) -> dict[str, float]:
    # Center prompts before computing the covariance spectrum.
    X = X.astype(np.float64, copy=False)
    X = X - X.mean(axis=0, keepdims=True)

    # Singular values of centered prompt matrix; covariance eigenvalues are s^2/(n-1).
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    if s.size == 0:
        return {
            "participation_ratio": float("nan"),
            "effective_rank": float("nan"),
            "pc1_ratio": float("nan"),
            "pc5_ratio": float("nan"),
            "pc10_ratio": float("nan"),
        }

    evals = (s ** 2) / max(X.shape[0] - 1, 1)
    total = float(evals.sum())
    if total <= 0:
        return {
            "participation_ratio": float("nan"),
            "effective_rank": float("nan"),
            "pc1_ratio": float("nan"),
            "pc5_ratio": float("nan"),
            "pc10_ratio": float("nan"),
        }

    p = evals / total
    entropy = -float(np.sum(p[p > 0] * np.log(p[p > 0])))
    participation_ratio = float((total ** 2) / np.sum(evals ** 2))

    return {
        "participation_ratio": participation_ratio,
        "effective_rank": float(np.exp(entropy)),
        "pc1_ratio": float(np.sum(evals[:1]) / total),
        "pc5_ratio": float(np.sum(evals[:5]) / total),
        "pc10_ratio": float(np.sum(evals[:10]) / total),
    }


def compute_rank_metrics(
    residuals: dict[str, np.ndarray],
    prompt_ids: list[str],
) -> dict[str, list[float]]:
    n_layers = residuals[prompt_ids[0]].shape[0]
    metrics = {
        "participation_ratio": [],
        "effective_rank": [],
        "pc1_ratio": [],
        "pc5_ratio": [],
        "pc10_ratio": [],
    }
    for layer_idx in range(n_layers):
        X = np.stack([residuals[pid][layer_idx] for pid in prompt_ids], axis=0)
        vals = _spectrum_metrics(X)
        for key, value in vals.items():
            metrics[key].append(value)
    return metrics


def summarize_late(metrics: dict[str, list[float]], late_fraction: float = 0.2) -> dict[str, float]:
    n_layers = len(metrics["participation_ratio"])
    start = max(int(np.floor(n_layers * (1.0 - late_fraction))), 0)
    summary = {"late_start_layer": start}
    for key, values in metrics.items():
        arr = np.asarray(values, dtype=np.float64)
        summary[f"{key}_last"] = float(arr[-1])
        summary[f"{key}_late_mean"] = float(np.nanmean(arr[start:]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute covariance-based rank metrics from merged L8 residuals.")
    parser.add_argument("--residuals-npz", type=Path, required=True, help="Path to merged L8_residuals.npz")
    parser.add_argument("--out-json", type=Path, default=None, help="Where to save metrics JSON")
    parser.add_argument("--max-prompts", type=int, default=2000, help="Optional prompt subsample for speed")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prompt_ids, residuals = _load_residuals(args.residuals_npz)
    prompt_ids = _subsample_prompt_ids(prompt_ids, args.max_prompts, args.seed)
    metrics = compute_rank_metrics(residuals, prompt_ids)
    summary = summarize_late(metrics)

    payload = {
        "residuals_npz": str(args.residuals_npz),
        "n_prompts_used": len(prompt_ids),
        "metrics": metrics,
        "summary": summary,
    }

    out_json = args.out_json or args.residuals_npz.with_name("L8_rank_metrics.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[l8-rank] wrote {out_json}")


if __name__ == "__main__":
    main()
