#!/usr/bin/env python3
"""Direction stability check (0A analog) for Phase 0 multi-model steering.

Computes direction stability from existing per-worker acts data.
Since each worker processed an independent subset of ~300 records,
we can compute the corrective direction from each worker independently
and measure the cosine similarity between them (split-half reliability).

For paper-quality bootstrap (1000 resamples), we'd need per-record acts.
This split-half check is a lower bound on stability.

Output:
  results/exp8/plots/data/phase0_direction_stability.json

Usage:
  uv run python -m src.poc.exp8.direction_stability
  # Or on Modal (with volume):
  python -m src.poc.exp8.direction_stability --base-dir /results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.poc.cross_model.config import MODEL_REGISTRY, get_spec

MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "deepseek_v2_lite", "olmo2_7b"]


def compute_split_half_stability(model_name: str, base_dir: str = "results") -> dict | None:
    """Compute direction from each worker independently, measure cosine."""
    spec = get_spec(model_name)
    all_layers = list(range(1, spec.n_layers))
    d_model = spec.d_model

    work = Path(base_dir) / f"cross_model/{model_name}/directions_work/acts"
    act_files = sorted(work.glob("w*.npz"))
    if len(act_files) < 2:
        print(f"  {model_name}: only {len(act_files)} worker file(s), need 2+")
        return None

    # Compute direction per worker
    per_worker_dirs: list[dict[int, np.ndarray]] = []
    for f in act_files:
        with np.load(f) as d:
            dirs = {}
            for li in all_layers:
                it_sum = d[f"it_sum_{li}"].astype(np.float64)
                pt_sum = d[f"pt_sum_{li}"].astype(np.float64)
                n_it = int(d[f"it_count_{li}"])
                n_pt = int(d[f"pt_count_{li}"])
                if n_it == 0 or n_pt == 0:
                    continue
                mean_it = it_sum / n_it
                mean_pt = pt_sum / n_pt
                vec = mean_it - mean_pt
                norm = np.linalg.norm(vec)
                dirs[li] = vec / (norm + 1e-12)
            per_worker_dirs.append(dirs)

    # Compute pairwise cosine between worker directions at each layer
    n_workers = len(per_worker_dirs)
    layer_cosines = {}
    for li in all_layers:
        cosines = []
        for i in range(n_workers):
            for j in range(i + 1, n_workers):
                if li in per_worker_dirs[i] and li in per_worker_dirs[j]:
                    cos = float(np.dot(per_worker_dirs[i][li], per_worker_dirs[j][li]))
                    cosines.append(cos)
        if cosines:
            layer_cosines[li] = {
                "mean_cosine": float(np.mean(cosines)),
                "min_cosine": float(np.min(cosines)),
                "n_pairs": len(cosines),
            }

    # Also compare each worker to the combined direction
    combined_path = Path(base_dir) / f"cross_model/{model_name}/directions/corrective_directions.npz"
    worker_vs_combined = {}
    if combined_path.exists():
        with np.load(combined_path) as d:
            for li in all_layers:
                key = f"layer_{li}"
                if key in d:
                    combined_dir = d[key].astype(np.float64)
                    cosines = []
                    for w in per_worker_dirs:
                        if li in w:
                            cos = float(np.dot(w[li], combined_dir))
                            cosines.append(cos)
                    if cosines:
                        worker_vs_combined[li] = {
                            "mean_cosine": float(np.mean(cosines)),
                            "min_cosine": float(np.min(cosines)),
                        }

    # Summary for corrective layers
    corrective_layers = list(range(spec.corrective_onset, spec.n_layers))
    corr_cosines = [layer_cosines[li]["mean_cosine"] for li in corrective_layers if li in layer_cosines]
    corr_vs_combined = [worker_vs_combined[li]["mean_cosine"] for li in corrective_layers if li in worker_vs_combined]

    result = {
        "model": model_name,
        "n_workers": n_workers,
        "n_layers": spec.n_layers,
        "corrective_onset": spec.corrective_onset,
        "split_half_per_layer": {str(k): v for k, v in layer_cosines.items()},
        "worker_vs_combined_per_layer": {str(k): v for k, v in worker_vs_combined.items()},
        "corrective_summary": {
            "split_half_mean": float(np.mean(corr_cosines)) if corr_cosines else None,
            "split_half_min": float(np.min(corr_cosines)) if corr_cosines else None,
            "worker_vs_combined_mean": float(np.mean(corr_vs_combined)) if corr_vs_combined else None,
            "worker_vs_combined_min": float(np.min(corr_vs_combined)) if corr_vs_combined else None,
            "stable": bool(np.mean(corr_cosines) > 0.90) if corr_cosines else False,
        },
    }

    s = result["corrective_summary"]
    print(f"  {model_name}: split-half cosine = {s['split_half_mean']:.4f} "
          f"(min {s['split_half_min']:.4f}), "
          f"worker-vs-combined = {s['worker_vs_combined_mean']:.4f} "
          f"{'STABLE' if s['stable'] else 'UNSTABLE'}", flush=True)

    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-dir", default="results")
    args = p.parse_args()

    print("Computing direction stability (split-half) for all models...")
    all_results = {}
    for m in MODELS:
        result = compute_split_half_stability(m, args.base_dir)
        if result:
            all_results[m] = result

    # Save
    out_dir = Path("results/exp8/plots/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase0_direction_stability.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")

    # Summary table
    print("\n=== Direction Stability Summary ===")
    print(f"{'Model':<20} {'Split-Half':<12} {'vs Combined':<12} {'Status'}")
    print("-" * 60)
    for m in MODELS:
        if m in all_results:
            s = all_results[m]["corrective_summary"]
            status = "STABLE" if s["stable"] else "UNSTABLE"
            print(f"{m:<20} {s['split_half_mean']:.4f}       {s['worker_vs_combined_mean']:.4f}       {status}")
        else:
            print(f"{m:<20} N/A")


if __name__ == "__main__":
    main()
