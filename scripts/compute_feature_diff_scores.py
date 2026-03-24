#!/usr/bin/env python3
"""B0 Step 3: Compute IT-PT feature diff_scores (Method 3).

diff_score[f, l] = (count_IT[f,l] / total_IT_tokens[l]) - (count_PT[f,l] / total_PT_tokens[l])

Positive diff_score: feature fires MORE in IT than PT (IT-amplified).
Negative diff_score: feature fires LESS in IT (PT-amplified, IT-suppressed).

Input:
  results/exp3/it_16k_l0_big_affine_t512/feature_importance_summary.npz
  results/exp3/pt_16k_l0_big_affine_t512/feature_importance_summary.npz
      Keys: count_l{i} [16384] and sum_l{i} [16384] for i in 0..33

Output:
  results/exp6/feature_diff_scores/layer_{20..33}.json
      {feature_idx_str: diff_score_float}  (sorted by |diff_score| descending)

Usage:
    uv run python scripts/compute_feature_diff_scores.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--it-dir", default="results/exp3/it_16k_l0_big_affine_t512")
    p.add_argument("--pt-dir", default="results/exp3/pt_16k_l0_big_affine_t512")
    p.add_argument("--layers", default="20,21,22,23,24,25,26,27,28,29,30,31,32,33")
    p.add_argument("--out-dir", default="results/exp6/feature_diff_scores")
    args = p.parse_args()

    it_dir = Path(args.it_dir)
    pt_dir = Path(args.pt_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]

    # Load feature importance summaries
    it_npz_path = it_dir / "feature_importance_summary.npz"
    pt_npz_path = pt_dir / "feature_importance_summary.npz"

    print(f"Loading IT from {it_npz_path}...", flush=True)
    it_data = np.load(str(it_npz_path))
    print(f"Loading PT from {pt_npz_path}...", flush=True)
    pt_data = np.load(str(pt_npz_path))

    for layer_idx in layers:
        it_counts = it_data[f"count_l{layer_idx}"].astype(np.float64)  # [n_features]
        pt_counts = pt_data[f"count_l{layer_idx}"].astype(np.float64)

        # Total tokens = sum of all feature activations as a proxy
        # (more precisely: total_tokens = number of positions × batch, but count
        # is number of times each feature fires, so we use the sum of counts as proxy)
        it_total = it_counts.sum()
        pt_total = pt_counts.sum()

        if it_total < 1 or pt_total < 1:
            print(f"  Layer {layer_idx}: empty counts, skipping", flush=True)
            continue

        # Activation frequency per feature
        it_freq = it_counts / it_total
        pt_freq = pt_counts / pt_total
        diff = it_freq - pt_freq  # positive = IT uses more, negative = PT uses more

        # Save as JSON (sorted by abs value descending for easy inspection)
        scores = {str(i): round(float(diff[i]), 6) for i in range(len(diff))}
        sorted_scores = dict(sorted(scores.items(), key=lambda x: -abs(float(x[1]))))

        out_path = out_dir / f"layer_{layer_idx}.json"
        with open(out_path, "w") as f:
            json.dump(sorted_scores, f)

        # Summary
        n_it_amplified = int((diff > 0).sum())
        n_pt_amplified = int((diff < 0).sum())
        top10 = sorted(scores.items(), key=lambda x: -float(x[1]))[:5]
        bot10 = sorted(scores.items(), key=lambda x: float(x[1]))[:5]
        print(f"Layer {layer_idx}: {n_it_amplified} IT-amplified, {n_pt_amplified} PT-amplified", flush=True)
        print(f"  Top IT-amplified: {[(k, f'{v:.4f}') for k, v in top10]}", flush=True)
        print(f"  Top PT-amplified: {[(k, f'{v:.4f}') for k, v in bot10]}", flush=True)
        print(f"  Saved → {out_path}", flush=True)

    print("\n=== Feature diff scores complete ===")


if __name__ == "__main__":
    main()
