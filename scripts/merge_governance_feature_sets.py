#!/usr/bin/env python3
"""B0 Step 4: Merge classification, enrichment, and diff-score results into feature sets.

Combines:
  Method 1+2: classified as GOVERNANCE AND enrichment ≥ 2.0
              → sorted by enrichment ratio, sliced to top-10/50/100/500/all
  Method 3:   IT-amplified: diff_score > 0, top-100 by magnitude
              IT-suppressed: diff_score < 0, top-100 by |magnitude| (for B3 controls)
  Controls:   random_100 (random feature indices, seed 42)
              method12_content_top100 (classified CONTENT, enrichment-sorted)

Output:
  results/exp6/governance_feature_sets.json
  {
    "layer_20": {
      "method12_all": [feat_idx, ...],
      "method12_top500": [...],
      "method12_top100": [...],
      "method12_top50": [...],
      "method12_top10": [...],
      "method3_it_amplified_top100": [...],
      "method3_it_suppressed_top100": [...],
      "random_100": [...],
      "method12_content_top100": [...],
    }, ...
  }

Usage:
    uv run python scripts/merge_governance_feature_sets.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--classifications-dir", default="results/exp6/feature_classifications")
    p.add_argument("--enrichment-dir", default="results/exp6/structural_enrichment")
    p.add_argument("--diff-scores-dir", default="results/exp6/feature_diff_scores")
    p.add_argument("--layers", default="20,21,22,23,24,25,26,27,28,29,30,31,32,33")
    p.add_argument("--out-path", default="results/exp6/governance_feature_sets.json")
    p.add_argument("--enrichment-threshold", type=float, default=2.0)
    p.add_argument("--n-features", type=int, default=16384)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    cls_dir = Path(args.classifications_dir)
    enr_dir = Path(args.enrichment_dir)
    diff_dir = Path(args.diff_scores_dir)
    layers = [int(x) for x in args.layers.split(",")]

    out: dict[str, dict] = {}

    for layer_idx in layers:
        layer_key = f"layer_{layer_idx}"
        print(f"\nLayer {layer_idx}:", flush=True)

        # Load classification results
        cls_path = cls_dir / f"layer_{layer_idx}.json"
        classifications: dict[str, str] = {}
        if cls_path.exists():
            with open(cls_path) as f:
                classifications = json.load(f)
            print(f"  Classifications: {len(classifications)} features", flush=True)
        else:
            print(f"  WARNING: No classifications found at {cls_path}", flush=True)

        # Load enrichment ratios
        enr_path = enr_dir / f"layer_{layer_idx}.json"
        enrichments: dict[str, float] = {}
        if enr_path.exists():
            with open(enr_path) as f:
                enrichments = json.load(f)
            print(f"  Enrichment: {len(enrichments)} features", flush=True)
        else:
            print(f"  WARNING: No enrichment data at {enr_path}", flush=True)

        # Load diff scores
        diff_path = diff_dir / f"layer_{layer_idx}.json"
        diff_scores: dict[str, float] = {}
        if diff_path.exists():
            with open(diff_path) as f:
                diff_scores = json.load(f)
            print(f"  Diff scores: {len(diff_scores)} features", flush=True)
        else:
            print(f"  WARNING: No diff scores at {diff_path}", flush=True)

        # ── Method 1+2: GOVERNANCE classification AND enrichment ≥ threshold ──
        # Default enrichment 3.0 (> threshold 2.0) when enrichment file is missing,
        # so all GOVERNANCE-classified features pass when structural enrichment couldn't be computed.
        default_enrichment = 3.0 if not enr_path.exists() else 0.0
        gov_feats = []
        for fidx_str, cat in classifications.items():
            if cat == "GOVERNANCE":
                enr = enrichments.get(fidx_str, default_enrichment)
                if enr >= args.enrichment_threshold:
                    gov_feats.append((int(fidx_str), enr))
        # Sort by enrichment ratio descending
        gov_feats.sort(key=lambda x: -x[1])
        gov_indices = [f for f, _ in gov_feats]

        print(f"  Method 1+2 (GOVERNANCE ∩ enrichment≥{args.enrichment_threshold}): {len(gov_indices)} features", flush=True)

        # ── Method 1+2: CONTENT features (for B3 control) ────────────────────
        content_feats = []
        for fidx_str, cat in classifications.items():
            if cat == "CONTENT":
                enr = enrichments.get(fidx_str, 0.0)
                content_feats.append((int(fidx_str), enr))
        content_feats.sort(key=lambda x: -x[1])
        content_indices = [f for f, _ in content_feats]

        # ── Method 3: IT-amplified and IT-suppressed ──────────────────────────
        it_amplified = [(int(fidx), score) for fidx, score in diff_scores.items() if float(score) > 0]
        it_amplified.sort(key=lambda x: -x[1])
        it_amplified_indices = [f for f, _ in it_amplified[:100]]

        it_suppressed = [(int(fidx), score) for fidx, score in diff_scores.items() if float(score) < 0]
        it_suppressed.sort(key=lambda x: x[1])  # most negative first
        it_suppressed_indices = [f for f, _ in it_suppressed[:100]]

        print(f"  Method 3 IT-amplified: {len(it_amplified_indices)}", flush=True)
        print(f"  Method 3 IT-suppressed: {len(it_suppressed_indices)}", flush=True)

        # ── Random control ────────────────────────────────────────────────────
        all_feature_indices = list(range(args.n_features))
        random.Random(args.random_seed + layer_idx).shuffle(all_feature_indices)
        random_100 = all_feature_indices[:100]

        # ── Package ───────────────────────────────────────────────────────────
        out[layer_key] = {
            "method12_all": gov_indices,
            "method12_top500": gov_indices[:500],
            "method12_top100": gov_indices[:100],
            "method12_top50": gov_indices[:50],
            "method12_top10": gov_indices[:10],
            "method3_it_amplified_top100": it_amplified_indices,
            "method3_it_suppressed_top100": it_suppressed_indices,
            "random_100": random_100,
            "method12_content_top100": content_indices[:100],
        }

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nGovernance feature sets saved → {out_path}")

    # Global summary
    print("\nSummary:")
    for layer_key, sets in out.items():
        gov_all = len(sets["method12_all"])
        gov100 = len(sets["method12_top100"])
        amp100 = len(sets["method3_it_amplified_top100"])
        print(f"  {layer_key}: {gov_all} governance total, top100={gov100}, IT-amplified={amp100}")


if __name__ == "__main__":
    main()
