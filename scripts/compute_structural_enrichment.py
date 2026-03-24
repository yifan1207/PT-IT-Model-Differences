#!/usr/bin/env python3
"""B0 Step 2: Compute structural-token enrichment ratio for each feature at corrective layers.

enrichment[f, l] = P(structural_token | feature_active at layer l) / P(structural_token)

Features with enrichment > 2.0 are considered "structural governance features"
(they fire preferentially when the model produces structural/discourse tokens).

Input:
  results/exp3/it_16k_l0_big_affine_t512/features.npz
      {record_id: object array of shape [n_steps, n_layers]}
      Each cell contains a sparse feature representation (indices of active features).
  results/exp3/it_16k_l0_big_affine_t512/results.jsonl
      Per-step generated tokens including token_str.

Output:
  results/exp6/structural_enrichment/layer_{20..33}.json
      {feature_idx_str: enrichment_ratio_float}

Usage:
    uv run python scripts/compute_structural_enrichment.py
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _classify_token(token_str: str) -> str:
    """Classify a token string into a word category (structural/discourse/content/etc)."""
    from src.poc.exp3.analysis.word_categories import classify_generated_tokens_by_word
    tokens = [{"token_str": token_str}]
    cats = classify_generated_tokens_by_word(tokens)
    return cats[0] if cats else "OTHER"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir",
                   default="results/exp3/it_16k_l0_big_affine_t512")
    p.add_argument("--layers", default="20,21,22,23,24,25,26,27,28,29,30,31,32,33")
    p.add_argument("--out-dir", default="results/exp6/structural_enrichment")
    p.add_argument("--min-activations", type=int, default=5,
                   help="Minimum activation count to compute enrichment (avoids noise).")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",")]
    layer_set = set(layers)

    # Load features.npz
    features_path = results_dir / "features.npz"
    print(f"Loading features from {features_path}...", flush=True)
    if not features_path.exists():
        print(
            f"WARNING: {features_path} not found — per-step feature activations were not saved "
            "for this exp3 run. Skipping structural enrichment computation.\n"
            "merge_governance_feature_sets.py will treat all GOVERNANCE-classified features as "
            "passing the enrichment threshold (using default enrichment = 3.0).",
            flush=True,
        )
        return
    features_data = np.load(str(features_path), allow_pickle=True)
    print(f"  {len(features_data.files)} records", flush=True)

    # Load results.jsonl to get generated_tokens and their token_str
    results_path = results_dir / "results.jsonl"
    print(f"Loading results from {results_path}...", flush=True)
    record_tokens: dict[str, list[dict]] = {}
    with open(results_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            rid = rec.get("record_id") or rec.get("prompt_id", "")
            if rid and "generated_tokens" in rec:
                record_tokens[rid] = rec["generated_tokens"]
    print(f"  {len(record_tokens)} records with generated_tokens", flush=True)

    # Precompute token categories for all records
    print("Classifying token types...", flush=True)
    record_categories: dict[str, list[str]] = {}
    for rid, tokens in record_tokens.items():
        record_categories[rid] = [_classify_token(t.get("token_str", "")) for t in tokens]

    # Count structural tokens globally (for baseline P(structural))
    total_tokens = 0
    structural_tokens = 0
    governance_cats = {"STRUCTURAL", "DISCOURSE"}
    for cats in record_categories.values():
        total_tokens += len(cats)
        structural_tokens += sum(1 for c in cats if c in governance_cats)

    p_structural = structural_tokens / max(total_tokens, 1)
    print(f"P(structural) = {p_structural:.4f} ({structural_tokens}/{total_tokens} tokens)", flush=True)

    # For each corrective layer, count feature activations and structural co-activations
    for layer_idx in layers:
        print(f"\nProcessing layer {layer_idx}...", flush=True)

        # n_active[f] = number of steps where feature f is active at this layer
        # n_active_structural[f] = number of those steps where token is structural/discourse
        n_active: dict[int, int] = defaultdict(int)
        n_active_structural: dict[int, int] = defaultdict(int)

        for record_id in features_data.files:
            if record_id not in record_categories:
                continue
            cats = record_categories[record_id]
            feats_arr = features_data[record_id]  # object array [n_steps, n_layers]

            if feats_arr.ndim < 2 or feats_arr.shape[0] == 0:
                continue

            n_steps = min(feats_arr.shape[0], len(cats))
            if layer_idx >= feats_arr.shape[1]:
                continue

            for step_idx in range(n_steps):
                cell = feats_arr[step_idx, layer_idx]
                if cell is None:
                    continue
                # cell can be: array of active feature indices, or a tuple (indices, values)
                if isinstance(cell, np.ndarray):
                    active_feats = cell.astype(int).tolist()
                elif isinstance(cell, (list, tuple)) and len(cell) == 2:
                    active_feats = np.asarray(cell[0]).astype(int).tolist()
                else:
                    continue

                is_structural = cats[step_idx] in governance_cats
                for f in active_feats:
                    n_active[f] += 1
                    if is_structural:
                        n_active_structural[f] += 1

        print(f"  {len(n_active)} active features found", flush=True)

        # Compute enrichment ratio
        enrichment: dict[str, float] = {}
        for f, count in n_active.items():
            if count < args.min_activations:
                continue
            p_struct_given_active = n_active_structural[f] / count
            ratio = p_struct_given_active / max(p_structural, 1e-6)
            enrichment[str(f)] = round(ratio, 4)

        out_path = out_dir / f"layer_{layer_idx}.json"
        with open(out_path, "w") as f_out:
            json.dump(enrichment, f_out)

        # Summary stats
        if enrichment:
            vals = list(enrichment.values())
            n_governance = sum(1 for v in vals if v >= 2.0)
            print(f"  {len(enrichment)} features with ≥{args.min_activations} activations", flush=True)
            print(f"  {n_governance} with enrichment ≥ 2.0 (governance candidates)", flush=True)
            print(f"  Max enrichment: {max(vals):.2f}, Mean: {sum(vals)/len(vals):.2f}", flush=True)
        print(f"  Saved → {out_path}", flush=True)

    print("\n=== Structural enrichment computation complete ===")


if __name__ == "__main__":
    main()
