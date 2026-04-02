#!/usr/bin/env python3
"""1B: Commitment delay under steering — post-processing.

Reads logit_lens_top1_{condition}.npz files produced during the A1 α-sweep
(with --collect-logit-lens) and computes the commitment layer at each α value.

Commitment layer for a generated step = earliest layer ℓ such that:
  top1[ℓ] == top1[n_layers-1]  AND  top1[ℓ'] == top1[n_layers-1] for all ℓ' > ℓ
("no flip-back" requirement, matching collect_L1L2.py)

Output per model:
  results/cross_model/{model}/exp6/commitment_vs_alpha.json

Usage:
  python scripts/phase0_commitment_vs_alpha.py --model-name llama31_8b
  python scripts/phase0_commitment_vs_alpha.py  # all models
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec


def _commitment_layer(top1_row: np.ndarray) -> int:
    """Compute commitment layer for one generated step.

    Args:
        top1_row: [n_layers] int array — top-1 predicted token at each layer.

    Returns:
        Earliest layer ℓ where top1[ℓ] == top1[-1] and all subsequent layers
        also match top1[-1].  Returns n_layers-1 if no early commitment.
    """
    n_layers = len(top1_row)
    final_tok = top1_row[-1]
    for l in range(n_layers):
        if top1_row[l] == final_tok:
            # Check stability: all subsequent layers must also match
            if all(top1_row[l2] == final_tok for l2 in range(l + 1, n_layers)):
                return l
    return n_layers - 1


def process_model(model_name: str, skip_first_n: int = 5) -> dict:
    """Process all logit_lens_top1 files for a model.

    Args:
        skip_first_n: Skip the first N generated tokens (early tokens are noisy).

    Returns dict mapping condition_name → {alpha, mean_commitment, median_commitment,
    std_commitment, n_tokens}.
    """
    spec = get_spec(model_name)
    merged_dir = Path(f"results/cross_model/{model_name}/exp6/merged_A1_{model_name}_it_v1")

    if not merged_dir.exists():
        print(f"[1B] SKIP {model_name}: merged dir not found at {merged_dir}")
        return {}

    # Find all logit_lens_top1_*.npz files
    ll_files = sorted(merged_dir.glob("logit_lens_top1_*.npz"))
    if not ll_files:
        # Also check per-worker dirs
        worker_dirs = sorted(Path(f"results/cross_model/{model_name}/exp6").glob(
            f"A1_{model_name}_it_v1_w*"
        ))
        for wd in worker_dirs:
            ll_files.extend(sorted(wd.glob("logit_lens_top1_*.npz")))

    if not ll_files:
        print(f"[1B] SKIP {model_name}: no logit_lens_top1_*.npz files found")
        return {}

    print(f"[1B] {model_name}: processing {len(ll_files)} logit-lens files", flush=True)

    # Also load scores.jsonl to map condition names to alpha values
    scores_path = merged_dir / "scores.jsonl"
    alpha_map: dict[str, float] = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                row = json.loads(line.strip())
                cond = row.get("condition", "")
                alpha = row.get("alpha")
                if alpha is not None:
                    alpha_map[cond] = float(alpha)

    results: dict[str, dict] = {}

    for ll_path in ll_files:
        # Extract condition name from filename: logit_lens_top1_{condition}.npz
        condition = ll_path.stem.replace("logit_lens_top1_", "")

        with np.load(ll_path) as data:
            all_commitments: list[int] = []

            for record_id in data.files:
                arr = data[record_id]  # [n_generated_tokens, n_layers]
                if arr.ndim != 2:
                    continue

                # Skip first N tokens (noisy) and compute commitment per step
                for step in range(skip_first_n, arr.shape[0]):
                    cl = _commitment_layer(arr[step])
                    all_commitments.append(cl)

            if all_commitments:
                commitments = np.array(all_commitments, dtype=np.float64)
                alpha = alpha_map.get(condition)

                results[condition] = {
                    "condition": condition,
                    "alpha": alpha,
                    "mean_commitment": float(np.mean(commitments)),
                    "median_commitment": float(np.median(commitments)),
                    "std_commitment": float(np.std(commitments)),
                    "n_tokens": len(all_commitments),
                    "n_layers": spec.n_layers,
                    "skip_first_n": skip_first_n,
                }

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="1B: Commitment delay under steering")
    p.add_argument("--model-name", default="", help="Single model (empty = all)")
    p.add_argument("--skip-first-n", type=int, default=5,
                   help="Skip first N generated tokens (default 5)")
    args = p.parse_args()

    models = [args.model_name] if args.model_name else list(MODEL_REGISTRY.keys())

    all_results: dict[str, dict] = {}

    for model in models:
        results = process_model(model, skip_first_n=args.skip_first_n)
        if results:
            # Save per-model
            out_dir = Path(f"results/cross_model/{model}/exp6")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "commitment_vs_alpha.json"
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"[1B] {model}: {len(results)} conditions -> {out_path}")

            all_results[model] = results

    # Save combined summary
    if all_results:
        # Extract (model, alpha, median_commitment) for plotting
        summary = []
        for model, conditions in all_results.items():
            for cond_name, cond_data in conditions.items():
                if cond_data.get("alpha") is not None:
                    summary.append({
                        "model": model,
                        "condition": cond_name,
                        "alpha": cond_data["alpha"],
                        "median_commitment": cond_data["median_commitment"],
                        "mean_commitment": cond_data["mean_commitment"],
                        "std_commitment": cond_data["std_commitment"],
                        "n_tokens": cond_data["n_tokens"],
                    })

        summary_path = Path("results/cross_model/plots/data/commitment_vs_alpha.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[1B] Combined summary: {len(summary)} entries -> {summary_path}")


if __name__ == "__main__":
    main()
