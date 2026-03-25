#!/usr/bin/env python3
"""Create a 50-example gold standard set for LLM judge validation.

Samples 50 outputs from merged_A1_it/sample_outputs.jsonl:
  - 10 outputs per α level: {-5, -1, 0, 1, 5}
  - Only GOV-CONV category (most natural for G1/G2 human annotation)
  - Stratified across α values for coverage of governance change

Exports:
  data/gold_standard_v1.csv   — fill human_g1, human_g2 columns manually

Usage:
    uv run python scripts/create_gold_standard.py
    uv run python scripts/create_gold_standard.py --source-dir results/exp6/merged_A1_it
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


_ALPHA_STRATA = [-5.0, -1.0, 0.0, 1.0, 5.0]  # 5 strata × 10 samples = 50 total
_N_PER_STRATUM = 10
_TARGET_CATEGORY = "GOV-CONV"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="results/exp6/merged_A1_it")
    p.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    p.add_argument("--output", default="data/gold_standard_v1.csv")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    samples_path = Path(args.source_dir) / "sample_outputs.jsonl"
    if not samples_path.exists():
        raise FileNotFoundError(f"sample_outputs.jsonl not found: {samples_path}")

    # Load dataset for question lookup (both v1 and v2)
    records_by_id: dict[str, dict] = {}
    v1_path = Path("data/exp6_dataset.jsonl")
    if v1_path.exists():
        for line in open(v1_path):
            r = json.loads(line)
            records_by_id[r["id"]] = r
    if Path(args.dataset).exists():
        for line in open(args.dataset):
            r = json.loads(line)
            records_by_id[r["id"]] = r

    # Load all sample outputs
    all_samples: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for line in open(samples_path):
        s = json.loads(line)
        key = (s["condition"], s["record_id"])
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Category: use from sample directly (stored by run.py), or from dataset lookup
        cat = s.get("category", "")
        if not cat:
            rec = records_by_id.get(s["record_id"])
            cat = rec.get("category", "") if rec else ""
        if cat != _TARGET_CATEGORY:
            continue

        rec = records_by_id.get(s["record_id"])
        prompt = s.get("prompt", "")
        if not prompt and rec and rec.get("formats"):
            prompt = list(rec["formats"].values())[0]

        all_samples.append({
            "condition": s["condition"],
            "record_id": s["record_id"],
            "prompt": prompt,
            "generated_text": s.get("generated_text", ""),
            "alpha": _extract_alpha(s["condition"]),
        })

    print(f"Loaded {len(all_samples)} GOV-CONV samples from {samples_path}")

    # Stratify by alpha level
    rng = random.Random(args.seed)
    selected: list[dict] = []
    for alpha in _ALPHA_STRATA:
        stratum = [s for s in all_samples if _matches_alpha(s["alpha"], alpha)]
        rng.shuffle(stratum)
        chosen = stratum[:_N_PER_STRATUM]
        if len(chosen) < _N_PER_STRATUM:
            print(f"  Warning: only {len(chosen)} samples for α={alpha} (target {_N_PER_STRATUM})")
        selected.extend(chosen)
        print(f"  α={alpha:+.0f}: {len(chosen)} samples selected")

    # If we couldn't fill all strata, pad from remaining
    if len(selected) < 50:
        used_keys = {(s["condition"], s["record_id"]) for s in selected}
        remaining = [s for s in all_samples if (s["condition"], s["record_id"]) not in used_keys]
        rng.shuffle(remaining)
        selected.extend(remaining[:50 - len(selected)])
        print(f"  Padded with {50 - len(selected)} additional samples")

    # Shuffle final set for annotation (don't reveal stratum to annotator)
    rng.shuffle(selected)

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "id", "condition", "record_id", "alpha",
            "prompt", "generated_text",
            "human_g1", "human_g2",
            "notes",
        ])
        writer.writeheader()
        for i, s in enumerate(selected):
            writer.writerow({
                "id": i + 1,
                "condition": s["condition"],
                "record_id": s["record_id"],
                "alpha": s["alpha"],
                "prompt": s["prompt"][:200],
                "generated_text": s["generated_text"][:400],
                "human_g1": "",   # fill manually (1-5)
                "human_g2": "",   # fill manually (1-5)
                "notes": "",
            })

    print(f"\nGold standard ({len(selected)} examples) → {out_path}")
    print("\nInstructions:")
    print("  Fill 'human_g1' (1-5) and 'human_g2' (1-5) for each row.")
    print("  G1: Structure/format quality (1=incoherent, 5=excellent)")
    print("  G2: Assistant register (1=raw web text, 5=clearly assistant)")
    print("  Then run: uv run python scripts/compute_cohens_kappa.py")


def _extract_alpha(condition: str) -> float | None:
    """Extract alpha value from condition name like 'A1_alpha_-5'."""
    import re
    m = re.search(r"alpha_([+-]?\d+(?:\.\d+)?)", condition)
    if m:
        return float(m.group(1))
    if "baseline" in condition:
        return 1.0  # baseline = alpha 1.0
    return None


def _matches_alpha(actual: float | None, target: float) -> bool:
    if actual is None:
        return False
    return abs(actual - target) < 0.01


if __name__ == "__main__":
    main()
