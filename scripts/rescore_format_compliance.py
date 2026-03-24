#!/usr/bin/env python3
"""Post-hoc rescore format_compliance using the improved prompt-text-based detector.

The old scorer used pre-assigned expected_format metadata (wrong for most IFEval records).
This script re-detects the format constraint from prompt text and rescores all conditions.

Usage:
    uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_A1_it
    uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_A2_pt
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.poc.exp6.benchmarks.governance import _check_format_compliance


def score_dir(merged_dir: Path, dataset_path: Path) -> None:
    samples_path = merged_dir / "sample_outputs.jsonl"
    if not samples_path.exists():
        print(f"No sample_outputs.jsonl at {merged_dir}")
        return

    records = {
        json.loads(l)["id"]: json.loads(l)
        for l in open(dataset_path)
    }

    # Deduplicate by (condition, record_id) and score
    seen: set[tuple] = set()
    by_condition: dict[str, list[float]] = defaultdict(list)
    skipped = 0
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            key = (s.get("condition", ""), s.get("record_id", ""))
            if key in seen:
                continue
            seen.add(key)
            rec = records.get(s["record_id"])
            if not rec:
                continue
            prompt = rec.get("formats", {}).get("B", "") or rec.get("prompt", "")
            result = _check_format_compliance(s.get("generated_text", ""), prompt)
            if result is not None:
                by_condition[s["condition"]].append(result)
            else:
                skipped += 1

    print(f"\n{'condition':45s} {'format_compliance':>18s}  n  (skipped={skipped})")
    print("-" * 75)
    for cond in sorted(by_condition):
        scores = by_condition[cond]
        mean = sum(scores) / len(scores) if scores else float("nan")
        print(f"{cond:45s} {mean:18.3f}  {len(scores)}")

    # Update scores.csv
    scores_csv = merged_dir / "scores.csv"
    existing: list[dict] = []
    if scores_csv.exists():
        with open(scores_csv) as f:
            existing = list(csv.DictReader(f))

    # Remove old format_compliance rows
    existing = [r for r in existing if r.get("benchmark") != "format_compliance"]

    # Infer metadata from existing rows per condition
    meta_by_cond: dict[str, dict] = {}
    for r in existing:
        c = r.get("condition", "")
        if c not in meta_by_cond:
            meta_by_cond[c] = {
                "experiment": r.get("experiment", ""),
                "method": r.get("method", ""),
                "alpha": r.get("alpha", ""),
                "beta": r.get("beta", ""),
                "layers": r.get("layers", ""),
            }

    new_rows = []
    for cond, scores in by_condition.items():
        mean = sum(scores) / len(scores) if scores else float("nan")
        meta = meta_by_cond.get(cond, {})
        new_rows.append({
            "condition": cond,
            "benchmark": "format_compliance",
            "metric": "compliance",
            "value": mean,
            "n": len(scores),
            "experiment": meta.get("experiment", ""),
            "method": meta.get("method", ""),
            "alpha": meta.get("alpha", ""),
            "beta": meta.get("beta", ""),
            "layers": meta.get("layers", ""),
        })

    all_rows = existing + new_rows
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(scores_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nUpdated {scores_csv} (+{len(new_rows)} format_compliance rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged-dir", required=True)
    p.add_argument("--dataset", default="data/exp6_dataset.jsonl")
    args = p.parse_args()
    score_dir(Path(args.merged_dir), Path(args.dataset))


if __name__ == "__main__":
    main()
