#!/usr/bin/env python3
"""Post-hoc score coherent_assistant_rate on existing sample_outputs.jsonl files.

Reads sample_outputs.jsonl, computes the deterministic coherent_assistant_rate metric
per (condition, record_id) pair, and writes results to coherent_assistant_scores.jsonl.
Also updates scores.csv with the new metric.

Usage:
    uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp06_corrective_direction_steering/merged_A1_it
    uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp06_corrective_direction_steering/merged_A2_pt
"""
from __future__ import annotations
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.poc.exp06_corrective_direction_steering.benchmarks.governance import _coherent_assistant_response


def score_dir(merged_dir: Path) -> None:
    samples_path = merged_dir / "sample_outputs.jsonl"
    if not samples_path.exists():
        print(f"No sample_outputs.jsonl at {merged_dir}")
        return

    # Read all samples, deduplicate by (condition, record_id)
    seen: set[tuple] = set()
    rows: list[dict] = []
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("condition", ""), r.get("record_id", ""))
            if key not in seen:
                seen.add(key)
                rows.append(r)

    # Score
    by_condition: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        score = _coherent_assistant_response(r.get("generated_text", ""))
        by_condition[r.get("condition", "")].append(score)

    # Summarise per condition
    print(f"\n{'condition':40s} {'coherent_assistant_rate':>22s}  n")
    print("-" * 70)
    for cond in sorted(by_condition):
        scores = by_condition[cond]
        mean = sum(scores) / len(scores) if scores else float("nan")
        print(f"{cond:40s} {mean:22.3f}  {len(scores)}")

    # Load existing scores.csv and add/replace coherent_assistant_rate rows
    scores_csv = merged_dir / "scores.csv"
    existing: list[dict] = []
    if scores_csv.exists():
        with open(scores_csv) as f:
            existing = list(csv.DictReader(f))

    # Remove old coherent_assistant_rate rows
    existing = [r for r in existing if r.get("benchmark") != "coherent_assistant_rate"]

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
            "benchmark": "coherent_assistant_rate",
            "metric": "rate",
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
        # Union of all keys across all rows (handles schema differences between A/B experiments)
        seen_keys: dict[str, None] = {}
        for row in all_rows:
            seen_keys.update(dict.fromkeys(row.keys()))
        fieldnames = list(seen_keys.keys())
        with open(scores_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([{k: row.get(k, "") for k in fieldnames} for row in all_rows])
        print(f"\nUpdated {scores_csv} (+{len(new_rows)} coherent_assistant_rate rows)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged-dir", required=True)
    args = p.parse_args()
    score_dir(Path(args.merged_dir))


if __name__ == "__main__":
    main()
