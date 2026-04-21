#!/usr/bin/env python3
"""Re-score all existing merged experiment directories with G1/G2/S1/S2 LLM judge.

Uses eval_dataset_v2.jsonl if available, falls back to exp6_dataset.jsonl.
The LLM judge is post-hoc — no model re-generation needed.
Only the LLM judge scores (G1/G2/S1/S2) are re-computed;
mmlu_forced_choice and format_compliance_v2 require new model runs with v2 dataset.

Usage:
    uv run python scripts/rescore_v2_all.py              # all known dirs
    uv run python scripts/rescore_v2_all.py --dry-run    # print what would run
    uv run python scripts/rescore_v2_all.py --overwrite  # delete existing scores

Adjust MERGED_DIRS below if you have additional experiment directories.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MERGED_DIRS = [
    "results/exp6/merged_A1_it",
    "results/exp6/merged_A1_early_it",
    "results/exp6/merged_A1_mid_it",
    "results/exp6/merged_A2_pt",
    "results/exp6/merged_B1_it",
    "results/exp6/merged_B2_it",
    "results/exp6/merged_B2_pt",
    "results/exp6/merged_B3_it",
    "results/exp6/merged_B4_it",
    "results/exp6/merged_B5_it",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true", help="Print commands without running")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing v2 judge scores")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tasks", nargs="+", default=["g1", "g2", "s1", "s2"])
    p.add_argument("--model", default="google/gemini-2.5-flash")
    p.add_argument("--provider", default="auto", choices=["auto", "gemini", "openrouter"])
    args = p.parse_args()

    dataset = "data/eval_dataset_v2.jsonl"
    if not Path(dataset).exists():
        dataset = "data/exp6_dataset.jsonl"
        print(f"Warning: eval_dataset_v2.jsonl not found, using {dataset}")

    existing_dirs = [d for d in MERGED_DIRS if (Path(d) / "sample_outputs.jsonl").exists()]
    missing_dirs = [d for d in MERGED_DIRS if not (Path(d) / "sample_outputs.jsonl").exists()]

    print(f"Found {len(existing_dirs)} dirs with sample_outputs.jsonl")
    if missing_dirs:
        print(f"Skipping {len(missing_dirs)} dirs (no sample_outputs.jsonl):")
        for d in missing_dirs:
            print(f"  {d}")
    print()

    for merged_dir in existing_dirs:
        out_file = Path(merged_dir) / "llm_judge_v2_scores.jsonl"
        already_has = out_file.exists() and out_file.stat().st_size > 0

        cmd = [
            "uv", "run", "python", "scripts/llm_judge.py",
            "--merged-dir", merged_dir,
            "--dataset", dataset,
            "--model", args.model,
            "--provider", args.provider,
            "--workers", str(args.workers),
            "--tasks", *args.tasks,
        ]
        if args.overwrite:
            cmd.append("--overwrite")

        status = "RESCORE" if already_has else "SCORE"
        if args.overwrite and already_has:
            status = "OVERWRITE"

        print(f"[{status}] {merged_dir}")
        if args.dry_run:
            print(f"  Command: {' '.join(cmd)}")
            continue

        result = subprocess.run(cmd, capture_output=False)
        if result.returncode != 0:
            print(f"  ERROR: {merged_dir} failed with code {result.returncode}")
        else:
            print(f"  Done: {merged_dir}")
        print()

    if args.dry_run:
        print(f"\nDry run complete. {len(existing_dirs)} dirs would be scored.")


if __name__ == "__main__":
    main()
