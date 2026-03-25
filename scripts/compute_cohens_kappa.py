#!/usr/bin/env python3
"""Compute Cohen's κ between human annotations and LLM judge scores.

Workflow:
  1. Run create_gold_standard.py → data/gold_standard_v1.csv
  2. Fill human_g1, human_g2 columns manually
  3. Run LLM judge on the same outputs (done via llm_judge_exp6.py)
  4. Run this script to compute κ

Usage:
    uv run python scripts/compute_cohens_kappa.py
    uv run python scripts/compute_cohens_kappa.py \\
        --gold data/gold_standard_v1.csv \\
        --judge-scores results/exp6/merged_A1_it/llm_judge_v2_scores.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def _weighted_cohens_kappa(
    y_human: list[int],
    y_judge: list[int],
    n_classes: int,
) -> float:
    """Compute linearly-weighted Cohen's κ for ordinal ratings.

    κ = (P_o - P_e) / (1 - P_e)
    where weights w_ij = 1 - |i-j| / (n_classes - 1)  (linear weighting)
    """
    n = len(y_human)
    if n == 0:
        return float("nan")

    # Build weight matrix
    weights = [
        [1.0 - abs(i - j) / (n_classes - 1) for j in range(n_classes)]
        for i in range(n_classes)
    ]

    # Observed weighted agreement
    p_o = sum(weights[y_human[k] - 1][y_judge[k] - 1] for k in range(n)) / n

    # Expected weighted agreement (marginal distributions)
    human_counts = [0] * n_classes
    judge_counts = [0] * n_classes
    for k in range(n):
        human_counts[y_human[k] - 1] += 1
        judge_counts[y_judge[k] - 1] += 1

    p_e = sum(
        weights[i][j] * (human_counts[i] / n) * (judge_counts[j] / n)
        for i in range(n_classes)
        for j in range(n_classes)
    )

    if p_e >= 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default="data/gold_standard_v1.csv")
    p.add_argument("--judge-scores", default="results/exp6/merged_A1_it/llm_judge_v2_scores.jsonl")
    p.add_argument("--threshold", type=float, default=0.70,
                   help="Minimum κ to proceed (default 0.70)")
    args = p.parse_args()

    gold_path = Path(args.gold)
    scores_path = Path(args.judge_scores)

    if not gold_path.exists():
        print(f"Gold standard not found: {gold_path}")
        print("Run create_gold_standard.py and fill in human annotations first.")
        sys.exit(1)
    if not scores_path.exists():
        print(f"Judge scores not found: {scores_path}")
        print("Run llm_judge_exp6.py on the merged_A1_it directory first.")
        sys.exit(1)

    # Load gold standard
    gold: list[dict] = []
    with open(gold_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h_g1 = row.get("human_g1", "").strip()
            h_g2 = row.get("human_g2", "").strip()
            if not h_g1 or not h_g2:
                continue  # skip unannotated rows
            try:
                gold.append({
                    "condition": row["condition"],
                    "record_id": row["record_id"],
                    "human_g1": int(h_g1),
                    "human_g2": int(h_g2),
                })
            except ValueError:
                print(f"  Skipping row with non-integer annotation: {row}")

    print(f"Loaded {len(gold)} annotated examples from {gold_path}")
    if len(gold) == 0:
        print("No annotated examples found. Please fill in human_g1 and human_g2 columns.")
        sys.exit(1)

    # Load LLM judge scores
    judge_scores: dict[tuple[str, str, str], int] = {}
    for line in open(scores_path):
        try:
            r = json.loads(line)
            if r.get("score", -1) >= 0:
                judge_scores[(r["condition"], r["record_id"], r["task"])] = r["score"]
        except Exception:
            pass

    print(f"Loaded {len(judge_scores)} judge score entries from {scores_path}")

    # Match gold to judge scores
    g1_human, g1_judge = [], []
    g2_human, g2_judge = [], []
    unmatched = 0

    for entry in gold:
        cond = entry["condition"]
        rid = entry["record_id"]

        j_g1 = judge_scores.get((cond, rid, "g1"))
        j_g2 = judge_scores.get((cond, rid, "g2"))

        if j_g1 is not None:
            g1_human.append(entry["human_g1"])
            g1_judge.append(j_g1)
        else:
            unmatched += 1

        if j_g2 is not None:
            g2_human.append(entry["human_g2"])
            g2_judge.append(j_g2)

    if unmatched > 0:
        print(f"  Warning: {unmatched} gold entries had no matching judge scores")

    # Compute κ
    print(f"\n{'='*50}")
    print("Cohen's κ Results (linearly weighted)")
    print(f"{'='*50}")

    kappa_g1 = float("nan")
    kappa_g2 = float("nan")

    if g1_human:
        kappa_g1 = _weighted_cohens_kappa(g1_human, g1_judge, n_classes=5)
        mean_human = sum(g1_human) / len(g1_human)
        mean_judge = sum(g1_judge) / len(g1_judge)
        exact_agree = sum(1 for h, j in zip(g1_human, g1_judge) if h == j) / len(g1_human)
        within1 = sum(1 for h, j in zip(g1_human, g1_judge) if abs(h-j) <= 1) / len(g1_human)
        print(f"\nG1 (Output Governance Quality), n={len(g1_human)}")
        print(f"  Human mean: {mean_human:.2f}, Judge mean: {mean_judge:.2f}")
        print(f"  Exact agreement: {exact_agree:.1%}")
        print(f"  Within-1 agreement: {within1:.1%}")
        print(f"  Cohen's κ (weighted): {kappa_g1:.3f}", end="")
        if kappa_g1 >= args.threshold:
            print(f"  ✓ PASS (≥{args.threshold})")
        else:
            print(f"  ✗ FAIL (<{args.threshold}) — revise rubric/calibration examples")

    if g2_human:
        kappa_g2 = _weighted_cohens_kappa(g2_human, g2_judge, n_classes=5)
        mean_human = sum(g2_human) / len(g2_human)
        mean_judge = sum(g2_judge) / len(g2_judge)
        exact_agree = sum(1 for h, j in zip(g2_human, g2_judge) if h == j) / len(g2_human)
        within1 = sum(1 for h, j in zip(g2_human, g2_judge) if abs(h-j) <= 1) / len(g2_human)
        print(f"\nG2 (Conversational Register), n={len(g2_human)}")
        print(f"  Human mean: {mean_human:.2f}, Judge mean: {mean_judge:.2f}")
        print(f"  Exact agreement: {exact_agree:.1%}")
        print(f"  Within-1 agreement: {within1:.1%}")
        print(f"  Cohen's κ (weighted): {kappa_g2:.3f}", end="")
        if kappa_g2 >= args.threshold:
            print(f"  ✓ PASS (≥{args.threshold})")
        else:
            print(f"  ✗ FAIL (<{args.threshold}) — revise rubric/calibration examples")

    print(f"\n{'='*50}")
    if not (kappa_g1 >= args.threshold and kappa_g2 >= args.threshold):
        print("\nACTION REQUIRED: κ below threshold.")
        print("  1. Review disagreements (see below)")
        print("  2. Revise rubric calibration examples in llm_judge_exp6.py")
        print("  3. Re-run judge on gold set and re-compute κ")
        # Show worst disagreements
        print("\nWorst G1 disagreements (|human - judge| >= 2):")
        with open(gold_path) as f:
            gold_rows = list(csv.DictReader(f))
        for i, (h, j) in enumerate(zip(g1_human, g1_judge)):
            if abs(h - j) >= 2 and i < len(gold_rows):
                row = gold_rows[i]
                print(f"  Human={h}, Judge={j}: {row.get('prompt','')[:60]}...")
                print(f"    Response: {row.get('generated_text','')[:80]}...")
    else:
        print("\nBoth κ values pass threshold. LLM judge is validated for use at scale.")
        print(f"Report: 'Our LLM judge (claude-sonnet-4-6) achieved κ(G1)={kappa_g1:.2f}, "
              f"κ(G2)={kappa_g2:.2f} against human annotations on a 50-example gold standard.'")


if __name__ == "__main__":
    main()
