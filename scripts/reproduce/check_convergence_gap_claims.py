#!/usr/bin/env python3
"""Check convergence-gap paper claims against committed summary artifacts.

This is a CPU-only summary audit. It verifies the paper-facing numbers for the
second convergence-gap / commitment-delay draft without rerunning GPU jobs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


NumberFn = Callable[[Path], float]


@dataclass(frozen=True)
class Claim:
    name: str
    expected: float
    read: NumberFn
    tolerance: float = 5e-4


def load_json(repo: Path, relpath: str):
    with (repo / relpath).open() as f:
        return json.load(f)


def load_csv(repo: Path, relpath: str) -> list[dict[str, str]]:
    with (repo / relpath).open(newline="") as f:
        return list(csv.DictReader(f))


def exp22(key: str, column: str = "estimate_it_minus_pt") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp22_endpoint_deconfounded_table.csv")
        for row in rows:
            if row["key"] == key:
                return float(row[column])
        raise KeyError(key)

    return _read


def exp11(condition: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp11_matched_prefix_mlp_graft/plots/"
            "exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json",
        )
        return float(data["dense_family_means"][condition][metric])

    return _read


def exp14(group: str, condition: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp14_symmetric_matched_prefix_causality/"
            "exp13exp14_full_20260416/exp13_full_summary.json",
        )
        return float(data["dense_family_means"][group][condition])

    return _read


def exp19(window: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp19_late_mlp_specificity_controls/"
            "exp19B_core120_h100x8_20260424_050421_analysis/"
            "exp19B_summary_light.json",
        )
        return float(data["dense5_pooled"][window]["kl_to_own_final"][metric]["mean"])

    return _read


CLAIMS = [
    Claim("Exp22 endpoint-matched raw late KL", 0.425078139, exp22("endpoint_matched_raw_late_kl")),
    Claim("Exp22 endpoint-matched tuned late KL", 0.762124300, exp22("endpoint_matched_tuned_late_kl")),
    Claim("Exp22 endpoint-free adjacent JS", 0.052171032, exp22("endpoint_free_remaining_adj_js")),
    Claim("Exp22 endpoint-free future top1 flips", 0.202966851, exp22("endpoint_free_future_top1_flips")),
    Claim("Exp22 min matched retention", 0.796257811, exp22("quality_min_matched_retention")),
    Claim("Exp22 max SMD after matching", 0.056949275, exp22("quality_max_smd_after")),
    Claim("Exp11 early graft final20 KL delta", -0.034725368, exp11("B_early_raw", "final_20pct_delta_kl_to_own_final")),
    Claim("Exp11 mid graft final20 KL delta", -0.045361679, exp11("B_mid_raw", "final_20pct_delta_kl_to_own_final")),
    Claim("Exp11 late graft final20 KL delta", 0.341448775, exp11("B_late_raw", "final_20pct_delta_kl_to_own_final")),
    Claim("Exp14 IT-side early swap final20 KL delta", -0.102870663, exp14("it_side_final20_kl_delta", "D_early_ptswap")),
    Claim("Exp14 IT-side mid swap final20 KL delta", -0.227702778, exp14("it_side_final20_kl_delta", "D_mid_ptswap")),
    Claim("Exp14 IT-side late swap final20 KL delta", -0.508619582, exp14("it_side_final20_kl_delta", "D_late_ptswap")),
    Claim("Exp19 true late graft KL delta", 0.327112878, exp19("late", "true_delta")),
    Claim("Exp19 random late residual-projection KL delta", 0.003236096, exp19("late", "rand_resproj_delta")),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    failures: list[str] = []
    for claim in CLAIMS:
        observed = claim.read(repo)
        diff = abs(observed - claim.expected)
        status = "ok" if diff <= claim.tolerance else "FAIL"
        print(f"{status:4s} {claim.name}: observed={observed:.9f} expected={claim.expected:.9f}")
        if status != "ok" or math.isnan(observed):
            failures.append(f"{claim.name}: observed {observed}, expected {claim.expected}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(" -", failure)
        return 1
    print(f"\nAll {len(CLAIMS)} convergence-gap claims passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
