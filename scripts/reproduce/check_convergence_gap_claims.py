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
import subprocess
import sys
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


def exp22_template_audit(key: str, column: str = "estimate_it_minus_pt") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp22_template_raw_public600_effects.csv")
        for row in rows:
            if row["key"] == key:
                return float(row[column])
        raise KeyError(key)

    return _read


def exp22_template_length(model: str, variant: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp22_template_raw_public600_lengths.csv")
        for row in rows:
            if row["model"] == model and row["variant"] == variant:
                return float(row[column])
        raise KeyError((model, variant, column))

    return _read


def exp22_template_quality(key: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(repo, "results/paper_synthesis/exp22_template_raw_public600_audit.json")
        return float(data["quality"][key])

    return _read


def exp22_fixed_history(key: str, column: str = "estimate") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp22_fixed_history_template_audit_effects.csv")
        for row in rows:
            if row["key"] == key:
                return float(row[column])
        raise KeyError(key)

    return _read


def exp22_fixed_history_quality(key: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(repo, "results/paper_synthesis/exp22_fixed_history_template_audit.json")
        return float(data["quality"][key])

    return _read


def exp22_fixed_history_pt_teacher(key: str, column: str = "estimate") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit_effects.csv")
        for row in rows:
            if row["key"] == key:
                return float(row[column])
        raise KeyError(key)

    return _read


def exp22_fixed_history_pt_teacher_quality(key: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(repo, "results/paper_synthesis/exp22_fixed_history_pt_teacher_audit.json")
        return float(data["quality"][key])

    return _read


def exp9_nsteps(model: str, branch: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(repo, "results/exp09_cross_model_observational_replication/data/exp9_summary.json")
        return float(data[model][f"tuned_lens_raw_kl_final_n_steps_{branch}"])

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


def exp11_model(model: str, condition: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp11_matched_prefix_mlp_graft/plots/"
            "exp11_exp3_600rand_v11_depthablation_full/depth_ablation_metrics.json",
        )
        for row in data["models"]:
            if row["model"] == model:
                return float(row["pipelines"][condition]["regions"]["final_20pct"]["kl_to_own_final"][metric])
        raise KeyError((model, condition, metric))

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


def exp14_model(model: str, condition: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp14_symmetric_matched_prefix_causality/"
            "exp13exp14_full_20260416/exp13_full_summary.json",
        )
        return float(
            data["models"][model]["it_side"][condition]["regions"]["final_20pct"][
                "kl_to_own_final"
            ]["delta"]
        )

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


def exp55(window: str, side: str, region: str, column: str = "estimate") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp55_late_window_robustness_effects.csv")
        for row in rows:
            if (
                row["level"] == "dense5_model_mean"
                and row["window_name"] == window
                and row["side"] == side
                and row["region"] == region
                and row["metric"] == "kl_to_own_final"
            ):
                return float(row[column])
        raise KeyError((window, side, region, column))

    return _read


def check_generated_reporting_tables(repo: Path) -> None:
    script = repo / "scripts/analysis/build_convergence_gap_reporting_tables.py"
    if not script.exists():
        raise FileNotFoundError(script)
    subprocess.run(
        [sys.executable, str(script), "--repo", str(repo), "--check"],
        cwd=repo,
        check=True,
    )


EXP55_TABLE_CLAIMS = [
    Claim("Exp55 prelate final20 graft KL delta", -0.000919429, exp55("prelate_half", "it_graft_into_pt", "final_20pct")),
    Claim("Exp55 prelate final20 swap KL delta", -0.383723697, exp55("prelate_half", "pt_swap_into_it", "final_20pct")),
    Claim("Exp55 prelate edited-window graft KL delta", 0.068550355, exp55("prelate_half", "it_graft_into_pt", "graft_window")),
    Claim("Exp55 prelate edited-window swap KL delta", -2.674308927, exp55("prelate_half", "pt_swap_into_it", "graft_window")),
    Claim("Exp55 late-full final20 graft KL delta", 0.070077778, exp55("late_full", "it_graft_into_pt", "final_20pct")),
    Claim("Exp55 late-full final20 graft KL CI low", -0.047642406, exp55("late_full", "it_graft_into_pt", "final_20pct", "ci95_low")),
    Claim("Exp55 late-full final20 graft KL CI high", 0.188532657, exp55("late_full", "it_graft_into_pt", "final_20pct", "ci95_high")),
    Claim("Exp55 late-full final20 swap KL delta", -0.625367707, exp55("late_full", "pt_swap_into_it", "final_20pct")),
    Claim("Exp55 late-full final20 swap KL CI low", -1.075560776, exp55("late_full", "pt_swap_into_it", "final_20pct", "ci95_low")),
    Claim("Exp55 late-full final20 swap KL CI high", -0.201026364, exp55("late_full", "pt_swap_into_it", "final_20pct", "ci95_high")),
    Claim("Exp55 late-full edited-window graft KL delta", 0.364734734, exp55("late_full", "it_graft_into_pt", "graft_window")),
    Claim("Exp55 late-full edited-window graft KL CI low", 0.107971200, exp55("late_full", "it_graft_into_pt", "graft_window", "ci95_low")),
    Claim("Exp55 late-full edited-window graft KL CI high", 0.628987279, exp55("late_full", "it_graft_into_pt", "graft_window", "ci95_high")),
    Claim("Exp55 late-full edited-window swap KL delta", -1.604592178, exp55("late_full", "pt_swap_into_it", "graft_window")),
    Claim("Exp55 late-front final20 graft KL delta", 0.008264746, exp55("late_front_half", "it_graft_into_pt", "final_20pct")),
    Claim("Exp55 late-front final20 swap KL delta", -0.351917537, exp55("late_front_half", "pt_swap_into_it", "final_20pct")),
    Claim("Exp55 late-front edited-window graft KL delta", 0.142462530, exp55("late_front_half", "it_graft_into_pt", "graft_window")),
    Claim("Exp55 late-front edited-window swap KL delta", -1.320787294, exp55("late_front_half", "pt_swap_into_it", "graft_window")),
    Claim("Exp55 late-center final20 graft KL delta", 0.022144812, exp55("late_center_half", "it_graft_into_pt", "final_20pct")),
    Claim("Exp55 late-center final20 swap KL delta", -0.351788969, exp55("late_center_half", "pt_swap_into_it", "final_20pct")),
    Claim("Exp55 late-center edited-window graft KL delta", 0.093934874, exp55("late_center_half", "it_graft_into_pt", "graft_window")),
    Claim("Exp55 late-center edited-window swap KL delta", -0.700689994, exp55("late_center_half", "pt_swap_into_it", "graft_window")),
    Claim("Exp55 late-terminal final20 graft KL delta", 0.050348030, exp55("late_terminal_half", "it_graft_into_pt", "final_20pct")),
    Claim("Exp55 late-terminal final20 swap KL delta", -0.442889594, exp55("late_terminal_half", "pt_swap_into_it", "final_20pct")),
    Claim("Exp55 late-terminal edited-window graft KL delta", 0.039998630, exp55("late_terminal_half", "it_graft_into_pt", "graft_window")),
    Claim("Exp55 late-terminal edited-window swap KL delta", -0.403257171, exp55("late_terminal_half", "pt_swap_into_it", "graft_window")),
    Claim("Exp55 terminal-quarter final20 graft KL delta", 0.032808853, exp55("terminal_quarter", "it_graft_into_pt", "final_20pct")),
    Claim("Exp55 terminal-quarter final20 swap KL delta", -0.347234229, exp55("terminal_quarter", "pt_swap_into_it", "final_20pct")),
    Claim("Exp55 terminal-quarter edited-window graft KL delta", -0.012300406, exp55("terminal_quarter", "it_graft_into_pt", "graft_window")),
    Claim("Exp55 terminal-quarter edited-window swap KL delta", -0.133966810, exp55("terminal_quarter", "pt_swap_into_it", "graft_window")),
]


CLAIMS = [
    Claim("Exp22 endpoint-matched raw late KL", 0.425078139, exp22("endpoint_matched_raw_late_kl")),
    Claim("Exp22 endpoint-matched tuned late KL", 0.762124300, exp22("endpoint_matched_tuned_late_kl")),
    Claim("Exp22 endpoint-free adjacent JS", 0.052171032, exp22("endpoint_free_remaining_adj_js")),
    Claim("Exp22 endpoint-free future top1 flips", 0.202966851, exp22("endpoint_free_future_top1_flips")),
    Claim("Exp22 min matched retention", 0.796257811, exp22("quality_min_matched_retention")),
    Claim("Exp22 max SMD after matching", 0.056949275, exp22("quality_max_smd_after")),
    Claim("Exp22 fixed-history native paired raw late KL", 1.181038884, exp22_fixed_history("it_native_native_fixed_raw_late_kl")),
    Claim("Exp22 fixed-history native paired raw late KL CI low", 1.152865413, exp22_fixed_history("it_native_native_fixed_raw_late_kl", "ci95_low")),
    Claim("Exp22 fixed-history native paired raw late KL CI high", 1.211134281, exp22_fixed_history("it_native_native_fixed_raw_late_kl", "ci95_high")),
    Claim("Exp22 fixed-history raw paired raw late KL", 0.548991758, exp22_fixed_history("it_native_raw_fixed_raw_late_kl")),
    Claim("Exp22 fixed-history raw paired raw late KL CI low", 0.530884516, exp22_fixed_history("it_native_raw_fixed_raw_late_kl", "ci95_low")),
    Claim("Exp22 fixed-history raw paired raw late KL CI high", 0.567593100, exp22_fixed_history("it_native_raw_fixed_raw_late_kl", "ci95_high")),
    Claim("Exp22 fixed-history template-delta paired raw late KL", 0.632047126, exp22_fixed_history("it_native_template_delta_raw_late_kl")),
    Claim("Exp22 fixed-history template-delta paired raw late KL CI low", 0.610986807, exp22_fixed_history("it_native_template_delta_raw_late_kl", "ci95_low")),
    Claim("Exp22 fixed-history template-delta paired raw late KL CI high", 0.654425299, exp22_fixed_history("it_native_template_delta_raw_late_kl", "ci95_high")),
    Claim("Exp22 fixed-history native CEM raw late KL", 0.547576836, exp22_fixed_history("it_native_native_fixed_cem_raw_late_kl")),
    Claim("Exp22 fixed-history native CEM raw late KL CI low", 0.502218445, exp22_fixed_history("it_native_native_fixed_cem_raw_late_kl", "ci95_low")),
    Claim("Exp22 fixed-history native CEM raw late KL CI high", 0.593797636, exp22_fixed_history("it_native_native_fixed_cem_raw_late_kl", "ci95_high")),
    Claim("Exp22 fixed-history raw CEM raw late KL", 0.201747060, exp22_fixed_history("it_native_raw_fixed_cem_raw_late_kl")),
    Claim("Exp22 fixed-history raw CEM raw late KL CI low", 0.157706271, exp22_fixed_history("it_native_raw_fixed_cem_raw_late_kl", "ci95_low")),
    Claim("Exp22 fixed-history raw CEM raw late KL CI high", 0.246511620, exp22_fixed_history("it_native_raw_fixed_cem_raw_late_kl", "ci95_high")),
    Claim("Exp22 fixed-history template-delta CEM raw late KL", 0.356874057, exp22_fixed_history("it_native_template_delta_cem_raw_late_kl")),
    Claim("Exp22 fixed-history template-delta CEM raw late KL CI low", 0.305791420, exp22_fixed_history("it_native_template_delta_cem_raw_late_kl", "ci95_low")),
    Claim("Exp22 fixed-history template-delta CEM raw late KL CI high", 0.410733598, exp22_fixed_history("it_native_template_delta_cem_raw_late_kl", "ci95_high")),
    Claim("Exp22 fixed-history min CEM retention", 0.998994912, exp22_fixed_history_quality("min_retained_fraction")),
    Claim("Exp22 fixed-history max SMD", 0.060525782, exp22_fixed_history_quality("max_smd_after")),
    Claim("Exp22 fixed-history malformed rate", 0.0, exp22_fixed_history_quality("max_malformed_rate")),
    Claim("Exp22 fixed-history missing aligned rows", 0.0, exp22_fixed_history_quality("missing_aligned_step_rows"), tolerance=0.5),
    Claim("Exp54 PT-teacher native paired raw late KL", 0.609906464, exp22_fixed_history_pt_teacher("pt_raw_native_fixed_raw_late_kl")),
    Claim("Exp54 PT-teacher native paired raw late KL CI low", 0.134735569, exp22_fixed_history_pt_teacher("pt_raw_native_fixed_raw_late_kl", "ci95_low")),
    Claim("Exp54 PT-teacher native paired raw late KL CI high", 1.079030946, exp22_fixed_history_pt_teacher("pt_raw_native_fixed_raw_late_kl", "ci95_high")),
    Claim("Exp54 PT-teacher raw paired raw late KL", 0.428647668, exp22_fixed_history_pt_teacher("pt_raw_raw_fixed_raw_late_kl")),
    Claim("Exp54 PT-teacher raw paired raw late KL CI low", 0.154965367, exp22_fixed_history_pt_teacher("pt_raw_raw_fixed_raw_late_kl", "ci95_low")),
    Claim("Exp54 PT-teacher raw paired raw late KL CI high", 0.638592198, exp22_fixed_history_pt_teacher("pt_raw_raw_fixed_raw_late_kl", "ci95_high")),
    Claim("Exp54 PT-teacher template-delta paired raw late KL", 0.181258796, exp22_fixed_history_pt_teacher("pt_raw_template_delta_raw_late_kl")),
    Claim("Exp54 PT-teacher template-delta paired raw late KL CI low", -0.104763423, exp22_fixed_history_pt_teacher("pt_raw_template_delta_raw_late_kl", "ci95_low")),
    Claim("Exp54 PT-teacher template-delta paired raw late KL CI high", 0.467281015, exp22_fixed_history_pt_teacher("pt_raw_template_delta_raw_late_kl", "ci95_high")),
    Claim("Exp54 PT-teacher min CEM retention", 0.991434541, exp22_fixed_history_pt_teacher_quality("min_retained_fraction")),
    Claim("Exp54 PT-teacher max SMD", 0.154679129, exp22_fixed_history_pt_teacher_quality("max_smd_after")),
    Claim("Exp54 PT-teacher malformed rate", 0.0, exp22_fixed_history_pt_teacher_quality("max_malformed_rate")),
    Claim("Exp54 PT-teacher missing aligned rows", 0.0, exp22_fixed_history_pt_teacher_quality("missing_aligned_step_rows"), tolerance=0.5),
    Claim("Exp9 Gemma PT token steps", 1273606, exp9_nsteps("gemma3_4b", "pt"), tolerance=0.5),
    Claim("Exp9 Gemma IT token steps", 810347, exp9_nsteps("gemma3_4b", "it"), tolerance=0.5),
    Claim("Exp9 Llama PT token steps", 517579, exp9_nsteps("llama31_8b", "pt"), tolerance=0.5),
    Claim("Exp9 Llama IT token steps", 499240, exp9_nsteps("llama31_8b", "it"), tolerance=0.5),
    Claim("Exp9 Qwen PT token steps", 482319, exp9_nsteps("qwen3_4b", "pt"), tolerance=0.5),
    Claim("Exp9 Qwen IT token steps", 636979, exp9_nsteps("qwen3_4b", "it"), tolerance=0.5),
    Claim("Exp9 Mistral PT token steps", 1380081, exp9_nsteps("mistral_7b", "pt"), tolerance=0.5),
    Claim("Exp9 Mistral IT token steps", 501093, exp9_nsteps("mistral_7b", "it"), tolerance=0.5),
    Claim("Exp9 OLMo PT token steps", 370171, exp9_nsteps("olmo2_7b", "pt"), tolerance=0.5),
    Claim("Exp9 OLMo IT token steps", 166698, exp9_nsteps("olmo2_7b", "it"), tolerance=0.5),
    Claim("Exp9 DeepSeek PT token steps", 184658, exp9_nsteps("deepseek_v2_lite", "pt"), tolerance=0.5),
    Claim("Exp9 DeepSeek IT token steps", 163877, exp9_nsteps("deepseek_v2_lite", "it"), tolerance=0.5),
    Claim("Exp11 early graft final20 KL delta", -0.034725368, exp11("B_early_raw", "final_20pct_delta_kl_to_own_final")),
    Claim("Exp11 mid graft final20 KL delta", -0.045361679, exp11("B_mid_raw", "final_20pct_delta_kl_to_own_final")),
    Claim("Exp11 late graft final20 KL delta", 0.341448775, exp11("B_late_raw", "final_20pct_delta_kl_to_own_final")),
    Claim("Exp14 IT-side early swap final20 KL delta", -0.102870663, exp14("it_side_final20_kl_delta", "D_early_ptswap")),
    Claim("Exp14 IT-side mid swap final20 KL delta", -0.227702778, exp14("it_side_final20_kl_delta", "D_mid_ptswap")),
    Claim("Exp14 IT-side late swap final20 KL delta", -0.508619582, exp14("it_side_final20_kl_delta", "D_late_ptswap")),
    Claim("Exp19 true late graft KL delta", 0.327112878, exp19("late", "true_delta")),
    Claim("Exp19 random late residual-projection KL delta", 0.003236096, exp19("late", "rand_resproj_delta")),
    Claim("Exp11 Gemma late graft final20 KL delta", 0.609353042, exp11_model("gemma3_4b", "B_late_raw", "delta"), tolerance=5e-4),
    Claim("Exp11 Qwen late graft final20 KL delta", 0.490619414, exp11_model("qwen3_4b", "B_late_raw", "delta"), tolerance=5e-4),
    Claim("Exp11 Llama late graft final20 KL delta", 0.310460229, exp11_model("llama31_8b", "B_late_raw", "delta"), tolerance=5e-4),
    Claim("Exp11 Mistral late graft final20 KL delta", 0.115389272, exp11_model("mistral_7b", "B_late_raw", "delta"), tolerance=5e-4),
    Claim("Exp11 OLMo late graft final20 KL delta", 0.181421916, exp11_model("olmo2_7b", "B_late_raw", "delta"), tolerance=5e-4),
    Claim("Exp14 Gemma late swap final20 KL delta", -0.822406870, exp14_model("gemma3_4b", "D_late_ptswap"), tolerance=5e-4),
    Claim("Exp14 Qwen late swap final20 KL delta", -1.015086601, exp14_model("qwen3_4b", "D_late_ptswap"), tolerance=5e-4),
    Claim("Exp14 Llama late swap final20 KL delta", -0.290872606, exp14_model("llama31_8b", "D_late_ptswap"), tolerance=5e-4),
    Claim("Exp14 Mistral late swap final20 KL delta", -0.272626565, exp14_model("mistral_7b", "D_late_ptswap"), tolerance=5e-4),
    Claim("Exp14 OLMo late swap final20 KL delta", -0.142105269, exp14_model("olmo2_7b", "D_late_ptswap"), tolerance=5e-4),
    *EXP55_TABLE_CLAIMS,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    check_generated_reporting_tables(repo)
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
