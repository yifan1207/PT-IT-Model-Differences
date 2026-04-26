#!/usr/bin/env python3
"""Check manuscript headline numbers against committed summary artifacts.

This is intentionally a summary-artifact audit. It does not rerun GPU
generation; it verifies that the numbers quoted in the paper can be recovered
mechanically from the released JSON/CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


NumberFn = Callable[[Path], float]


@dataclass(frozen=True)
class ClaimCheck:
    claim: str
    source: str
    expected: float
    observed_fn: NumberFn
    tolerance: float = 5e-4
    digits: int = 6


def load_json(repo: Path, relpath: str) -> Any:
    with (repo / relpath).open() as f:
        return json.load(f)


def load_csv(repo: Path, relpath: str) -> list[dict[str, str]]:
    with (repo / relpath).open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def exp9_cg(lens: str, *, exclude: set[str] | None = None) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp09_cross_model_observational_replication/data/"
            "convergence_gap_values.json",
        )[lens]
        values = [
            float(row["CG"])
            for model, row in data.items()
            if not exclude or model not in exclude
        ]
        return mean(values)

    return _read


def exp16_js(pair: str, region: str, agg: str = "regions_weighted") -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp16_matched_prefix_js_gap/"
            "exp16_js_replay_runpod_20260422_075307/js_summary.json",
        )
        return float(data["dense5"]["pairs"][pair][agg][region]["mean"])

    return _read


def exp11_depth(condition: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp11_matched_prefix_mlp_graft/plots/"
            "exp11_exp3_600rand_v11_depthablation_full/"
            "depth_ablation_metrics.json",
        )
        return float(data["dense_family_means"][condition][metric])

    return _read


def exp14_dense(group: str, condition: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp14_symmetric_matched_prefix_causality/"
            "exp13exp14_full_20260416/exp13_full_summary.json",
        )
        return float(data["dense_family_means"][group][condition])

    return _read


def exp19_random(window: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp19_late_mlp_specificity_controls/"
            "exp19B_core120_h100x8_20260424_050421_analysis/"
            "exp19B_summary_light.json",
        )
        return float(data["dense5_pooled"][window]["kl_to_own_final"][metric]["mean"])

    return _read


def exp20_condition(
    mode: str, condition: str, token_class: str, *, margin: bool = False
) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp20_divergence_token_counterfactual/"
            "factorial_validation_holdout_fast_20260425_2009_with_early/"
            "validation_analysis/summary.json",
        )
        row = data["pooled"][mode]["conditions"][condition]
        if margin:
            return float(row["late_margin_mean"])
        return float(row["class_fractions"][token_class])

    return _read


def exp20_margin_drop(condition: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp20_divergence_token_counterfactual/"
            "factorial_validation_holdout_fast_20260425_2009_with_early/"
            "validation_analysis/summary.json",
        )
        conditions = data["pooled"]["native"]["conditions"]
        return float(conditions["C_it_chat"]["late_margin_mean"]) - float(
            conditions[condition]["late_margin_mean"]
        )

    return _read


def exp21_window(condition: str, window: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp21_productive_opposition/"
            "exp21_full_productive_opposition_clean_20260426_053736/"
            "analysis/summary.json",
        )
        return float(
            data["pooled"]["native"]["conditions"]["first_diff"][condition][
                "windows"
            ][window][metric]
        )

    return _read


def exp21_effect(effect: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp21_productive_opposition/"
            "exp21_full_productive_opposition_clean_20260426_053736/"
            "analysis/effects.csv",
        )
        for row in rows:
            if (
                row["mode"] == "native"
                and row["model"] == "dense5"
                and row["event_kind"] == "first_diff"
                and row["effect"] == effect
            ):
                return float(row["mean"])
        raise KeyError(effect)

    return _read


def exp15_llm_pairwise(comparison: str, criterion: str) -> NumberFn:
    key = f"pairwise_{criterion.lower()}"

    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp15_symmetric_behavioral_causality/plots/"
            "exp15_eval_core_600_t512_dense5/exp15_behavior_summary.json",
        )
        return float(data["dense_pairwise"][comparison][key]["target_win_rate_resolved"])

    return _read


def exp15_human_pairwise(comparison: str, criterion: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp15_symmetric_behavioral_causality/human_eval/"
            "human_eval_summary.json",
        )
        for row in data["pairwise_primary"]:
            if row["comparison"] == comparison and row["criterion"] == criterion:
                return float(row["human_resolved_target_rate"])
        raise KeyError((comparison, criterion))

    return _read


CHECKS: list[ClaimCheck] = [
    ClaimCheck(
        "Six-family tuned final-half convergence gap",
        "exp09 convergence_gap_values.json",
        0.3979263649605938,
        exp9_cg("tuned"),
    ),
    ClaimCheck(
        "Six-family raw final-half convergence gap",
        "exp09 convergence_gap_values.json",
        0.729235079781585,
        exp9_cg("raw"),
    ),
    ClaimCheck(
        "Raw final-half convergence gap excluding Gemma and DeepSeek",
        "exp09 convergence_gap_values.json",
        0.7120960062343834,
        exp9_cg("raw", exclude={"gemma3_4b", "deepseek_v2_lite"}),
    ),
    ClaimCheck(
        "Matched-prefix JS(A', C), pre-corrective",
        "exp16 js_summary.json",
        0.10566313034221163,
        exp16_js("JS_AC", "pre_corrective"),
    ),
    ClaimCheck(
        "Matched-prefix JS(A', C), final 20%",
        "exp16 js_summary.json",
        0.16909400464260788,
        exp16_js("JS_AC", "final_20pct"),
    ),
    ClaimCheck(
        "Depth ablation early graft final-20% KL delta",
        "exp11 depth_ablation_metrics.json",
        -0.034725368342970596,
        exp11_depth("B_early_raw", "final_20pct_delta_kl_to_own_final"),
    ),
    ClaimCheck(
        "Depth ablation middle graft final-20% KL delta",
        "exp11 depth_ablation_metrics.json",
        -0.045361678671667495,
        exp11_depth("B_mid_raw", "final_20pct_delta_kl_to_own_final"),
    ),
    ClaimCheck(
        "Depth ablation late graft final-20% KL delta",
        "exp11 depth_ablation_metrics.json",
        0.34144877467282564,
        exp11_depth("B_late_raw", "final_20pct_delta_kl_to_own_final"),
    ),
    ClaimCheck(
        "Symmetric PT-side late graft KL delta",
        "exp14 exp13_full_summary.json",
        0.3376970321801641,
        exp14_dense("pt_side_final20_kl_delta", "B_late_raw"),
    ),
    ClaimCheck(
        "Symmetric IT-side late PT-swap KL delta",
        "exp14 exp13_full_summary.json",
        -0.5086195820051482,
        exp14_dense("it_side_final20_kl_delta", "D_late_ptswap"),
    ),
    ClaimCheck(
        "Late random-control true KL effect",
        "exp19 exp19B_summary_light.json",
        0.3271128779620055,
        exp19_random("late", "true_delta"),
    ),
    ClaimCheck(
        "Late random-control residual-projection KL effect",
        "exp19 exp19B_summary_light.json",
        0.003236096460556932,
        exp19_random("late", "rand_resproj_delta"),
    ),
    ClaimCheck(
        "Late random-control true-minus-random specificity",
        "exp19 exp19B_summary_light.json",
        0.32387678150144855,
        exp19_random("late", "true_minus_rand_resproj"),
    ),
    ClaimCheck(
        "Raw-shared PT+IT mid IT-token fraction",
        "exp20 factorial validation summary.json",
        0.2604760308414348,
        exp20_condition("raw_shared", "B_mid_raw", "it"),
    ),
    ClaimCheck(
        "Raw-shared PT+IT late IT-token fraction",
        "exp20 factorial validation summary.json",
        0.1756620851491787,
        exp20_condition("raw_shared", "B_late_raw", "it"),
    ),
    ClaimCheck(
        "Raw-shared IT+PT mid PT-token fraction",
        "exp20 factorial validation summary.json",
        0.31210191082802546,
        exp20_condition("raw_shared", "D_mid_ptswap", "pt"),
    ),
    ClaimCheck(
        "Raw-shared IT+PT late PT-token fraction",
        "exp20 factorial validation summary.json",
        0.20817968488099228,
        exp20_condition("raw_shared", "D_late_ptswap", "pt"),
    ),
    ClaimCheck(
        "Native IT-host late-swap margin drop",
        "exp20 factorial validation summary.json",
        13.247869201660155,
        exp20_margin_drop("D_late_ptswap"),
        tolerance=1e-3,
    ),
    ClaimCheck(
        "Pure IT late MLP support for IT token",
        "exp21 summary.json",
        0.7891920499525173,
        exp21_window("C_it_chat", "exp11_late", "support_it_token"),
    ),
    ClaimCheck(
        "Pure IT late full-update IT-vs-PT margin",
        "exp21 summary.json",
        0.7679580554758016,
        exp21_window("C_it_chat", "exp11_late", "margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "Pure IT late negative-parallel IT-vs-PT margin",
        "exp21 summary.json",
        -0.004641470088490418,
        exp21_window("C_it_chat", "exp11_late", "opposition_margin_it_vs_pt"),
    ),
    ClaimCheck(
        "PT-host late-graft MLP margin gain",
        "exp21 effects.csv",
        0.0035434878629669805,
        exp21_effect("B_late_minus_A:margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "PT-host mid+late minus mid MLP margin gain",
        "exp21 effects.csv",
        0.011877210732542606,
        exp21_effect("B_midlate_minus_B_mid:margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "Late-weight main effect on MLP margin",
        "exp21 effects.csv",
        0.14779308325151191,
        exp21_effect("late_weight_effect:margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "Upstream-context main effect on MLP margin",
        "exp21 effects.csv",
        0.402698090194721,
        exp21_effect("upstream_context_effect:margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "Late-weight x upstream-context interaction on MLP margin",
        "exp21 effects.csv",
        0.2884991907770899,
        exp21_effect("late_interaction:margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "LLM judge resolved G2: PT late graft over PT baseline",
        "exp15 behavior summary.json",
        0.5631364562118126,
        exp15_llm_pairwise("pt_late_vs_a", "g2"),
    ),
    ClaimCheck(
        "LLM judge resolved G2: IT baseline over late PT swap",
        "exp15 behavior summary.json",
        0.7714025500910747,
        exp15_llm_pairwise("it_c_vs_dlate", "g2"),
    ),
    ClaimCheck(
        "Human resolved G2: IT baseline over late PT swap",
        "exp15 human_eval_summary.json",
        0.705607476635514,
        exp15_human_pairwise("it_c_vs_dlate", "G2"),
    ),
    ClaimCheck(
        "Human resolved G2: PT late graft over PT baseline",
        "exp15 human_eval_summary.json",
        0.6054421768707483,
        exp15_human_pairwise("pt_late_vs_a", "G2"),
    ),
]


def fmt(value: float, digits: int) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def run(repo: Path) -> int:
    rows: list[tuple[ClaimCheck, float | None, str, str]] = []
    failures = 0
    for check in CHECKS:
        try:
            observed = check.observed_fn(repo)
            delta = abs(observed - check.expected)
            status = "PASS" if delta <= check.tolerance else "FAIL"
            if status == "FAIL":
                failures += 1
            rows.append((check, observed, fmt(delta, check.digits), status))
        except Exception as exc:  # pragma: no cover - CLI reporting path
            failures += 1
            rows.append((check, None, type(exc).__name__, "ERROR"))

    headers = ["status", "claim", "observed", "expected", "abs_delta", "source"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join("---" for _ in headers) + "|")
    for check, observed, delta, status in rows:
        observed_s = "missing" if observed is None else fmt(observed, check.digits)
        expected_s = fmt(check.expected, check.digits)
        print(
            "| "
            + " | ".join(
                [
                    status,
                    check.claim,
                    observed_s,
                    expected_s,
                    delta,
                    check.source,
                ]
            )
            + " |"
        )

    print()
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} checks passed.")
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to two directories above this script.",
    )
    args = parser.parse_args()
    return run(args.repo_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
