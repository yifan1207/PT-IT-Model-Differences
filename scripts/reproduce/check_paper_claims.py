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


def exp22_table(key: str, column: str = "estimate_it_minus_pt") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, "results/paper_synthesis/exp22_endpoint_deconfounded_table.csv")
        for row in rows:
            if row["key"] == key:
                return float(row[column])
        raise KeyError(key)

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


def exp21_content_effect(effect: str, field: str = "mean") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp21_productive_opposition/"
            "exp21_content_reasoning_20260427_0943_h100x8/"
            "analysis/effects.csv",
        )
        for row in rows:
            if (
                row["mode"] == "raw_shared"
                and row["model"] == "dense5"
                and row["event_kind"] == "first_diff"
                and row["effect"] == effect
            ):
                return float(row[field])
        raise KeyError((effect, field))

    return _read


def exp23_effect(run_name: str, effect: str, field: str = "estimate") -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp23_midlate_interaction_suite/"
            f"{run_name}/analysis/exp23_summary.json",
        )
        return float(data["residual_factorial"]["effects"]["common_it"][effect][field])

    return _read


def exp23_subgroup(
    run_name: str, group: str, value: str, field: str = "estimate"
) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp23_midlate_interaction_suite/"
            f"{run_name}/analysis/subgroups/exp23_subgroup_summary.json",
        )
        for row in data["rows"]:
            if (
                row["readout"] == "common_it"
                and row["effect"] == "interaction"
                and row["group"] == group
                and row["value"] == value
            ):
                return float(row[field])
        raise KeyError((run_name, group, value, field))

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
        "Dense-5 tuned final-half convergence gap",
        "exp09 convergence_gap_values.json",
        0.40973461108427894,
        exp9_cg("tuned", exclude={"deepseek_v2_lite"}),
    ),
    ClaimCheck(
        "Dense-5 raw final-half convergence gap",
        "exp09 convergence_gap_values.json",
        0.7712311073328106,
        exp9_cg("raw", exclude={"deepseek_v2_lite"}),
    ),
    ClaimCheck(
        "Dense-5 raw final-half convergence gap excluding Gemma",
        "exp09 convergence_gap_values.json",
        0.7120960062343834,
        exp9_cg("raw", exclude={"gemma3_4b", "deepseek_v2_lite"}),
    ),
    ClaimCheck(
        "Endpoint-matched raw late KL gap",
        "exp22_endpoint_deconfounded_table.csv",
        0.42507813938209227,
        exp22_table("endpoint_matched_raw_late_kl"),
    ),
    ClaimCheck(
        "Endpoint-matched raw late KL gap CI low",
        "exp22_endpoint_deconfounded_table.csv",
        0.3558056663988748,
        exp22_table("endpoint_matched_raw_late_kl", "ci95_low"),
    ),
    ClaimCheck(
        "Endpoint-matched raw late KL gap CI high",
        "exp22_endpoint_deconfounded_table.csv",
        0.49267175872107266,
        exp22_table("endpoint_matched_raw_late_kl", "ci95_high"),
    ),
    ClaimCheck(
        "Endpoint-matched tuned late KL gap",
        "exp22_endpoint_deconfounded_table.csv",
        0.762124299866896,
        exp22_table("endpoint_matched_tuned_late_kl"),
    ),
    ClaimCheck(
        "Endpoint-matched tuned late KL gap CI low",
        "exp22_endpoint_deconfounded_table.csv",
        0.7091858737942381,
        exp22_table("endpoint_matched_tuned_late_kl", "ci95_low"),
    ),
    ClaimCheck(
        "Endpoint-matched tuned late KL gap CI high",
        "exp22_endpoint_deconfounded_table.csv",
        0.8141641585027306,
        exp22_table("endpoint_matched_tuned_late_kl", "ci95_high"),
    ),
    ClaimCheck(
        "Endpoint-matched remaining adjacent JS gap",
        "exp22_endpoint_deconfounded_table.csv",
        0.052171032353954726,
        exp22_table("endpoint_free_remaining_adj_js"),
    ),
    ClaimCheck(
        "Endpoint-matched future top-1 flips gap",
        "exp22_endpoint_deconfounded_table.csv",
        0.20296685082331023,
        exp22_table("endpoint_free_future_top1_flips"),
    ),
    ClaimCheck(
        "Endpoint matching minimum retained fraction",
        "exp22_endpoint_deconfounded_table.csv",
        0.7962578113120108,
        exp22_table("quality_min_matched_retention"),
    ),
    ClaimCheck(
        "Endpoint matching max post-match SMD",
        "exp22_endpoint_deconfounded_table.csv",
        0.056949274990143434,
        exp22_table("quality_max_smd_after"),
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
        "Exp23 primary late given PT upstream",
        "exp23 primary exp23_summary.json",
        0.5720075457553511,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "late_it_given_pt_upstream",
        ),
    ),
    ClaimCheck(
        "Exp23 primary late given IT upstream",
        "exp23 primary exp23_summary.json",
        3.2072035352029644,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "late_it_given_it_upstream",
        ),
    ),
    ClaimCheck(
        "Exp23 primary upstream-context effect",
        "exp23 primary exp23_summary.json",
        4.238709540575966,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "upstream_context_effect",
        ),
    ),
    ClaimCheck(
        "Exp23 primary late-stack main effect",
        "exp23 primary exp23_summary.json",
        1.8896055404791579,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "late_weight_effect",
        ),
    ),
    ClaimCheck(
        "Exp23 primary upstream x late interaction",
        "exp23 primary exp23_summary.json",
        2.6351959894476136,
        exp23_effect("exp23_dense5_full_h100x8_20260426_sh4_rw4", "interaction"),
    ),
    ClaimCheck(
        "Exp23 primary interaction CI low",
        "exp23 primary exp23_summary.json",
        2.5404429873591,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 primary interaction CI high",
        "exp23 primary exp23_summary.json",
        2.7335680538313816,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 holdout GOV-CONV subgroup interaction",
        "exp23 primary subgroup summary.json",
        2.0509100734312526,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "prompt_category",
            "GOV-CONV",
        ),
    ),
    ClaimCheck(
        "Exp23 holdout GOV-FORMAT subgroup interaction",
        "exp23 primary subgroup summary.json",
        3.61092726012482,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "prompt_category",
            "GOV-FORMAT",
        ),
    ),
    ClaimCheck(
        "Exp23 holdout SAFETY subgroup interaction",
        "exp23 primary subgroup summary.json",
        2.8332766414218478,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "prompt_category",
            "SAFETY",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction",
        "exp23 content/reasoning exp23_summary.json",
        1.6803632792866057,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction CI low",
        "exp23 content/reasoning exp23_summary.json",
        1.5929663162872967,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction CI high",
        "exp23 content/reasoning exp23_summary.json",
        1.7752462505766033,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning upstream-context effect",
        "exp23 content/reasoning exp23_summary.json",
        3.529437865202374,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "upstream_context_effect",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning late given PT upstream",
        "exp23 content/reasoning exp23_summary.json",
        -1.4062707862194421,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "late_it_given_pt_upstream",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning late given IT upstream",
        "exp23 content/reasoning exp23_summary.json",
        0.27409249306716343,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "late_it_given_it_upstream",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning late-stack main effect",
        "exp23 content/reasoning exp23_summary.json",
        -0.5660891465761394,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "late_weight_effect",
        ),
    ),
    ClaimCheck(
        "Exp23 CONTENT-FACT subgroup interaction",
        "exp23 content/reasoning subgroup summary.json",
        1.8421921828806194,
        exp23_subgroup(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "prompt_category",
            "CONTENT-FACT",
        ),
    ),
    ClaimCheck(
        "Exp23 CONTENT-REASON subgroup interaction",
        "exp23 content/reasoning subgroup summary.json",
        1.336117588486396,
        exp23_subgroup(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "prompt_category",
            "CONTENT-REASON",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning GOV-FORMAT subgroup interaction",
        "exp23 content/reasoning subgroup summary.json",
        1.9727237747609636,
        exp23_subgroup(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "prompt_category",
            "GOV-FORMAT",
        ),
    ),
    ClaimCheck(
        "Exp21 content/reasoning late-weight MLP margin",
        "exp21 content/reasoning effects.csv",
        0.041969978748188225,
        exp21_content_effect("late_weight_effect:margin_writein_it_vs_pt"),
    ),
    ClaimCheck(
        "Exp21 content/reasoning residual-opposing component",
        "exp21 content/reasoning effects.csv",
        -5.502802526305325e-06,
        exp21_content_effect("late_weight_effect:opposition_margin_it_vs_pt"),
        tolerance=5e-7,
        digits=8,
    ),
    ClaimCheck(
        "Exp21 content/reasoning remaining/token-specific component",
        "exp21 content/reasoning effects.csv",
        0.04881583150857773,
        exp21_content_effect("late_weight_effect:remainder_margin_it_vs_pt"),
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
