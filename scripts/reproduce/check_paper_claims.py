#!/usr/bin/env python3
"""Check manuscript headline numbers against committed summary artifacts.

This is intentionally a summary-artifact audit. It does not rerun GPU
generation; it verifies that the numbers quoted in the paper can be recovered
mechanically from the released JSON/CSV files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


NumberFn = Callable[[Path], float]
_EXP43_NONOUTLIER_CACHE: dict[tuple[str, str, str], float] = {}


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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty list")
    pos = (len(ordered) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def exact_family_bootstrap_ci(values: list[float], field: str) -> float:
    n = len(values)
    boot_means = [
        mean([values[idx] for idx in sample])
        for sample in itertools.product(range(n), repeat=n)
    ]
    if field == "ci95_low":
        return percentile(boot_means, 2.5)
    if field == "ci95_high":
        return percentile(boot_means, 97.5)
    raise KeyError(field)


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


def exp16_js(
    pair: str,
    region: str,
    agg: str = "regions_weighted",
    field: str = "mean",
) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp16_matched_prefix_js_gap/"
            "exp16_js_replay_runpod_20260422_075307/js_summary.json",
        )
        return float(data["dense5"]["pairs"][pair][agg][region][field])

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


def exp11_depth_family_ci(condition: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp11_matched_prefix_mlp_graft/plots/"
            "exp11_exp3_600rand_v11_depthablation_full/"
            "depth_ablation_metrics.json",
        )
        values = [
            float(
                row["pipelines"][condition]["regions"]["final_20pct"][
                    "kl_to_own_final"
                ]["delta"]
            )
            for row in data["models"]
            if row["model"] != "deepseek_v2_lite"
        ]
        return exact_family_bootstrap_ci(values, field)

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


def exp14_dense_family_ci(group: str, condition: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp14_symmetric_matched_prefix_causality/"
            "exp13exp14_full_20260416/exp13_full_summary.json",
        )
        values = [
            float(
                row[group][condition]["regions"]["final_20pct"]["kl_to_own_final"][
                    "delta"
                ]
            )
            for model, row in data["models"].items()
            if model != "deepseek_v2_lite"
        ]
        return exact_family_bootstrap_ci(values, field)

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


def exp23_model_effect(
    run_name: str,
    effect: str,
    model: str,
    field: str = "estimate",
) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp23_midlate_interaction_suite/"
            f"{run_name}/analysis/exp23_summary.json",
        )
        return float(
            data["residual_factorial"]["effects"]["common_it"][effect]["model_cis"][
                model
            ][field]
        )

    return _read


def exp23_model_summary(
    run_name: str,
    effect: str,
    summary: str,
    *,
    exclude: set[str] | None = None,
) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp23_midlate_interaction_suite/"
            f"{run_name}/analysis/exp23_summary.json",
        )
        model_cis = data["residual_factorial"]["effects"]["common_it"][effect][
            "model_cis"
        ]
        values = [
            float(row["estimate"])
            for model, row in model_cis.items()
            if not exclude or model not in exclude
        ]
        if summary == "median":
            return percentile(values, 50)
        if summary == "trimmed_minmax_mean":
            ordered = sorted(values)
            if len(ordered) < 3:
                raise ValueError("trimmed_minmax_mean requires at least 3 values")
            return mean(ordered[1:-1])
        if summary == "mean":
            return mean(values)
        raise KeyError(summary)

    return _read


def exp23_leave_one_out(
    run_name: str,
    effect: str,
    held_out_model: str,
    field: str = "estimate",
) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp23_midlate_interaction_suite/"
            f"{run_name}/analysis/exp23_summary.json",
        )
        return float(
            data["residual_factorial"]["effects"]["common_it"][effect][
                "leave_one_out"
            ][held_out_model][field]
        )

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


def exp23_compatibility(run_name: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp23_midlate_interaction_suite/"
            f"{run_name}/analysis/compatibility_permutation/"
            "exp23_compatibility_permutation_summary.json",
        )
        if metric in data["summary"]["pooled_model_mean"]:
            return float(data["summary"]["pooled_model_mean"][metric])
        if metric == "p_upper":
            return float(data["permutation"]["p_upper"])
        if metric == "null_q99.9":
            return float(data["permutation"]["null_quantiles"]["q99.9"])
        raise KeyError((run_name, metric))

    return _read


def exp23_position_sensitivity(stratum: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_position_sensitivity_table.csv",
        )
        for row in rows:
            if row["position_stratum"] == stratum:
                return float(row[column])
        raise KeyError((stratum, column))

    return _read


def exp23_dense6_effect(effect: str, readout: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_dense6_core/exp23_dense6_core_effects.csv",
        )
        for row in rows:
            if row["effect"] == effect and row["readout"] == readout:
                return float(row[column])
        raise KeyError((effect, readout, column))

    return _read


def exp23_dense6_native_shift(readout: str, scope: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_dense6_core/exp23_dense6_core_effects.csv",
        )
        by_effect = {
            row["effect"]: row
            for row in rows
            if row["readout"] == readout
        }
        column = f"{scope}_estimate"
        return (
            float(by_effect["late_weight_effect"][column])
            + float(by_effect["upstream_context_effect"][column])
        )

    return _read


def exp23_dense6_interaction_share(readout: str, scope: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_dense6_core/exp23_dense6_core_effects.csv",
        )
        by_effect = {
            row["effect"]: row
            for row in rows
            if row["readout"] == readout
        }
        column = f"{scope}_estimate"
        native = (
            float(by_effect["late_weight_effect"][column])
            + float(by_effect["upstream_context_effect"][column])
        )
        return float(by_effect["interaction"][column]) / native

    return _read


def exp23_dense6_late_effect_amplification(readout: str, scope: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_dense6_core/exp23_dense6_core_effects.csv",
        )
        by_effect = {
            row["effect"]: row
            for row in rows
            if row["readout"] == readout
        }
        column = f"{scope}_estimate"
        return (
            float(by_effect["late_it_given_it_upstream"][column])
            / float(by_effect["late_it_given_pt_upstream"][column])
        )

    return _read


def exp23_dense6_family_interaction_stat(stat: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_dense6_core/exp23_dense6_family_effects.csv",
        )
        values = [
            float(row["interaction"])
            for row in rows
            if row["readout"] == "common_it"
        ]
        if stat == "min":
            return min(values)
        if stat == "max":
            return max(values)
        if stat == "median":
            ordered = sorted(values)
            mid = len(ordered) // 2
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        raise KeyError(stat)

    return _read


def exp23_dense6_family_share_stat(stat: str, *, exclude: set[str] | None = None) -> NumberFn:
    def _read(repo: Path) -> float:
        values: list[float] = []
        sources = [
            "results/exp23_midlate_interaction_suite/"
            "exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_effects.csv",
            "results/exp24_32b_external_validity/"
            "exp24_qwen25_32b_full_eval_v21_20260427_194839/analysis/"
            "exp23_midlate_interaction_suite/exp23_effects.csv",
        ]
        for source in sources:
            rows = load_csv(repo, source)
            family_names = sorted(
                {
                    row["scope"].split(":", 1)[1]
                    for row in rows
                    if row["readout"] == "common_it"
                    and row["scope"].startswith("family:")
                }
            )
            for model in family_names:
                if exclude and model in exclude:
                    continue
                by_effect = {
                    row["effect"]: row
                    for row in rows
                    if row["readout"] == "common_it"
                    and row["scope"] == f"family:{model}"
                }
                native = (
                    float(by_effect["late_weight_effect"]["estimate"])
                    + float(by_effect["upstream_context_effect"]["estimate"])
                )
                values.append(float(by_effect["interaction"]["estimate"]) / native)
        if stat == "min":
            return min(values)
        if stat == "max":
            return max(values)
        if stat == "median":
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[mid]
            return (ordered[mid - 1] + ordered[mid]) / 2.0
        raise KeyError(stat)

    return _read


def exp23_dense6_position(stratum: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_dense6_core/exp23_dense6_position_sensitivity.csv",
        )
        for row in rows:
            if row["position_stratum"] == stratum:
                return float(row[column])
        raise KeyError((stratum, column))

    return _read


def exp40_effect(scope: str, metric: str, column: str = "estimate", readout: str = "common_it") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp40_prelate_commitment_control/"
            "exp40_exp20_layerwise_proxy_20260503_110001/analysis/effects.csv",
        )
        for row in rows:
            if (
                row["readout"] == readout
                and row["analysis"] == "scope"
                and row["scope"] == scope
                and row["metric"] == metric
            ):
                return float(row[column])
        raise KeyError((scope, metric, column, readout))

    return _read


def exp40_regression(scope: str, metric: str, column: str = "estimate", readout: str = "common_it") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp40_prelate_commitment_control/"
            "exp40_exp20_layerwise_proxy_20260503_110001/analysis/effects.csv",
        )
        for row in rows:
            if (
                row["readout"] == readout
                and row["analysis"] == "regression"
                and row["scope"] == scope
                and row["metric"] == metric
            ):
                return float(row[column])
        raise KeyError((scope, metric, column, readout))

    return _read


def exp36_summary(readout: str, metric: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp36_offmanifold_validation/"
            "exp36_offmanifold_dense5_full_a100x8_20260502_233904/"
            "analysis/summary.json",
        )
        value = data["readouts"][readout][metric]
        if column:
            value = value[column]
        return float(value)

    return _read


def exp36_anomaly(subset: str, readout: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp36_offmanifold_validation/"
            "exp36_offmanifold_dense5_full_a100x8_20260502_233904/"
            "analysis/anomaly_effects.csv",
        )
        for row in rows:
            if row["subset"] == subset and row["readout"] == readout:
                return float(row[column])
        raise KeyError((subset, readout, column))

    return _read


def exp36_random_control(readout: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp36_offmanifold_validation/"
            "exp36_offmanifold_dense5_full_a100x8_20260502_233904/"
            "analysis/random_control_effects.csv",
        )
        for row in rows:
            if (
                row["scope"] == "Dense-5 family mean"
                and row["readout"] == readout
                and row["control"] == "signed_permutation"
            ):
                return float(row[column])
        raise KeyError((readout, column))

    return _read


def exp37_effect(key: str, column: str, readout: str = "common_it") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp37_random_prefix_baseline/"
            "exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/"
            "analysis/effects.csv",
        )
        for row in rows:
            if (
                row["key"] == key
                and row["readout"] == readout
                and row["effect"] == "interaction"
                and row["scope"] == "dense5"
            ):
                return float(row[column])
        raise KeyError((key, column, readout))

    return _read


def exp37_token_support(scope: str, metric: str, column: str = "fraction") -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp37_random_prefix_baseline/"
            "exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/"
            "analysis/token_support_control/summary.json",
        )
        return float(data["comparison"][scope]["family_balanced"][metric][column])

    return _read


def token_support_bool(metric: str, column: str = "fraction") -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/first_divergence_token_support/"
            "dense5_llm_gpt55_20260503_121500/summary.json",
        )
        return float(data["deterministic"]["boolean_support"][metric][column])

    return _read


def token_support_weighted(field: str, value: str, column: str = "fraction") -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/first_divergence_token_support/"
            "dense5_llm_gpt55_20260503_121500/summary.json",
        )
        rows = data["llm"]["population_weighted"][field]
        for row in rows:
            if row["value"] == value:
                return float(row[column])
        raise KeyError((field, value, column))

    return _read


def token_support_llm_n() -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/first_divergence_token_support/"
            "dense5_llm_gpt55_20260503_121500/summary.json",
        )
        return float(data["llm"]["n"])

    return _read


def exp23_position_family(stratum: str, model: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_position_sensitivity_per_family.csv",
        )
        for row in rows:
            if row["position_stratum"] == stratum and row["model"] == model:
                return float(row[column])
        raise KeyError((stratum, model, column))

    return _read


def exp23_dense5_repro_snapshot(model: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        path = (
            repo
            / "results/exp23_midlate_interaction_suite/"
            / "exp23_dense5_full_h100x8_20260426_sh4_rw4/residual_factorial/raw_shared"
            / model
            / "records.jsonl.gz"
        )
        total = pos0 = ge3 = ge5 = 0
        with gzip.open(path, "rt") as f:
            for line in f:
                row = json.loads(line)
                event = (row.get("events") or {}).get("first_diff") or {}
                if not event.get("valid"):
                    continue
                step = int(event["event"]["step"])
                total += 1
                pos0 += step == 0
                ge3 += step >= 3
                ge5 += step >= 5
        if field == "events":
            return float(total)
        if field == "pos0_frac":
            return pos0 / total
        if field == "ge3_frac":
            return ge3 / total
        if field == "ge5_frac":
            return ge5 / total
        raise KeyError((model, field))

    return _read


def exp24_repro_snapshot(field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp24_32b_external_validity/"
            "exp24_32b_position_sensitivity.csv",
        )
        counts: dict[str, float] = {}
        for row in rows:
            if row["metric"] == "interaction":
                counts[row["position_filter"]] = float(row["n_records"])
        total = counts["all"]
        if field == "events":
            return total
        if field == "pos0_frac":
            return counts["step_0"] / total
        if field == "ge3_frac":
            return counts["step_ge3"] / total
        if field == "ge5_frac":
            return counts["step_ge5"] / total
        raise KeyError(field)

    return _read


def exp23_position_category_mix(position_filter: str, axis: str, category: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_position_category_mix.csv",
        )
        for row in rows:
            if (
                row["position_filter"] == position_filter
                and row["axis"] == axis
                and row["category"] == category
            ):
                return float(row[column])
        raise KeyError((position_filter, axis, category, column))

    return _read


def exp23_position_prompt_category(position_filter: str, prompt_category: str, metric: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp23_position_prompt_category_effects.csv",
        )
        for row in rows:
            if (
                row["position_filter"] == position_filter
                and row["prompt_category"] == prompt_category
                and row["metric"] == metric
            ):
                return float(row[column])
        raise KeyError((position_filter, prompt_category, metric, column))

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


def exp24_residual_effect(readout: str, effect: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp24_32b_external_validity/"
            "exp24_qwen25_32b_full_eval_v21_20260427_194839/analysis/"
            "exp23_midlate_interaction_suite/exp23_summary.json",
        )
        return float(data["residual_factorial"]["effects"][readout][effect][column])

    return _read


def exp24_late_effect_amplification(readout: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp24_32b_external_validity/"
            "exp24_qwen25_32b_full_eval_v21_20260427_194839/analysis/"
            "exp23_midlate_interaction_suite/exp23_summary.json",
        )
        effects = data["residual_factorial"]["effects"][readout]
        return (
            float(effects["late_it_given_it_upstream"]["estimate"])
            / float(effects["late_it_given_pt_upstream"]["estimate"])
        )

    return _read


def exp24_kl_effect(side: str, effect: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp24_32b_external_validity/"
            "exp24_qwen25_32b_full_eval_v21_20260427_194839/analysis/"
            "part_a_mlp_kl/exp23_midlate_kl_factorial_effects.csv",
        )
        for row in rows:
            if row["model"] == "qwen25_32b" and row["side"] == side and row["effect"] == effect:
                return float(row[column])
        raise KeyError((side, effect, column))

    return _read


def exp24_position(position_filter: str, metric: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp24_32b_external_validity/"
            "exp24_32b_position_sensitivity.csv",
        )
        for row in rows:
            if row["position_filter"] == position_filter and row["metric"] == metric:
                return float(row[column])
        raise KeyError((position_filter, metric, column))

    return _read


def exp24_position_prompt_category(position_filter: str, prompt_category: str, metric: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp24_32b_external_validity/"
            "exp24_32b_position_prompt_category_effects.csv",
        )
        for row in rows:
            if (
                row["position_filter"] == position_filter
                and row["prompt_category"] == prompt_category
                and row["metric"] == metric
            ):
                return float(row[column])
        raise KeyError((position_filter, prompt_category, metric, column))

    return _read


def exp25_olmo_stage(transition: str, column: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/paper_synthesis/exp25_olmo_stage_full_20260428_0905/"
            "olmo_stage_progression_table.csv",
        )
        for row in rows:
            if row["transition"] == transition:
                return float(row[column])
        raise KeyError((transition, column))

    return _read


def exp27_primary(variant: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp27_natural_rollout_residual_opposition_ntp/"
            "exp27_full_dense5_combined_20260430_2050/"
            "analysis/exp27_summary.json",
        )
        return float(data["primary"][variant][field])

    return _read


def exp34_model(model: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/paper_synthesis/exp34_dense5_final_readout_crosscoder/"
            "combined_dense5_20260503_0018/exp34_dense5_crosscoder_summary.json",
        )
        for row in data["models"]:
            if row["model"] == model:
                return float(row[field])
        raise KeyError((model, field))

    return _read


def exp34_model_drop_share(model: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/paper_synthesis/exp34_dense5_final_readout_crosscoder/"
            "combined_dense5_20260503_0018/exp34_dense5_crosscoder_summary.json",
        )
        for row in data["models"]:
            if row["model"] == model:
                return float(row["causal_top200_interaction_drop"]) / float(row["interaction_full"])
        raise KeyError(model)

    return _read


def exp38_gate(run_key: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_qwen_olmo_final_summary_20260503/analysis/"
            "exp38_qwen_olmo_decision_summary.json",
        )
        return float(data["runs"][run_key]["success_gates"][field])

    return _read


def exp38_qwen_final2_drop_share() -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_qwen3_4b_final2_d81920_k64_20260503_0451_a100x2/"
            "selected_d81920_k64/analysis/summary.json",
        )
        for row in data["effects"]:
            if row.get("feature_set") == "causal_top" and int(row.get("k", -1)) == 200:
                return float(row["interaction_drop_mean"]) / float(row["interaction_full_mean"])
        raise KeyError("qwen_final2 causal_top k=200")

    return _read


def crosscoder_effects_drop_share(path: str, k: int) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, path)
        for row in rows:
            if row.get("feature_set") == "causal_top" and int(float(row.get("k", -1))) == k:
                return float(row["interaction_drop_mean"]) / float(row["interaction_full_mean"])
        raise KeyError((path, k))

    return _read


def exp38_layer(run_key: str, layer: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_qwen_olmo_final_summary_20260503/analysis/"
            "exp38_qwen_olmo_decision_summary.json",
        )
        return float(data["runs"][run_key]["metrics_by_layer"][layer][field])

    return _read


def exp40_terminal_grid_row(path: str, config: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_json(repo, path)
        for row in rows:
            if row["config"] == config:
                return float(row[field])
        raise KeyError((path, config, field))

    return _read


def exp40_terminal_grid_effect(root: str, config: str, feature_set: str, k: int, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(repo, f"{root}/grid/{config}/analysis/effects.csv")
        vals = []
        for row in rows:
            if row["feature_set"] != feature_set:
                continue
            if int(float(row["k"])) != k:
                continue
            vals.append(float(row[field]))
        if not vals:
            raise KeyError((root, config, feature_set, k, field))
        return float(sum(vals) / len(vals))

    return _read


def exp40_olmo30_selected(field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp40_terminal_crosscoder_hardening/"
            "exp40_olmo2_7b_layer30_grid_20260503_1210_a100x4_localshm_compact/"
            "olmo2_7b/layer30_grid/analysis/olmo30_selected_candidate.json",
        )
        return float(data["selected"][field])

    return _read


def exp39_validation(field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp39_causal_feature_interpretation/"
            "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/"
            "autointerp/label_validation.json",
        )
        return float(data[field])

    return _read


def exp39_taxonomy_summary_count(category: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp39_causal_feature_interpretation/"
            "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/"
            "analysis/exp39_causal_paper_taxonomy_summary_v3.json",
        )
        return float(data["overall_counts"].get(category, 0))

    return _read


def exp39_taxonomy_group(group: str) -> NumberFn:
    groups = {
        "instruction_tuned_behavior_readout": {
            "assistant_register_or_user_facing_style",
            "alignment_safety_or_advice_boundary",
            "instruction_following_or_format_control",
            "response_structure_or_answer_readout",
            "evaluation_or_multiple_choice_scaffold",
            "code_or_tool_syntax_readout",
        },
        "surface_tokenization_adjacent": {
            "surface_punctuation_or_tokenization",
        },
        "artifact_or_unclear": {
            "dataset_or_repetition_artifact",
            "rare_unicode_or_web_artifact",
            "generic_frequency_or_unclear",
        },
    }

    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp39_causal_feature_interpretation/"
            "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/"
            "analysis/exp39_causal_paper_taxonomy_summary_v3.json",
        )
        categories = groups[group]
        return float(sum(data["overall_counts"].get(category, 0) for category in categories))

    return _read


def exp39_taxonomy_overall(category: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp39_causal_feature_interpretation/"
            "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/"
            "analysis/causal_paper_taxonomy_overall_v3.csv",
        )
        for row in rows:
            if row["paper_category"] == category:
                return float(row[field])
        raise KeyError((category, field))

    return _read


def exp39_taxonomy_set(name: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp39_causal_feature_interpretation/"
            "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/"
            "analysis/all_paper_taxonomy_set_tests_v4.csv",
        )
        for row in rows:
            if row["set"] == name:
                return float(row[field])
        raise KeyError((name, field))

    return _read


def exp39_taxonomy_category(category: str, field: str) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp39_causal_feature_interpretation/"
            "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345/"
            "analysis/all_paper_taxonomy_category_by_role_v4.csv",
        )
        for row in rows:
            if row["paper_category"] == category:
                return float(row[field])
        raise KeyError((category, field))

    return _read


def exp41_structure_count() -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp41_causal_feature_bucket_steering/"
            "exp41_terminal_bucket_logit_full_h100x8_20260503_1520/"
            "bucket_manifest/strict_primary/bucket_features.csv",
        )
        return float(sum(1 for row in rows if row["bucket"] == "structure_readout" and row["include_primary"] == "True"))

    return _read


def exp41_structure_interaction_drop(alpha: float, condition_kind: str, model: str | None = None) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp41_causal_feature_bucket_steering/"
            "exp41_terminal_bucket_logit_full_h100x8_20260503_1520/"
            "analysis/bucket_effects_by_model.csv",
        )
        vals = []
        for row in rows:
            if row["source_bucket"] != "structure_readout":
                continue
            if row["condition_kind"] != condition_kind:
                continue
            if abs(float(row["alpha"]) - alpha) > 1e-9:
                continue
            if model is not None and row["model"] != model:
                continue
            vals.append(float(row["interaction_drop_mean"]))
        if not vals:
            raise KeyError((alpha, condition_kind, model))
        return float(sum(vals) / len(vals))

    return _read


def exp42_family_estimate(name: str, field: str = "estimate", family: str | None = None) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp42_terminal_feature_upstream_conditioning/"
            "exp42_full_4fam_h100x8_20260503_155212/analysis/"
            "feature_gating_summary.json",
        )
        row = data["family_estimates"][name]
        if family is not None:
            return float(row["family_means"][family])
        return float(row[field])

    return _read


def exp42_effect(feature_set: str, k: int, field: str, model: str | None = None) -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp42_terminal_feature_upstream_conditioning/"
            "exp42_full_4fam_h100x8_20260503_155212/analysis/"
            "feature_gating_effects.csv",
        )
        vals = []
        for row in rows:
            if row["feature_set"] != feature_set:
                continue
            if int(float(row["k"])) != k:
                continue
            if model is not None and row["model"] != model:
                continue
            vals.append(float(row[field]))
        if not vals:
            raise KeyError((feature_set, k, field, model))
        return float(sum(vals) / len(vals))

    return _read


def exp43_family_balanced(effect: str, metric: str, field: str = "estimate") -> NumberFn:
    def _read(repo: Path) -> float:
        rows = load_csv(
            repo,
            "results/exp43_feature_rescue_handoff/"
            "exp43_full_h100x8_clean_20260503_182947/"
            "analysis/primary_family_balanced_effects.csv",
        )
        for row in rows:
            if row["effect"] == effect and row["metric"] == metric:
                return float(row[field])
        raise KeyError((effect, metric, field))

    return _read


def exp43_direct_rescue(model: str, metric: str) -> NumberFn:
    def _read(repo: Path) -> float:
        data = load_json(
            repo,
            "results/exp43_feature_rescue_handoff/"
            "exp43_full_h100x8_clean_20260503_182947/"
            "analysis/exp43_summary.json",
        )
        for row in data["primary_causal_rows"]:
            if row["model"] == model:
                return float(row[metric])
        raise KeyError((model, metric))

    return _read


def _exp43_nonoutlier_values(repo: Path, effect: str, metric: str) -> dict[str, dict[str, list[float]]]:
    models = {"llama31_8b", "mistral_7b", "qwen3_4b"}
    run_root = repo / "results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947"
    grouped: dict[str, dict[str, list[float]]] = {model: {} for model in models}
    if effect == "causal_top":
        for model in sorted(models):
            path = run_root / "raw" / model / "records.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    if row.get("record_type") != "rescue":
                        continue
                    if row.get("feature_set") != "causal_top" or row.get("control_mode") != "feature_delta":
                        continue
                    if int(row.get("k", 0)) != 200 or abs(float(row.get("alpha", 0.0)) - 1.0) > 1e-9:
                        continue
                    value = row.get(metric)
                    if value is not None and math.isfinite(float(value)):
                        grouped[model].setdefault(str(row["prompt_id"]), []).append(float(value))
    else:
        control = {
            "causal_minus_causal_matched_random": "causal_matched_random",
            "causal_minus_causal_same_delta_random": "causal_same_delta_random",
        }[effect]
        rows = load_csv(
            repo,
            "results/exp43_feature_rescue_handoff/"
            "exp43_full_h100x8_clean_20260503_182947/"
            "analysis/rescue_control_differences_event_rows.csv",
        )
        for row in rows:
            model = row["model"]
            if model not in models:
                continue
            if row["control"] != control:
                continue
            if int(float(row["k"])) != 200 or abs(float(row["alpha"]) - 1.0) > 1e-9:
                continue
            value = row.get(metric)
            if value not in (None, "") and math.isfinite(float(value)):
                grouped[model].setdefault(str(row["prompt_id"]), []).append(float(value))
    return grouped


def _exp43_nonoutlier_family_balanced(repo: Path, effect: str, metric: str, field: str) -> float:
    key = (effect, metric, field)
    if key in _EXP43_NONOUTLIER_CACHE:
        return _EXP43_NONOUTLIER_CACHE[key]
    grouped = _exp43_nonoutlier_values(repo, effect, metric)
    family_means = []
    for model in sorted(grouped):
        by_prompt = grouped[model]
        vals = [value for cluster in by_prompt.values() for value in cluster]
        family_means.append(mean(vals))
    estimate = mean(family_means)
    if field == "estimate":
        _EXP43_NONOUTLIER_CACHE[key] = estimate
        return estimate

    rng = random.Random(0)
    boot = []
    for _ in range(5000):
        sampled_family_means = []
        for model in sorted(grouped):
            by_prompt = grouped[model]
            clusters = list(by_prompt.values())
            sampled = []
            for _cluster in clusters:
                sampled.extend(rng.choice(clusters))
            sampled_family_means.append(mean(sampled))
        boot.append(mean(sampled_family_means))
    if field == "ci_low":
        out = percentile(boot, 2.5)
    elif field == "ci_high":
        out = percentile(boot, 97.5)
    else:
        raise KeyError(field)
    _EXP43_NONOUTLIER_CACHE[key] = out
    return out


def exp43_nonoutlier_family_balanced(effect: str, metric: str, field: str = "estimate") -> NumberFn:
    def _read(repo: Path) -> float:
        return _exp43_nonoutlier_family_balanced(repo, effect, metric, field)

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
        "Matched-prefix prompt-mean JS(A', C), pre-corrective",
        "exp16 js_summary.json",
        0.12087137830346867,
        exp16_js("JS_AC", "pre_corrective", "regions_prompt_mean"),
    ),
    ClaimCheck(
        "Matched-prefix prompt-mean JS(A', C), pre-corrective CI low",
        "exp16 js_summary.json",
        0.11868562148698142,
        exp16_js("JS_AC", "pre_corrective", "regions_prompt_mean", "ci95_low"),
    ),
    ClaimCheck(
        "Matched-prefix prompt-mean JS(A', C), pre-corrective CI high",
        "exp16 js_summary.json",
        0.12310125203197614,
        exp16_js("JS_AC", "pre_corrective", "regions_prompt_mean", "ci95_high"),
    ),
    ClaimCheck(
        "Matched-prefix prompt-mean JS(A', C), final 20%",
        "exp16 js_summary.json",
        0.19558364230213074,
        exp16_js("JS_AC", "final_20pct", "regions_prompt_mean"),
    ),
    ClaimCheck(
        "Matched-prefix prompt-mean JS(A', C), final 20% CI low",
        "exp16 js_summary.json",
        0.19271804235830423,
        exp16_js("JS_AC", "final_20pct", "regions_prompt_mean", "ci95_low"),
    ),
    ClaimCheck(
        "Matched-prefix prompt-mean JS(A', C), final 20% CI high",
        "exp16 js_summary.json",
        0.19848691088959977,
        exp16_js("JS_AC", "final_20pct", "regions_prompt_mean", "ci95_high"),
    ),
    ClaimCheck(
        "Depth ablation early graft final-20% KL delta",
        "exp11 depth_ablation_metrics.json",
        -0.034725368342970596,
        exp11_depth("B_early_raw", "final_20pct_delta_kl_to_own_final"),
    ),
    ClaimCheck(
        "Depth ablation early graft final-20% KL delta CI low",
        "exp11 depth_ablation_metrics.json",
        -0.09601013479360884,
        exp11_depth_family_ci("B_early_raw", "ci95_low"),
    ),
    ClaimCheck(
        "Depth ablation early graft final-20% KL delta CI high",
        "exp11 depth_ablation_metrics.json",
        0.018944847273831518,
        exp11_depth_family_ci("B_early_raw", "ci95_high"),
    ),
    ClaimCheck(
        "Depth ablation middle graft final-20% KL delta",
        "exp11 depth_ablation_metrics.json",
        -0.045361678671667495,
        exp11_depth("B_mid_raw", "final_20pct_delta_kl_to_own_final"),
    ),
    ClaimCheck(
        "Depth ablation middle graft final-20% KL delta CI low",
        "exp11 depth_ablation_metrics.json",
        -0.1138431449095277,
        exp11_depth_family_ci("B_mid_raw", "ci95_low"),
    ),
    ClaimCheck(
        "Depth ablation middle graft final-20% KL delta CI high",
        "exp11 depth_ablation_metrics.json",
        0.02311977698591131,
        exp11_depth_family_ci("B_mid_raw", "ci95_high"),
    ),
    ClaimCheck(
        "Depth ablation late graft final-20% KL delta",
        "exp11 depth_ablation_metrics.json",
        0.34144877467282564,
        exp11_depth("B_late_raw", "final_20pct_delta_kl_to_own_final"),
    ),
    ClaimCheck(
        "Depth ablation late graft final-20% KL delta CI low",
        "exp11 depth_ablation_metrics.json",
        0.18081652114234793,
        exp11_depth_family_ci("B_late_raw", "ci95_low"),
    ),
    ClaimCheck(
        "Depth ablation late graft final-20% KL delta CI high",
        "exp11 depth_ablation_metrics.json",
        0.5020810282033033,
        exp11_depth_family_ci("B_late_raw", "ci95_high"),
    ),
    ClaimCheck(
        "Symmetric PT-side late graft KL delta",
        "exp14 exp13_full_summary.json",
        0.3376970321801641,
        exp14_dense("pt_side_final20_kl_delta", "B_late_raw"),
    ),
    ClaimCheck(
        "Symmetric PT-side late graft KL delta CI low",
        "exp14 exp13_full_summary.json",
        0.18184266954259196,
        exp14_dense_family_ci("pt_side", "B_late_raw", "ci95_low"),
    ),
    ClaimCheck(
        "Symmetric PT-side late graft KL delta CI high",
        "exp14 exp13_full_summary.json",
        0.5039044664954922,
        exp14_dense_family_ci("pt_side", "B_late_raw", "ci95_high"),
    ),
    ClaimCheck(
        "Symmetric IT-side early PT-swap KL delta CI low",
        "exp14 exp13_full_summary.json",
        -0.25641013561189754,
        exp14_dense_family_ci("it_side", "D_early_ptswap", "ci95_low"),
    ),
    ClaimCheck(
        "Symmetric IT-side early PT-swap KL delta CI high",
        "exp14 exp13_full_summary.json",
        0.027417360401097366,
        exp14_dense_family_ci("it_side", "D_early_ptswap", "ci95_high"),
    ),
    ClaimCheck(
        "Symmetric IT-side middle PT-swap KL delta CI low",
        "exp14 exp13_full_summary.json",
        -0.3710696790816494,
        exp14_dense_family_ci("it_side", "D_mid_ptswap", "ci95_low"),
    ),
    ClaimCheck(
        "Symmetric IT-side middle PT-swap KL delta CI high",
        "exp14 exp13_full_summary.json",
        -0.08628748310834142,
        exp14_dense_family_ci("it_side", "D_mid_ptswap", "ci95_high"),
    ),
    ClaimCheck(
        "Symmetric IT-side late PT-swap KL delta",
        "exp14 exp13_full_summary.json",
        -0.5086195820051482,
        exp14_dense("it_side_final20_kl_delta", "D_late_ptswap"),
    ),
    ClaimCheck(
        "Symmetric IT-side late PT-swap KL delta CI low",
        "exp14 exp13_full_summary.json",
        -0.8280586533684673,
        exp14_dense_family_ci("it_side", "D_late_ptswap", "ci95_low"),
    ),
    ClaimCheck(
        "Symmetric IT-side late PT-swap KL delta CI high",
        "exp14 exp13_full_summary.json",
        -0.22406725064182915,
        exp14_dense_family_ci("it_side", "D_late_ptswap", "ci95_high"),
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
        "Exp23 Dense-6 late given PT upstream",
        "exp23_dense6_core_effects.csv",
        0.6394436240874645,
        exp23_dense6_effect("late_it_given_pt_upstream", "common_it", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 late given IT upstream",
        "exp23_dense6_core_effects.csv",
        3.076473895656491,
        exp23_dense6_effect("late_it_given_it_upstream", "common_it", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 upstream-context effect",
        "exp23_dense6_core_effects.csv",
        3.874074864700922,
        exp23_dense6_effect("upstream_context_effect", "common_it", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 late-stack main effect",
        "exp23_dense6_core_effects.csv",
        1.8579587598719778,
        exp23_dense6_effect("late_weight_effect", "common_it", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 upstream x late interaction",
        "exp23_dense6_core_effects.csv",
        2.4370302715690264,
        exp23_dense6_effect("interaction", "common_it", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 interaction CI low",
        "exp23_dense6_core_effects.csv",
        2.3526391879414423,
        exp23_dense6_effect("interaction", "common_it", "dense6_ci95_low"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 interaction CI high",
        "exp23_dense6_core_effects.csv",
        2.5214213551966105,
        exp23_dense6_effect("interaction", "common_it", "dense6_ci95_high"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed interaction",
        "exp23_dense6_core_effects.csv",
        1.708858200882832,
        exp23_dense6_effect("interaction", "common_it", "gemma_removed_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-PT interaction",
        "exp23_dense6_core_effects.csv",
        2.4211752822401453,
        exp23_dense6_effect("interaction", "common_pt", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-IT late-effect amplification",
        "exp23_dense6_core_effects.csv",
        4.811172994408784,
        exp23_dense6_late_effect_amplification("common_it", "dense6"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed common-IT late-effect amplification",
        "exp23_dense6_core_effects.csv",
        3.287918267084384,
        exp23_dense6_late_effect_amplification("common_it", "gemma_removed"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-PT late-effect amplification",
        "exp23_dense6_core_effects.csv",
        4.657065006190824,
        exp23_dense6_late_effect_amplification("common_pt", "dense6"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed common-PT late-effect amplification",
        "exp23_dense6_core_effects.csv",
        3.2791219540280427,
        exp23_dense6_late_effect_amplification("common_pt", "gemma_removed"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 native diagonal margin shift",
        "exp23_dense6_core_effects.csv",
        5.7320336245728996,
        exp23_dense6_native_shift("common_it", "dense6"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 interaction share of native diagonal shift",
        "exp23_dense6_core_effects.csv",
        0.4251598003754928,
        exp23_dense6_interaction_share("common_it", "dense6"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed native diagonal margin shift",
        "exp23_dense6_core_effects.csv",
        4.9419090994874795,
        exp23_dense6_native_shift("common_it", "gemma_removed"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed interaction share",
        "exp23_dense6_core_effects.csv",
        0.345789079985315,
        exp23_dense6_interaction_share("common_it", "gemma_removed"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-PT native diagonal margin shift",
        "exp23_dense6_core_effects.csv",
        5.722973271427784,
        exp23_dense6_native_shift("common_pt", "dense6"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-PT interaction share",
        "exp23_dense6_core_effects.csv",
        0.4230624829803029,
        exp23_dense6_interaction_share("common_pt", "dense6"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-PT Gemma-removed native shift",
        "exp23_dense6_core_effects.csv",
        4.996099175713341,
        exp23_dense6_native_shift("common_pt", "gemma_removed"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 common-PT Gemma-removed interaction share",
        "exp23_dense6_core_effects.csv",
        0.3476480822207691,
        exp23_dense6_interaction_share("common_pt", "gemma_removed"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 family interaction minimum",
        "exp23_dense6_family_effects.csv",
        1.25331787109375,
        exp23_dense6_family_interaction_stat("min"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 family interaction maximum",
        "exp23_dense6_family_effects.csv",
        6.077890625,
        exp23_dense6_family_interaction_stat("max"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 family interaction median",
        "exp23_dense6_family_effects.csv",
        1.655386667783214,
        exp23_dense6_family_interaction_stat("median"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 family interaction-share minimum",
        "exp23_dense5/exp24 exp23_effects.csv",
        0.23392797687327452,
        exp23_dense6_family_share_stat("min"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 family interaction-share maximum",
        "exp23_dense5/exp24 exp23_effects.csv",
        0.6277090157982217,
        exp23_dense6_family_share_stat("max"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 family interaction-share median",
        "exp23_dense5/exp24 exp23_effects.csv",
        0.3787078537526653,
        exp23_dense6_family_share_stat("median"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed family share range minimum",
        "exp23_dense5/exp24 exp23_effects.csv",
        0.23392797687327452,
        exp23_dense6_family_share_stat("min", exclude={"gemma3_4b"}),
    ),
    ClaimCheck(
        "Exp23 Dense-6 Gemma-removed family share range maximum",
        "exp23_dense5/exp24 exp23_effects.csv",
        0.3936659862894504,
        exp23_dense6_family_share_stat("max", exclude={"gemma3_4b"}),
    ),
    ClaimCheck(
        "Exp23 Dense-6 position >=1 interaction",
        "exp23_dense6_position_sensitivity.csv",
        2.078516884791658,
        exp23_dense6_position("positions >=1", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 position >=3 interaction",
        "exp23_dense6_position_sensitivity.csv",
        1.4342981832789896,
        exp23_dense6_position("positions >=3", "dense6_estimate"),
    ),
    ClaimCheck(
        "Exp23 Dense-6 position >=5 interaction",
        "exp23_dense6_position_sensitivity.csv",
        1.4799920390405037,
        exp23_dense6_position("position >=5", "dense6_estimate"),
    ),
    *[
        ClaimCheck(
            f"Minimal reproducibility snapshot {model} {field}",
            "exp23 dense5 raw_shared records",
            expected,
            exp23_dense5_repro_snapshot(model, field),
        )
        for model, field, expected in [
            ("gemma3_4b", "events", 600.0),
            ("gemma3_4b", "pos0_frac", 0.525),
            ("gemma3_4b", "ge3_frac", 0.125),
            ("gemma3_4b", "ge5_frac", 0.06),
            ("llama31_8b", "events", 600.0),
            ("llama31_8b", "pos0_frac", 0.5933333333333334),
            ("llama31_8b", "ge3_frac", 0.28),
            ("llama31_8b", "ge5_frac", 0.17333333333333334),
            ("qwen3_4b", "events", 600.0),
            ("qwen3_4b", "pos0_frac", 0.4866666666666667),
            ("qwen3_4b", "ge3_frac", 0.3516666666666667),
            ("qwen3_4b", "ge5_frac", 0.235),
            ("mistral_7b", "events", 597.0),
            ("mistral_7b", "pos0_frac", 0.3082077051926298),
            ("mistral_7b", "ge3_frac", 0.3132328308207705),
            ("mistral_7b", "ge5_frac", 0.20100502512562815),
            ("olmo2_7b", "events", 586.0),
            ("olmo2_7b", "pos0_frac", 0.6006825938566553),
            ("olmo2_7b", "ge3_frac", 0.2713310580204778),
            ("olmo2_7b", "ge5_frac", 0.16040955631399317),
        ]
    ],
    *[
        ClaimCheck(
            f"Minimal reproducibility snapshot qwen25_32b {field}",
            "exp24_32b_position_sensitivity.csv",
            expected,
            exp24_repro_snapshot(field),
        )
        for field, expected in [
            ("events", 1397.0),
            ("pos0_frac", 0.38797423049391555),
            ("ge3_frac", 0.44953471725125266),
            ("ge5_frac", 0.30565497494631355),
        ]
    ],
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
        2.5381331012542363,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 primary interaction CI high",
        "exp23 primary exp23_summary.json",
        2.7355495852654523,
        exp23_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp36 endpoint interaction common-IT",
        "exp36 summary.json",
        2.64851678944894,
        exp36_summary("common_it", "endpoint_interaction", "estimate"),
    ),
    ClaimCheck(
        "Exp36 endpoint interaction common-IT CI low",
        "exp36 summary.json",
        2.5477349113866485,
        exp36_summary("common_it", "endpoint_interaction", "ci95_low"),
    ),
    ClaimCheck(
        "Exp36 endpoint delta from Exp23",
        "exp36 summary.json",
        0.013320800001326294,
        exp36_summary("common_it", "exp23_endpoint_delta", ""),
    ),
    ClaimCheck(
        "Exp36 PT-to-IT interpolation slope",
        "exp36 summary.json",
        2.7021306712781774,
        exp36_summary("common_it", "slope", "estimate"),
    ),
    ClaimCheck(
        "Exp36 PT-to-IT interpolation slope CI low",
        "exp36 summary.json",
        2.601459001103588,
        exp36_summary("common_it", "slope", "ci95_low"),
    ),
    ClaimCheck(
        "Exp36 PT-to-IT interpolation slope CI high",
        "exp36 summary.json",
        2.7989111041624213,
        exp36_summary("common_it", "slope", "ci95_high"),
    ),
    ClaimCheck(
        "Exp36 low-anomaly half interaction",
        "exp36 anomaly_effects.csv",
        2.783980627591492,
        exp36_anomaly("low_anomaly_half", "common_it", "interaction"),
    ),
    ClaimCheck(
        "Exp36 low-anomaly half interaction CI low",
        "exp36 anomaly_effects.csv",
        2.64226523363606,
        exp36_anomaly("low_anomaly_half", "common_it", "ci95_low"),
    ),
    ClaimCheck(
        "Exp36 signed-permutation random/observed ratio",
        "exp36 random_control_effects.csv",
        0.25264588477770933,
        exp36_random_control("common_it", "random_abs_over_observed_abs"),
    ),
    ClaimCheck(
        "Exp37 source-balanced random local interaction",
        "exp37 effects.csv",
        1.6716716595066665,
        exp37_effect("random_local_disagreement__source_balanced", "estimate"),
    ),
    ClaimCheck(
        "Exp37 source-balanced random local ratio",
        "exp37 effects.csv",
        0.6311377879758742,
        exp37_effect("random_local_disagreement__source_balanced", "ratio_to_first_diff"),
    ),
    ClaimCheck(
        "Exp37 pre-divergence future-token interaction",
        "exp37 effects.csv",
        0.08207303801785182,
        exp37_effect("prediv_future_pair__shared_prediv", "estimate"),
    ),
    ClaimCheck(
        "Exp37 pre-divergence future-token interaction CI low",
        "exp37 effects.csv",
        -0.05763438322703556,
        exp37_effect("prediv_future_pair__shared_prediv", "ci95_low"),
    ),
    ClaimCheck(
        "Exp37 pre-divergence future-token ratio",
        "exp37 effects.csv",
        0.030986584819134605,
        exp37_effect("prediv_future_pair__shared_prediv", "ratio_to_first_diff"),
    ),
    ClaimCheck(
        "Exp37 token support true first-diff position 0",
        "exp37 token_support_control summary.json",
        0.502778059809857,
        exp37_token_support("first_diff__reference", "position_0"),
    ),
    ClaimCheck(
        "Exp37 token support random local position >=3",
        "exp37 token_support_control summary.json",
        0.9487884335219119,
        exp37_token_support("random_local_disagreement__source_balanced", "position_ge_3"),
    ),
    ClaimCheck(
        "Exp37 token support random local position >=5",
        "exp37 token_support_control summary.json",
        0.9122868874694541,
        exp37_token_support("random_local_disagreement__source_balanced", "position_ge_5"),
    ),
    ClaimCheck(
        "Exp37 token support first-diff any content",
        "exp37 token_support_control summary.json",
        0.5967986576797526,
        exp37_token_support("first_diff__reference", "any_content_token"),
    ),
    ClaimCheck(
        "Exp37 token support random local any content",
        "exp37 token_support_control summary.json",
        0.6858233281261132,
        exp37_token_support("random_local_disagreement__source_balanced", "any_content_token"),
    ),
    ClaimCheck(
        "Exp37 token support first-diff any format",
        "exp37 token_support_control summary.json",
        0.35500477930036983,
        exp37_token_support("first_diff__reference", "any_format_token"),
    ),
    ClaimCheck(
        "Exp37 token support random local any format",
        "exp37 token_support_control summary.json",
        0.2631593047068337,
        exp37_token_support("random_local_disagreement__source_balanced", "any_format_token"),
    ),
    ClaimCheck(
        "First-divergence support generated position 0 fraction",
        "first_divergence_token_support summary.json",
        0.5025142474019444,
        token_support_bool("position_0"),
    ),
    ClaimCheck(
        "First-divergence support generated position >=3 fraction",
        "first_divergence_token_support summary.json",
        0.2681863895407308,
        token_support_bool("position_ge_3"),
    ),
    ClaimCheck(
        "First-divergence support generated position >=5 fraction",
        "first_divergence_token_support summary.json",
        0.16594032852832719,
        token_support_bool("position_ge_5"),
    ),
    ClaimCheck(
        "First-divergence support pure surface-format fraction",
        "first_divergence_token_support summary.json",
        0.022795843110962118,
        token_support_bool("both_surface_format"),
    ),
    ClaimCheck(
        "First-divergence support any content-token fraction",
        "first_divergence_token_support summary.json",
        0.596714716728126,
        token_support_bool("any_content_token"),
    ),
    ClaimCheck(
        "First-divergence support any format-token fraction",
        "first_divergence_token_support summary.json",
        0.35501173315454243,
        token_support_bool("any_format_token"),
    ),
    ClaimCheck(
        "First-divergence LLM judged sample n",
        "first_divergence_token_support summary.json",
        749.0,
        token_support_llm_n(),
    ),
    ClaimCheck(
        "First-divergence LLM substantive weighted fraction",
        "first_divergence_token_support summary.json",
        0.7343898928358606,
        token_support_weighted("llm_substantive", "True"),
    ),
    ClaimCheck(
        "First-divergence LLM substantive weighted CI low",
        "first_divergence_token_support summary.json",
        0.7080377494821853,
        token_support_weighted("llm_substantive", "True", "ci95_low"),
    ),
    ClaimCheck(
        "First-divergence LLM substantive weighted CI high",
        "first_divergence_token_support summary.json",
        0.7587230200004381,
        token_support_weighted("llm_substantive", "True", "ci95_high"),
    ),
    ClaimCheck(
        "First-divergence LLM semantic-content weighted fraction",
        "first_divergence_token_support summary.json",
        0.30212041609300516,
        token_support_weighted("llm_category", "semantic_content"),
    ),
    ClaimCheck(
        "First-divergence LLM structural-format weighted fraction",
        "first_divergence_token_support summary.json",
        0.22037039389556823,
        token_support_weighted("llm_category", "structural_instruction_format"),
    ),
    ClaimCheck(
        "First-divergence LLM safety weighted fraction",
        "first_divergence_token_support summary.json",
        0.14533774552206602,
        token_support_weighted("llm_category", "safety_refusal_helpfulness"),
    ),
    ClaimCheck(
        "First-divergence LLM discourse weighted fraction",
        "first_divergence_token_support summary.json",
        0.16690677391987604,
        token_support_weighted("llm_category", "discourse_style_opening"),
    ),
    ClaimCheck(
        "First-divergence LLM surface-format weighted fraction",
        "first_divergence_token_support summary.json",
        0.08393264751458093,
        token_support_weighted("llm_category", "surface_format_low"),
    ),
    ClaimCheck(
        "Exp40 no-commitment subset interaction",
        "exp40 effects.csv",
        2.433592810805311,
        exp40_effect("boundary_it_le_zero", "interaction"),
    ),
    ClaimCheck(
        "Exp40 no-commitment subset interaction CI low",
        "exp40 effects.csv",
        2.284572178945279,
        exp40_effect("boundary_it_le_zero", "interaction", "ci95_low"),
    ),
    ClaimCheck(
        "Exp40 no-commitment subset interaction CI high",
        "exp40 effects.csv",
        2.58326602422306,
        exp40_effect("boundary_it_le_zero", "interaction", "ci95_high"),
    ),
    ClaimCheck(
        "Exp40 lowest boundary-margin tercile interaction",
        "exp40 effects.csv",
        2.4115178659923044,
        exp40_effect("boundary_it_low_tercile", "interaction"),
    ),
    ClaimCheck(
        "Exp40 lowest boundary-margin tercile interaction CI low",
        "exp40 effects.csv",
        2.2632523066640764,
        exp40_effect("boundary_it_low_tercile", "interaction", "ci95_low"),
    ),
    ClaimCheck(
        "Exp40 lowest boundary-margin tercile interaction CI high",
        "exp40 effects.csv",
        2.559536429649379,
        exp40_effect("boundary_it_low_tercile", "interaction", "ci95_high"),
    ),
    ClaimCheck(
        "Exp40 state-level IT-upstream coefficient",
        "exp40 effects.csv",
        2.598430860276424,
        exp40_regression("state_level_all_support", "it_upstream_provenance_coef"),
    ),
    ClaimCheck(
        "Exp40 state-level IT-upstream coefficient CI low",
        "exp40 effects.csv",
        2.4881395571629077,
        exp40_regression("state_level_all_support", "it_upstream_provenance_coef", "ci95_low"),
    ),
    ClaimCheck(
        "Exp40 state-level IT-upstream coefficient CI high",
        "exp40 effects.csv",
        2.7053516554242503,
        exp40_regression("state_level_all_support", "it_upstream_provenance_coef", "ci95_high"),
    ),
    ClaimCheck(
        "Exp40 pair-level zero boundary-delta interaction",
        "exp40 effects.csv",
        1.8369208468503628,
        exp40_regression("pair_level_delta_adjusted", "interaction_at_zero_boundary_delta"),
    ),
    ClaimCheck(
        "Exp40 pair-level zero boundary-delta interaction CI low",
        "exp40 effects.csv",
        1.7253462438147371,
        exp40_regression("pair_level_delta_adjusted", "interaction_at_zero_boundary_delta", "ci95_low"),
    ),
    ClaimCheck(
        "Exp40 pair-level zero boundary-delta interaction CI high",
        "exp40 effects.csv",
        1.9430040299977105,
        exp40_regression("pair_level_delta_adjusted", "interaction_at_zero_boundary_delta", "ci95_high"),
    ),
    ClaimCheck(
        "Exp23 Gemma interaction",
        "exp23 primary exp23_summary.json",
        6.077890625,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "gemma3_4b",
        ),
    ),
    ClaimCheck(
        "Exp23 Gemma interaction CI low",
        "exp23 primary exp23_summary.json",
        5.72,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "gemma3_4b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Gemma interaction CI high",
        "exp23 primary exp23_summary.json",
        6.43855078125,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "gemma3_4b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 Llama interaction",
        "exp23 primary exp23_summary.json",
        1.25331787109375,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "llama31_8b",
        ),
    ),
    ClaimCheck(
        "Exp23 Llama interaction CI low",
        "exp23 primary exp23_summary.json",
        1.1038143513997394,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "llama31_8b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Llama interaction CI high",
        "exp23 primary exp23_summary.json",
        1.4172403767903645,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "llama31_8b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 Qwen interaction",
        "exp23 primary exp23_summary.json",
        1.4641145833333333,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "qwen3_4b",
        ),
    ),
    ClaimCheck(
        "Exp23 Qwen interaction CI low",
        "exp23 primary exp23_summary.json",
        1.3152317708333332,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "qwen3_4b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Qwen interaction CI high",
        "exp23 primary exp23_summary.json",
        1.6185019531249998,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "qwen3_4b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 Mistral interaction",
        "exp23 primary exp23_summary.json",
        2.5339981155778895,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mistral_7b",
        ),
    ),
    ClaimCheck(
        "Exp23 Mistral interaction CI low",
        "exp23 primary exp23_summary.json",
        2.3492364164572863,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mistral_7b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Mistral interaction CI high",
        "exp23 primary exp23_summary.json",
        2.7178221838358456,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mistral_7b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 OLMo interaction",
        "exp23 primary exp23_summary.json",
        1.846658752233095,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "olmo2_7b",
        ),
    ),
    ClaimCheck(
        "Exp23 OLMo interaction CI low",
        "exp23 primary exp23_summary.json",
        1.6737299222588133,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "olmo2_7b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 OLMo interaction CI high",
        "exp23 primary exp23_summary.json",
        2.0264709759084027,
        exp23_model_effect(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "olmo2_7b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 family interaction median",
        "exp23 primary exp23_summary.json",
        1.846658752233095,
        exp23_model_summary(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "median",
        ),
    ),
    ClaimCheck(
        "Exp23 family interaction trimmed mean",
        "exp23 primary exp23_summary.json",
        1.9482571503814394,
        exp23_model_summary(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "trimmed_minmax_mean",
        ),
    ),
    ClaimCheck(
        "Exp23 family interaction excluding Gemma and Mistral",
        "exp23 primary exp23_summary.json",
        1.5213637355533927,
        exp23_model_summary(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mean",
            exclude={"gemma3_4b", "mistral_7b"},
        ),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed interaction",
        "exp23 primary exp23_summary.json",
        1.774522330559517,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "gemma3_4b",
        ),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed interaction CI low",
        "exp23 primary exp23_summary.json",
        1.6941470529485825,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "gemma3_4b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed interaction CI high",
        "exp23 primary exp23_summary.json",
        1.8602784042875087,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "gemma3_4b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 Llama-removed interaction",
        "exp23 primary exp23_summary.json",
        2.9806655190360796,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "llama31_8b",
        ),
    ),
    ClaimCheck(
        "Exp23 Llama-removed interaction CI low",
        "exp23 primary exp23_summary.json",
        2.8686141448123674,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "llama31_8b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Llama-removed interaction CI high",
        "exp23 primary exp23_summary.json",
        3.096613603094612,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "llama31_8b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 Qwen-removed interaction",
        "exp23 primary exp23_summary.json",
        2.927966340976184,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "qwen3_4b",
        ),
    ),
    ClaimCheck(
        "Exp23 Qwen-removed interaction CI low",
        "exp23 primary exp23_summary.json",
        2.8134245621096934,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "qwen3_4b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Qwen-removed interaction CI high",
        "exp23 primary exp23_summary.json",
        3.0432283165390843,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "qwen3_4b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 Mistral-removed interaction",
        "exp23 primary exp23_summary.json",
        2.6604954579150446,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mistral_7b",
        ),
    ),
    ClaimCheck(
        "Exp23 Mistral-removed interaction CI low",
        "exp23 primary exp23_summary.json",
        2.544466285202148,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mistral_7b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 Mistral-removed interaction CI high",
        "exp23 primary exp23_summary.json",
        2.7752855719597807,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "mistral_7b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 OLMo-removed interaction",
        "exp23 primary exp23_summary.json",
        2.832330298751243,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "olmo2_7b",
        ),
    ),
    ClaimCheck(
        "Exp23 OLMo-removed interaction CI low",
        "exp23 primary exp23_summary.json",
        2.717073292717862,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "olmo2_7b",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 OLMo-removed interaction CI high",
        "exp23 primary exp23_summary.json",
        2.945469679175909,
        exp23_leave_one_out(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "interaction",
            "olmo2_7b",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 position >=1 interaction",
        "exp23_position_sensitivity_table.csv",
        2.248233419644727,
        exp23_position_sensitivity("positions >=1", "dense5_interaction"),
    ),
    ClaimCheck(
        "Exp23 position >=1 interaction CI low",
        "exp23_position_sensitivity_table.csv",
        2.111713621144889,
        exp23_position_sensitivity("positions >=1", "dense5_ci_low"),
    ),
    ClaimCheck(
        "Exp23 position >=1 interaction CI high",
        "exp23_position_sensitivity_table.csv",
        2.381941370294744,
        exp23_position_sensitivity("positions >=1", "dense5_ci_high"),
    ),
    ClaimCheck(
        "Exp23 position >=3 interaction",
        "exp23_position_sensitivity_table.csv",
        1.5171470715271445,
        exp23_position_sensitivity("positions >=3", "dense5_interaction"),
    ),
    ClaimCheck(
        "Exp23 position >=3 interaction CI low",
        "exp23_position_sensitivity_table.csv",
        1.3609315405695073,
        exp23_position_sensitivity("positions >=3", "dense5_ci_low"),
    ),
    ClaimCheck(
        "Exp23 position >=3 interaction CI high",
        "exp23_position_sensitivity_table.csv",
        1.6777874011648348,
        exp23_position_sensitivity("positions >=3", "dense5_ci_high"),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed position >=3 interaction",
        "exp23_position_sensitivity_table.csv",
        0.788100506075597,
        exp23_position_sensitivity("positions >=3", "gemma_removed_interaction"),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed position >=3 interaction CI low",
        "exp23_position_sensitivity_table.csv",
        0.6999715202300372,
        exp23_position_sensitivity("positions >=3", "gemma_removed_ci_low"),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed position >=3 interaction CI high",
        "exp23_position_sensitivity_table.csv",
        0.8797333487976914,
        exp23_position_sensitivity("positions >=3", "gemma_removed_ci_high"),
    ),
    ClaimCheck(
        "Exp23 Gemma position >=3 interaction CI low",
        "exp23_position_sensitivity_per_family.csv",
        3.7064583333333334,
        exp23_position_family("positions >=3", "gemma3_4b", "ci_low"),
    ),
    ClaimCheck(
        "Exp23 Qwen position >=3 interaction CI low",
        "exp23_position_sensitivity_per_family.csv",
        0.9480061463270142,
        exp23_position_family("positions >=3", "qwen3_4b", "ci_low"),
    ),
    ClaimCheck(
        "Exp23 Llama position >=3 interaction CI low",
        "exp23_position_sensitivity_per_family.csv",
        0.0118896484375,
        exp23_position_family("positions >=3", "llama31_8b", "ci_low"),
    ),
    ClaimCheck(
        "Exp23 Mistral position >=3 interaction CI low",
        "exp23_position_sensitivity_per_family.csv",
        1.0083347259358288,
        exp23_position_family("positions >=3", "mistral_7b", "ci_low"),
    ),
    ClaimCheck(
        "Exp23 OLMo position >=3 interaction CI low",
        "exp23_position_sensitivity_per_family.csv",
        0.4928541087504453,
        exp23_position_family("positions >=3", "olmo2_7b", "ci_low"),
    ),
    ClaimCheck(
        "Exp23 position >=5 interaction",
        "exp23_position_sensitivity_table.csv",
        1.6388563719071523,
        exp23_position_sensitivity("position >=5", "dense5_interaction"),
    ),
    ClaimCheck(
        "Exp23 position >=5 interaction CI low",
        "exp23_position_sensitivity_table.csv",
        1.3924069821936453,
        exp23_position_sensitivity("position >=5", "dense5_ci_low"),
    ),
    ClaimCheck(
        "Exp23 position >=5 interaction CI high",
        "exp23_position_sensitivity_table.csv",
        1.8802611244216922,
        exp23_position_sensitivity("position >=5", "dense5_ci_high"),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed position >=5 interaction",
        "exp23_position_sensitivity_table.csv",
        0.8259142148839407,
        exp23_position_sensitivity("position >=5", "gemma_removed_interaction"),
    ),
    ClaimCheck(
        "Exp23 position >=3 category-mix total records",
        "exp23_position_category_mix.csv",
        800.0,
        exp23_position_category_mix("step_ge3", "prompt_category", "GOV-CONV", "total"),
    ),
    ClaimCheck(
        "Exp23 position >=3 GOV-CONV count",
        "exp23_position_category_mix.csv",
        700.0,
        exp23_position_category_mix("step_ge3", "prompt_category", "GOV-CONV", "count"),
    ),
    ClaimCheck(
        "Exp23 position >=3 GOV-FORMAT count",
        "exp23_position_category_mix.csv",
        61.0,
        exp23_position_category_mix("step_ge3", "prompt_category", "GOV-FORMAT", "count"),
    ),
    ClaimCheck(
        "Exp23 position >=3 SAFETY count",
        "exp23_position_category_mix.csv",
        39.0,
        exp23_position_category_mix("step_ge3", "prompt_category", "SAFETY", "count"),
    ),
    ClaimCheck(
        "Exp23 position >=3 IT CONTENT-token count",
        "exp23_position_category_mix.csv",
        476.0,
        exp23_position_category_mix("step_ge3", "it_token_category", "CONTENT", "count"),
    ),
    ClaimCheck(
        "Exp23 position >=3 IT FUNCTION_OTHER-token count",
        "exp23_position_category_mix.csv",
        199.0,
        exp23_position_category_mix("step_ge3", "it_token_category", "FUNCTION_OTHER", "count"),
    ),
    ClaimCheck(
        "Exp23 position >=3 IT FORMAT-token count",
        "exp23_position_category_mix.csv",
        125.0,
        exp23_position_category_mix("step_ge3", "it_token_category", "FORMAT", "count"),
    ),
    ClaimCheck(
        "Exp23 GOV-CONV all-position interaction",
        "exp23_position_prompt_category_effects.csv",
        2.0509100734312526,
        exp23_position_prompt_category("all", "GOV-CONV", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Exp23 GOV-CONV position >=3 interaction",
        "exp23_position_prompt_category_effects.csv",
        1.512368607090077,
        exp23_position_prompt_category("step_ge3", "GOV-CONV", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Exp23 GOV-FORMAT position >=3 interaction",
        "exp23_position_prompt_category_effects.csv",
        2.277389090401786,
        exp23_position_prompt_category("step_ge3", "GOV-FORMAT", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Exp23 GOV-FORMAT position >=3 clusters",
        "exp23_position_prompt_category_effects.csv",
        61.0,
        exp23_position_prompt_category("step_ge3", "GOV-FORMAT", "interaction", "n_prompt_clusters"),
    ),
    ClaimCheck(
        "Exp23 SAFETY position >=3 interaction",
        "exp23_position_prompt_category_effects.csv",
        0.6419631231398809,
        exp23_position_prompt_category("step_ge3", "SAFETY", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Exp23 SAFETY position >=3 clusters",
        "exp23_position_prompt_category_effects.csv",
        39.0,
        exp23_position_prompt_category("step_ge3", "SAFETY", "interaction", "n_prompt_clusters"),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed position >=5 interaction CI low",
        "exp23_position_sensitivity_table.csv",
        0.7123552879374079,
        exp23_position_sensitivity("position >=5", "gemma_removed_ci_low"),
    ),
    ClaimCheck(
        "Exp23 Gemma-removed position >=5 interaction CI high",
        "exp23_position_sensitivity_table.csv",
        0.9388008422222021,
        exp23_position_sensitivity("position >=5", "gemma_removed_ci_high"),
    ),
    ClaimCheck(
        "Exp23 Llama position >=5 interaction",
        "exp23_position_sensitivity_per_family.csv",
        0.11598557692307693,
        exp23_position_family("position >=5", "llama31_8b", "interaction"),
    ),
    ClaimCheck(
        "Exp23 Llama position >=5 interaction CI low",
        "exp23_position_sensitivity_per_family.csv",
        -0.046980168269230765,
        exp23_position_family("position >=5", "llama31_8b", "ci_low"),
    ),
    ClaimCheck(
        "Exp23 Llama position >=5 interaction CI high",
        "exp23_position_sensitivity_per_family.csv",
        0.2692307692307692,
        exp23_position_family("position >=5", "llama31_8b", "ci_high"),
    ),
    ClaimCheck(
        "Exp23 primary IT compatibility boost",
        "exp23 compatibility permutation summary.json",
        5.556307535299774,
        exp23_compatibility(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "it_compatibility_boost",
        ),
    ),
    ClaimCheck(
        "Exp23 primary PT compatibility boost",
        "exp23 compatibility permutation summary.json",
        2.9211115458521597,
        exp23_compatibility(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "pt_compatibility_boost",
        ),
    ),
    ClaimCheck(
        "Exp23 primary label-swap null q99.9",
        "exp23 compatibility permutation summary.json",
        0.23852594495783128,
        exp23_compatibility(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "null_q99.9",
        ),
    ),
    ClaimCheck(
        "Exp23 primary label-swap p-value",
        "exp23 compatibility permutation summary.json",
        4.999750012499375e-05,
        exp23_compatibility(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "p_upper",
        ),
        tolerance=1e-9,
        digits=8,
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
        "Exp23 assistant-marker subgroup clusters",
        "exp23 primary subgroup summary.json",
        551.0,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "assistant_marker_event",
            "assistant_marker",
            "n_prompt_clusters",
        ),
    ),
    ClaimCheck(
        "Exp23 assistant-marker subgroup interaction",
        "exp23 primary subgroup summary.json",
        3.3040487511538137,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "assistant_marker_event",
            "assistant_marker",
        ),
    ),
    ClaimCheck(
        "Exp23 non-assistant-marker subgroup clusters",
        "exp23 primary subgroup summary.json",
        2432.0,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "assistant_marker_event",
            "non_assistant_marker",
            "n_prompt_clusters",
        ),
    ),
    ClaimCheck(
        "Exp23 non-assistant-marker subgroup interaction",
        "exp23 primary subgroup summary.json",
        2.4822755059501542,
        exp23_subgroup(
            "exp23_dense5_full_h100x8_20260426_sh4_rw4",
            "assistant_marker_event",
            "non_assistant_marker",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction",
        "exp23 content/reasoning exp23_summary.json",
        1.8121837830772685,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction CI low",
        "exp23 content/reasoning exp23_summary.json",
        1.7211691821951594,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
            "ci95_low",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction CI high",
        "exp23 content/reasoning exp23_summary.json",
        1.9014917989489144,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
            "ci95_high",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction event records",
        "exp23 content/reasoning exp23_summary.json",
        5889.0,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
            "n_units",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning interaction prompt clusters",
        "exp23 content/reasoning exp23_summary.json",
        2983.0,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "interaction",
            "n_prompt_clusters",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning upstream-context effect",
        "exp23 content/reasoning exp23_summary.json",
        3.7124395930343383,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "upstream_context_effect",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning late given PT upstream",
        "exp23 content/reasoning exp23_summary.json",
        -1.175818325521803,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "late_it_given_pt_upstream",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning late given IT upstream",
        "exp23 content/reasoning exp23_summary.json",
        0.636365457555465,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "late_it_given_it_upstream",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning late-stack main effect",
        "exp23 content/reasoning exp23_summary.json",
        -0.269726433983169,
        exp23_effect(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "late_weight_effect",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning IT compatibility boost",
        "exp23 content/reasoning compatibility permutation summary.json",
        4.618531484572973,
        exp23_compatibility(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "it_compatibility_boost",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning PT compatibility boost",
        "exp23 content/reasoning compatibility permutation summary.json",
        2.8063477014957043,
        exp23_compatibility(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "pt_compatibility_boost",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning label-swap null q99.9",
        "exp23 content/reasoning compatibility permutation summary.json",
        0.17777057094258042,
        exp23_compatibility(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "null_q99.9",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning label-swap p-value",
        "exp23 content/reasoning compatibility permutation summary.json",
        4.999750012499375e-05,
        exp23_compatibility(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "p_upper",
        ),
        tolerance=1e-9,
        digits=8,
    ),
    ClaimCheck(
        "Exp23 CONTENT-FACT subgroup interaction",
        "exp23 content/reasoning subgroup summary.json",
        1.9563506699546729,
        exp23_subgroup(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "prompt_category",
            "CONTENT-FACT",
        ),
    ),
    ClaimCheck(
        "Exp23 CONTENT-REASON subgroup interaction",
        "exp23 content/reasoning subgroup summary.json",
        1.3526539918509761,
        exp23_subgroup(
            "exp23_content_reasoning_residual_20260427_0930_h100x8",
            "prompt_category",
            "CONTENT-REASON",
        ),
    ),
    ClaimCheck(
        "Exp23 content/reasoning GOV-FORMAT subgroup interaction",
        "exp23 content/reasoning subgroup summary.json",
        2.2838367915811766,
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
        "Exp21 content/reasoning remaining/token-specific component",
        "exp21 content/reasoning effects.csv",
        0.04881583150857773,
        exp21_content_effect("late_weight_effect:remainder_margin_it_vs_pt"),
    ),
    ClaimCheck(
        "Exp27 no-opposition PT NLL hurt",
        "exp27_summary.json",
        0.0004003626910650624,
        exp27_primary("noopp", "pt_nll_delta"),
    ),
    ClaimCheck(
        "Exp27 no-opposition IT NLL hurt",
        "exp27_summary.json",
        0.04324806564819389,
        exp27_primary("noopp", "it_nll_delta"),
    ),
    ClaimCheck(
        "Exp27 no-opposition IT-PT NLL hurt",
        "exp27_summary.json",
        0.04284770295712883,
        exp27_primary("noopp", "it_minus_pt_nll_delta"),
    ),
    ClaimCheck(
        "Exp27 no-opposition IT-PT NLL hurt CI low",
        "exp27_summary.json",
        0.04026022867670021,
        exp27_primary("noopp", "it_minus_pt_nll_delta_ci_low"),
    ),
    ClaimCheck(
        "Exp27 no-opposition IT-PT NLL hurt CI high",
        "exp27_summary.json",
        0.04525020986159794,
        exp27_primary("noopp", "it_minus_pt_nll_delta_ci_high"),
    ),
    ClaimCheck(
        "Exp27 norm-preserving IT-PT NLL hurt",
        "exp27_summary.json",
        0.03356535901970479,
        exp27_primary("normpres_noopp", "it_minus_pt_nll_delta"),
    ),
    ClaimCheck(
        "Exp27 flip-opposition IT-PT NLL hurt",
        "exp27_summary.json",
        0.07438064366204855,
        exp27_primary("flipopp", "it_minus_pt_nll_delta"),
    ),
    ClaimCheck(
        "Exp27 random-removal IT-PT NLL hurt",
        "exp27_summary.json",
        -0.03441470189085599,
        exp27_primary("randremove", "it_minus_pt_nll_delta"),
    ),
    ClaimCheck(
        "Exp27 no-opposition IT-PT true-logit drop",
        "exp27_summary.json",
        6.542463737939043,
        exp27_primary("noopp", "it_minus_pt_true_logit_drop"),
    ),
    ClaimCheck(
        "Exp27 no-opposition IT-PT true-logit drop CI low",
        "exp27_summary.json",
        6.402845100091115,
        exp27_primary("noopp", "it_minus_pt_true_logit_drop_ci_low"),
    ),
    ClaimCheck(
        "Exp27 norm-preserving IT-PT true-logit drop",
        "exp27_summary.json",
        5.836304256624378,
        exp27_primary("normpres_noopp", "it_minus_pt_true_logit_drop"),
    ),
    ClaimCheck(
        "Exp27 residual-norm random-removal IT-PT true-logit drop",
        "exp27_summary.json",
        7.2153228103302816,
        exp27_primary("randremove_resnorm", "it_minus_pt_true_logit_drop"),
    ),
    ClaimCheck(
        "Qwen2.5-32B residual-state interaction",
        "exp24 exp23_summary.json",
        1.4462016821760917,
        exp24_residual_effect("common_it", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B residual-state interaction CI low",
        "exp24 exp23_summary.json",
        1.321259842519685,
        exp24_residual_effect("common_it", "interaction", "ci95_low"),
    ),
    ClaimCheck(
        "Qwen2.5-32B late IT given PT upstream",
        "exp24 exp23_summary.json",
        0.9766240157480315,
        exp24_residual_effect("common_it", "late_it_given_pt_upstream", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B late IT given IT upstream",
        "exp24 exp23_summary.json",
        2.422825697924123,
        exp24_residual_effect("common_it", "late_it_given_it_upstream", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B late-effect amplification",
        "exp24 exp23_summary.json",
        2.4808172427220043,
        exp24_late_effect_amplification("common_it"),
    ),
    ClaimCheck(
        "Qwen2.5-32B common-PT residual-state interaction",
        "exp24 exp23_summary.json",
        1.478458303507516,
        exp24_residual_effect("common_pt", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B position >=3 interaction",
        "exp24_32b_position_sensitivity.csv",
        1.0200537420382165,
        exp24_position("step_ge3", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B position >=3 interaction CI low",
        "exp24_32b_position_sensitivity.csv",
        0.8526062400477707,
        exp24_position("step_ge3", "interaction", "ci95_low"),
    ),
    ClaimCheck(
        "Qwen2.5-32B position >=5 interaction",
        "exp24_32b_position_sensitivity.csv",
        0.68567037470726,
        exp24_position("step_ge5", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B position >=10 interaction",
        "exp24_32b_position_sensitivity.csv",
        0.8860554245283019,
        exp24_position("step_ge10", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B position >=3 content-fact interaction",
        "exp24_32b_position_prompt_category_effects.csv",
        1.4680316091954022,
        exp24_position_prompt_category("step_ge3", "CONTENT-FACT", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B position >=3 content-reason interaction",
        "exp24_32b_position_prompt_category_effects.csv",
        1.580105633802817,
        exp24_position_prompt_category("step_ge3", "CONTENT-REASON", "interaction", "estimate"),
    ),
    ClaimCheck(
        "Qwen2.5-32B raw-KL IT-side interaction",
        "exp24 raw-KL effects.csv",
        0.46455907661454976,
        exp24_kl_effect("it", "I_it", "mean"),
    ),
    ClaimCheck(
        "Qwen2.5-32B raw-KL PT-side interaction",
        "exp24 raw-KL effects.csv",
        -0.12499038208182434,
        exp24_kl_effect("pt", "I_pt", "mean"),
    ),
    ClaimCheck(
        "OLMo stage Base->SFT interaction",
        "exp25 olmo_stage_progression_table.csv",
        0.7822269640470806,
        exp25_olmo_stage("PT->SFT", "interaction"),
    ),
    ClaimCheck(
        "OLMo stage Base->SFT interaction CI low",
        "exp25 olmo_stage_progression_table.csv",
        0.6805445260880431,
        exp25_olmo_stage("PT->SFT", "interaction_ci_low"),
    ),
    ClaimCheck(
        "OLMo stage SFT->DPO interaction",
        "exp25 olmo_stage_progression_table.csv",
        0.1352249775583483,
        exp25_olmo_stage("SFT->DPO", "interaction"),
    ),
    ClaimCheck(
        "OLMo stage DPO->RLVR interaction",
        "exp25 olmo_stage_progression_table.csv",
        0.01596030042918455,
        exp25_olmo_stage("DPO->RLVR", "interaction"),
    ),
    ClaimCheck(
        "OLMo stage DPO->RLVR interaction CI low",
        "exp25 olmo_stage_progression_table.csv",
        0.007711909871244635,
        exp25_olmo_stage("DPO->RLVR", "interaction_ci_low"),
    ),
    ClaimCheck(
        "OLMo stage Base->RLVR interaction",
        "exp25 olmo_stage_progression_table.csv",
        1.9304599371379554,
        exp25_olmo_stage("PT->RLVR", "interaction"),
    ),
    ClaimCheck(
        "OLMo stage Base->RLVR interaction CI high",
        "exp25 olmo_stage_progression_table.csv",
        2.1123002976687695,
        exp25_olmo_stage("PT->RLVR", "interaction_ci_high"),
    ),
    ClaimCheck(
        "OLMo stage Base->SFT mid+late IT-token transfer",
        "exp25 olmo_stage_progression_table.csv",
        0.40834845735027225,
        exp25_olmo_stage("PT->SFT", "b_midlate_it_match_rate"),
    ),
    ClaimCheck(
        "OLMo stage SFT->DPO mid+late IT-token transfer",
        "exp25 olmo_stage_progression_table.csv",
        0.5709156193895871,
        exp25_olmo_stage("SFT->DPO", "b_midlate_it_match_rate"),
    ),
    ClaimCheck(
        "OLMo stage Base->SFT first-diff events",
        "exp25 olmo_stage_progression_table.csv",
        551.0,
        exp25_olmo_stage("PT->SFT", "first_diff_events"),
    ),
    ClaimCheck(
        "Exp34 Gemma terminal crosscoder top-200 drop",
        "exp34_dense5_crosscoder_summary.json",
        1.6945490056818182,
        exp34_model("gemma3_4b", "causal_top200_interaction_drop"),
    ),
    ClaimCheck(
        "Exp34 Gemma terminal crosscoder matched-random drop",
        "exp34_dense5_crosscoder_summary.json",
        -0.30980705492424243,
        exp34_model("gemma3_4b", "causal_matched_random200_drop_mean"),
    ),
    ClaimCheck(
        "Exp34 Gemma terminal crosscoder drop share",
        "exp34_dense5_crosscoder_summary.json",
        0.2793708688753974,
        exp34_model_drop_share("gemma3_4b"),
    ),
    ClaimCheck(
        "Exp34 Gemma terminal crosscoder top-500 drop share",
        "exp34_gemma effects.csv",
        0.28273724144795004,
        crosscoder_effects_drop_share(
            "results/exp34_dense5_final_readout_crosscoder/"
            "exp34_gemma3_4b_full_20260502_2110_a100x8_bs16/gemma3_4b/"
            "selected_d81920_k64/analysis/effects.csv",
            500,
        ),
    ),
    ClaimCheck(
        "Exp34 Llama terminal crosscoder top-200 drop",
        "exp34_dense5_crosscoder_summary.json",
        0.5987082741477273,
        exp34_model("llama31_8b", "causal_top200_interaction_drop"),
    ),
    ClaimCheck(
        "Exp34 Llama terminal crosscoder matched-random drop",
        "exp34_dense5_crosscoder_summary.json",
        -0.2094970703125,
        exp34_model("llama31_8b", "causal_matched_random200_drop_mean"),
    ),
    ClaimCheck(
        "Exp34 Llama terminal crosscoder drop share",
        "exp34_dense5_crosscoder_summary.json",
        0.4845227156121652,
        exp34_model_drop_share("llama31_8b"),
    ),
    ClaimCheck(
        "Exp30 Llama terminal crosscoder top-500 drop share",
        "exp30 effects.csv",
        0.5174794261042074,
        crosscoder_effects_drop_share(
            "results/exp30_final_readout_crosscoder_mediation/"
            "exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/"
            "selected_d131072_k64/analysis/effects.csv",
            500,
        ),
    ),
    ClaimCheck(
        "Exp34 Mistral terminal crosscoder top-200 drop",
        "exp34_dense5_crosscoder_summary.json",
        0.6844608123569794,
        exp34_model("mistral_7b", "causal_top200_interaction_drop"),
    ),
    ClaimCheck(
        "Exp34 Mistral terminal crosscoder matched-random drop",
        "exp34_dense5_crosscoder_summary.json",
        -0.09979262013729977,
        exp34_model("mistral_7b", "causal_matched_random200_drop_mean"),
    ),
    ClaimCheck(
        "Exp34 Mistral terminal crosscoder drop share",
        "exp34_dense5_crosscoder_summary.json",
        0.2607540796033454,
        exp34_model_drop_share("mistral_7b"),
    ),
    ClaimCheck(
        "Exp34 Mistral terminal crosscoder top-500 drop share",
        "exp34_mistral effects.csv",
        0.28873239436619713,
        crosscoder_effects_drop_share(
            "results/exp34_dense5_final_readout_crosscoder/"
            "exp34_mistral_7b_full_20260502_1124/mistral_7b/"
            "selected_d131072_k64/analysis/effects.csv",
            500,
        ),
    ),
    ClaimCheck(
        "Exp38 Qwen final-two crosscoder top-200 drop",
        "exp38_qwen_olmo_decision_summary.json",
        0.32359397194602274,
        exp38_gate("qwen_final2_clean_pass", "causal_top200_interaction_drop"),
    ),
    ClaimCheck(
        "Exp38 Qwen final-two crosscoder matched-random drop",
        "exp38_qwen_olmo_decision_summary.json",
        -0.033422111742424244,
        exp38_gate("qwen_final2_clean_pass", "causal_matched_random200_interaction_drop_mean"),
    ),
    ClaimCheck(
        "Exp38 Qwen final-two crosscoder drop share",
        "exp38 qwen final-two summary.json",
        0.3746184402557093,
        exp38_qwen_final2_drop_share(),
    ),
    ClaimCheck(
        "Exp38 Qwen final-two crosscoder top-500 drop share",
        "exp38 qwen final-two effects.csv",
        0.3816149869473165,
        crosscoder_effects_drop_share(
            "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_qwen3_4b_final2_d81920_k64_20260503_0451_a100x2/"
            "selected_d81920_k64/analysis/effects.csv",
            500,
        ),
    ),
    ClaimCheck(
        "Exp38 Qwen layer-34 IT VE",
        "exp38_qwen_olmo_decision_summary.json",
        0.9595943056046963,
        exp38_layer("qwen_final2_clean_pass", "layer_34", "heldout_variance_explained_it"),
    ),
    ClaimCheck(
        "Exp38 Qwen layer-35 IT VE",
        "exp38_qwen_olmo_decision_summary.json",
        0.9701936151832342,
        exp38_layer("qwen_final2_clean_pass", "layer_35", "heldout_variance_explained_it"),
    ),
    ClaimCheck(
        "Exp38 OLMo final-two diagnostic top-200 drop",
        "exp38_qwen_olmo_decision_summary.json",
        0.5982546506228147,
        exp38_gate("olmo_original_final2_causal_pass_quality_fail", "causal_top200_interaction_drop"),
    ),
    ClaimCheck(
        "Exp38 OLMo final-two diagnostic matched-random drop",
        "exp38_qwen_olmo_decision_summary.json",
        -0.06800053855107567,
        exp38_gate("olmo_original_final2_causal_pass_quality_fail", "causal_matched_random200_interaction_drop_mean"),
    ),
    ClaimCheck(
        "Exp38 OLMo layer-30 IT VE quality failure",
        "exp38_qwen_olmo_decision_summary.json",
        0.6157933473587036,
        exp38_layer(
            "olmo_original_final2_causal_pass_quality_fail",
            "layer_30",
            "heldout_variance_explained_it",
        ),
    ),
    ClaimCheck(
        "Exp40 Qwen layer-33 best-grid IT VE",
        "exp40 qwen layer33 grid_summary.json",
        0.7434459030628204,
        exp40_terminal_grid_row(
            "results/exp40_terminal_crosscoder_hardening/"
            "exp40_qwen3_4b_layer33_grid_20260503_1115_a100x4_localtmp_compact/"
            "qwen3_4b/layer33_grid/grid_summary.json",
            "qwen_l33_d196608_k96",
            "ve_it",
        ),
    ),
    ClaimCheck(
        "Exp40 Qwen layer-33 best-grid top-200 drop",
        "exp40 qwen layer33 effects.csv",
        0.09778645833333334,
        exp40_terminal_grid_effect(
            "results/exp40_terminal_crosscoder_hardening/"
            "exp40_qwen3_4b_layer33_grid_20260503_1115_a100x4_localtmp_compact/"
            "qwen3_4b/layer33_grid",
            "qwen_l33_d196608_k96",
            "causal_top",
            200,
            "interaction_drop_mean",
        ),
    ),
    ClaimCheck(
        "Exp40 OLMo layer-30 selected-grid IT VE",
        "exp40 olmo layer30 selected_candidate.json",
        0.6348804533481598,
        exp40_olmo30_selected("ve_it"),
    ),
    ClaimCheck(
        "Exp40 OLMo layer-30 selected-grid top-200 drop",
        "exp40 olmo layer30 selected_candidate.json",
        0.45155696278994845,
        exp40_olmo30_selected("causal_top200_interaction_drop"),
    ),
    ClaimCheck(
        "Exp40 OLMo layer-30 selected-grid matched-random drop",
        "exp40 olmo layer30 selected_candidate.json",
        -0.08282137736422089,
        exp40_olmo30_selected("causal_matched_random200_interaction_drop_mean"),
    ),
    ClaimCheck(
        "Exp39 autointerp feature count",
        "exp39 label_validation.json",
        300.0,
        exp39_validation("n_features"),
    ),
    ClaimCheck(
        "Exp39 autointerp mean AUROC",
        "exp39 label_validation.json",
        0.8777546296296297,
        exp39_validation("mean_auroc"),
    ),
    ClaimCheck(
        "Exp39 causal behavior/readout feature count",
        "exp39_causal_paper_taxonomy_summary_v3.json",
        42.0,
        exp39_taxonomy_group("instruction_tuned_behavior_readout"),
    ),
    ClaimCheck(
        "Exp39 causal surface/tokenization feature count",
        "exp39_causal_paper_taxonomy_summary_v3.json",
        17.0,
        exp39_taxonomy_group("surface_tokenization_adjacent"),
    ),
    ClaimCheck(
        "Exp39 causal artifact/unclear feature count",
        "exp39_causal_paper_taxonomy_summary_v3.json",
        41.0,
        exp39_taxonomy_group("artifact_or_unclear"),
    ),
    ClaimCheck(
        "Exp41 structure-readout feature count",
        "exp41 bucket_features.csv",
        12.0,
        exp41_structure_count(),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 0 mean interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.0,
        exp41_structure_interaction_drop(0.0, "feature_bucket"),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 0.5 mean interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.03232858986910464,
        exp41_structure_interaction_drop(0.5, "feature_bucket"),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 1.0 mean interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.0658083273218305,
        exp41_structure_interaction_drop(1.0, "feature_bucket"),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 1.5 mean interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.10312832524029654,
        exp41_structure_interaction_drop(1.5, "feature_bucket"),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 2.0 mean interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.14902497436932383,
        exp41_structure_interaction_drop(2.0, "feature_bucket"),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 2.0 matched-random mean",
        "exp41 bucket_effects_by_model.csv",
        -0.04965972900390625,
        exp41_structure_interaction_drop(2.0, "matched_random"),
    ),
    ClaimCheck(
        "Exp41 structure-readout alpha 2.0 same-delta-random mean",
        "exp41 bucket_effects_by_model.csv",
        0.029755577841396107,
        exp41_structure_interaction_drop(2.0, "same_delta_random"),
    ),
    ClaimCheck(
        "Exp41 Gemma structure-readout alpha 2.0 interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.05625,
        exp41_structure_interaction_drop(2.0, "feature_bucket", model="gemma3_4b"),
    ),
    ClaimCheck(
        "Exp41 Llama structure-readout alpha 2.0 interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.09130757649739583,
        exp41_structure_interaction_drop(2.0, "feature_bucket", model="llama31_8b"),
    ),
    ClaimCheck(
        "Exp41 Mistral structure-readout alpha 2.0 interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.33867252931323283,
        exp41_structure_interaction_drop(2.0, "feature_bucket", model="mistral_7b"),
    ),
    ClaimCheck(
        "Exp41 Qwen structure-readout alpha 2.0 interaction drop",
        "exp41 bucket_effects_by_model.csv",
        0.10986979166666666,
        exp41_structure_interaction_drop(2.0, "feature_bucket", model="qwen3_4b"),
    ),
    ClaimCheck(
        "Exp42 top-200 absolute causal feature gate mean",
        "exp42 feature_gating_effects.csv",
        1.0226069353573286,
        exp42_effect("causal_top", 200, "feature_causal_gate_mean"),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate vs matched-random estimate",
        "exp42 feature_gating_summary.json",
        1.2736012551346552,
        exp42_family_estimate("feature_causal_gate__causal_minus_causal_matched_random__k200"),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate vs matched-random CI low",
        "exp42 feature_gating_summary.json",
        1.1981219387288735,
        exp42_family_estimate("feature_causal_gate__causal_minus_causal_matched_random__k200", "ci_low"),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate vs matched-random CI high",
        "exp42 feature_gating_summary.json",
        1.3505247632287054,
        exp42_family_estimate("feature_causal_gate__causal_minus_causal_matched_random__k200", "ci_high"),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate vs top-active estimate",
        "exp42 feature_gating_summary.json",
        2.0614680428560854,
        exp42_family_estimate("feature_causal_gate__causal_minus_top_active_noncausal__k200"),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate vs top-active CI low",
        "exp42 feature_gating_summary.json",
        1.9526834287819153,
        exp42_family_estimate("feature_causal_gate__causal_minus_top_active_noncausal__k200", "ci_low"),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate vs top-active CI high",
        "exp42 feature_gating_summary.json",
        2.1751352306570437,
        exp42_family_estimate("feature_causal_gate__causal_minus_top_active_noncausal__k200", "ci_high"),
    ),
    ClaimCheck(
        "Exp42 top-200 margin-weighted activation gate vs matched-random estimate",
        "exp42 feature_gating_summary.json",
        0.4227289413772968,
        exp42_family_estimate(
            "activation_gate_decoder_margin_weighted__causal_minus_causal_matched_random__k200"
        ),
    ),
    ClaimCheck(
        "Exp42 top-200 margin-weighted activation gate vs matched-random CI low",
        "exp42 feature_gating_summary.json",
        0.3648426126601312,
        exp42_family_estimate(
            "activation_gate_decoder_margin_weighted__causal_minus_causal_matched_random__k200",
            "ci_low",
        ),
    ),
    ClaimCheck(
        "Exp42 top-200 margin-weighted activation gate vs matched-random CI high",
        "exp42 feature_gating_summary.json",
        0.48144207303777403,
        exp42_family_estimate(
            "activation_gate_decoder_margin_weighted__causal_minus_causal_matched_random__k200",
            "ci_high",
        ),
    ),
    ClaimCheck(
        "Exp42 top-200 causal gate position>=3 mean",
        "exp42 feature_gating_effects.csv",
        0.735790508779449,
        exp42_effect("causal_top", 200, "feature_causal_gate_pos3_mean"),
    ),
    ClaimCheck(
        "Exp42 Gemma top-200 absolute causal gate",
        "exp42 feature_gating_effects.csv",
        1.9826106770833334,
        exp42_effect("causal_top", 200, "feature_causal_gate_mean", model="gemma3_4b"),
    ),
    ClaimCheck(
        "Exp42 Llama top-200 absolute causal gate",
        "exp42 feature_gating_effects.csv",
        0.9222066243489583,
        exp42_effect("causal_top", 200, "feature_causal_gate_mean", model="llama31_8b"),
    ),
    ClaimCheck(
        "Exp42 Mistral top-200 absolute causal gate",
        "exp42 feature_gating_effects.csv",
        0.8155490996649917,
        exp42_effect("causal_top", 200, "feature_causal_gate_mean", model="mistral_7b"),
    ),
    ClaimCheck(
        "Exp42 Qwen top-200 absolute causal gate",
        "exp42 feature_gating_effects.csv",
        0.37006134033203125,
        exp42_effect("causal_top", 200, "feature_causal_gate_mean", model="qwen3_4b"),
    ),
    ClaimCheck(
        "Exp42 Gemma top-200 causal gate vs matched-random",
        "exp42 feature_gating_summary.json",
        2.4323589409722226,
        exp42_family_estimate(
            "feature_causal_gate__causal_minus_causal_matched_random__k200",
            family="gemma3_4b",
        ),
    ),
    ClaimCheck(
        "Exp42 Llama top-200 causal gate vs matched-random",
        "exp42 feature_gating_summary.json",
        1.2535394965277777,
        exp42_family_estimate(
            "feature_causal_gate__causal_minus_causal_matched_random__k200",
            family="llama31_8b",
        ),
    ),
    ClaimCheck(
        "Exp42 Mistral top-200 causal gate vs matched-random",
        "exp42 feature_gating_summary.json",
        0.9824513190954773,
        exp42_family_estimate(
            "feature_causal_gate__causal_minus_causal_matched_random__k200",
            family="mistral_7b",
        ),
    ),
    ClaimCheck(
        "Exp42 Qwen top-200 causal gate vs matched-random",
        "exp42 feature_gating_summary.json",
        0.42605526394314236,
        exp42_family_estimate(
            "feature_causal_gate__causal_minus_causal_matched_random__k200",
            family="qwen3_4b",
        ),
    ),
    ClaimCheck(
        "Exp43 three-family direct top-200 rescue",
        "exp43 primary_family_balanced_effects.csv",
        0.49437065972222216,
        exp43_nonoutlier_family_balanced("causal_top", "rescue_gain"),
    ),
    ClaimCheck(
        "Exp43 three-family direct top-200 rescue CI low",
        "exp43 primary_family_balanced_effects.csv",
        0.4507979343740185,
        exp43_nonoutlier_family_balanced("causal_top", "rescue_gain", "ci_low"),
        tolerance=2e-3,
    ),
    ClaimCheck(
        "Exp43 three-family direct top-200 rescue CI high",
        "exp43 primary_family_balanced_effects.csv",
        0.5394371798636192,
        exp43_nonoutlier_family_balanced("causal_top", "rescue_gain", "ci_high"),
        tolerance=2e-3,
    ),
    ClaimCheck(
        "Exp43 three-family direct rescue fraction",
        "exp43 primary_family_balanced_effects.csv",
        0.08055635817397248,
        exp43_nonoutlier_family_balanced("causal_top", "rescue_fraction"),
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus matched-random",
        "exp43 primary_family_balanced_effects.csv",
        0.5605922725243696,
        exp43_nonoutlier_family_balanced("causal_minus_causal_matched_random", "rescue_gain_causal_minus_control"),
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus matched-random CI low",
        "exp43 primary_family_balanced_effects.csv",
        0.5095765456287151,
        exp43_nonoutlier_family_balanced("causal_minus_causal_matched_random", "rescue_gain_causal_minus_control", "ci_low"),
        tolerance=2e-3,
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus matched-random CI high",
        "exp43 primary_family_balanced_effects.csv",
        0.6129771347514482,
        exp43_nonoutlier_family_balanced("causal_minus_causal_matched_random", "rescue_gain_causal_minus_control", "ci_high"),
        tolerance=2e-3,
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus matched-random fraction",
        "exp43 primary_family_balanced_effects.csv",
        0.10793944544225549,
        exp43_nonoutlier_family_balanced("causal_minus_causal_matched_random", "rescue_fraction_causal_minus_control"),
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus same-delta-random",
        "exp43 primary_family_balanced_effects.csv",
        0.4706316619537938,
        exp43_nonoutlier_family_balanced("causal_minus_causal_same_delta_random", "rescue_gain_causal_minus_control"),
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus same-delta-random CI low",
        "exp43 primary_family_balanced_effects.csv",
        0.42670571772754473,
        exp43_nonoutlier_family_balanced("causal_minus_causal_same_delta_random", "rescue_gain_causal_minus_control", "ci_low"),
        tolerance=2e-3,
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus same-delta-random CI high",
        "exp43 primary_family_balanced_effects.csv",
        0.5166974646566346,
        exp43_nonoutlier_family_balanced("causal_minus_causal_same_delta_random", "rescue_gain_causal_minus_control", "ci_high"),
        tolerance=2e-3,
    ),
    ClaimCheck(
        "Exp43 three-family rescue minus same-delta-random fraction",
        "exp43 primary_family_balanced_effects.csv",
        0.08273564836912005,
        exp43_nonoutlier_family_balanced("causal_minus_causal_same_delta_random", "rescue_fraction_causal_minus_control"),
    ),
    ClaimCheck(
        "Exp43 Llama direct rescue",
        "exp43 exp43_summary.json",
        0.6273225911458333,
        exp43_direct_rescue("llama31_8b", "rescue_gain_mean"),
    ),
    ClaimCheck(
        "Exp43 Mistral direct rescue",
        "exp43 exp43_summary.json",
        0.7552083333333334,
        exp43_direct_rescue("mistral_7b", "rescue_gain_mean"),
    ),
    ClaimCheck(
        "Exp43 Qwen direct rescue",
        "exp43 exp43_summary.json",
        0.1005810546875,
        exp43_direct_rescue("qwen3_4b", "rescue_gain_mean"),
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
