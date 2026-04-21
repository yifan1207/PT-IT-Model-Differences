#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
DENSE5_MODELS = [m for m in MODEL_ORDER if m != "deepseek_v2_lite"]
PT_PIPELINES = ["B_early_raw", "B_mid_raw", "B_late_raw"]
IT_PIPELINES = ["D_early_ptswap", "D_mid_ptswap", "D_late_ptswap"]
SUMMARY_METRICS = ["kl_to_own_final", "cross_kl", "delta_cosine", "residual_cosine"]
MECHANISM_METRICS = [
    "anti_top1_proj",
    "anti_top1_cosine",
    "support_teacher_proj",
    "support_teacher_cosine",
    "anti_kl_final_proj",
    "anti_kl_final_cosine",
    "orth_remainder_norm",
    "teacher_token_rank_gain",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_jsonl(path: Path):
    with open(path, "rb") as f:
        for raw in f:
            if not raw.strip():
                continue
            yield json.loads(raw.decode("utf-8", errors="ignore"))


def _collect_model_dirs(run_root: Path) -> list[tuple[str, Path]]:
    if (run_root / "merged").exists():
        base = run_root / "merged"
    else:
        base = run_root
    rows: list[tuple[str, Path]] = []
    subdirs = {p.name: p for p in base.iterdir() if p.is_dir()}
    for model in MODEL_ORDER:
        path = subdirs.get(model)
        if path is not None:
            rows.append((model, path))
    return rows


def _slice_mean(values: list[Any] | None, start: int, end: int) -> float | None:
    if values is None:
        return None
    subset = [float(v) for v in values[start:end] if v is not None]
    if not subset:
        return None
    return sum(subset) / len(subset)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x <= 1e-12 or den_y <= 1e-12:
        return None
    return num / (den_x * den_y)


def _collapse_category(raw: str) -> str:
    if raw in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FORMAT"}:
        return "FORMAT"
    if raw == "CONTENT":
        return "CONTENT"
    return "FUNCTION_OTHER"


def _final_region(config: dict[str, Any]) -> tuple[int, int]:
    n_layers = int(config["n_layers"])
    late_layers = config.get("late_mechanism_layers")
    if late_layers:
        return int(late_layers[0]), int(late_layers[-1]) + 1
    width = max(1, math.ceil(0.2 * n_layers))
    return n_layers - width, n_layers


def _prompt_delta(rows: list[dict[str, Any]], pipeline_key: str, baseline_key: str, field: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        pipeline = row.get(pipeline_key)
        baseline = row.get(baseline_key)
        if not pipeline or not baseline:
            continue
        p_val = pipeline.get(field)
        b_val = baseline.get(field)
        if p_val is None or b_val is None:
            continue
        vals.append(float(p_val) - float(b_val))
    return _mean(vals)


def _step_region_means(
    *,
    step_metrics_path: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[tuple[str, str, int], dict[str, float]]]:
    final_start, final_end = _final_region(config)
    windows = config.get("graft_windows_by_pipeline") or {}
    region_sums: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0.0}))
    mechanism_targets: dict[tuple[str, str, int], dict[str, float]] = {}

    for row in _iter_jsonl(step_metrics_path):
        pipeline = row["pipeline"]
        step = int(row["step"])
        prompt_id = row["prompt_id"]
        metrics = row["metrics"]
        pipeline_regions = {"final_20pct": (final_start, final_end)}
        for window_name, window in windows.items():
            pipeline_regions[f"window::{window_name}"] = (
                int(window["start_layer"]),
                int(window["end_layer_exclusive"]),
            )
        if pipeline in windows:
            window = windows[pipeline]
            pipeline_regions["graft_window"] = (
                int(window["start_layer"]),
                int(window["end_layer_exclusive"]),
            )
        for region_name, (start, end) in pipeline_regions.items():
            region_payload = region_sums[pipeline].setdefault(region_name, {})
            for metric_name in SUMMARY_METRICS:
                mean_value = _slice_mean(metrics.get(metric_name, []), start, end)
                metric_stats = region_payload.setdefault(metric_name, {"sum": 0.0, "count": 0.0})
                if mean_value is not None:
                    metric_stats["sum"] += mean_value
                    metric_stats["count"] += 1.0
        if pipeline in {"A_prime_raw", "B_late_raw", "C_it_chat", "D_late_ptswap"}:
            late_kl_mean = _slice_mean(metrics.get("kl_to_own_final", []), final_start, final_end)
            late_delta_cosine_mean = _slice_mean(metrics.get("delta_cosine", []), final_start, final_end)
            if late_kl_mean is not None and late_delta_cosine_mean is not None:
                mechanism_targets[(prompt_id, pipeline, step)] = {
                    "late_kl_mean": late_kl_mean,
                    "late_delta_cosine_mean": late_delta_cosine_mean,
                }

    region_means: dict[str, Any] = {}
    for pipeline, regions in region_sums.items():
        dst_regions: dict[str, Any] = {}
        for region_name, metric_map in regions.items():
            dst_regions[region_name] = {}
            for metric_name, stats in metric_map.items():
                count = int(stats["count"])
                dst_regions[region_name][metric_name] = {
                    "mean": (stats["sum"] / stats["count"]) if count else None,
                    "count": count,
                }
        region_means[pipeline] = dst_regions
    return region_means, mechanism_targets


def _mechanism_summary(
    *,
    mechanism_path: Path,
    mechanism_targets: dict[tuple[str, str, int], dict[str, float]],
) -> dict[str, Any]:
    pipeline_totals: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0.0}))
    by_teacher_cat: dict[str, dict[str, dict[str, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: {"sum": 0.0, "count": 0.0}))
    )
    corr_xs: dict[str, list[float]] = defaultdict(list)
    corr_ys: dict[str, list[float]] = defaultdict(list)
    corr_by_pipeline_xs: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    corr_by_pipeline_ys: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for row in _iter_jsonl(mechanism_path):
        pipeline = row["pipeline"]
        teacher_cat = _collapse_category(row.get("teacher_token_category_collapsed", "FUNCTION_OTHER"))
        predictors = {
            "anti_top1_proj": _mean([float(v) for v in row.get("anti_top1_proj", [])]),
            "anti_top1_cosine": _mean([float(v) for v in row.get("anti_top1_cosine", [])]),
            "support_teacher_proj": _mean([float(v) for v in row.get("support_teacher_proj", [])]),
            "support_teacher_cosine": _mean([float(v) for v in row.get("support_teacher_cosine", [])]),
            "anti_kl_final_proj": _mean([float(v) for v in row.get("anti_kl_final_proj", [])]),
            "anti_kl_final_cosine": _mean([float(v) for v in row.get("anti_kl_final_cosine", [])]),
            "orth_remainder_norm": _mean([float(v) for v in row.get("orth_remainder_norm", [])]),
            "teacher_token_rank_gain": _mean([float(v) for v in row.get("teacher_token_rank_gain", [])]),
        }
        for metric_name, mean_value in predictors.items():
            if mean_value is None:
                continue
            pipeline_totals[pipeline][metric_name]["sum"] += mean_value
            pipeline_totals[pipeline][metric_name]["count"] += 1.0
            by_teacher_cat[pipeline][teacher_cat][metric_name]["sum"] += mean_value
            by_teacher_cat[pipeline][teacher_cat][metric_name]["count"] += 1.0

        target = mechanism_targets.get((row["prompt_id"], pipeline, int(row["step"])))
        if target is None:
            continue
        late_kl_mean = target["late_kl_mean"]
        delta_cosine_mean = target["late_delta_cosine_mean"]
        corr_xs["delta_cosine"].append(delta_cosine_mean)
        corr_ys["delta_cosine"].append(late_kl_mean)
        corr_by_pipeline_xs[pipeline]["delta_cosine"].append(delta_cosine_mean)
        corr_by_pipeline_ys[pipeline]["delta_cosine"].append(late_kl_mean)
        for metric_name, mean_value in predictors.items():
            if mean_value is None:
                continue
            corr_xs[metric_name].append(mean_value)
            corr_ys[metric_name].append(late_kl_mean)
            corr_by_pipeline_xs[pipeline][metric_name].append(mean_value)
            corr_by_pipeline_ys[pipeline][metric_name].append(late_kl_mean)

    pipeline_means: dict[str, Any] = {}
    for pipeline, metric_map in pipeline_totals.items():
        pipeline_means[pipeline] = {
            metric_name: (stats["sum"] / stats["count"]) if stats["count"] else None
            for metric_name, stats in metric_map.items()
        }
    by_teacher_means: dict[str, Any] = {}
    for pipeline, cat_map in by_teacher_cat.items():
        by_teacher_means[pipeline] = {}
        for category, metric_map in cat_map.items():
            by_teacher_means[pipeline][category] = {
                metric_name: (stats["sum"] / stats["count"]) if stats["count"] else None
                for metric_name, stats in metric_map.items()
            }

    predictive_corr = {
        metric_name: _pearson(xs, corr_ys[metric_name])
        for metric_name, xs in corr_xs.items()
    }
    predictive_corr_by_pipeline: dict[str, Any] = {}
    for pipeline, metric_map in corr_by_pipeline_xs.items():
        predictive_corr_by_pipeline[pipeline] = {
            metric_name: _pearson(xs, corr_by_pipeline_ys[pipeline][metric_name])
            for metric_name, xs in metric_map.items()
        }
    return {
        "late_mean_by_pipeline": pipeline_means,
        "late_mean_by_teacher_category": by_teacher_means,
        "predictive_correlation_overall": predictive_corr,
        "predictive_correlation_by_pipeline": predictive_corr_by_pipeline,
    }


def _summarize_model(model: str, model_dir: Path) -> dict[str, Any]:
    config = _read_json(model_dir / "config.json")
    prompt_rows = list(_iter_jsonl(model_dir / "prompt_summaries.jsonl"))
    region_means, mechanism_targets = _step_region_means(
        step_metrics_path=model_dir / "step_metrics.jsonl",
        config=config,
    )
    final_start, final_end = _final_region(config)
    final_region = {
        "start_layer": final_start,
        "end_layer_exclusive": final_end,
        "display_range": f"{final_start}-{final_end - 1}",
    }

    windows = config.get("graft_windows_by_pipeline") or {}
    pt_side: dict[str, Any] = {}
    for pipeline in PT_PIPELINES:
        pt_side[pipeline] = {
            "graft_window": windows.get(pipeline),
            "regions": {
                "graft_window": {},
                "final_20pct": {},
            },
            "prompt_level": {
                "delta_mean_commitment_layer_kl_0.1": _prompt_delta(
                    prompt_rows,
                    pipeline_key=f"pipeline_{pipeline.lower()}",
                    baseline_key="pipeline_a_prime_raw",
                    field="mean_commitment_layer_kl_0.1",
                ),
            },
        }
        for region_name in ("graft_window", "final_20pct"):
            pipeline_region = region_means.get(pipeline, {}).get(region_name, {})
            baseline_key = f"window::{pipeline}" if region_name == "graft_window" else region_name
            baseline_region = region_means.get("A_prime_raw", {}).get(baseline_key, {})
            for metric_name in SUMMARY_METRICS:
                pipeline_mean = pipeline_region.get(metric_name, {}).get("mean")
                baseline_mean = baseline_region.get(metric_name, {}).get("mean")
                delta = None
                if pipeline_mean is not None and baseline_mean is not None:
                    delta = pipeline_mean - baseline_mean
                pt_side[pipeline]["regions"][region_name][metric_name] = {
                    "pipeline_mean": pipeline_mean,
                    "baseline_mean": baseline_mean,
                    "delta": delta,
                }

    it_side: dict[str, Any] = {}
    for pipeline in IT_PIPELINES:
        it_side[pipeline] = {
            "graft_window": windows.get(pipeline),
            "regions": {
                "graft_window": {},
                "final_20pct": {},
            },
            "prompt_level": {
                "delta_mean_commitment_layer_kl_0.1": _prompt_delta(
                    prompt_rows,
                    pipeline_key=f"pipeline_{pipeline.lower()}",
                    baseline_key="pipeline_c_it_chat",
                    field="mean_commitment_layer_kl_0.1",
                ),
            },
        }
        for region_name in ("graft_window", "final_20pct"):
            pipeline_region = region_means.get(pipeline, {}).get(region_name, {})
            baseline_key = f"window::{pipeline}" if region_name == "graft_window" else region_name
            baseline_region = region_means.get("C_it_chat", {}).get(baseline_key, {})
            for metric_name in SUMMARY_METRICS:
                pipeline_mean = pipeline_region.get(metric_name, {}).get("mean")
                baseline_mean = baseline_region.get(metric_name, {}).get("mean")
                delta = None
                if pipeline_mean is not None and baseline_mean is not None:
                    delta = pipeline_mean - baseline_mean
                it_side[pipeline]["regions"][region_name][metric_name] = {
                    "pipeline_mean": pipeline_mean,
                    "baseline_mean": baseline_mean,
                    "delta": delta,
                }

    mechanism_summary = {}
    mechanism_path = model_dir / "mechanism_metrics.jsonl"
    if mechanism_path.exists():
        mechanism_summary = _mechanism_summary(
            mechanism_path=mechanism_path,
            mechanism_targets=mechanism_targets,
        )

    return {
        "model": model,
        "model_dir": str(model_dir),
        "n_layers": int(config["n_layers"]),
        "final_region": final_region,
        "primary_readout_names_by_pipeline": config.get("primary_readout_names_by_pipeline"),
        "raw_readout_names_by_pipeline": config.get("raw_readout_names_by_pipeline"),
        "pt_side": pt_side,
        "it_side": it_side,
        "mechanism": mechanism_summary,
    }


def _dense_mean(models_payload: dict[str, Any], side_key: str, pipeline: str, region_name: str, metric_name: str) -> float | None:
    vals: list[float] = []
    for model in DENSE5_MODELS:
        metric = (
            models_payload.get(model, {})
            .get(side_key, {})
            .get(pipeline, {})
            .get("regions", {})
            .get(region_name, {})
            .get(metric_name, {})
            .get("delta")
        )
        if metric is not None:
            vals.append(float(metric))
    return _mean(vals)


def _dense_mechanism_mean(models_payload: dict[str, Any], pipeline: str, metric_name: str) -> float | None:
    vals: list[float] = []
    for model in DENSE5_MODELS:
        value = (
            models_payload.get(model, {})
            .get("mechanism", {})
            .get("late_mean_by_pipeline", {})
            .get(pipeline, {})
            .get(metric_name)
        )
        if value is not None:
            vals.append(float(value))
    return _mean(vals)


def _dense_mechanism_corr(models_payload: dict[str, Any], metric_name: str) -> float | None:
    vals: list[float] = []
    for model in DENSE5_MODELS:
        value = (
            models_payload.get(model, {})
            .get("mechanism", {})
            .get("predictive_correlation_overall", {})
            .get(metric_name)
        )
        if value is not None:
            vals.append(float(value))
    return _mean(vals)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exp13 full + exp14 combined causal outputs.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root directory containing merged per-model outputs.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional summary output path. Defaults to <run-root>/exp13_full_summary.json",
    )
    args = parser.parse_args()

    models_payload: dict[str, Any] = {}
    for model, model_dir in _collect_model_dirs(args.run_root):
        models_payload[model] = _summarize_model(model, model_dir)

    dense_family_means = {
        "pt_side_final20_kl_delta": {
            pipeline: _dense_mean(models_payload, "pt_side", pipeline, "final_20pct", "kl_to_own_final")
            for pipeline in PT_PIPELINES
        },
        "it_side_final20_kl_delta": {
            pipeline: _dense_mean(models_payload, "it_side", pipeline, "final_20pct", "kl_to_own_final")
            for pipeline in IT_PIPELINES
        },
        "pt_side_final20_delta_cosine": {
            pipeline: _dense_mean(models_payload, "pt_side", pipeline, "final_20pct", "delta_cosine")
            for pipeline in PT_PIPELINES
        },
        "it_side_final20_delta_cosine": {
            pipeline: _dense_mean(models_payload, "it_side", pipeline, "final_20pct", "delta_cosine")
            for pipeline in IT_PIPELINES
        },
        "mechanism_late_mean_by_pipeline": {
            pipeline: {
                metric_name: _dense_mechanism_mean(models_payload, pipeline, metric_name)
                for metric_name in MECHANISM_METRICS
            }
            for pipeline in ("A_prime_raw", "B_late_raw", "C_it_chat", "D_late_ptswap")
        },
        "mechanism_predictive_correlation": {
            metric_name: _dense_mechanism_corr(models_payload, metric_name)
            for metric_name in [
                "delta_cosine",
                "anti_top1_proj",
                "anti_top1_cosine",
                "support_teacher_proj",
                "support_teacher_cosine",
                "anti_kl_final_proj",
                "anti_kl_final_cosine",
            ]
        },
    }

    summary = {
        "analysis": "exp13_full_exp14",
        "run_root": str(args.run_root),
        "models": models_payload,
        "dense_family_means": dense_family_means,
    }
    out_path = args.out or (args.run_root / "exp13_full_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[exp13-full] wrote {out_path}")


if __name__ == "__main__":
    main()
