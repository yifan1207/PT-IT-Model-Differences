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
DENSE5 = [model for model in MODEL_ORDER if model != "deepseek_v2_lite"]
WINDOWS = ["early", "mid", "late"]
METRICS = [
    "kl_to_own_final",
    "delta_cosine",
    "next_token_rank",
    "cross_kl",
    "residual_divergence",
    "support_teacher_proj",
    "anti_top1_proj",
    "local_teacher_rank_gain",
]


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _mean(values: list[float]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return sum(kept) / len(kept) if kept else None


def _slice_mean(values: list[Any] | None, start: int, end: int) -> float | None:
    if values is None:
        return None
    return _mean([float(value) for value in values[start:end] if value is not None])


def _model_dirs(run_root: Path) -> list[tuple[str, Path]]:
    base = run_root / "merged" if (run_root / "merged").exists() else run_root
    by_name = {path.name: path for path in base.iterdir() if path.is_dir()}
    return [(model, by_name[model]) for model in MODEL_ORDER if model in by_name]


def _final_region(config: dict[str, Any]) -> tuple[int, int]:
    layers = config.get("final_region_layers")
    if layers:
        return int(layers[0]), int(layers[-1]) + 1
    n_layers = int(config["n_layers"])
    width = max(1, math.ceil(0.2 * n_layers))
    return n_layers - width, n_layers


def _side_for_pipeline(pipeline: str) -> str | None:
    if pipeline.startswith("B_"):
        return "pt_side"
    if pipeline.startswith("D_"):
        return "it_side"
    return None


def analyze_model(model: str, model_dir: Path) -> dict[str, Any]:
    config = json.loads((model_dir / "config.json").read_text())
    final_start, final_end = _final_region(config)
    sums: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, int]] = set()
    duplicates = 0
    step_rows = 0
    for row in _iter_jsonl(model_dir / "step_metrics.jsonl"):
        key = (str(row["pipeline"]), str(row["prompt_id"]), int(row["step"]))
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        step_rows += 1
        metrics = row.get("metrics", {})
        for metric in METRICS:
            value = _slice_mean(metrics.get(metric), final_start, final_end)
            if value is not None:
                sums[row["pipeline"]][metric].append(value)

    means = {
        pipeline: {
            metric: _mean(values)
            for metric, values in metric_map.items()
        }
        for pipeline, metric_map in sums.items()
    }
    deltas: dict[str, dict[str, dict[str, float | None]]] = {"pt_side": {}, "it_side": {}}
    for pipeline, metric_map in means.items():
        side = _side_for_pipeline(pipeline)
        if side is None:
            continue
        baseline = "A_prime_raw" if side == "pt_side" else "C_it_chat"
        deltas[side][pipeline] = {}
        for metric in METRICS:
            p_val = metric_map.get(metric)
            b_val = means.get(baseline, {}).get(metric)
            deltas[side][pipeline][metric] = (
                float(p_val) - float(b_val)
                if p_val is not None and b_val is not None
                else None
            )

    return {
        "model": model,
        "model_dir": str(model_dir),
        "n_layers": int(config["n_layers"]),
        "final_region": {
            "start_layer": final_start,
            "end_layer_exclusive": final_end,
            "display_range": f"{final_start}-{final_end - 1}",
        },
        "pipelines": config.get("pipelines", []),
        "graft_windows_by_pipeline": config.get("graft_windows_by_pipeline", {}),
        "n_step_rows": step_rows,
        "duplicates": duplicates,
        "means": means,
        "deltas": deltas,
    }


def _dense_mean(models: dict[str, Any], side: str, pipeline: str, metric: str) -> float | None:
    values = []
    for model in DENSE5:
        payload = models.get(model)
        if not payload:
            continue
        value = payload.get("deltas", {}).get(side, {}).get(pipeline, {}).get(metric)
        if value is not None:
            values.append(float(value))
    return _mean(values)


def _pipelines_by_side(models: dict[str, Any], side: str) -> list[str]:
    names: set[str] = set()
    for payload in models.values():
        names.update(payload.get("deltas", {}).get(side, {}).keys())
    return sorted(names)


def run_analysis(run_root: Path) -> dict[str, Any]:
    models = {
        model: analyze_model(model, model_dir)
        for model, model_dir in _model_dirs(run_root)
    }
    dense: dict[str, Any] = {"pt_side": {}, "it_side": {}}
    for side in ("pt_side", "it_side"):
        for pipeline in _pipelines_by_side(models, side):
            dense[side][pipeline] = {
                metric: _dense_mean(models, side, pipeline, metric)
                for metric in METRICS
            }
    contrasts: dict[str, Any] = {"pt_side": {}, "it_side": {}}
    for side, prefix in (("pt_side", "B"), ("it_side", "D")):
        side_rows = dense.get(side, {})
        if not side_rows:
            continue
        for window in WINDOWS:
            true_metrics = side_rows.get(f"{prefix}_{window}_true", {})
            identity_metrics = side_rows.get(f"{prefix}_{window}_identity", {})
            layerperm_metrics = side_rows.get(f"{prefix}_{window}_layerperm", {})
            contrasts[side][window] = {}
            for metric in METRICS:
                true_value = true_metrics.get(metric)
                identity_value = identity_metrics.get(metric)
                layerperm_value = layerperm_metrics.get(metric)
                rand_resproj_values = []
                rand_norm_values = []
                for pipeline, metric_map in side_rows.items():
                    if f"{prefix}_{window}_rand_resproj_" in pipeline and metric_map.get(metric) is not None:
                        rand_resproj_values.append(float(metric_map[metric]))
                    if f"{prefix}_{window}_rand_norm_" in pipeline and metric_map.get(metric) is not None:
                        rand_norm_values.append(float(metric_map[metric]))
                rand_resproj_mean = _mean(rand_resproj_values)
                rand_norm_mean = _mean(rand_norm_values)
                contrasts[side][window][metric] = {
                    "true": true_value,
                    "identity": identity_value,
                    "layerperm": layerperm_value,
                    "rand_resproj_mean": rand_resproj_mean,
                    "rand_norm_mean": rand_norm_mean,
                    "true_minus_identity": (
                        float(true_value) - float(identity_value)
                        if true_value is not None and identity_value is not None
                        else None
                    ),
                    "true_minus_layerperm": (
                        float(true_value) - float(layerperm_value)
                        if true_value is not None and layerperm_value is not None
                        else None
                    ),
                    "true_minus_rand_resproj_mean": (
                        float(true_value) - float(rand_resproj_mean)
                        if true_value is not None and rand_resproj_mean is not None
                        else None
                    ),
                    "true_minus_rand_norm_mean": (
                        float(true_value) - float(rand_norm_mean)
                        if true_value is not None and rand_norm_mean is not None
                        else None
                    ),
                }
    pt_contrasts = contrasts.get("pt_side", {})
    key_metrics = ["kl_to_own_final", "local_teacher_rank_gain", "support_teacher_proj"]
    late_specificity: dict[str, Any] = {}
    for metric in key_metrics:
        by_window: dict[str, Any] = {}
        for window in WINDOWS:
            metric_payload = pt_contrasts.get(window, {}).get(metric, {})
            margin = metric_payload.get("true_minus_rand_resproj_mean")
            by_window[window] = {
                "true": metric_payload.get("true"),
                "rand_resproj_mean": metric_payload.get("rand_resproj_mean"),
                "true_minus_rand_resproj_mean": margin,
            }
        late_margin = by_window.get("late", {}).get("true_minus_rand_resproj_mean")
        early_margin = by_window.get("early", {}).get("true_minus_rand_resproj_mean")
        mid_margin = by_window.get("mid", {}).get("true_minus_rand_resproj_mean")
        late_specificity[metric] = {
            "by_window": by_window,
            "late_minus_early_margin": (
                float(late_margin) - float(early_margin)
                if late_margin is not None and early_margin is not None
                else None
            ),
            "late_minus_mid_margin": (
                float(late_margin) - float(mid_margin)
                if late_margin is not None and mid_margin is not None
                else None
            ),
            "margin_ranking_desc": [
                window
                for window, _ in sorted(
                    (
                        (window, values["true_minus_rand_resproj_mean"])
                        for window, values in by_window.items()
                        if values["true_minus_rand_resproj_mean"] is not None
                    ),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )
            ],
        }
    return {
        "analysis": "exp19_late_mlp_specificity_controls",
        "run_root": str(run_root),
        "dense5_models": [model for model in DENSE5 if model in models],
        "models": models,
        "dense5_mean_deltas": dense,
        "dense5_control_contrasts": contrasts,
        "dense5_late_specificity_summary": late_specificity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp19 late-MLP specificity controls.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    summary = run_analysis(args.run_root)
    out = args.out or (args.run_root / "exp19_specificity_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[exp19] wrote {out}")


if __name__ == "__main__":
    main()
