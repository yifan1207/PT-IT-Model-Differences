#!/usr/bin/env python3
"""Aggregate Exp18 pure-flow summaries into one run-level analysis JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODELS = [
    "gemma3_4b",
    "llama31_8b",
    "qwen3_4b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
VARIANTS = ["pt", "it"]
DENSE5 = [m for m in MODELS if m != "deepseek_v2_lite"]
CATEGORIES = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]
WINDOWS = ["early", "mid_policy", "late_reconciliation"]
METRICS = [
    "support_target_delta",
    "repulsion_top10_delta",
    "margin_delta",
    "handoff_rate",
    "mean_first_top1_layer",
    "mean_first_top5_layer",
    "mean_first_top20_layer",
    "fraction_top1_displaced",
    "mean_top20_entries",
    "mean_top20_exits",
]


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def _category_block(summary: dict[str, Any], window: str, category: str) -> dict[str, Any]:
    return (
        summary.get("windows", {})
        .get(window, {})
        .get("by_token_category", {})
        .get(category, {})
    )


def _weighted_average(blocks: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    total = 0.0
    weight = 0.0
    total_count = 0
    for block in blocks:
        value = block.get(metric)
        count = int(block.get("count") or 0)
        if value is None or count <= 0:
            continue
        total += float(value) * count
        weight += count
        total_count += count
    return {
        "count": total_count,
        metric: _safe_div(total, weight),
    }


def _load_condition(summary_path: Path) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text())
    out = {
        "n_prompts": int(payload.get("n_prompts", 0)),
        "n_steps": int(payload.get("n_steps", 0)),
        "windows": {},
    }
    for window in WINDOWS:
        out["windows"][window] = {"by_token_category": {}}
        for category in CATEGORIES:
            block = _category_block(payload, window, category)
            out["windows"][window]["by_token_category"][category] = {
                metric: block.get(metric) for metric in METRICS
            }
            out["windows"][window]["by_token_category"][category]["count"] = int(block.get("count") or 0)
    return out


def _pool_conditions(conditions: dict[str, dict[str, Any]], models: list[str]) -> dict[str, Any]:
    pooled: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_conditions = [conditions[f"{model}:{variant}"] for model in models if f"{model}:{variant}" in conditions]
        pooled_variant = {
            "n_conditions": len(variant_conditions),
            "n_prompts": sum(c["n_prompts"] for c in variant_conditions),
            "n_steps": sum(c["n_steps"] for c in variant_conditions),
            "windows": {},
        }
        for window in WINDOWS:
            pooled_variant["windows"][window] = {"by_token_category": {}}
            for category in CATEGORIES:
                blocks = [
                    c["windows"][window]["by_token_category"][category]
                    for c in variant_conditions
                ]
                pooled_variant["windows"][window]["by_token_category"][category] = {
                    metric: _weighted_average(blocks, metric).get(metric)
                    for metric in METRICS
                }
                pooled_variant["windows"][window]["by_token_category"][category]["count"] = sum(
                    int(block.get("count") or 0) for block in blocks
                )
        pooled[variant] = pooled_variant
    return pooled


def _quality_checks(conditions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = [f"{model}:{variant}" for model in MODELS for variant in VARIANTS]
    missing = [key for key in expected if key not in conditions]
    warnings: list[str] = []
    for key, condition in sorted(conditions.items()):
        if condition["n_prompts"] != 600:
            warnings.append(f"{key}: expected 600 prompts, found {condition['n_prompts']}")
        if condition["n_steps"] <= 0:
            warnings.append(f"{key}: no generated steps")
        for window in WINDOWS:
            for category in CATEGORIES:
                block = condition["windows"][window]["by_token_category"][category]
                if int(block.get("count") or 0) <= 0:
                    warnings.append(f"{key}: empty {window}/{category}")
    return {
        "expected_conditions": expected,
        "missing_conditions": missing,
        "warning_count": len(warnings),
        "warnings": warnings,
        "ok": not missing and not warnings,
    }


def analyze_run(run_dir: Path) -> dict[str, Any]:
    conditions: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for variant in VARIANTS:
            summary_path = run_dir / "pure_flow" / model / variant / "pure_flow_summary.json"
            if summary_path.exists():
                conditions[f"{model}:{variant}"] = _load_condition(summary_path)
    return {
        "run_dir": str(run_dir),
        "conditions": conditions,
        "pooled": {
            "dense5": _pool_conditions(conditions, DENSE5),
            "all6": _pool_conditions(conditions, MODELS),
            "deepseek_only": _pool_conditions(conditions, ["deepseek_v2_lite"]),
        },
        "quality": _quality_checks(conditions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp18 pure-flow summaries.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = analyze_run(args.run_dir)
    out_path = args.out or (args.run_dir / "pure_flow_aggregate_summary.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[exp18] wrote {out_path}")


if __name__ == "__main__":
    main()
