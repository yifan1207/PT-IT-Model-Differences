#!/usr/bin/env python3
"""Plot an aggregate Exp18 pure-flow overview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CATEGORIES = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]
COLORS = {
    "FORMAT": "#2A9D8F",
    "CONTENT": "#457B9D",
    "FUNCTION_OTHER": "#8D99AE",
}


def _metric(payload: dict, variant: str, window: str, category: str, metric: str) -> float:
    value = (
        payload.get("pooled", {})
        .get("dense5", {})
        .get(variant, {})
        .get("windows", {})
        .get(window, {})
        .get("by_token_category", {})
        .get(category, {})
        .get(metric)
    )
    return float(value) if value is not None else 0.0


def _model_metric(payload: dict, model: str, variant: str, window: str, category: str, metric: str) -> float:
    value = (
        payload.get("conditions", {})
        .get(f"{model}:{variant}", {})
        .get("windows", {})
        .get(window, {})
        .get("by_token_category", {})
        .get(category, {})
        .get(metric)
    )
    return float(value) if value is not None else 0.0


def make_plot(payload: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    x = np.arange(len(CATEGORIES))
    width = 0.36
    for idx, variant in enumerate(["pt", "it"]):
        offsets = x + (idx - 0.5) * width
        vals = [_metric(payload, variant, "mid_policy", cat, "margin_delta") for cat in CATEGORIES]
        axes[0, 0].bar(offsets, vals, width=width, label=f"{variant} mid")
    axes[0, 0].set_xticks(x, CATEGORIES)
    axes[0, 0].set_title("Dense5 Mid-Window Margin")
    axes[0, 0].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend()

    for idx, variant in enumerate(["pt", "it"]):
        offsets = x + (idx - 0.5) * width
        vals = [_metric(payload, variant, "late_reconciliation", cat, "margin_delta") for cat in CATEGORIES]
        axes[0, 1].bar(offsets, vals, width=width, label=f"{variant} late")
    axes[0, 1].set_xticks(x, CATEGORIES)
    axes[0, 1].set_title("Dense5 Late-Window Margin")
    axes[0, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    for idx, variant in enumerate(["pt", "it"]):
        offsets = x + (idx - 0.5) * width
        vals = [_metric(payload, variant, "late_reconciliation", cat, "handoff_rate") for cat in CATEGORIES]
        axes[1, 0].bar(offsets, vals, width=width, label=variant)
    axes[1, 0].set_xticks(x, CATEGORIES)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Dense5 Handoff Rate")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend()

    dense_models = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
    late_minus_mid = [
        _model_metric(payload, model, "it", "late_reconciliation", "FORMAT", "margin_delta")
        - _model_metric(payload, model, "it", "mid_policy", "FORMAT", "margin_delta")
        for model in dense_models
    ]
    axes[1, 1].bar(
        np.arange(len(dense_models)),
        late_minus_mid,
        color=COLORS["FORMAT"],
    )
    axes[1, 1].set_xticks(np.arange(len(dense_models)), dense_models, rotation=20, ha="right")
    axes[1, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[1, 1].set_title("IT FORMAT: Late Minus Mid Margin")
    axes[1, 1].grid(axis="y", alpha=0.25)

    quality = payload.get("quality", {})
    fig.suptitle(
        f"Exp18 Pure-Flow Overview\nmissing={len(quality.get('missing_conditions', []))} warnings={quality.get('warning_count', 0)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp18 pure-flow aggregate summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    out_path = args.out or args.summary.with_name("exp18_pure_flow_overview.png")
    make_plot(payload, out_path)
    print(f"[exp18] wrote {out_path}")


if __name__ == "__main__":
    main()
