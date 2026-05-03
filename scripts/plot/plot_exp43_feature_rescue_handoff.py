#!/usr/bin/env python
"""Plot Exp43 feature-rescue and middle-to-terminal summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


MODELS = ("gemma3_4b", "llama31_8b", "mistral_7b", "qwen3_4b")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", "None"} else 0.0


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_primary_rescue(analysis_dir: Path) -> None:
    rescue = _rows(analysis_dir / "rescue_effects.csv")
    controls = _rows(analysis_dir / "rescue_control_differences.csv")
    causal = {
        row["model"]: row
        for row in rescue
        if row.get("feature_set") == "causal_top"
        and row.get("control_mode") == "feature_delta"
        and row.get("k") == "200"
        and float(row.get("alpha", -1)) == 1.0
    }
    matched = {
        row["model"]: row
        for row in controls
        if row.get("control") == "causal_matched_random"
        and row.get("k") == "200"
        and float(row.get("alpha", -1)) == 1.0
    }
    same_delta = {
        row["model"]: row
        for row in controls
        if row.get("control") == "causal_same_delta_random"
        and row.get("k") == "200"
        and float(row.get("alpha", -1)) == 1.0
    }

    x = range(len(MODELS))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.bar([i - width for i in x], [_float(causal[m], "rescue_gain_mean") for m in MODELS], width, label="Targeted rescue")
    ax.bar([i for i in x], [_float(matched[m], "rescue_gain_causal_minus_control_mean") for m in MODELS], width, label="Minus matched-random")
    ax.bar([i + width for i in x], [_float(same_delta[m], "rescue_gain_causal_minus_control_mean") for m in MODELS], width, label="Minus same-delta random")
    ax.set_xticks(list(x), MODELS, rotation=20, ha="right")
    ax.set_ylabel("IT-margin rescue gain (logits)")
    ax.set_title("Exp43 primary feature-rescue effect, k=200 alpha=1")
    ax.legend(frameon=False)
    _save(fig, analysis_dir / "exp43_primary_rescue_gain.png")


def plot_family_balanced(analysis_dir: Path) -> None:
    rows = [
        row
        for row in _rows(analysis_dir / "primary_family_balanced_effects.csv")
        if "rescue_gain" in row.get("metric", "")
    ]
    labels = [
        "Targeted",
        "Targeted - matched random",
        "Targeted - same-delta random",
    ]
    effects = [
        "causal_top",
        "causal_minus_causal_matched_random",
        "causal_minus_causal_same_delta_random",
    ]
    by_effect = {row["effect"]: row for row in rows}
    estimates = [_float(by_effect[e], "estimate") for e in effects]
    lows = [_float(by_effect[e], "ci_low") for e in effects]
    highs = [_float(by_effect[e], "ci_high") for e in effects]
    yerr = [[est - lo for est, lo in zip(estimates, lows, strict=False)], [hi - est for est, hi in zip(estimates, highs, strict=False)]]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.bar(range(len(effects)), estimates, color=["#315c72", "#7c9a54", "#b46b45"])
    ax.errorbar(range(len(effects)), estimates, yerr=yerr, fmt="none", ecolor="#111111", capsize=4, linewidth=1.2)
    ax.set_xticks(range(len(effects)), labels, rotation=18, ha="right")
    ax.set_ylabel("Family-balanced rescue gain (logits)")
    ax.set_title("Exp43 family-balanced primary estimates")
    _save(fig, analysis_dir / "exp43_family_balanced_rescue_gain.png")


def plot_middle_probe(analysis_dir: Path) -> None:
    rows = _rows(analysis_dir / "middle_probe_effects.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    metrics = [
        ("margin_drop_mean", "Margin drop"),
        ("activation_drop_mean", "Activation drop"),
        ("decoder_margin_weighted_drop_mean", "Decoder-weighted drop"),
    ]
    width = 0.38
    for ax, (metric, title) in zip(axes, metrics, strict=False):
        early = {
            row["model"]: row
            for row in rows
            if row.get("window") == "early"
            and row.get("feature_set") == "causal_top"
            and row.get("k") == "200"
        }
        mid = {
            row["model"]: row
            for row in rows
            if row.get("window") == "mid"
            and row.get("feature_set") == "causal_top"
            and row.get("k") == "200"
        }
        x = range(len(MODELS))
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.bar([i - width / 2 for i in x], [_float(early[m], metric) for m in MODELS], width, label="early")
        ax.bar([i + width / 2 for i in x], [_float(mid[m], metric) for m in MODELS], width, label="mid")
        ax.set_title(title)
        ax.set_xticks(list(x), MODELS, rotation=25, ha="right")
    axes[0].set_ylabel("Drop after removing middle-window causal features")
    axes[-1].legend(frameon=False)
    _save(fig, analysis_dir / "exp43_middle_probe.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_primary_rescue(args.analysis_dir)
    plot_family_balanced(args.analysis_dir)
    plot_middle_probe(args.analysis_dir)


if __name__ == "__main__":
    main()
