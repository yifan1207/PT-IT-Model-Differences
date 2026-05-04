#!/usr/bin/env python3
"""Paper-facing plot for random-disagreement selection baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_MODELS = ("llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b")
KEYS = (
    "first_diff__reference",
    "random_local_disagreement__source_balanced",
    "random_local_disagreement__pt_rollout",
    "random_local_disagreement__it_rollout",
    "prediv_future_pair__shared_prediv",
)
LABELS = (
    "First\ndivergence",
    "Random local\nbalanced",
    "Random local\nPT path",
    "Random local\nIT path",
    "Pre-divergence\nfuture pair",
)


def _family_mean(summary: dict, key: str, readout: str, models: tuple[str, ...]) -> float:
    model_payloads = summary["summaries"][key][readout]["interaction"]["models"]
    vals = []
    for model in models:
        estimate = model_payloads[model]["estimate"]
        if estimate is not None:
            vals.append(float(estimate))
    if not vals:
        raise ValueError(f"No finite model estimates for {key}")
    return float(np.mean(vals))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/exp37_random_prefix_baseline/"
            "exp37_full_dense5_auth_xetfast_h100x8_20260503_002609/"
            "analysis/summary.json"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/paper_synthesis/exp37_core_small_selection_baseline/selection_baselines_core_small.png"),
    )
    parser.add_argument("--readout", default="common_it")
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    models = tuple(args.models)
    vals = [_family_mean(summary, key, args.readout, models) for key in KEYS]
    ref = vals[0]
    shares = [100.0 * val / ref for val in vals]

    colors = ["#315F7D", "#76A978", "#A1C98A", "#C9A64B", "#B75B5E"]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(len(KEYS))
    ax.bar(x, vals, color=colors, width=0.72)
    ax.axhline(ref, color="#315F7D", linestyle="--", linewidth=1.2, alpha=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel("Upstream x late interaction (logits)")
    ax.set_title("First divergence is a high-signal, not arbitrary, support")
    ax.set_ylim(min(-0.08, min(vals) - 0.15), max(vals) * 1.25)

    for idx, (value, share) in enumerate(zip(vals, shares, strict=True)):
        label = f"{value:+.2f}\n{share:.0f}%"
        ax.text(idx, value + max(vals) * 0.035, label, ha="center", va="bottom", fontsize=9)

    ax.text(
        0.99,
        0.95,
        "Core-small family-balanced means",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
