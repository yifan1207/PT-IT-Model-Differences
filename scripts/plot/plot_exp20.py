#!/usr/bin/env python3
"""Plot Exp20 divergence-token counterfactual summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONDITIONS = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
]

COLORS = {
    "A_pt_raw": "#4C566A",
    "B_early_raw": "#88C0D0",
    "B_mid_raw": "#5E81AC",
    "B_late_raw": "#2E3440",
    "C_it_chat": "#A3BE8C",
    "D_early_ptswap": "#EBCB8B",
    "D_mid_ptswap": "#D08770",
    "D_late_ptswap": "#BF616A",
}


def _mean(payload: dict, path: list[str], default: float = 0.0) -> float:
    cur = payload
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    if isinstance(cur, dict):
        cur = cur.get("mean")
    try:
        if cur is None:
            return default
        return float(cur)
    except (TypeError, ValueError):
        return default


def _fraction(counter_block: dict, key: str) -> float:
    try:
        value = counter_block.get(key, {}).get("fraction")
        return float(value) if value is not None else 0.0
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _bucket(summary: dict, pool: str) -> dict:
    return summary.get("pooled", {}).get(pool, {})


def make_plot(summary: dict, out_path: Path, pool: str) -> None:
    bucket = _bucket(summary, pool)
    readout = bucket.get("readouts", {}).get("first_diff", {}).get("by_condition", {})
    divergence = bucket.get("divergence", {}).get("first_diff", {})
    pairwise = bucket.get("pairwise_agreement", {})

    fig, axes = plt.subplots(2, 2, figsize=(17, 10))
    x = np.arange(len(CONDITIONS))

    it_follow = [
        _fraction(readout.get(condition, {}).get("token_class_at_step", {}), "it")
        for condition in CONDITIONS
    ]
    pt_follow = [
        _fraction(readout.get(condition, {}).get("token_class_at_step", {}), "pt")
        for condition in CONDITIONS
    ]
    width = 0.38
    axes[0, 0].bar(x - width / 2, pt_follow, width=width, color="#4C566A", label="matches PT token")
    axes[0, 0].bar(x + width / 2, it_follow, width=width, color="#A3BE8C", label="matches IT token")
    axes[0, 0].set_xticks(x, CONDITIONS, rotation=25, ha="right")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("Token chosen at first PT/IT divergence")
    axes[0, 0].grid(axis="y", alpha=0.25)
    axes[0, 0].legend()

    for idx, window in enumerate(["mid_policy", "late_reconciliation"]):
        offsets = x + (idx - 0.5) * width
        values = [
            _mean(
                readout,
                [
                    condition,
                    "windows",
                    window,
                    "it_minus_pt_margin.total_delta",
                ],
            )
            for condition in CONDITIONS
        ]
        axes[0, 1].bar(offsets, values, width=width, label=window)
    axes[0, 1].set_xticks(x, CONDITIONS, rotation=25, ha="right")
    axes[0, 1].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[0, 1].set_title("IT-token minus PT-token margin change")
    axes[0, 1].grid(axis="y", alpha=0.25)
    axes[0, 1].legend()

    anchor_pairs = [
        "A_pt_raw__B_early_raw",
        "A_pt_raw__B_mid_raw",
        "A_pt_raw__B_late_raw",
        "A_pt_raw__C_it_chat",
        "C_it_chat__D_early_ptswap",
        "C_it_chat__D_mid_ptswap",
        "C_it_chat__D_late_ptswap",
    ]
    agreement = [
        _mean(pairwise, [pair, "agreement_fraction"])
        for pair in anchor_pairs
    ]
    first_div = [
        _mean(pairwise, [pair, "first_divergence_step"])
        for pair in anchor_pairs
    ]
    axes[1, 0].bar(np.arange(len(anchor_pairs)), agreement, color="#5E81AC")
    axes[1, 0].set_xticks(np.arange(len(anchor_pairs)), anchor_pairs, rotation=25, ha="right")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_title("Free-running pairwise agreement")
    axes[1, 0].grid(axis="y", alpha=0.25)
    ax2 = axes[1, 0].twinx()
    ax2.plot(np.arange(len(anchor_pairs)), first_div, color="#BF616A", marker="o", label="first div step")
    ax2.set_ylabel("Mean first divergence step")

    winners_it = [
        _fraction(readout.get(condition, {}).get("winner", {}), "it")
        for condition in CONDITIONS
    ]
    winners_pt = [
        _fraction(readout.get(condition, {}).get("winner", {}), "pt")
        for condition in CONDITIONS
    ]
    axes[1, 1].bar(x - width / 2, winners_pt, width=width, color="#4C566A", label="PT wins final layer")
    axes[1, 1].bar(x + width / 2, winners_it, width=width, color="#A3BE8C", label="IT wins final layer")
    axes[1, 1].set_xticks(x, CONDITIONS, rotation=25, ha="right")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Readout final-layer winner at shared prefix")
    axes[1, 1].grid(axis="y", alpha=0.25)
    axes[1, 1].legend()

    q = summary.get("quality", {})
    present = divergence.get("present", 0)
    step = divergence.get("step", {}).get("mean")
    fig.suptitle(
        f"Exp20 Divergence Token Counterfactuals ({pool}) | first_diff={present}, mean_step={step}, warnings={q.get('warning_count', 0)}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp20 summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--pool", default="dense5", choices=["dense5", "all6", "deepseek_only"])
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    out_path = args.out or args.summary.with_name(f"exp20_{args.pool}_overview.png")
    make_plot(summary, out_path, args.pool)
    print(f"[exp20] wrote {out_path}")


if __name__ == "__main__":
    main()
