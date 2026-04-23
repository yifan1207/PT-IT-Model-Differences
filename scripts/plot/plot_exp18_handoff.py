#!/usr/bin/env python3
"""Plot compact Exp18 handoff diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLLAPSED = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]


def _metric(block: dict, category: str, key: str) -> float:
    value = block.get("by_token_category", {}).get(category, {}).get(key)
    return float(value) if value is not None else 0.0


def make_plot(summary: dict, out_dir: Path) -> None:
    dense = summary["pooled"]["dense5"]
    windows = dense["windows"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    x = np.arange(len(COLLAPSED))
    width = 0.24
    for idx, window in enumerate(["early", "mid_policy", "late_reconciliation"]):
        axes[0].bar(
            x + (idx - 1) * width,
            [_metric(windows.get(window, {}), cat, "mean_teacher_rank_gain") for cat in COLLAPSED],
            width=width,
            label=window,
        )
    axes[0].axhline(0, color="black", lw=0.8, alpha=0.5)
    axes[0].set_xticks(x, COLLAPSED)
    axes[0].set_title("Target rank gain by disjoint window")
    axes[0].set_ylabel("Mean rank gain")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(
        x,
        [_metric(dense["handoff"], cat, "handoff_rate") for cat in COLLAPSED],
        color=["#2A9D8F", "#457B9D", "#8D99AE"],
    )
    axes[1].set_xticks(x, COLLAPSED)
    axes[1].set_title("Mid-selected and late-reconciled")
    axes[1].set_ylabel("Handoff rate")
    axes[1].set_ylim(0, 1)
    axes[1].grid(axis="y", alpha=0.25)

    for idx, window in enumerate(["B_early_raw", "B_mid_raw", "B_late_raw"]):
        axes[2].bar(
            x + (idx - 1) * width,
            [_metric(windows.get(window, {}), cat, "fraction_top1_displaced") for cat in COLLAPSED],
            width=width,
            label=window,
        )
    axes[2].set_xticks(x, COLLAPSED)
    axes[2].set_title("Continuity: A' to B_window top-1 changes")
    axes[2].set_ylabel("Top-1 displacement fraction")
    axes[2].set_ylim(0, 1)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend()

    fig.suptitle("Exp18 Mid-to-Late Token Handoff", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "exp18_handoff_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp18 handoff summary.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/exp18_midlate_token_handoff/matched_prefix_latest/summary.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    out_dir = args.out_dir or args.summary.parent
    make_plot(summary, out_dir)
    print(f"[exp18] wrote {out_dir / 'exp18_handoff_summary.png'}")


if __name__ == "__main__":
    main()
