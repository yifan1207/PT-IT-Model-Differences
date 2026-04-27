#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


PT_CONDITIONS = ["A_prime_raw", "B_mid_raw", "B_late_raw", "B_midlate_raw"]
IT_CONDITIONS = ["C_it_chat", "D_mid_ptswap", "D_late_ptswap", "D_midlate_ptswap"]


def _effect_lookup(summary: dict) -> dict[str, dict]:
    return {
        row["effect"]: row
        for row in summary.get("effects", [])
        if row.get("model") == "dense5"
    }


def _plot_value(value):
    return float(value) if value is not None else math.nan


def _format_effect(row: dict) -> str:
    mean = row.get("mean")
    lo = row.get("ci_low")
    hi = row.get("ci_high")
    if mean is None:
        return "NA"
    if lo is None or hi is None:
        return f"{float(mean):.3f} [NA, NA]"
    return f"{float(mean):.3f} [{float(lo):.3f}, {float(hi):.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp23 mid x late KL factorial summary.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    means = summary["dense5_condition_means"]
    effects = _effect_lookup(summary)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    panels = [
        (axes[0], PT_CONDITIONS, ["A'", "mid", "late", "mid+late"], "PT host"),
        (axes[1], IT_CONDITIONS, ["C", "mid swap", "late swap", "mid+late swap"], "IT host"),
    ]
    for ax, conditions, labels, title in panels:
        values = [_plot_value(means.get(condition)) for condition in conditions]
        ax.bar(labels, values, color=["#808080", "#4c78a8", "#f58518", "#54a24b"])
        ax.set_title(title)
        ax.set_ylabel("Final-20% KL(layer || own final)")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)

    i_pt = effects.get("I_pt", {})
    i_it = effects.get("I_collapse_it", {})
    fig.suptitle(
        "Exp23 mid x late KL factorial\n"
        f"PT interaction={_format_effect(i_pt)}  |  "
        f"IT collapse interaction={_format_effect(i_it)}"
        if i_pt.get("mean") is not None and i_it.get("mean") is not None
        else "Exp23 mid x late KL factorial"
    )
    fig.tight_layout()
    out = args.out or (args.summary.parent / "exp23_midlate_kl_factorial.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"[exp23] wrote {out}")


if __name__ == "__main__":
    main()
