#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


WINDOW_ORDER = {"early": 0, "mid": 1, "late": 2}


def _short_label(name: str) -> str:
    label = name.replace("B_", "").replace("D_", "")
    label = label.replace("rand_resproj_", "act-match ")
    label = label.replace("rand_norm_", "norm-match ")
    label = label.replace("layerperm", "layer-perm")
    return label.replace("_", " ")


def _ordered(names: list[str]) -> list[str]:
    def key(name: str):
        window_idx = 9
        for window_name, idx in WINDOW_ORDER.items():
            if f"_{window_name}_" in f"_{name}_":
                window_idx = idx
                break
        if name.endswith("true"):
            control_idx = 0
        elif "identity" in name:
            control_idx = 1
        elif "layerperm" in name:
            control_idx = 2
        elif "rand_resproj" in name:
            control_idx = 3
        elif "rand_norm" in name:
            control_idx = 4
        else:
            control_idx = 9
        return (window_idx, control_idx, name)

    return sorted(names, key=key)


def _plot_side(ax, payload: dict, side: str, metric: str, title: str) -> None:
    rows = payload["dense5_mean_deltas"].get(side, {})
    names = _ordered(list(rows))
    values = [rows[name].get(metric) for name in names]
    colors = []
    for name in names:
        if name.endswith("true"):
            colors.append("#274c77")
        elif "identity" in name:
            colors.append("#5c7c5c")
        else:
            colors.append("#8aa6c1")
    ax.bar(range(len(names)), [0.0 if value is None else value for value in values], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([_short_label(name) for name in names], rotation=35, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Exp19 specificity-control summaries.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = json.loads(args.summary.read_text())
    out_dir = args.out_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    _plot_side(axes[0], payload, "pt_side", "kl_to_own_final", "PT side: final-20% KL delay vs matched controls")
    _plot_side(axes[1], payload, "pt_side", "local_teacher_rank_gain", "PT side: teacher-rank gain vs matched controls")
    fig.suptitle("Exp19 depth-matched PT window controls: true IT grafts vs matched nulls")
    path = out_dir / "exp19_specificity_main.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    print(f"[exp19] wrote {path}")


if __name__ == "__main__":
    main()
