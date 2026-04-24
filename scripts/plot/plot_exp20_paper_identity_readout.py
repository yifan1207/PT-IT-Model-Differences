#!/usr/bin/env python3
"""Create the paper-facing Exp20 identity-vs-readout figure."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


IDENTITY_EFFECTS = [
    "PT+IT mid vs late IT-token transfer",
    "IT+PT mid vs late PT-token transfer",
]
MARGIN_GAIN_EFFECT = "IT late-minus-mid margin gain"
NECESSITY_EFFECT = "Pure IT late margin minus IT+PT late margin"


def _read_effects(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    effects: dict[tuple[str, str], dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["mode"], row["effect"])
            mean = float(row["mean"])
            effects[key] = {
                "mean": mean,
                "lo": float(row["ci_low"]),
                "hi": float(row["ci_high"]),
                "n": float(row["n"]),
            }
    return effects


def _barh_ci(ax: plt.Axes, labels: list[str], values: list[dict[str, float]], color: str) -> None:
    y = np.arange(len(labels))
    means = np.array([v["mean"] for v in values])
    xerr = np.array([
        means - np.array([v["lo"] for v in values]),
        np.array([v["hi"] for v in values]) - means,
    ])
    ax.barh(y, means, color=color, alpha=0.92)
    ax.errorbar(means, y, xerr=xerr, fmt="none", ecolor="black", capsize=4, lw=1.8)
    ax.set_yticks(y, labels)
    ax.axvline(0, color="black", lw=1.0)
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()


def make_plot(effects_csv: Path, out_png: Path, out_pdf: Path | None = None) -> None:
    effects = _read_effects(effects_csv)

    identity_values = [effects[("raw_shared", effect)] for effect in IDENTITY_EFFECTS]
    margin_values = [
        effects[("raw_shared", MARGIN_GAIN_EFFECT)],
        effects[("native", MARGIN_GAIN_EFFECT)],
    ]
    necessity_values = [
        effects[("raw_shared", NECESSITY_EFFECT)],
        effects[("native", NECESSITY_EFFECT)],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    _barh_ci(
        axes[0],
        ["PT+IT mid\nminus late", "IT+PT mid\nminus late"],
        identity_values,
        "#5E81AC",
    )
    axes[0].set_title("A. Mid transfers token identity more")
    axes[0].set_xlabel("Fraction-point advantage")
    axes[0].set_xlim(0, 0.24)
    axes[0].text(
        0.01,
        1.08,
        "raw-shared first-divergent token",
        transform=axes[0].transAxes,
        fontsize=10,
        color="#4C566A",
    )

    _barh_ci(
        axes[1],
        ["raw-shared", "native IT chat"],
        margin_values,
        "#BF616A",
    )
    axes[1].set_title("B. Late raises IT-token margin")
    axes[1].set_xlabel("Late minus mid margin gain (logits)")
    axes[1].set_xlim(0, 22.5)

    _barh_ci(
        axes[2],
        ["raw-shared", "native IT chat"],
        necessity_values,
        "#A3BE8C",
    )
    axes[2].set_title("C. PT late swap weakens IT readout")
    axes[2].set_xlabel("Pure IT late margin minus IT+PT late (logits)")
    axes[2].set_xlim(0, 12.5)

    fig.suptitle(
        "Exp20: first-divergent-token counterfactuals separate identity from readout",
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.035,
        "Bars show dense-5 prompt-bootstrap 95% CIs. DeepSeek is excluded from the pooled dense-family claim.",
        ha="center",
        fontsize=10,
        color="#4C566A",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    if out_pdf:
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effects-csv",
        type=Path,
        default=Path(
            "results/exp20_divergence_token_counterfactual/"
            "full_runpod_20260423_2148_combined_final/deep_dive/key_effects_ci_dense5.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results/exp20_divergence_token_counterfactual/"
            "full_runpod_20260423_2148_combined_final/deep_dive/exp20_paper_identity_readout.png"
        ),
    )
    parser.add_argument("--pdf", type=Path, default=None)
    args = parser.parse_args()
    pdf_path = args.pdf
    if pdf_path is None:
        pdf_path = args.out.with_suffix(".pdf")
    make_plot(args.effects_csv, args.out, pdf_path)
    print(f"[exp20] wrote {args.out}")
    print(f"[exp20] wrote {pdf_path}")


if __name__ == "__main__":
    main()
