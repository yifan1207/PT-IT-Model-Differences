"""Create dense-family-only layerwise KL plots for the paper.

This reads the committed Exp9 summary JSON rather than raw arrays, so the
figure can be regenerated from the review artifact without probe tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DENSE5 = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]


def plot_dense_layerwise(data_path: Path, out_path: Path, lens: str = "tuned") -> None:
    data = json.loads(data_path.read_text())
    lens_data = data[lens]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), sharey=False)
    axes_flat = axes.ravel()
    fig.suptitle(
        "Layerwise Delayed Stabilization in Five Dense PT/IT Pairs",
        fontsize=13,
        fontweight="bold",
    )

    for idx, model in enumerate(DENSE5):
        ax = axes_flat[idx]
        rec = lens_data[model]
        layers = np.arange(rec["n_layers"])
        pt = np.asarray(rec["mean_kl_pt_per_layer"], dtype=float)
        it = np.asarray(rec["mean_kl_it_per_layer"], dtype=float)

        ax.plot(layers, pt, color="#2f6db3", linestyle="--", linewidth=2, label="PT")
        ax.plot(layers, it, color="#c93d3d", linestyle="-", linewidth=2, label="IT")
        ax.fill_between(layers, pt, it, where=it >= pt, color="#c93d3d", alpha=0.12)
        ax.axvspan(rec["half_start"], rec["n_layers"] - 1, color="#555555", alpha=0.06)
        ax.set_title(rec["label"], fontsize=10.5)
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel("Mean KL to own final (nats)")
        ax.set_xlim(0, rec["n_layers"] - 1)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)

    axes_flat[-1].axis("off")
    axes_flat[-1].text(
        0.0,
        0.9,
        "Tuned-lens decoded\n"
        "native prompting\n"
        "gray band: final half\n\n"
        "Dense-only view for the\n"
        "main-paper context figure.",
        fontsize=10,
        va="top",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("results/exp09_cross_model_observational_replication/data/convergence_gap_values.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/exp09_cross_model_observational_replication/plots/L2_mean_kl_per_layer_tuned_dense5.png"),
    )
    parser.add_argument("--lens", choices=["tuned", "raw"], default="tuned")
    args = parser.parse_args()
    plot_dense_layerwise(args.data, args.out, args.lens)


if __name__ == "__main__":
    main()
