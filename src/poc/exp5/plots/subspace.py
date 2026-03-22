from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_checkpoint_metrics(
    subspace_rows: dict[str, dict[int, dict[str, float]]],
    output_path: str | Path,
) -> None:
    """Plot subspace shift metrics for all ablation conditions.

    Per Exp D design (§4 of the research plan): produces the full
    (condition × checkpoint × metric) table as a multi-line plot so that
    each condition can be compared against the baseline at every checkpoint layer.

    Parameters
    ----------
    subspace_rows : {condition_name: {layer_idx: {metric: float}}}
        Output of summarise_checkpoint_shift keyed by condition name.
    output_path : path to save the figure.
    """
    if not subspace_rows:
        return

    metrics = ["cka_to_baseline", "cosine_to_baseline", "participation_ratio"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for condition_name, summary in subspace_rows.items():
            layers = sorted(l for l in summary if metric in summary[l])
            if not layers:
                continue
            values = [summary[l][metric] for l in layers]
            ax.plot(layers, values, marker="o", label=condition_name)
        ax.set_title(metric.replace("_", " "))
        ax.set_xlabel("checkpoint layer")
        # Only draw a legend if there are labelled artists
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=6, loc="best")

    fig.suptitle("Exp5 — Checkpoint Subspace Metrics by Condition")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
