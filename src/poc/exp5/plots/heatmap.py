from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_phase_benchmark_heatmap(rows: list[dict], output_path: str | Path) -> None:
    if not rows:
        return
    methods = sorted({row["condition"] for row in rows})
    benchmarks = sorted({row["benchmark"] for row in rows})
    matrix = np.full((len(methods), len(benchmarks)), np.nan, dtype=np.float32)
    for row in rows:
        i = methods.index(row["condition"])
        j = benchmarks.index(row["benchmark"])
        matrix[i, j] = row["value"]

    fig, ax = plt.subplots(figsize=(1.8 * len(benchmarks), 0.5 * len(methods) + 3))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(benchmarks)))
    ax.set_xticklabels(benchmarks, rotation=30, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title("Exp5 — Phase × Benchmark Score Heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

