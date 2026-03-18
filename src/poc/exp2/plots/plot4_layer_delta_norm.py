"""
Plot 4: Layer delta norm per layer — mean ||h_i − h_{i-1}||₂ by category.

Measures how much each transformer layer changes the residual stream.
A large delta at a layer means that layer is doing substantial computation
(attention + MLP together). Comparing IC vs OOC vs R reveals whether
some layers are differentially engaged for each task type.

  X-axis : transformer layer (0–33)
  Y-axis : mean layer delta norm across all prompts × all generation steps
  Lines  : IC (blue), OOC (orange), R (green) with ±1 SEM ribbon

Note: delta[0] = ||h_0|| (approximation; embedding norm, not true delta).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


CATEGORY_STYLE = {
    "in_context":     {"color": "#2196F3", "label": "IC (in-context)"},
    "out_of_context": {"color": "#FF9800", "label": "OOC (out-of-context)"},
    "reasoning":      {"color": "#4CAF50", "label": "R (reasoning)"},
}


def _aggregate(results: list[dict], key: str) -> dict[str, np.ndarray]:
    rows: dict[str, list] = defaultdict(list)
    for r in results:
        cat = r["category"]
        for step_vals in r[key]:
            rows[cat].append(step_vals)
    return {cat: np.array(v, dtype=float) for cat, v in rows.items()}


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cat_arrays = _aggregate(results, "layer_delta_norm")
    if not cat_arrays:
        print("  Plot 4 skipped — no data")
        return
    n_layers = next(iter(cat_arrays.values())).shape[1]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5))

    for cat, style in CATEGORY_STYLE.items():
        arr = cat_arrays.get(cat)
        if arr is None or arr.size == 0:
            continue
        mean = arr.mean(axis=0)
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
        ax.plot(layers, mean, color=style["color"],
                label=f"{style['label']}  (n={arr.shape[0]})", lw=2)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=style["color"])

    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Mean layer delta norm  ||h_i − h_{i−1}||₂", fontsize=11)
    ax.set_title(
        "Plot 4: Layer delta norm per layer by category\n"
        "delta[i] = ||residual_stream_after_layer_i − residual_stream_after_layer_{i-1}||₂",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_layers - 1)
    fig.tight_layout()

    out_path = Path(output_dir) / "plot4_layer_delta_norm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 4 saved → {out_path}")
