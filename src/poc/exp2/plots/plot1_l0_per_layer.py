"""
Plot 1: Mean L0 (active feature count) per layer, by category.

L0 at layer i = number of transcoder features with non-zero activation
(computed by encoding the MLP input through the transcoder encoder).

Hypothesis: if IC prompts produce more focused feature activations
(lower L0) while R prompts require broader feature engagement (higher L0),
this would suggest mechanistically distinct computation modes.

  X-axis : transformer layer (0–33)
  Y-axis : mean L0 across all prompts × all generation steps in that category
  Lines  : IC (blue), OOC (orange), R (green) with ±1 SEM ribbon
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


CATEGORY_STYLE = {
    "in_context":    {"color": "#2196F3", "label": "IC (in-context)"},
    "out_of_context": {"color": "#FF9800", "label": "OOC (out-of-context)"},
    "reasoning":     {"color": "#4CAF50", "label": "R (reasoning)"},
}


def _aggregate_by_category(results: list[dict], key: str) -> dict[str, np.ndarray]:
    """Flatten results[i][key] into per-category arrays of shape [N_obs, N_layers].

    key must point to a list[step][layer] of floats.
    Each (prompt, step) pair contributes one row.
    """
    rows: dict[str, list] = defaultdict(list)
    for r in results:
        cat = r["category"]
        for step_vals in r[key]:
            rows[cat].append(step_vals)
    return {cat: np.array(v, dtype=float) for cat, v in rows.items()}


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cat_arrays = _aggregate_by_category(results, "l0")
    if not cat_arrays:
        print("  Plot 1 skipped — no data")
        return
    n_layers = next(iter(cat_arrays.values())).shape[1]
    layers = np.arange(n_layers)

    fig, ax = plt.subplots(figsize=(11, 5))

    for cat, style in CATEGORY_STYLE.items():
        arr = cat_arrays.get(cat)
        if arr is None or arr.size == 0:
            continue
        mean = arr.mean(axis=0)          # [n_layers]
        sem = arr.std(axis=0) / np.sqrt(arr.shape[0])
        ax.plot(layers, mean, color=style["color"], label=f"{style['label']}  (n={arr.shape[0]})", lw=2)
        ax.fill_between(layers, mean - sem, mean + sem, alpha=0.2, color=style["color"])

    ax.set_xlabel("Transformer layer", fontsize=11)
    ax.set_ylabel("Mean active features (L0)", fontsize=11)
    ax.set_title(
        "Plot 1: Mean L0 per layer by task category\n"
        "L0 = active transcoder feature count per MLP; averaged over all prompts × generation steps",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_layers - 1)
    fig.tight_layout()

    out_path = Path(output_dir) / "plot1_l0_per_layer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 1 saved → {out_path}")
