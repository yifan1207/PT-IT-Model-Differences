"""
Plot 7: Logit-lens entropy per layer, by category.

Logit-lens entropy at layer i = H(softmax(h_i @ W_U)_real), the entropy of
the output distribution if we early-exited at layer i. This reveals at what
depth each category "crystallises" its prediction:
  - A sharp drop in entropy at layer L → the model commits to its answer at L.
  - IC prompts should crystallise early (high-confidence retrieval).
  - R prompts may crystallise later (computation needs more layers).
  - OOC prompts may never fully crystallise (persistent uncertainty).

  X-axis : transformer layer (0–33)
  Y-axis : mean logit-lens entropy (nats) across all prompts × generation steps
  Lines  : IC (blue), OOC (orange), R (green) with ±1 SEM ribbon
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

    cat_arrays = _aggregate(results, "logit_lens_entropy")
    if not cat_arrays:
        print("  Plot 7 skipped — no data")
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
    ax.set_ylabel("Mean logit-lens entropy (nats)", fontsize=11)
    ax.set_title(
        "Plot 7: Logit-lens entropy per layer by category\n"
        "entropy[layer] = H(softmax(residual_stream[layer] @ W_U)  over real tokens)",
        fontsize=10,
    )
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_layers - 1)
    fig.tight_layout()

    out_path = Path(output_dir) / "plot7_logit_lens_entropy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 7 saved → {out_path}")
