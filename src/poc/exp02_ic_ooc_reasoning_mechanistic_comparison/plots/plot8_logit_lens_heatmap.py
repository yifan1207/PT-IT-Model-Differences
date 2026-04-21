"""
Plot 8: Logit-lens entropy heatmap [layer × generation step] per category.

Shows how the entropy trajectory (layer-depth vs generation-step) evolves
across all three task types. The "crystallisation front" — where entropy
drops sharply — reveals when the model commits to a prediction.

  X-axis : generation step
  Y-axis : transformer layer (top = 0)
  Color  : mean logit-lens entropy across all prompts in that category
  Panels : IC | OOC | R
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


CATEGORY_ORDER = ["in_context", "out_of_context", "reasoning"]
CATEGORY_LABEL = {
    "in_context":     "IC",
    "out_of_context": "OOC",
    "reasoning":      "R",
}


def _build_ll_heatmaps(results: list[dict]) -> dict[str, np.ndarray]:
    """category → np.ndarray [n_layers, max_steps] of mean logit-lens entropy."""
    max_steps: dict[str, int] = defaultdict(int)
    for r in results:
        max_steps[r["category"]] = max(max_steps[r["category"]], len(r["logit_lens_entropy"]))

    n_layers = len(results[0]["logit_lens_entropy"][0]) if results else 34

    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for cat in CATEGORY_ORDER:
        ms = max_steps.get(cat, 1)
        sums[cat] = np.zeros((n_layers, ms), dtype=float)
        counts[cat] = np.zeros((n_layers, ms), dtype=float)

    for r in results:
        cat = r["category"]
        for step, layer_vals in enumerate(r["logit_lens_entropy"]):
            for layer, val in enumerate(layer_vals):
                sums[cat][layer, step] += val
                counts[cat][layer, step] += 1

    heatmaps = {}
    for cat in CATEGORY_ORDER:
        cnt = counts.get(cat)
        if cnt is None or cnt.sum() == 0:
            continue
        with np.errstate(invalid="ignore"):
            hm = np.where(cnt > 0, sums[cat] / cnt, np.nan)
        heatmaps[cat] = hm
    return heatmaps


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    heatmaps = _build_ll_heatmaps(results)
    if not heatmaps:
        print("  Plot 8 skipped — no data")
        return

    cats = [c for c in CATEGORY_ORDER if c in heatmaps]
    n_panels = len(cats)

    # Shared colour scale
    vmin = min(np.nanmin(hm) for hm in heatmaps.values())
    vmax = max(np.nanmax(hm) for hm in heatmaps.values())

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        hm = heatmaps[cat]
        n_layers, n_steps = hm.shape
        im = ax.imshow(
            hm, aspect="auto", origin="upper",
            extent=[-0.5, n_steps - 0.5, n_layers - 0.5, -0.5],
            vmin=vmin, vmax=vmax, cmap="viridis_r",
        )
        ax.set_title(CATEGORY_LABEL[cat], fontsize=13, fontweight="bold")
        ax.set_xlabel("Generation step", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Transformer layer", fontsize=10)
        plt.colorbar(im, ax=ax, label="Mean logit-lens entropy (nats)",
                     fraction=0.046, pad=0.04)

    fig.suptitle(
        "Plot 8: Logit-lens entropy per (layer, generation step) by category\n"
        "Low entropy = model has committed to a confident prediction at that layer/step",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot8_logit_lens_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 8 saved → {out_path}")
