"""
Plot 2: L0 heatmap [layer × generation step], one panel per category.

Shows where (which layer) and when (which generation step) feature sparsity
changes. A denser pattern (higher L0) at certain layers or steps would suggest
those layers are doing more "work" at that point in generation.

  X-axis : generation step (0 → max_gen_tokens)
  Y-axis : transformer layer (0–33, top=0)
  Color  : mean L0 across all prompts in that category at that (layer, step)

Three side-by-side panels: IC | OOC | R
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


def _build_l0_heatmaps(results: list[dict]) -> dict[str, np.ndarray]:
    """For each category, build a 2-D array [layer, step] of mean L0.

    Different prompts may have different numbers of generated steps.
    We find the max step count per category and zero-pad shorter sequences.
    Returns dict: category → np.ndarray [n_layers, max_steps]
    """
    # Accumulate (sum, count) arrays
    from collections import defaultdict
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}

    # First pass: determine max steps per category
    max_steps: dict[str, int] = defaultdict(int)
    for r in results:
        cat = r["category"]
        max_steps[cat] = max(max_steps[cat], len(r["l0"]))

    n_layers = len(results[0]["l0"][0]) if results else 34

    for cat in CATEGORY_ORDER:
        ms = max_steps.get(cat, 1)
        sums[cat] = np.zeros((n_layers, ms), dtype=float)
        counts[cat] = np.zeros((n_layers, ms), dtype=float)

    for r in results:
        cat = r["category"]
        l0_steps = r["l0"]  # list[step][layer]
        for step, layer_vals in enumerate(l0_steps):
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

    heatmaps = _build_l0_heatmaps(results)
    if not heatmaps:
        print("  Plot 2 skipped — no data")
        return

    cats = [c for c in CATEGORY_ORDER if c in heatmaps]
    n_panels = len(cats)
    # Cap at 99th percentile so sparse high-activity spikes don't wash out the colour scale.
    all_vals = np.concatenate([hm[~np.isnan(hm)].ravel() for hm in heatmaps.values()])
    vmax = float(np.percentile(all_vals, 99)) if all_vals.size > 0 else 200.0

    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        hm = heatmaps[cat]
        n_layers, n_steps = hm.shape
        im = ax.imshow(
            hm, aspect="auto", origin="upper",
            extent=[-0.5, n_steps - 0.5, n_layers - 0.5, -0.5],
            vmin=0, vmax=vmax, cmap="YlOrRd",
        )
        ax.set_title(CATEGORY_LABEL[cat], fontsize=13, fontweight="bold")
        ax.set_xlabel("Generation step", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Transformer layer", fontsize=10)
        plt.colorbar(im, ax=ax, label="Mean L0", fraction=0.046, pad=0.04)

    fig.suptitle(
        "Plot 2: Mean L0 per (layer, generation step) by category\n"
        "Color = mean active feature count (L0) across all prompts in that category",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot2_l0_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 2 saved → {out_path}")
