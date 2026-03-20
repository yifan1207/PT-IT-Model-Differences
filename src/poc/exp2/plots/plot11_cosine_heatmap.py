"""
Plot 11: cos(delta_i, h_{i-1}) heatmap [layer × generation step] per category.

Companion to Plot 10 (which averaged over all steps).  Here every (layer, step)
cell shows the mean cosine similarity across all prompts in that category,
letting you see how layer-wise computation evolves throughout generation.

  X-axis : generation step  (0 → max generated tokens for that category)
  Y-axis : transformer layer (0 at top, 33 at bottom)
  Color  : mean cos(delta_i, h_{i-1}) across all prompts
             > 0  →  layer reinforces the residual stream direction   (warm)
             ≈ 0  →  layer adds orthogonal / genuinely new information (neutral)
             < 0  →  layer suppresses / contradicts the stream         (cool)
  Panels : IC | OOC | R

Layer 0 is always NaN (no h_{i-1} available) and is shown in grey.
The colour scale is shared across all panels and centred at 0 so warm/cool
asymmetry is immediately visible.
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


def _build_cos_heatmaps(results: list[dict]) -> dict[str, np.ndarray]:
    """Build per-category mean cos(delta_i, h_{i-1}) heatmaps.

    Returns dict: category → np.ndarray [n_layers, max_steps], NaN where no
    data exists (including layer 0 which has no h_{i-1}).
    """
    max_steps: dict[str, int] = defaultdict(int)
    for r in results:
        max_steps[r["category"]] = max(max_steps[r["category"]],
                                       len(r["layer_delta_cosine"]))

    n_layers = len(results[0]["layer_delta_cosine"][0]) if results else 34

    sums:   dict[str, np.ndarray] = {}
    counts: dict[str, np.ndarray] = {}
    for cat in CATEGORY_ORDER:
        ms = max_steps.get(cat, 1)
        sums[cat]   = np.zeros((n_layers, ms), dtype=float)
        counts[cat] = np.zeros((n_layers, ms), dtype=float)

    for r in results:
        cat = r["category"]
        for step, layer_vals in enumerate(r["layer_delta_cosine"]):
            for layer, val in enumerate(layer_vals):
                if not np.isnan(val):
                    sums[cat][layer, step]   += val
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

    if not results or "layer_delta_cosine" not in results[0]:
        print("  Plot 11 skipped — no layer_delta_cosine data")
        return

    heatmaps = _build_cos_heatmaps(results)
    if not heatmaps:
        print("  Plot 11 skipped — no data")
        return

    cats = [c for c in CATEGORY_ORDER if c in heatmaps]

    # Shared diverging colour scale centred at 0.
    # Ignore NaN (layer 0) when computing limits.
    all_finite = np.concatenate([
        hm[~np.isnan(hm)].ravel() for hm in heatmaps.values()
    ])
    abs_max = float(np.nanmax(np.abs(all_finite))) if all_finite.size > 0 else 1.0
    # Cap at a reasonable limit so outliers don't wash out the scale.
    abs_max = min(abs_max, 1.0)

    cmap = plt.get_cmap("RdBu_r")     # red = positive (reinforcing), blue = negative (suppressing)
    cmap.set_bad(color="#CCCCCC")      # NaN cells (layer 0) shown in grey

    fig, axes = plt.subplots(1, len(cats), figsize=(7 * len(cats), 8), sharey=True)
    if len(cats) == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        hm = heatmaps[cat]
        n_layers, n_steps = hm.shape
        im = ax.imshow(
            hm,
            aspect="auto",
            origin="upper",
            extent=[-0.5, n_steps - 0.5, n_layers - 0.5, -0.5],
            vmin=-abs_max,
            vmax=abs_max,
            cmap=cmap,
        )
        ax.set_title(CATEGORY_LABEL[cat], fontsize=13, fontweight="bold")
        ax.set_xlabel("Generation step", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Transformer layer", fontsize=10)
        plt.colorbar(
            im, ax=ax,
            label="Mean cos(delta_i, h_{i−1})",
            fraction=0.046, pad=0.04,
        )

    fig.suptitle(
        "Plot 11: cos(delta_i, h_{i−1}) per (layer, generation step) by category\n"
        "Red = layer reinforces stream direction  ·  "
        "Blue = layer suppresses stream  ·  "
        "Grey = layer 0 (no prior residual)\n"
        "Horizontal bands = layers that consistently amplify or suppress the stream; "
        "vertical stripes = generation steps with unusual computation",
        fontsize=9,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot11_cosine_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 11 saved → {out_path}")
