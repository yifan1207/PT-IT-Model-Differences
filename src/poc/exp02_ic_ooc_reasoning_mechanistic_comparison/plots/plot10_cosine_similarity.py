"""
Plot 10: Cosine similarity between each layer's delta and its input residual.

For every forward pass, at every layer i:
    delta_i = h_i - h_{i-1}          (what this layer added to the stream)
    cos_i   = dot(delta_i, h_{i-1}) / (||delta_i|| * ||h_{i-1}||)

  cos_i > 0  →  layer reinforces the existing stream direction
  cos_i ≈ 0  →  layer adds orthogonal / genuinely new information
  cos_i < 0  →  layer suppresses / contradicts the existing stream

We average cos_i across all prompts and all generation steps for each category
and plot it as a function of layer depth.

Panel A: Mean cos_i per layer, one line per category (IC / OOC / R).
Panel B: Same, split by prompt-relative generation band (early / mid / late third).

This directly answers: do IC, OOC, and R prompts cause different layer-wise
computation patterns in the residual stream?
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


CATEGORY_STYLE = {
    "in_context":     {"color": "#2196F3", "label": "IC (in-context)",     "ls": "-"},
    "out_of_context": {"color": "#FF9800", "label": "OOC (out-of-context)", "ls": "--"},
    "reasoning":      {"color": "#4CAF50", "label": "R (reasoning)",        "ls": "-."},
}
CATEGORY_ORDER = ["in_context", "out_of_context", "reasoning"]


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity between two 1-D arrays. Returns float('nan') if either is None."""
    if a is None or b is None:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _mean_cos_profile(results: list[dict], category: str) -> np.ndarray | None:
    """Mean cosine(delta_i, h_{i-1}) per layer for one category.

    Averages over all prompts and all generation steps.
    Returns float array [N_LAYERS], with nan at layer 0 (no h_{i-1} available).
    """
    if not results:
        return None
    n_layers = len(results[0]["layer_delta_cosine"][0]) if results else 34
    sums = np.zeros(n_layers)
    counts = np.zeros(n_layers)

    for r in results:
        if r["category"] != category:
            continue
        for step_cos in r["layer_delta_cosine"]:   # one list per generation step
            for layer, val in enumerate(step_cos):
                if not np.isnan(val):
                    sums[layer] += val
                    counts[layer] += 1

    if counts.sum() == 0:
        return None
    with np.errstate(invalid="ignore"):
        profile = np.where(counts > 0, sums / counts, np.nan)
    return profile


def _split_into_thirds(steps: list) -> list[list]:
    """Split a list of per-step values into prompt-relative early/mid/late thirds."""
    n = len(steps)
    if n == 0:
        return [[], [], []]
    bands: list[list] = [[], [], []]
    for idx, val in enumerate(steps):
        band = min((idx * 3) // n, 2)
        bands[band].append(val)
    return bands


def _mean_cos_profile_band(results: list[dict], category: str, band_idx: int) -> np.ndarray | None:
    """Mean cosine profile for one category restricted to one generation band."""
    if not results:
        return None
    n_layers = len(results[0]["layer_delta_cosine"][0]) if results else 34
    sums = np.zeros(n_layers)
    counts = np.zeros(n_layers)

    for r in results:
        if r["category"] != category:
            continue
        band_steps = _split_into_thirds(r["layer_delta_cosine"])[band_idx]
        for step_cos in band_steps:
            for layer, val in enumerate(step_cos):
                if not np.isnan(val):
                    sums[layer] += val
                    counts[layer] += 1

    if counts.sum() == 0:
        return None
    with np.errstate(invalid="ignore"):
        profile = np.where(counts > 0, sums / counts, np.nan)
    return profile


def _plot_profiles(ax, profiles: dict, title: str, layers: np.ndarray) -> None:
    for cat in CATEGORY_ORDER:
        profile = profiles.get(cat)
        if profile is None:
            continue
        style = CATEGORY_STYLE[cat]
        valid = ~np.isnan(profile)
        ax.plot(layers[valid], profile[valid],
                color=style["color"], ls=style["ls"], lw=2, label=style["label"])
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Mean cos(delta_i, h_{i-1})", fontsize=10)
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8)


def _plot_deviation_profiles(
    ax,
    band_profiles: dict,
    overall_profiles: dict,
    title: str,
    layers: np.ndarray,
) -> None:
    """Plot band_profile − overall_profile per category (deviation from overall mean).

    This reveals temporal variation that is invisible on an absolute scale because
    cos(delta_i, h_{i-1}) is dominated by the network's structural layer pattern.
    """
    any_plotted = False
    for cat in CATEGORY_ORDER:
        band = band_profiles.get(cat)
        overall = overall_profiles.get(cat)
        if band is None or overall is None:
            continue
        style = CATEGORY_STYLE[cat]
        delta = band - overall
        valid = ~np.isnan(delta)
        ax.plot(layers[valid], delta[valid],
                color=style["color"], ls=style["ls"], lw=2, label=style["label"])
        any_plotted = True
    ax.axhline(0, color="black", lw=0.8, ls=":", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("Δ cos (band − overall)", fontsize=10)
    ax.set_title(title, fontsize=9)
    if any_plotted:
        ax.legend(fontsize=8)


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not results or "layer_delta_cosine" not in results[0]:
        print("  Plot 10 skipped — no layer_delta_cosine data")
        return

    n_layers = len(results[0]["layer_delta_cosine"][0])
    layers = np.arange(n_layers)

    # sharey=False: Panel A shows absolute values; panels B/C/D show deviations
    # (a different axis range), so they must not share Y.
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)

    # Panel A: overall absolute profile
    profiles_all = {cat: _mean_cos_profile(results, cat) for cat in CATEGORY_ORDER}
    _plot_profiles(axes[0], profiles_all, "All steps (absolute)", layers)

    # Panels B/C/D: deviation of each generation band from overall mean.
    # cos(delta_i, h_{i-1}) reflects structural layer behaviour (mostly fixed by
    # weights), so the absolute curves for bands are nearly identical to the overall.
    # Plotting the deviation makes temporal variation visible.
    band_labels = [
        "Early third\n(Δ from overall)",
        "Mid third\n(Δ from overall)",
        "Late third\n(Δ from overall)",
    ]
    for b_idx, (ax, label) in enumerate(zip(axes[1:], band_labels)):
        profiles_band = {
            cat: _mean_cos_profile_band(results, cat, b_idx)
            for cat in CATEGORY_ORDER
        }
        _plot_deviation_profiles(ax, profiles_band, profiles_all, label, layers)

    fig.suptitle(
        "Plot 10: cos(delta_i, h_{i−1}) per layer — does the layer reinforce or replace the residual stream?\n"
        "Panel A: absolute cos profile. Panels B/C/D: deviation from overall mean per generation third.\n"
        "Non-zero deviations in B/C/D reveal whether layer computation changes as generation progresses.",
        fontsize=9,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot10_cosine_similarity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 10 saved → {out_path}")
