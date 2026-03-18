"""
Plot 10: Cosine similarity between category mean logit-lens entropy profiles.

A synthetic measure of "representational similarity" between categories:
computes the cosine similarity between the mean logit-lens entropy vectors
(one value per layer) for each pair of categories.

This is a scalar summary of how similar the layer-wise entropy profile is
between IC, OOC, and R. Also shows a breakdown across generation steps —
the first, middle, and last third of generated tokens — to capture whether
similarity changes as generation progresses.

Panel A: Bar chart of cosine similarity for each category pair.
Panel B: Cosine similarity vs generation step thirds (early/mid/late).

Interpretation:
  - High similarity IC–R → reasoning uses similar layer activations to retrieval.
  - Low similarity → mechanistically distinct computation modes.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


CATEGORY_ORDER = ["in_context", "out_of_context", "reasoning"]
CATEGORY_LABEL = {
    "in_context":     "IC",
    "out_of_context": "OOC",
    "reasoning":      "R",
}
PAIR_STYLE = {
    ("in_context", "out_of_context"): {"color": "#9C27B0", "label": "IC vs OOC"},
    ("in_context", "reasoning"):      {"color": "#F44336", "label": "IC vs R"},
    ("out_of_context", "reasoning"):  {"color": "#795548", "label": "OOC vs R"},
}


def _cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float("nan")


def _mean_ll_profile(results: list[dict], category: str) -> np.ndarray | None:
    """Mean logit-lens entropy vector over layers for one category."""
    rows = []
    for r in results:
        if r["category"] != category:
            continue
        for layer_vals in r["logit_lens_entropy"]:
            rows.append(layer_vals)
    if not rows:
        return None
    return np.array(rows, dtype=float).mean(axis=0)


def _split_prompt_into_thirds(steps: list[list[float]]) -> list[list[list[float]]]:
    """Split one prompt's steps into prompt-relative early/mid/late thirds."""
    n_steps = len(steps)
    if n_steps == 0:
        return [[], [], []]

    bands = [[], [], []]
    for step_idx, layer_vals in enumerate(steps):
        band_idx = min((step_idx * 3) // n_steps, 2)
        bands[band_idx].append(layer_vals)
    return bands


def _mean_ll_profile_for_band(results: list[dict], category: str, band_idx: int) -> np.ndarray | None:
    rows = []
    for r in results:
        if r["category"] != category:
            continue
        for layer_vals in _split_prompt_into_thirds(r["logit_lens_entropy"])[band_idx]:
            rows.append(layer_vals)
    if not rows:
        return None
    return np.array(rows, dtype=float).mean(axis=0)


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not results:
        print("  Plot 10 skipped — no data")
        return

    # ── Panel A: overall cosine similarity ──────────────────────────────────
    profiles = {cat: _mean_ll_profile(results, cat) for cat in CATEGORY_ORDER}

    pairs = list(PAIR_STYLE.keys())
    pair_sims = {pair: _cosine(profiles[pair[0]], profiles[pair[1]]) for pair in pairs}

    # ── Panel B: cosine similarity by prompt-relative generation thirds ─────
    band_labels = ["Early steps", "Mid steps", "Late steps"]
    band_sims: dict[tuple, list[float]] = {pair: [] for pair in pairs}
    for b_idx in range(3):
        for pair in pairs:
            cat_a, cat_b = pair
            p_a = _mean_ll_profile_for_band(results, cat_a, b_idx)
            p_b = _mean_ll_profile_for_band(results, cat_b, b_idx)
            band_sims[pair].append(_cosine(p_a, p_b))

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, (ax_bar, ax_band) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel A bar chart
    bar_x = np.arange(len(pairs))
    bar_vals = [pair_sims[p] for p in pairs]
    bar_colors = [PAIR_STYLE[p]["color"] for p in pairs]
    bar_labels = [PAIR_STYLE[p]["label"] for p in pairs]
    bars = ax_bar.bar(bar_x, bar_vals, color=bar_colors, width=0.5)
    ax_bar.set_xticks(bar_x)
    ax_bar.set_xticklabels(bar_labels, fontsize=10)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Cosine similarity", fontsize=11)
    ax_bar.set_title("Panel A: Overall profile similarity\n(cosine of mean logit-lens entropy vectors)", fontsize=9)
    for bar, val in zip(bars, bar_vals):
        if np.isnan(val):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, 0.02,
                        "n/a", ha="center", va="bottom", fontsize=9)
        else:
            ax_bar.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Panel B grouped bars
    bw = 0.25
    for i, pair in enumerate(pairs):
        xs = np.arange(3) + i * bw
        ax_band.bar(xs, band_sims[pair], width=bw,
                    color=PAIR_STYLE[pair]["color"], label=PAIR_STYLE[pair]["label"], alpha=0.85)
    ax_band.set_xticks(np.arange(3) + bw)
    ax_band.set_xticklabels(band_labels, fontsize=10)
    ax_band.set_ylim(0, 1)
    ax_band.set_ylabel("Cosine similarity", fontsize=11)
    ax_band.set_title("Panel B: Similarity by prompt-relative generation band\n(early / mid / late thirds)", fontsize=9)
    ax_band.legend(fontsize=9)

    fig.suptitle(
        "Plot 10: Representational similarity between category pairs\n"
        "Cosine similarity of mean logit-lens entropy profiles (per layer)",
        fontsize=10,
    )
    ax_band.text(
        0.01, 0.02,
        "n/a = insufficient data for that category pair/band",
        transform=ax_band.transAxes,
        fontsize=8,
        alpha=0.7,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot10_cosine_similarity.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 10 saved → {out_path}")
