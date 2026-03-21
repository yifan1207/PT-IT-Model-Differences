"""
Plot 4 (Exp4): Feature label category distribution before/after dip (E1a).

Tests P5: Feature Category Transition.

Two panels:
  A — Stacked bar chart: category fractions for three feature populations
      X-axis: pre-dip exclusive | surviving | post-dip exclusive
      Stacked bars: LEXICAL (blue), SEMANTIC (green), FORMAT (orange),
                    AMBIGUOUS (grey), UNLABELLED (light grey)
      Shown for PT and IT side by side within each population group.

  B — Population sizes and overlap
      Bar chart: number of features in each population for PT vs IT.
      Inset: Venn-style overlap percentages.

Interpretation:
  - If pre-dip bar has high LEXICAL fraction (>60%) and post-dip has high
    SEMANTIC fraction (>60%), prediction P5 is confirmed.
  - If IT's post-dip bar uniquely contains a high FORMAT fraction absent from
    PT's post-dip, it confirms IT-specific instruction/format features emerge
    after the phase transition.

Data source: output of feature_labels.summarise_population_categories().
Requires Neuronpedia API calls (run fetch_all_labels first).

If label data is not available, the plot prints a message and exits gracefully.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.poc.exp4.analysis.feature_labels import (
    extract_feature_populations,
    summarise_population_categories,
)


_CATEGORIES = ["LEXICAL", "SEMANTIC", "FORMAT", "AMBIGUOUS", "UNLABELLED"]
_COLORS = {
    "LEXICAL":    "#4878cf",
    "SEMANTIC":   "#6acc65",
    "FORMAT":     "#d65f5f",
    "AMBIGUOUS":  "#b8b8b8",
    "UNLABELLED": "#e8e8e8",
}


def make_plot(
    pt_features_path: str,
    it_features_path: str,
    pt_labels_path:   Optional[str],   # JSON: {pop → {feat_idx → label}}
    it_labels_path:   Optional[str],
    output_dir: str,
    dip_layer:  int = 11,
) -> None:
    """Generate the feature label distribution plot.

    Parameters
    ----------
    pt_features_path : path to PT exp4_features.npz
    it_features_path : path to IT exp4_features.npz
    pt_labels_path   : path to JSON with PT label data (from fetch_population_labels)
                       or None → plot population sizes only (no category breakdown)
    it_labels_path   : path to JSON with IT label data or None
    output_dir       : directory for output PNG
    dip_layer        : gate layer
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "plot4_feature_label_distribution.png"

    pre_layer  = dip_layer - 1
    post_layer = dip_layer + 1

    # Check features exist
    for path, label in [(pt_features_path, "PT"), (it_features_path, "IT")]:
        if not Path(path).exists():
            print(f"  Plot 4 (Exp4) skipped — {label} features not found: {path}")
            return

    # Extract populations
    pt_pops = extract_feature_populations(
        pt_features_path, pre_dip_layer=pre_layer, post_dip_layer=post_layer
    )
    it_pops = extract_feature_populations(
        it_features_path, pre_dip_layer=pre_layer, post_dip_layer=post_layer
    )

    # Load labels if available
    def _load_labels(labels_path: Optional[str]) -> Optional[dict]:
        if labels_path is None or not Path(labels_path).exists():
            return None
        with open(labels_path) as f:
            return json.load(f)

    pt_label_data = _load_labels(pt_labels_path)
    it_label_data = _load_labels(it_labels_path)

    has_labels = (pt_label_data is not None) and (it_label_data is not None)

    # Compute category distributions if labels available
    if has_labels:
        pt_cat = summarise_population_categories(pt_label_data, pt_pops)
        it_cat = summarise_population_categories(it_label_data, it_pops)
    else:
        pt_cat = it_cat = None

    # ── figure ────────────────────────────────────────────────────────────────
    if has_labels:
        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax_b = plt.subplots(1, 1, figsize=(8, 6))
        ax_a = None

    fig.suptitle(
        f"Exp4 — Feature Category Distribution Before/After Dip (E1a)\n"
        f"Pre-dip: L{pre_layer}  |  Post-dip: L{post_layer}  |  Dip: L{dip_layer}",
        fontsize=12, fontweight="bold",
    )

    # ── Panel A: stacked bar chart ─────────────────────────────────────────────
    if ax_a is not None and has_labels:
        populations   = ["pre_dip", "surviving", "post_dip"]
        pop_labels    = [f"Pre-dip\n(L{pre_layer} only)", "Surviving\n(both)", f"Post-dip\n(L{post_layer} only)"]
        n_pops        = len(populations)
        group_width   = 0.7
        bar_width     = group_width / 2.1
        x             = np.arange(n_pops)

        def _fracs(cat_summary: dict, pop: str) -> list[float]:
            d = cat_summary[pop]
            return [d.get(c, 0.0) for c in _CATEGORIES]

        for model_i, (label, cat_s) in enumerate([("PT", pt_cat), ("IT", it_cat)]):
            offset = (model_i - 0.5) * bar_width * 1.1
            bottoms = np.zeros(n_pops)
            for cat_j, cat in enumerate(_CATEGORIES):
                fracs = [cat_s[pop].get(cat, 0.0) if cat_s[pop].get(cat) is not None else 0.0
                         for pop in populations]
                bars = ax_a.bar(
                    x + offset, fracs, bar_width,
                    bottom=bottoms,
                    color=_COLORS[cat],
                    alpha=0.85 if label == "PT" else 0.6,
                    edgecolor="white" if label == "PT" else "grey",
                    linewidth=0.5,
                    label=f"{cat} ({label})" if model_i == 0 else None,
                )
                bottoms += np.array(fracs)

        # Legend with one entry per category
        handles = [plt.Rectangle((0, 0), 1, 1, color=_COLORS[c]) for c in _CATEGORIES]
        ax_a.legend(handles, _CATEGORIES, loc="upper right", fontsize=8,
                    title="Category")

        # x-axis
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(pop_labels, fontsize=9)
        ax_a.set_ylabel("Fraction of features")
        ax_a.set_title("Panel A — Category distribution per population\n"
                       "(solid=PT, semi-transparent=IT)")
        ax_a.set_ylim(0, 1.05)
        ax_a.set_yticks(np.arange(0, 1.1, 0.2))
        ax_a.axhline(0.6, color="black", linestyle="--", linewidth=0.7, alpha=0.4,
                     label="60% threshold (P5 criterion)")
        ax_a.grid(axis="y", alpha=0.3)

        # n_labelled annotation
        for model_i, (label, cat_s) in enumerate([("PT", pt_cat), ("IT", it_cat)]):
            offset = (model_i - 0.5) * bar_width * 1.1
            for pop_i, pop in enumerate(populations):
                n_lab = cat_s.get("n_labelled", {}).get(pop, 0)
                n_tot = cat_s.get("n_total",    {}).get(pop, 0)
                ax_a.text(pop_i + offset, 1.01, f"{label}\n{n_lab}/{n_tot}",
                          ha="center", va="bottom", fontsize=6)
    elif ax_a is not None:
        ax_a.text(0.5, 0.5, "No Neuronpedia labels available.\nRun fetch_population_labels first.",
                  transform=ax_a.transAxes, ha="center", va="center",
                  fontsize=11, color="grey")
        ax_a.set_title("Panel A — Feature categories (labels not available)")

    # ── Panel B: population sizes ──────────────────────────────────────────────
    pop_names = ["pre_dip_exclusive", "post_dip_exclusive", "surviving"]
    pop_display = ["Pre-dip\nexclusive", "Post-dip\nexclusive", "Surviving"]
    pt_sizes = [len(pt_pops[p]) for p in pop_names]
    it_sizes = [len(it_pops[p]) for p in pop_names]

    x_b     = np.arange(len(pop_names))
    width_b = 0.35

    ax_b.bar(x_b - width_b / 2, pt_sizes, width_b, label="PT", color="steelblue",  alpha=0.8)
    ax_b.bar(x_b + width_b / 2, it_sizes, width_b, label="IT", color="darkorange", alpha=0.8)

    # Value labels on bars
    for xi, (pt_v, it_v) in enumerate(zip(pt_sizes, it_sizes)):
        ax_b.text(xi - width_b / 2, pt_v + 0.5, str(pt_v), ha="center", va="bottom", fontsize=8)
        ax_b.text(xi + width_b / 2, it_v + 0.5, str(it_v), ha="center", va="bottom", fontsize=8)

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(pop_display, fontsize=10)
    ax_b.set_ylabel("Number of features (capped at 200)")
    ax_b.set_title("Panel B — Feature population sizes\n"
                   f"(frequency ≥5% of prompts; pre-dip=L{pre_layer}, post-dip=L{post_layer})")
    ax_b.legend()
    ax_b.grid(axis="y", alpha=0.3)

    # Annotation: n_prompts
    pt_n = pt_pops["n_prompts"]
    it_n = it_pops["n_prompts"]
    ax_b.text(0.98, 0.98, f"PT: {pt_n} prompts\nIT: {it_n} prompts",
              transform=ax_b.transAxes, ha="right", va="top", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
