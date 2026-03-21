"""
Plot 1 (Exp4): Adjacent-layer Jaccard similarity curve across the dip (E0b).

Tests P1: Feature Population Shift.

Panel A — Jaccard(L, L+1) for L ∈ {8,9,10,11,12,13}
  Two lines: PT (blue) and IT (orange).
  Error bars: ±1 SEM across prompts.
  Dip marker at layer 11 (grey shaded column).

Panel B — Feature death and birth counts per layer transition
  Stacked bar or twin-axis: deaths (features active at L but not L+1) in red,
  births (features active at L+1 but not L) in green.
  Comparison PT vs IT side by side.

Key reading:
  - If IT shows a sharper Jaccard valley at layers 10→11 vs PT, P1 is confirmed.
  - If Jaccard(10,12) is lower than Jaccard(10,11) or Jaccard(11,12) in IT, the
    dip is NOT just suppression — features actually turn over.

Compatible with:
  - exp4_features.npz from exp4 collection
  - exp3 .npz from exp3 collection (using load_features_exp3 loader)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.poc.exp4.analysis.jaccard import (
    load_features_exp3,
    load_features_exp4,
    compute_jaccard_stats,
    dip_summary,
)


_DIP_LAYER     = 11
_ANALYSIS_START = 8
_ANALYSIS_END   = 15  # exclusive → layers 8–14


def make_plot(
    pt_features_path: str,
    it_features_path: str,
    output_dir: str,
    dip_layer:   int = _DIP_LAYER,
    source:      str = "exp4",   # "exp3" or "exp4"
) -> None:
    """Generate the Jaccard curve comparison plot.

    Parameters
    ----------
    pt_features_path : path to PT active_features .npz
    it_features_path : path to IT active_features .npz
    output_dir       : directory for output PNG
    dip_layer        : gate layer index (default 11)
    source           : "exp3" or "exp4" (chooses the correct loader)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "plot1_jaccard_curve.png"

    # Load features
    load_fn = load_features_exp3 if source == "exp3" else load_features_exp4

    for path, label in [(pt_features_path, "PT"), (it_features_path, "IT")]:
        if not Path(path).exists():
            print(f"  Plot 1 (Exp4) skipped — {label} features not found: {path}")
            return

    pt_feats = load_fn(pt_features_path)
    it_feats = load_fn(it_features_path)

    pt_stats = compute_jaccard_stats(pt_feats,
                                     analysis_start=_ANALYSIS_START,
                                     analysis_end=_ANALYSIS_END,
                                     dip_layer=dip_layer)
    it_stats = compute_jaccard_stats(it_feats,
                                     analysis_start=_ANALYSIS_START,
                                     analysis_end=_ANALYSIS_END,
                                     dip_layer=dip_layer)

    pt_sum = dip_summary(pt_stats, dip_layer=dip_layer)
    it_sum = dip_summary(it_stats, dip_layer=dip_layer)

    # ── extract adjacent pair values ───────────────────────────────────────────
    adj_pairs = pt_stats["adjacent_pairs"]  # e.g. [(8,9),(9,10),(10,11),...,(13,14)]
    cross_pair = pt_stats["cross_dip_pair"]  # (10, 12)

    layers_for_plot = [p[0] for p in adj_pairs]  # [8,9,10,11,12,13]

    def _extract_adj(stats) -> tuple[list, list]:
        """Return (mean, sem) for each adjacent pair."""
        mean_ = [stats["mean_jaccard"].get(p, float("nan")) for p in adj_pairs]
        sem_  = [stats["sem_jaccard"].get(p,  float("nan")) for p in adj_pairs]
        return mean_, sem_

    pt_mean, pt_sem = _extract_adj(pt_stats)
    it_mean, it_sem = _extract_adj(it_stats)

    # cross-dip values as separate points
    pt_cross = pt_stats["mean_jaccard"].get(cross_pair, float("nan"))
    it_cross = it_stats["mean_jaccard"].get(cross_pair, float("nan"))
    pt_cross_sem = pt_stats["sem_jaccard"].get(cross_pair, float("nan"))
    it_cross_sem = it_stats["sem_jaccard"].get(cross_pair, float("nan"))

    # ── build figure ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Exp4 — Adjacent-Layer Jaccard: Feature Population Shift at the Dip (E0b)",
        fontsize=12, fontweight="bold"
    )

    # ── Panel A: Jaccard curve ─────────────────────────────────────────────────
    ax = axes[0]
    x   = np.array(layers_for_plot)
    xf  = x.astype(float)

    # Dip shading
    ax.axvspan(dip_layer - 0.4, dip_layer + 0.4, alpha=0.15, color="grey",
               label=f"Dip layer ({dip_layer})")

    # PT and IT Jaccard curves
    ax.errorbar(xf, pt_mean, yerr=pt_sem, fmt="o-", color="steelblue",
                linewidth=2, capsize=3, label="PT")
    ax.errorbar(xf, it_mean, yerr=it_sem, fmt="s-", color="darkorange",
                linewidth=2, capsize=3, label="IT")

    # Cross-dip marker (annotated separately with offset x-position)
    cross_x = (cross_pair[0] + cross_pair[1]) / 2.0  # midpoint between 10 and 12
    ax.errorbar([cross_x], [pt_cross], yerr=[pt_cross_sem],
                fmt="^", color="steelblue", markersize=9, capsize=3,
                alpha=0.7, label=f"PT J({cross_pair[0]},{cross_pair[1]}) cross-dip")
    ax.errorbar([cross_x], [it_cross], yerr=[it_cross_sem],
                fmt="v", color="darkorange", markersize=9, capsize=3,
                alpha=0.7, label=f"IT J({cross_pair[0]},{cross_pair[1]}) cross-dip")

    ax.set_xlabel("Layer L  (Jaccard for pair L, L+1)")
    ax.set_ylabel("Jaccard(L, L+1)")
    ax.set_title("Jaccard similarity between adjacent layers")
    ax.set_xticks(layers_for_plot)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel B: Feature death / birth counts ─────────────────────────────────
    ax2 = axes[1]
    width = 0.35
    x2   = np.arange(len(adj_pairs))

    pt_death = [pt_stats["feature_death"].get(p[0], float("nan")) for p in adj_pairs]
    it_death = [it_stats["feature_death"].get(p[0], float("nan")) for p in adj_pairs]
    pt_birth = [pt_stats["feature_birth"].get(p[0], float("nan")) for p in adj_pairs]
    it_birth = [it_stats["feature_birth"].get(p[0], float("nan")) for p in adj_pairs]

    pair_labels = [f"{p[0]}→{p[1]}" for p in adj_pairs]

    ax2.bar(x2 - width / 2, pt_death, width, label="PT deaths", color="steelblue",
            alpha=0.7)
    ax2.bar(x2 + width / 2, it_death, width, label="IT deaths", color="darkorange",
            alpha=0.7)
    ax2.bar(x2 - width / 2, [-b for b in pt_birth], width, label="PT births",
            color="steelblue", alpha=0.4, bottom=0,
            hatch="//")
    ax2.bar(x2 + width / 2, [-b for b in it_birth], width, label="IT births",
            color="darkorange", alpha=0.4, bottom=0,
            hatch="//")

    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvspan(
        x2[list(l for l, p in enumerate(adj_pairs) if p[0] == dip_layer - 1)][0] - 0.4
        if any(p[0] == dip_layer - 1 for p in adj_pairs) else -1,
        x2[list(l for l, p in enumerate(adj_pairs) if p[0] == dip_layer - 1)][0] + 0.4
        if any(p[0] == dip_layer - 1 for p in adj_pairs) else -1,
        alpha=0.15, color="grey"
    )
    ax2.set_xlabel("Layer transition")
    ax2.set_ylabel("Mean feature count per prompt")
    ax2.set_title("Feature deaths (↑) and births (↓, hatched) per transition")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(pair_labels, rotation=45)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # Annotation box
    note_lines = [
        f"PT  J(dip-1,dip+1)={pt_cross:.3f}",
        f"IT  J(dip-1,dip+1)={it_cross:.3f}",
        f"PT  deaths@dip={pt_sum['death_at_dip']:.1f}",
        f"IT  deaths@dip={it_sum['death_at_dip']:.1f}",
    ]
    ax2.text(0.97, 0.97, "\n".join(note_lines), transform=ax2.transAxes,
             ha="right", va="top", fontsize=7.5,
             bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
