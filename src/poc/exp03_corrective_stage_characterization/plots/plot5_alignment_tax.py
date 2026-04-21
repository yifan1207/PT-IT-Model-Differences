"""
Plot 5 (Exp3): Alignment tax quantification (Experiment 2b).

Visualises the fraction of total MLP computation that falls in the corrective
stage (layers >= boundary) vs the proposal stage.

Works on existing exp2 or exp3 results — no new inference needed.
Uses analysis/alignment_tax.py for the core computation.

Four panels:
  A: Per-layer mean delta norm as fraction of total (line plot by category)
     with SEM bands. IT solid / PT dashed overlay.
  B: Box plot of per-prompt alignment tax (corrective fraction) by category,
     IT and PT side by side.
  C: Mean alignment tax (bar chart, IT vs PT per category) with SEM error bars.
  D: Per-prompt tax scatter: IT vs PT paired by category.
     Reveals whether IT consistently taxes more than PT on the same categories.

REQUIRES: layer_delta_norm (available in exp2 results).
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.exp03_corrective_stage_characterization.analysis.alignment_tax import (
    compute_tax_by_category,
)
from src.poc.shared.constants import N_LAYERS

_BOUNDARY = 20

from src.poc.shared.plot_colors import SPLIT_COLORS as CATEGORY_COLORS, _DEFAULT_COLOR


def _per_layer_frac_with_sem(results: list[dict],
                              n_layers: int = N_LAYERS,
                              ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean ± SEM of delta-norm fraction of total, by category.

    For each prompt×step, computes layer_i_norm / sum_j(layer_j_norm),
    then aggregates per layer per category.
    """
    # sums and sum-of-squares over observations (prompt×step)
    sums   = defaultdict(lambda: np.zeros(n_layers))
    sqsums = defaultdict(lambda: np.zeros(n_layers))
    counts = defaultdict(lambda: np.zeros(n_layers))

    for r in results:
        cat = r.get("category", "unknown")
        for step_norms in r.get("layer_delta_norm", []):
            total = sum(v for v in step_norms if v is not None and not math.isnan(v))
            if total == 0:
                continue
            for layer, v in enumerate(step_norms[:n_layers]):
                if v is not None and not math.isnan(v):
                    frac = v / total
                    sums[cat][layer]   += frac
                    sqsums[cat][layer] += frac * frac
                    counts[cat][layer] += 1

    out = {}
    for cat in sums:
        n  = counts[cat]
        s  = sums[cat]
        sq = sqsums[cat]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / n, np.nan)
            var  = np.where(n > 1, sq / n - (s / n) ** 2, 0.0)
            sem  = np.where(n > 1, np.sqrt(np.maximum(var, 0.0) / n), 0.0)
        out[cat] = (mean, sem)
    return out


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or "layer_delta_norm" not in results[0]:
        print("  Plot 5 (Exp3) skipped — no layer_delta_norm data")
        return

    layers = np.arange(N_LAYERS)

    it_fracs_by_cat = _per_layer_frac_with_sem(results)
    it_tax          = compute_tax_by_category(results)

    pt_fracs_by_cat = _per_layer_frac_with_sem(pt_results) if pt_results else None
    pt_tax          = compute_tax_by_category(pt_results)  if pt_results else None

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: per-layer fraction with SEM bands ────────────────────────────
    ax_a.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
    ax_a.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4,
                 label=f"boundary (L{_BOUNDARY})")

    for cat, (m, s) in it_fracs_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax_a.plot(layers, m, lw=1.8, color=c, label=f"IT {cat}", alpha=0.85)
        ax_a.fill_between(layers,
                          np.maximum(0, m - s), m + s,
                          alpha=0.12, color=c)

    if pt_fracs_by_cat:
        for cat, (m, s) in pt_fracs_by_cat.items():
            c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
            ax_a.plot(layers, m, lw=1.4, color=c, ls="--", alpha=0.5)

    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean δ norm as fraction of total  (±SEM)")
    ax_a.set_title("Panel A — Per-layer computation fraction  IT solid / PT dashed\n"
                   "If IT shifts weight right of L20 → corrective stage is heavier")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: box plots of per-prompt tax IT vs PT ─────────────────────────
    all_cats_it  = sorted(it_tax.keys())
    all_cats_pt  = sorted(pt_tax.keys()) if pt_tax else []
    all_cats     = sorted(set(all_cats_it) | set(all_cats_pt))
    n_cats       = len(all_cats)
    box_offset   = 0.22
    positions_it = np.arange(n_cats) - box_offset
    positions_pt = np.arange(n_cats) + box_offset

    bp_it = ax_b.boxplot(
        [it_tax.get(c, {}).get("per_prompt_taxes", []) for c in all_cats],
        positions=positions_it,
        widths=0.38, patch_artist=True,
        medianprops={"color": "white", "lw": 2},
    )
    for patch in bp_it["boxes"]:
        patch.set_facecolor("#E65100")
        patch.set_alpha(0.8)

    if pt_tax:
        bp_pt = ax_b.boxplot(
            [pt_tax.get(c, {}).get("per_prompt_taxes", []) for c in all_cats],
            positions=positions_pt,
            widths=0.38, patch_artist=True,
            medianprops={"color": "black", "lw": 2},
        )
        for patch in bp_pt["boxes"]:
            patch.set_facecolor("#BBDEFB")
            patch.set_alpha(0.8)

    ax_b.set_xticks(np.arange(n_cats))
    ax_b.set_xticklabels(all_cats, rotation=20, ha="right", fontsize=8)
    ax_b.set_ylabel("Corrective fraction (alignment tax)")
    ax_b.set_ylim(0, 1)
    ax_b.set_title("Panel B — Alignment tax distribution  IT (orange) vs PT (blue)\n"
                   "Higher = more computation in corrective layers")
    ax_b.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    leg = [Patch(facecolor="#E65100", label="IT")]
    if pt_tax:
        leg.append(Patch(facecolor="#BBDEFB", label="PT"))
    ax_b.legend(handles=leg, fontsize=9)

    # ── Panel C: mean tax IT vs PT per category with SEM bars ─────────────────
    x_c   = np.arange(n_cats)
    bar_w = 0.35

    it_means = [it_tax.get(c, {}).get("mean_tax", float("nan")) for c in all_cats]
    it_stds  = [it_tax.get(c, {}).get("std_tax",  0.0)         for c in all_cats]
    it_ns    = [it_tax.get(c, {}).get("n_prompts", 1)           for c in all_cats]
    it_sems  = [s / math.sqrt(max(n, 1)) for s, n in zip(it_stds, it_ns)]

    ax_c.bar(x_c - bar_w / 2, it_means, bar_w,
             color="#E65100", alpha=0.85, label="IT",
             yerr=it_sems, capsize=4, error_kw={"elinewidth": 1.2})

    if pt_tax:
        pt_means = [pt_tax.get(c, {}).get("mean_tax", float("nan")) for c in all_cats]
        pt_stds  = [pt_tax.get(c, {}).get("std_tax",  0.0)         for c in all_cats]
        pt_ns    = [pt_tax.get(c, {}).get("n_prompts", 1)           for c in all_cats]
        pt_sems  = [s / math.sqrt(max(n, 1)) for s, n in zip(pt_stds, pt_ns)]
        ax_c.bar(x_c + bar_w / 2, pt_means, bar_w,
                 color="#1565C0", alpha=0.85, label="PT",
                 yerr=pt_sems, capsize=4, error_kw={"elinewidth": 1.2})

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(all_cats, rotation=20, ha="right", fontsize=9)
    ax_c.set_ylabel("Mean alignment tax  (±SEM)")
    ax_c.set_ylim(0, 1)
    ax_c.set_title("Panel C — Mean alignment tax by category  IT vs PT\n"
                   "Gap = how much more IT concentrates work in late layers")
    ax_c.legend(fontsize=9)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: scatter IT tax vs PT tax per category ────────────────────────
    if pt_tax:
        for cat in all_cats:
            it_v = it_tax.get(cat, {}).get("mean_tax", float("nan"))
            pt_v = pt_tax.get(cat, {}).get("mean_tax", float("nan"))
            if math.isnan(it_v) or math.isnan(pt_v):
                continue
            c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
            ax_d.scatter(pt_v, it_v, s=120, color=c, zorder=5, label=cat)
            ax_d.annotate(cat, (pt_v, it_v),
                          textcoords="offset points", xytext=(5, 3), fontsize=8)

        lo = min(ax_d.get_xlim()[0], ax_d.get_ylim()[0]) - 0.02
        hi = max(ax_d.get_xlim()[1], ax_d.get_ylim()[1]) + 0.02
        ax_d.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="--", alpha=0.6)
        ax_d.set_xlabel("PT mean alignment tax")
        ax_d.set_ylabel("IT mean alignment tax")
        ax_d.set_title("Panel D — IT vs PT alignment tax per category\n"
                       "Points above the diagonal = IT taxes more than PT")
        ax_d.grid(alpha=0.2)
    else:
        # IT only: show per-category tax as horizontal bars
        ax_d.barh(np.arange(n_cats), it_means, xerr=it_sems, capsize=4,
                  color="#E65100", alpha=0.8)
        ax_d.set_yticks(np.arange(n_cats))
        ax_d.set_yticklabels(all_cats, fontsize=9)
        ax_d.set_xlabel("Mean alignment tax  (±SEM)")
        ax_d.set_title("Panel D — IT alignment tax by category\n"
                       "(PT not available for comparison)")
        ax_d.grid(axis="x", alpha=0.2)

    fig.suptitle(
        "Exp3 Plot 5 — Alignment Tax: Fraction of Compute in Corrective Stage\n"
        "Higher tax = more MLP work in layers 20–33 (format / governance enforcement)",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot5_alignment_tax.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 5 (Exp3) saved → {out_path}")
