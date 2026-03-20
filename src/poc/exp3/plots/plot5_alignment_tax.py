"""
Plot 5 (Exp3): Alignment tax quantification (Experiment 2b).

Visualises the fraction of total MLP computation that falls in the corrective
stage (layers >= boundary) vs the proposal stage.

Works on existing exp2 or exp3 results — no new inference needed.
Uses analysis/alignment_tax.py for the core computation.

Current implementation — 2 panels:
  A: Per-layer mean delta norm as fraction of total (line plot by category).
     If pt_results is also provided, PT lines are overlaid as dashed.
  B: Box plot of per-prompt alignment tax (corrective fraction) by category.

To get the full PT-vs-IT comparison, pass both result files:
    run_plots.py --results it_results.json --pt-results pt_results.json

REQUIRES: layer_delta_norm (available in exp2 results).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.poc.exp3.analysis.alignment_tax import (
    compute_tax_by_category,
    compute_per_layer_contribution_fraction,
)

from src.poc.shared.constants import N_LAYERS

VARIANT_STYLE = {
    "pt": {"color": "#1565C0", "label": "PT (pretrained)",  "ls": "-"},
    "it": {"color": "#E65100", "label": "IT (instruction)", "ls": "--"},
}


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    """Render the alignment tax figure.

    Parameters
    ----------
    results : list[dict]
        Primary results (e.g. IT).
    output_dir : str
        Output directory.
    pt_results : list[dict] or None
        PT results for comparison.  If None, only the primary results are plotted.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or "layer_delta_norm" not in results[0]:
        print("  Plot 5 (Exp3) skipped — no layer_delta_norm data")
        return

    layers = np.arange(N_LAYERS)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: per-layer contribution fraction
    fracs_primary = compute_per_layer_contribution_fraction(results)
    for cat, frac in fracs_primary.items():
        axes[0].plot(layers, frac, lw=1.5, label=cat, alpha=0.7)
    if pt_results:
        fracs_pt = compute_per_layer_contribution_fraction(pt_results)
        for cat, frac in fracs_pt.items():
            axes[0].plot(layers, frac, lw=1.5, ls="--", alpha=0.5)

    axes[0].axvline(20, color="black", ls=":", lw=1.2, label="boundary (layer 20)")
    axes[0].set_xlabel("Transformer layer", fontsize=10)
    axes[0].set_ylabel("Mean δ norm as fraction of total", fontsize=10)
    axes[0].set_title("Per-layer computation fraction", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=8)

    # Panel B: per-prompt tax distribution
    tax_summary = compute_tax_by_category(results)
    positions = list(range(len(tax_summary)))
    cats = list(tax_summary.keys())
    axes[1].boxplot(
        [tax_summary[c]["per_prompt_taxes"] for c in cats],
        positions=positions,
        patch_artist=True,
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("Corrective fraction (alignment tax)", fontsize=10)
    axes[1].set_title("Alignment tax distribution by category", fontsize=10, fontweight="bold")
    axes[1].set_ylim(0, 1)

    fig.suptitle(
        "Alignment tax: fraction of total MLP computation in corrective stage (layers 20–33)\n"
        "Higher tax = more computation spent on governance / format enforcement",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot5_alignment_tax.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 5 (Exp3) saved → {out_path}")
