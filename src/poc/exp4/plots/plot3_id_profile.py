"""
Plot 3 (Exp4): Intrinsic Dimension (ID) profile — PT vs IT (E0c).

Tests P6: ID Geometry Confirmation.

Two panels:
  A — TwoNN ID per layer for PT and IT
      Smooth curve (PT) vs sharper contraction at dip (IT).
      Shading ±SEM would require bootstrap; we plot single-estimate curves.
      Also overlays MLE estimates as dashed lines for cross-validation.
      Grey column at dip layer; annotations at ID peak and dip.

  B — IT - PT ID difference per layer
      Positive = IT has higher ID at that layer (more diffuse).
      Negative = IT has lower ID at that layer (more compressed).
      Expected: IT lower at and after the dip (sharper compression).

Reference: Cheng et al. (ICLR 2025) found ID expansion→peak→contraction.
Our finding (hypothesis P6): IT sharpens this contraction at layer ~11.

TwoNN estimates are single-point estimates (no error bars without bootstrap).
Bootstrap CI is omitted here for runtime; a note is added to the plot.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.poc.exp4.analysis.intrinsic_dim import (
    compute_id_profile,
    compare_profiles,
)

_DIP_LAYER = 11
_N_LAYERS  = 34


def make_plot(
    pt_residuals_path: str,
    it_residuals_path: str,
    output_dir: str,
    dip_layer:  int  = _DIP_LAYER,
    run_mle:    bool = True,
) -> None:
    """Generate the ID profile comparison plot.

    Parameters
    ----------
    pt_residuals_path : path to PT exp4_residuals.npz
    it_residuals_path : path to IT exp4_residuals.npz
    output_dir        : directory for output PNG
    dip_layer         : gate layer index
    run_mle           : also compute and overlay MLE estimates
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "plot3_id_profile.png"

    for path, label in [(pt_residuals_path, "PT"), (it_residuals_path, "IT")]:
        if not Path(path).exists():
            print(f"  Plot 3 (Exp4) skipped — {label} residuals not found: {path}")
            return

    print("  Computing PT ID profile ...")
    pt_profile = compute_id_profile(pt_residuals_path, model_variant="pt",
                                    n_layers=_N_LAYERS, run_mle=run_mle)
    print("  Computing IT ID profile ...")
    it_profile = compute_id_profile(it_residuals_path, model_variant="it",
                                    n_layers=_N_LAYERS, run_mle=run_mle)

    comparison = compare_profiles(pt_profile, it_profile)

    layers   = np.arange(_N_LAYERS)
    pt_twonn = np.array(pt_profile["id_twonn"], dtype=float)
    it_twonn = np.array(it_profile["id_twonn"], dtype=float)
    pt_mle   = np.array(pt_profile["id_mle"],   dtype=float) if run_mle else None
    it_mle   = np.array(it_profile["id_mle"],   dtype=float) if run_mle else None

    diff = it_twonn - pt_twonn

    # ── figure ────────────────────────────────────────────────────────────────
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                                      gridspec_kw={"height_ratios": [2, 1]})
    fig.suptitle(
        "Exp4 — Intrinsic Dimension Profile PT vs IT (TwoNN) — E0c\n"
        f"n_samples={pt_profile['n_samples']}  |  Dip layer={dip_layer}",
        fontsize=12, fontweight="bold",
    )

    # ── Panel A ────────────────────────────────────────────────────────────────
    ax_a.axvspan(dip_layer - 0.5, dip_layer + 0.5, alpha=0.12, color="grey",
                 label=f"Dip layer ({dip_layer})")
    ax_a.axvline(dip_layer, color="grey", linewidth=1, linestyle="--", alpha=0.5)

    # TwoNN curves
    ax_a.plot(layers, pt_twonn, "o-", color="steelblue",  linewidth=2.5,
              markersize=5, label="PT  (TwoNN)")
    ax_a.plot(layers, it_twonn, "s-", color="darkorange", linewidth=2.5,
              markersize=5, label="IT  (TwoNN)")

    # MLE curves (dashed, for cross-validation)
    if run_mle and pt_mle is not None and it_mle is not None:
        ax_a.plot(layers, pt_mle, "--", color="steelblue",  linewidth=1.2,
                  alpha=0.5, label="PT  (MLE)")
        ax_a.plot(layers, it_mle, "--", color="darkorange", linewidth=1.2,
                  alpha=0.5, label="IT  (MLE)")

    # Annotate peak layers
    pt_peak = comparison["id_peak_layer_pt"]
    it_peak = comparison["id_peak_layer_it"]
    if pt_peak >= 0:
        ax_a.annotate(f"PT peak\nL{pt_peak}",
                      xy=(pt_peak, pt_twonn[pt_peak]),
                      xytext=(pt_peak - 2, pt_twonn[pt_peak] + 1.5),
                      arrowprops=dict(arrowstyle="->", color="steelblue"),
                      color="steelblue", fontsize=8)
    if it_peak >= 0:
        ax_a.annotate(f"IT peak\nL{it_peak}",
                      xy=(it_peak, it_twonn[it_peak]),
                      xytext=(it_peak + 1.5, it_twonn[it_peak] + 1.5),
                      arrowprops=dict(arrowstyle="->", color="darkorange"),
                      color="darkorange", fontsize=8)

    ax_a.set_ylabel("Intrinsic Dimension (TwoNN)")
    ax_a.set_title("Panel A — ID per layer  |  dashed = MLE cross-validation")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(axis="y", alpha=0.3)

    # Summary box
    summ = [
        f"Canonical dip: L{comparison['canonical_dip_layer']}",
        f"ID at dip  — PT: {comparison['id_at_dip_pt']:.2f}  IT: {comparison['id_at_dip_it']:.2f}",
        f"Dip sharpness PT: {comparison['dip_sharpness_pt']:.3f}",
        f"Dip sharpness IT: {comparison['dip_sharpness_it']:.3f}",
        f"Sharpening ratio (IT/PT): {comparison['sharpening_ratio']:.2f}x",
    ]
    ax_a.text(0.01, 0.98, "\n".join(summ), transform=ax_a.transAxes,
              va="top", fontsize=7.5,
              bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))

    # ── Panel B ────────────────────────────────────────────────────────────────
    ax_b.axhline(0, color="black", linewidth=0.8)
    ax_b.axvspan(dip_layer - 0.5, dip_layer + 0.5, alpha=0.12, color="grey")
    ax_b.axvline(dip_layer, color="grey", linewidth=1, linestyle="--", alpha=0.5)

    pos = np.where(diff >= 0, diff, 0)
    neg = np.where(diff <  0, diff, 0)
    ax_b.bar(layers, pos, color="#d62728", alpha=0.7, width=0.7, label="IT > PT")
    ax_b.bar(layers, neg, color="steelblue",  alpha=0.7, width=0.7, label="IT < PT")

    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("IT − PT  ID")
    ax_b.set_title("Panel B — IT minus PT intrinsic dimension  (negative = IT more compressed)")
    ax_b.set_xticks(layers)
    ax_b.set_xticklabels([str(i) for i in layers], fontsize=7)
    ax_b.legend(fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
