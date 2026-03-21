"""
Plot 2 (Exp4): Attention entropy profile — PT vs IT at dip layers (E0a).

Tests P2: Attention Divergence at the Dip.

Three panels:
  A — Per-layer mean attention entropy, all layers 0–33
      PT (blue) and IT (orange).  Shaded ±1 SEM.
      Vertical dashed line at dip layer.
      Global attention layers (5,11,17,23,29) annotated with 'G'.

  B — PT-IT absolute entropy difference per layer
      Highlights layers where attention patterns diverge most.
      Expected: peak divergence at layers 8–11 (dip region).

  C — Attention entropy heatmap (heads × layers)
      For PT and IT side by side.  Rows = attention heads (0–7).
      Columns = layers.  Colour = mean entropy across prompts.
      Shows which specific heads change most at the dip.

Interpretation guide:
  - Lower entropy = more focused attention (more concentrated).
  - If IT has lower entropy at dip layers → IT attention is more focused there.
  - If the divergence peak is at the dip layers (not elsewhere) → attention
    routing is the proximal mechanism of dip sharpening (P2 confirmed).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path

from src.poc.exp4.analysis.attention_entropy import (
    extract_attn_entropy,
    compute_mean_entropy_profile,
    compute_entropy_divergence,
    summarise_dip_region,
    GLOBAL_ATTN_LAYERS,
)

_DIP_LAYER = 11
_N_LAYERS  = 34
_N_HEADS   = 8  # Gemma 3 4B


def make_plot(
    pt_results: list[dict],
    it_results: list[dict],
    output_dir: str,
    dip_layer:  int = _DIP_LAYER,
) -> None:
    """Generate the attention entropy comparison plot.

    Parameters
    ----------
    pt_results   : list of exp4 result dicts for PT model
    it_results   : list of exp4 result dicts for IT model
    output_dir   : directory for output PNG
    dip_layer    : gate layer index (default 11)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / "plot2_attention_entropy.png"

    # Extract entropy data
    pt_ent_data = extract_attn_entropy(pt_results, n_layers=_N_LAYERS)
    it_ent_data = extract_attn_entropy(it_results, n_layers=_N_LAYERS)

    if pt_ent_data["n_records"] == 0 and it_ent_data["n_records"] == 0:
        print("  Plot 2 (Exp4) skipped — no attn_entropy data in results")
        return

    pt_profile = compute_mean_entropy_profile(pt_ent_data, n_layers=_N_LAYERS)
    it_profile = compute_mean_entropy_profile(it_ent_data, n_layers=_N_LAYERS)
    divergence = compute_entropy_divergence(pt_profile, it_profile, n_layers=_N_LAYERS)

    pt_dip_summary = summarise_dip_region(pt_profile, dip_layer=dip_layer)
    it_dip_summary = summarise_dip_region(it_profile, dip_layer=dip_layer)

    layers = np.arange(_N_LAYERS)
    pt_mean = np.array(pt_profile["mean_entropy"], dtype=float)
    it_mean = np.array(it_profile["mean_entropy"], dtype=float)
    pt_sem  = np.array(pt_profile["sem_entropy"],  dtype=float)
    it_sem  = np.array(it_profile["sem_entropy"],  dtype=float)
    abs_diff = np.array(divergence["abs_diff"], dtype=float)

    # Head-level means [n_layers, n_heads]
    def _head_matrix(profile) -> np.ndarray:
        mat = []
        n_heads = profile.get("mean_per_head", []) and len(profile["mean_per_head"][0]) or _N_HEADS
        for layer_heads in profile["mean_per_head"]:
            if layer_heads is None or len(layer_heads) == 0:
                mat.append([float("nan")] * n_heads)
            else:
                mat.append([float("nan") if np.isnan(v) else v for v in layer_heads])
        return np.array(mat, dtype=float)  # [n_layers, n_heads]

    pt_head_mat = _head_matrix(pt_profile)
    it_head_mat = _head_matrix(it_profile)

    # ── figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs  = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    ax_a  = fig.add_subplot(gs[0, :])   # Panel A: spans full top row
    ax_b  = fig.add_subplot(gs[1, 0])   # Panel B: divergence
    ax_c  = fig.add_subplot(gs[1, 1])   # Panel C: head heatmap difference

    fig.suptitle(
        "Exp4 — Attention Entropy Profile PT vs IT (E0a)\n"
        f"Dip layer = {dip_layer}  |  Global layers: {sorted(GLOBAL_ATTN_LAYERS)}",
        fontsize=12, fontweight="bold",
    )

    # ── Panel A: entropy profile ───────────────────────────────────────────────
    ax_a.axvspan(dip_layer - 0.4, dip_layer + 0.4, alpha=0.12, color="grey",
                 label=f"Dip layer ({dip_layer})")

    # Shade ±SEM
    ax_a.fill_between(layers, pt_mean - pt_sem, pt_mean + pt_sem,
                      alpha=0.15, color="steelblue")
    ax_a.fill_between(layers, it_mean - it_sem, it_mean + it_sem,
                      alpha=0.15, color="darkorange")

    ax_a.plot(layers, pt_mean, "o-", color="steelblue",  linewidth=2,
              markersize=4, label="PT")
    ax_a.plot(layers, it_mean, "s-", color="darkorange", linewidth=2,
              markersize=4, label="IT")

    # Mark global attention layers
    for g in sorted(GLOBAL_ATTN_LAYERS):
        ax_a.axvline(g, color="purple", alpha=0.2, linewidth=0.8, linestyle=":")
    ax_a.text(0.01, 0.97, "dotted = global attention layer", transform=ax_a.transAxes,
              fontsize=7.5, va="top", color="purple", alpha=0.7)

    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean attention entropy (nats)")
    ax_a.set_title("Panel A — Per-layer attention entropy across all heads")
    ax_a.set_xticks(layers)
    ax_a.set_xticklabels([str(i) for i in layers], fontsize=7)
    ax_a.legend(loc="upper right")
    ax_a.grid(axis="y", alpha=0.3)

    # ── Panel B: PT-IT divergence ──────────────────────────────────────────────
    bar_colors = ["#d62728" if abs_diff[l] == np.nanmax(abs_diff) else "steelblue"
                  for l in layers]
    ax_b.bar(layers, abs_diff, color="steelblue", alpha=0.75, width=0.7)

    # Colour bar over dip region
    dip_region = [l for l in layers if abs(l - dip_layer) <= 2]
    for l in dip_region:
        if not np.isnan(abs_diff[l]):
            ax_b.bar(l, abs_diff[l], color="darkorange", alpha=0.8, width=0.7)

    ax_b.axvline(dip_layer, color="grey", linestyle="--", alpha=0.6)
    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("|H_IT - H_PT| (nats)")
    ax_b.set_title("Panel B — PT-IT absolute entropy divergence per layer\n"
                   "(orange = dip region ±2)")
    ax_b.set_xticks(layers[::2])
    ax_b.set_xticklabels([str(i) for i in layers[::2]], fontsize=8)
    ax_b.grid(axis="y", alpha=0.3)

    # Annotation
    peak_layer = int(np.nanargmax(abs_diff))
    ax_b.annotate(f"peak: L{peak_layer}",
                  xy=(peak_layer, abs_diff[peak_layer]),
                  xytext=(peak_layer + 1.5, abs_diff[peak_layer] * 0.9),
                  arrowprops=dict(arrowstyle="->", color="black"),
                  fontsize=8)

    # ── Panel C: head-level heatmap (IT - PT difference) ─────────────────────
    head_diff = it_head_mat - pt_head_mat    # [n_layers, n_heads]
    # Rows = heads, columns = layers — so transpose for imshow
    head_diff_T = head_diff.T  # [n_heads, n_layers]

    vmax = np.nanquantile(np.abs(head_diff_T), 0.95)
    vmax = max(vmax, 0.01)  # avoid all-zero
    im = ax_c.imshow(
        head_diff_T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax_c.axvline(dip_layer, color="black", linewidth=1.5, alpha=0.5)
    ax_c.set_xlabel("Transformer Layer")
    ax_c.set_ylabel("Attention Head")
    ax_c.set_title("Panel C — Per-head entropy diff (IT − PT)\nRed = IT more spread, Blue = IT more focused")
    ax_c.set_xticks(range(0, _N_LAYERS, 4))
    ax_c.set_xticklabels(range(0, _N_LAYERS, 4), fontsize=8)
    ax_c.set_yticks(range(head_diff_T.shape[0]))
    ax_c.set_yticklabels([f"H{h}" for h in range(head_diff_T.shape[0])], fontsize=7)
    plt.colorbar(im, ax=ax_c, label="Entropy diff (nats)")

    # ── summary text ──────────────────────────────────────────────────────────
    summary_lines = [
        f"PT entropy at dip: {pt_dip_summary['entropy_at_dip']:.3f}",
        f"IT entropy at dip: {it_dip_summary['entropy_at_dip']:.3f}",
        f"Δ at dip: {it_dip_summary['entropy_at_dip'] - pt_dip_summary['entropy_at_dip']:+.3f}",
        f"PT dip-region mean: {pt_dip_summary['dip_region_mean']:.3f}",
        f"IT dip-region mean: {it_dip_summary['dip_region_mean']:.3f}",
        f"Peak divergence layer: L{peak_layer}",
    ]
    ax_b.text(0.98, 0.97, "\n".join(summary_lines),
              transform=ax_b.transAxes, ha="right", va="top", fontsize=7.5,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")
