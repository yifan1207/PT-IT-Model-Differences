"""
Plot 9 (Exp3): Pre-correction layers only (0–19) — IT vs PT by category (Exp 1c).

The core claim is that IT's corrective stage (layers 20–33) enforces format and
alignment on top of raw PT predictions. But if the IT/PT difference is already
visible in early layers (0–19), then post-training changed how IT processes the
PROPOSAL stage too, not just the correction.

This plot answers: is IC/OOC/R separation real in early layers, or is it an
artifact of the corrective stage?

Four panels:
  A: Mean L0 (transcoder sparsity) at proposal layers (0–19) per category, IT vs PT.
     If categories separate before layer 20, the effect is proposal-stage, not just correction.
  B: Mean layer_delta_norm at proposal layers per category, IT vs PT.
     How much does each layer edit the residual stream before the corrective stage?
  C: Mean layer_delta_cosine at proposal layers per category, IT vs PT.
     Sign = is the edit aligned with the existing residual (positive) or opposing (negative)?
  D: Mean residual_norm trajectory (layers 0–19) by category, IT solid / PT dashed.
     Does the residual stream magnitude grow differently across task types in early layers?
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.shared.constants import N_LAYERS

CATEGORY_COLORS = {
    "in_context":     "#2196F3",
    "out_of_context": "#FF9800",
    "reasoning":      "#4CAF50",
    "IC":             "#2196F3",
    "OOC":            "#FF9800",
    "R":              "#4CAF50",
    "GEN":            "#9C27B0",
}
_DEFAULT_COLOR = "#888888"
_BOUNDARY      = 20


def _per_layer_by_cat(results: list[dict], metric_key: str,
                       n_layers: int = N_LAYERS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean ± SEM for a [step][layer] metric, split by category."""
    by_cat: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n_layers)])
    for r in results:
        cat = r.get("category", "unknown")
        for step_vals in r.get(metric_key, []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    by_cat[cat][layer_i].append(float(v))
    out = {}
    for cat, layer_lists in by_cat.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layer_lists])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                      for v in layer_lists])
        out[cat] = (m, s)
    return out


def _per_layer_overall(results: list[dict], metric_key: str,
                        n_layers: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """Overall (all categories) per-layer mean ± SEM."""
    layer_vals: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        for step_vals in r.get(metric_key, []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    layer_vals[layer_i].append(float(v))
    m = np.array([np.mean(v) if v else float("nan") for v in layer_vals])
    s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                  for v in layer_vals])
    return m, s


def _make_panel(ax, results_by_cat, pt_overall_m, pt_overall_s,
                layers, metric_label, title, boundary=_BOUNDARY):
    """Shared panel renderer: IT by category (solid), PT overall (dashed)."""
    ax.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
    ax.axvline(boundary, color="black", lw=0.8, ls=":", alpha=0.4,
               label=f"boundary (L{boundary})")

    for cat, (m, s) in results_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax.plot(layers[:boundary], m[:boundary], lw=2, color=c, label=f"IT {cat}")
        ax.fill_between(layers[:boundary],
                        np.maximum(0, m[:boundary] - s[:boundary]),
                        m[:boundary] + s[:boundary],
                        alpha=0.12, color=c)

    if pt_overall_m is not None:
        ax.plot(layers[:boundary], pt_overall_m[:boundary],
                lw=1.8, color="#555", ls="--", alpha=0.7, label="PT (all cats)")
        ax.fill_between(layers[:boundary],
                        np.maximum(0, pt_overall_m[:boundary] - pt_overall_s[:boundary]),
                        pt_overall_m[:boundary] + pt_overall_s[:boundary],
                        alpha=0.08, color="#555")

    ax.set_xlabel("Transformer Layer (proposal stage only)")
    ax.set_ylabel(metric_label)
    ax.set_title(title)
    ax.set_xlim(-0.5, boundary - 0.5)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results:
        print("  Plot 9 (Exp3) skipped — no results")
        return

    layers = np.arange(N_LAYERS)

    it_l0_by_cat   = _per_layer_by_cat(results, "l0")
    it_norm_by_cat = _per_layer_by_cat(results, "layer_delta_norm")
    it_cos_by_cat  = _per_layer_by_cat(results, "layer_delta_cosine")
    it_rn_by_cat   = _per_layer_by_cat(results, "residual_norm")

    pt_l0_m, pt_l0_s     = (_per_layer_overall(pt_results, "l0")
                             if pt_results else (None, None))
    pt_norm_m, pt_norm_s = (_per_layer_overall(pt_results, "layer_delta_norm")
                             if pt_results else (None, None))
    pt_cos_m, pt_cos_s   = (_per_layer_overall(pt_results, "layer_delta_cosine")
                             if pt_results else (None, None))
    pt_rn_m, pt_rn_s     = (_per_layer_overall(pt_results, "residual_norm")
                             if pt_results else (None, None))

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    _make_panel(ax_a, it_l0_by_cat, pt_l0_m, pt_l0_s, layers,
                "Mean L0 (active transcoder features)",
                "Panel A — Transcoder sparsity (L0) in proposal layers\n"
                "If categories separate here → proposal stage is also task-dependent")

    _make_panel(ax_b, it_norm_by_cat, pt_norm_m, pt_norm_s, layers,
                "Mean layer δ norm",
                "Panel B — Edit magnitude in proposal layers\n"
                "How much does each layer update the residual before correction?")

    _make_panel(ax_c, it_cos_by_cat, pt_cos_m, pt_cos_s, layers,
                "Mean cos(δ_i,  h_{i−1})",
                "Panel C — Edit direction in proposal layers\n"
                "Negative = layer opposes the existing residual stream")

    _make_panel(ax_d, it_rn_by_cat, pt_rn_m, pt_rn_s, layers,
                "Mean residual ||h_i||",
                "Panel D — Residual stream magnitude in proposal layers\n"
                "Does residual grow differently across task types pre-correction?")

    fig.suptitle(
        "Exp3 Plot 9 — Pre-Correction Layers (0–19) Only  (Exp 1c)\n"
        "Does IT already differ from PT before the corrective stage?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot9_precorrection.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 9 (Exp3) saved → {out_path}")
