"""
Plot 4 (Exp3): Corrective stage magnitude stratified by token type (Experiment 2a).

Splits generated tokens into types (CONTENT, PUNCTUATION, DISCOURSE, STRUCTURAL, OTHER)
using analysis/token_types.py, then compares how much the corrective stage (layers 20+)
works per token type.

If IT's corrective stage is larger for DISCOURSE and STRUCTURAL tokens than for
CONTENT tokens, the corrective stage is governing output format, not core content.

Four panels:
  A: Mean layer_delta_norm at corrective layers (20–33) per token type
     Box plots: IT vs PT side by side per token type.
  B: Mean layer_delta_cosine at corrective layers (20–33) per token type
     Bar chart with SEM error bars: IT vs PT side by side.
     (cosine of residual delta with prior residual — negative = opposing stream)
  C: Per-layer mean delta norm, split by token type (IT only), with SEM bands.
  D: Token type frequency distribution (fraction of all generated tokens),
     IT vs PT side by side.

REQUIRES: generated_tokens + layer_delta_norm + layer_delta_cosine (all in exp2/3).
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.exp03_corrective_stage_characterization.analysis.token_types import stratify_by_token_type
from src.poc.shared.constants import N_LAYERS

_BOUNDARY = 20
_TOKEN_TYPES = ["CONTENT", "FUNCTION", "DISCOURSE", "STRUCTURAL", "PUNCTUATION", "OTHER"]
_TYPE_COLORS  = {
    "CONTENT":     "#1565C0",
    "FUNCTION":    "#00897B",
    "DISCOURSE":   "#E65100",
    "STRUCTURAL":  "#558B2F",
    "PUNCTUATION": "#6A1B9A",
    "OTHER":       "#78909C",
}


def _corrective_mean_by_type(results: list[dict], metric_key: str,
                               boundary: int = _BOUNDARY,
                               n_layers: int  = N_LAYERS) -> dict[str, list[float]]:
    """For each result, compute the mean metric value in corrective layers (>=boundary)
    for each token type, then return a dict of type → list of per-prompt means.
    """
    by_type: dict[str, list[float]] = defaultdict(list)

    for r in results:
        grouped = stratify_by_token_type(r, metric_key)
        for tok_type, step_lists in grouped.items():
            vals = []
            for step_vals in step_lists:
                for layer_i, v in enumerate(step_vals[:n_layers]):
                    if layer_i >= boundary and v is not None and not math.isnan(float(v)):
                        vals.append(float(v))
            if vals:
                by_type[tok_type].append(float(np.mean(vals)))

    return dict(by_type)


def _per_layer_mean_by_type(results: list[dict], metric_key: str,
                              n_layers: int = N_LAYERS
                              ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean ± SEM for each token type (collapsed across all prompts × steps)."""
    sums   = defaultdict(lambda: np.zeros(n_layers))
    sqsums = defaultdict(lambda: np.zeros(n_layers))
    counts = defaultdict(lambda: np.zeros(n_layers))

    for r in results:
        grouped = stratify_by_token_type(r, metric_key)
        for tok_type, step_lists in grouped.items():
            for step_vals in step_lists:
                for layer_i, v in enumerate(step_vals[:n_layers]):
                    if v is not None and not math.isnan(float(v)):
                        fv = float(v)
                        sums[tok_type][layer_i]   += fv
                        sqsums[tok_type][layer_i] += fv * fv
                        counts[tok_type][layer_i] += 1

    result = {}
    for tok_type in sums:
        n  = counts[tok_type]
        s  = sums[tok_type]
        sq = sqsums[tok_type]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(n > 0, s / n, np.nan)
            var  = np.where(n > 1, sq / n - (s / n) ** 2, 0.0)
            sem  = np.where(n > 1, np.sqrt(np.maximum(var, 0.0) / n), 0.0)
        result[tok_type] = (mean, sem)
    return result


def _token_type_frequencies(results: list[dict]) -> dict[str, float]:
    """Fraction of all generated tokens that fall into each type."""
    from src.poc.exp03_corrective_stage_characterization.analysis.token_types import classify_generated_tokens
    counts = defaultdict(int)
    total  = 0
    for r in results:
        types = classify_generated_tokens(r.get("generated_tokens", []))
        for t in types:
            counts[t] += 1
            total += 1
    if total == 0:
        return {t: 0.0 for t in _TOKEN_TYPES}
    return {t: counts[t] / total for t in _TOKEN_TYPES}


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or "generated_tokens" not in results[0]:
        print("  Plot 4 (Exp3) skipped — no generated_tokens data")
        return

    layers = np.arange(N_LAYERS)

    # ── compute metrics ────────────────────────────────────────────────────────
    it_norm_by_type  = _corrective_mean_by_type(results, "layer_delta_norm")
    it_cos_by_type   = _corrective_mean_by_type(results, "layer_delta_cosine")
    it_layer_norm    = _per_layer_mean_by_type(results, "layer_delta_norm")
    it_freqs         = _token_type_frequencies(results)

    pt_norm_by_type  = _corrective_mean_by_type(pt_results, "layer_delta_norm") if pt_results else None
    pt_cos_by_type   = _corrective_mean_by_type(pt_results, "layer_delta_cosine") if pt_results else None
    pt_freqs         = _token_type_frequencies(pt_results) if pt_results else None

    types_present = [t for t in _TOKEN_TYPES if t in it_norm_by_type]

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: box plot of corrective delta norm per type ───────────────────
    positions   = np.arange(len(types_present))
    box_width   = 0.35
    it_box_data = [it_norm_by_type.get(t, []) for t in types_present]
    it_bp = ax_a.boxplot(
        it_box_data,
        positions=positions - box_width / 2,
        widths=box_width, patch_artist=True,
        medianprops={"color": "white", "lw": 2},
    )
    for patch, tok_type in zip(it_bp["boxes"], types_present):
        patch.set_facecolor(_TYPE_COLORS[tok_type])
        patch.set_alpha(0.8)

    if pt_norm_by_type:
        pt_box_data = [pt_norm_by_type.get(t, []) for t in types_present]
        pt_bp = ax_a.boxplot(
            pt_box_data,
            positions=positions + box_width / 2,
            widths=box_width, patch_artist=True,
            medianprops={"color": "black", "lw": 2},
        )
        for patch in pt_bp["boxes"]:
            patch.set_facecolor("#BBDEFB")
            patch.set_alpha(0.8)

    ax_a.set_xticks(positions)
    ax_a.set_xticklabels(types_present, rotation=20, ha="right", fontsize=9)
    ax_a.set_ylabel("Mean δ norm in corrective layers (L20–33)")
    ax_a.set_title("Panel A — Corrective-stage edit size by token type\n"
                   "(IT coloured; PT light blue if available)")
    ax_a.grid(axis="y", alpha=0.25)
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=_TYPE_COLORS[t], label=f"IT {t}") for t in types_present]
    if pt_norm_by_type:
        handles.append(Patch(facecolor="#BBDEFB", label="PT (all types)"))
    ax_a.legend(handles=handles, fontsize=7, ncol=2)

    # ── Panel B: mean cosine by token type with SEM error bars ────────────────
    x_b   = np.arange(len(types_present))
    bar_w = 0.35

    it_cos_means = [np.mean(it_cos_by_type.get(t, [float("nan")])) for t in types_present]
    it_cos_sems  = [
        (np.std(it_cos_by_type[t]) / math.sqrt(len(it_cos_by_type[t]))
         if t in it_cos_by_type and len(it_cos_by_type[t]) > 1 else 0.0)
        for t in types_present
    ]
    ax_b.bar(x_b - bar_w / 2, it_cos_means, bar_w,
             color=[_TYPE_COLORS[t] for t in types_present], alpha=0.8, label="IT",
             yerr=it_cos_sems, capsize=4, error_kw={"elinewidth": 1.2})

    if pt_cos_by_type:
        pt_cos_means = [np.mean(pt_cos_by_type.get(t, [float("nan")])) for t in types_present]
        pt_cos_sems  = [
            (np.std(pt_cos_by_type[t]) / math.sqrt(len(pt_cos_by_type[t]))
             if t in pt_cos_by_type and len(pt_cos_by_type[t]) > 1 else 0.0)
            for t in types_present
        ]
        ax_b.bar(x_b + bar_w / 2, pt_cos_means, bar_w,
                 color="#BBDEFB", edgecolor="#1565C0", linewidth=1,
                 alpha=0.8, label="PT",
                 yerr=pt_cos_sems, capsize=4, error_kw={"elinewidth": 1.2})

    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(types_present, rotation=20, ha="right", fontsize=9)
    ax_b.set_ylabel("Mean cos(δ_i, h_{i-1}) in corrective layers  (±SEM)")
    ax_b.set_title("Panel B — Corrective-stage cosine similarity by token type\n"
                   "Negative = layer is actively opposing the residual stream")
    ax_b.legend(fontsize=9)
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel C: per-layer norm for IT by token type with SEM bands ────────────
    ax_c.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.5, label="dip (L11)")
    ax_c.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4,
                 label=f"boundary (L{_BOUNDARY})")

    for tok_type, (mean_arr, sem_arr) in it_layer_norm.items():
        if tok_type not in _TOKEN_TYPES:
            continue
        c = _TYPE_COLORS[tok_type]
        ax_c.plot(layers, mean_arr, lw=1.8, color=c, label=tok_type, alpha=0.85)
        ax_c.fill_between(layers,
                          np.maximum(0, mean_arr - sem_arr),
                          mean_arr + sem_arr,
                          alpha=0.12, color=c)

    ax_c.set_xlabel("Transformer Layer")
    ax_c.set_ylabel("Mean δ norm  (±SEM)")
    ax_c.set_title("Panel C — Per-layer delta norm by token type (IT)")
    ax_c.legend(fontsize=8, ncol=2)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: token type frequency distribution ────────────────────────────
    x_d    = np.arange(len(types_present))
    bar_w2 = 0.35
    it_freqs_vals = [it_freqs.get(t, 0) for t in types_present]
    ax_d.bar(x_d - bar_w2 / 2, it_freqs_vals, bar_w2,
             color=[_TYPE_COLORS[t] for t in types_present], alpha=0.8, label="IT")
    if pt_freqs:
        pt_freqs_vals = [pt_freqs.get(t, 0) for t in types_present]
        ax_d.bar(x_d + bar_w2 / 2, pt_freqs_vals, bar_w2,
                 color="#BBDEFB", edgecolor="#1565C0", linewidth=1,
                 alpha=0.8, label="PT")

    for xi, fv in zip(x_d - bar_w2 / 2, it_freqs_vals):
        ax_d.text(xi, fv + 0.002, f"{fv:.1%}", ha="center", va="bottom", fontsize=7)

    ax_d.set_xticks(x_d)
    ax_d.set_xticklabels(types_present, rotation=20, ha="right", fontsize=9)
    ax_d.set_ylabel("Fraction of all generated tokens")
    ax_d.set_title("Panel D — Token type frequency distribution\n"
                   "(IT should have more DISCOURSE/STRUCTURAL than PT)")
    ax_d.legend(fontsize=9)
    ax_d.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Exp3 Plot 4 — Corrective Stage Stratified by Token Type\n"
        "Hypothesis: IT corrects DISCOURSE/STRUCTURAL tokens more than CONTENT",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot4_token_stratification.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 4 (Exp3) saved → {out_path}")
