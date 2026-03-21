"""
E3.5 — Confidence-Stratified Commitment.

Purpose
-------
Controls for the "IT generates harder tokens" confound at the probability level:

  - Bin each generated token by its final-layer probability (how confident the model is).
  - Measure commitment delay (KL trajectory, commitment layer) within each bin.
  - If IT delays commitment MORE than PT even in the HIGH-confidence bin → the delay
    is not explained by token difficulty.

Confidence bins:
  HIGH:  final_layer_prob > 0.90
  MED:   0.50 ≤ final_layer_prob ≤ 0.90
  LOW:   final_layer_prob < 0.50

`final_layer_prob` is collected at the last layer (layer 33) from next_token_prob[step][-1].
Backward compatible: falls back to next_token_prob[step][-1] if final_layer_prob absent.

Four panels:
  A: Mean KL-to-final per layer, by confidence bin × model variant (6 lines).
     IT high (solid), IT med (solid), IT low (solid) + PT dashed counterparts.

  B: Commitment layer distribution per confidence bin — IT vs PT.
     3 side-by-side violin or histogram subplots (one per bin).
     Key: does IT commit later than PT within the same confidence bin?

  C: Mean layer_delta_norm per layer, by confidence bin.
     Shows whether corrective stage magnitude scales with token uncertainty.

  D: Summary bar chart — median commitment layer by (variant × bin).
     Error bars = IQR (25th–75th percentile).
     Clearest summary of the confound-control result.
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.shared.constants import N_LAYERS

_BOUNDARY     = 20
_KL_THRESHOLD = 0.1   # nats

# Confidence bins (inclusive)
_BINS = [
    ("HIGH", 0.90, 1.01, "#1B5E20", "-"),     # dark green solid IT
    ("MED",  0.50, 0.90, "#F57F17", "-"),     # amber solid IT
    ("LOW",  0.00, 0.50, "#B71C1C", "-"),     # dark red solid IT
]
_IT_ALPHA = 1.0
_PT_ALPHA = 0.55
_PT_LS    = "--"


def _get_final_layer_prob(r: dict, step_i: int) -> float | None:
    """Extract the final-layer probability for a given step.

    Tries 'final_layer_prob' first (new field), falls back to
    next_token_prob[step][-1] (always present when collect_emergence=True).
    """
    flp = r.get("final_layer_prob")
    if flp and step_i < len(flp):
        v = flp[step_i]
        if v is not None and not math.isnan(float(v)):
            return float(v)

    ntp = r.get("next_token_prob")
    if ntp and step_i < len(ntp) and ntp[step_i]:
        v = ntp[step_i][-1]
        if v is not None and not math.isnan(float(v)):
            return float(v)
    return None


def _bin_name(prob: float) -> str:
    if prob >= 0.90:
        return "HIGH"
    if prob >= 0.50:
        return "MED"
    return "LOW"


def _kl_by_confidence_bin(
    results: list[dict],
    n_layers: int = N_LAYERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean ± SEM kl_to_final per layer, grouped by confidence bin."""
    by_bin: dict[str, list[list[float]]] = {
        "HIGH": [[] for _ in range(n_layers)],
        "MED":  [[] for _ in range(n_layers)],
        "LOW":  [[] for _ in range(n_layers)],
    }
    for r in results:
        kl_data = r.get("kl_to_final", [])
        for step_i, step_kl in enumerate(kl_data):
            prob = _get_final_layer_prob(r, step_i)
            if prob is None:
                continue
            bname = _bin_name(prob)
            for li, v in enumerate(step_kl[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    by_bin[bname][li].append(float(v))
    out = {}
    for bname, layers_list in by_bin.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layers_list])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0. for v in layers_list])
        out[bname] = (m, s)
    return out


def _delta_by_confidence_bin(
    results: list[dict],
    n_layers: int = N_LAYERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean ± SEM layer_delta_norm per layer, grouped by confidence bin."""
    by_bin: dict[str, list[list[float]]] = {
        "HIGH": [[] for _ in range(n_layers)],
        "MED":  [[] for _ in range(n_layers)],
        "LOW":  [[] for _ in range(n_layers)],
    }
    for r in results:
        delta_data = r.get("layer_delta_norm", [])
        for step_i, step_delta in enumerate(delta_data):
            prob = _get_final_layer_prob(r, step_i)
            if prob is None:
                continue
            bname = _bin_name(prob)
            for li, v in enumerate(step_delta[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    by_bin[bname][li].append(float(v))
    out = {}
    for bname, layers_list in by_bin.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layers_list])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0. for v in layers_list])
        out[bname] = (m, s)
    return out


def _commitment_by_bin(
    results: list[dict],
    threshold: float = _KL_THRESHOLD,
    n_layers: int = N_LAYERS,
) -> dict[str, list[int]]:
    """Commitment layers grouped by confidence bin."""
    by_bin: dict[str, list[int]] = {"HIGH": [], "MED": [], "LOW": []}
    for r in results:
        kl_data = r.get("kl_to_final", [])
        for step_i, step_kl in enumerate(kl_data):
            prob = _get_final_layer_prob(r, step_i)
            if prob is None:
                continue
            bname = _bin_name(prob)
            for li, v in enumerate(step_kl[:n_layers]):
                if v is not None and not math.isnan(float(v)) and float(v) < threshold:
                    by_bin[bname].append(li)
                    break
    return by_bin


def _bin_step_counts(results: list[dict]) -> dict[str, int]:
    counts = {"HIGH": 0, "MED": 0, "LOW": 0}
    for r in results:
        kl_data = r.get("kl_to_final", [])
        for step_i in range(len(kl_data)):
            prob = _get_final_layer_prob(r, step_i)
            if prob is not None:
                counts[_bin_name(prob)] += 1
    return counts


def make_plot(
    results: list[dict],
    output_dir: str,
    pt_results: list[dict] | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("next_token_prob") and not results[0].get("final_layer_prob"):
        print("  Plot E3.5 skipped — no next_token_prob / final_layer_prob data.")
        return

    layers = np.arange(N_LAYERS)

    # ── Data extraction ───────────────────────────────────────────────────────
    it_kl_by_bin    = _kl_by_confidence_bin(results)
    it_delta_by_bin = _delta_by_confidence_bin(results)
    it_commit_by_bin = _commitment_by_bin(results)
    it_step_counts  = _bin_step_counts(results)

    pt_kl_by_bin:    dict = {}
    pt_delta_by_bin: dict = {}
    pt_commit_by_bin: dict = {}
    pt_step_counts:  dict = {}
    if pt_results:
        pt_kl_by_bin    = _kl_by_confidence_bin(pt_results)
        pt_delta_by_bin = _delta_by_confidence_bin(pt_results)
        pt_commit_by_bin = _commitment_by_bin(pt_results)
        pt_step_counts  = _bin_step_counts(pt_results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: KL trajectory by confidence bin ──────────────────────────────
    ax_a.axvspan(_BOUNDARY, N_LAYERS, alpha=0.04, color="grey")
    ax_a.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_a.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_a.axhline(_KL_THRESHOLD, color="green", lw=0.8, ls="--", alpha=0.5,
                 label=f"commit threshold")

    for bname, lo, hi, color, _ in _BINS:
        n = it_step_counts.get(bname, 0)
        if bname in it_kl_by_bin:
            m, s = it_kl_by_bin[bname]
            ax_a.plot(layers, m, lw=2, color=color, ls="-",
                      label=f"IT {bname} prob [{lo:.0%},{hi:.0%}) n={n:,}")
            ax_a.fill_between(layers, np.maximum(0, m-s), m+s, alpha=0.12, color=color)
        if pt_results and bname in pt_kl_by_bin:
            pm, ps = pt_kl_by_bin[bname]
            pt_n = pt_step_counts.get(bname, 0)
            ax_a.plot(layers, pm, lw=1.5, color=color, ls=_PT_LS, alpha=_PT_ALPHA,
                      label=f"PT {bname} n={pt_n:,}")
            ax_a.fill_between(layers, np.maximum(0, pm-ps), pm+ps,
                              alpha=0.05, color=color)

    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean KL(layer_i ∥ final) nats (±SEM)")
    ax_a.set_title("Panel A — KL trajectory by confidence bin\n"
                   "IT solid / PT dashed  |  Does IT delay commitment in ALL bins?")
    ax_a.legend(fontsize=7, ncol=2)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: commitment layer distribution per bin — IT vs PT ─────────────
    bins_hist = np.arange(0, N_LAYERS + 2) - 0.5
    bin_names = ["HIGH", "MED", "LOW"]
    bin_colors_map = {b[0]: b[3] for b in _BINS}

    for col_idx, bname in enumerate(bin_names):
        ax = ax_b
        if col_idx == 0:
            ax = ax_b
        # We draw all 3 in ax_b as overlapping histograms — use alpha + offset trick
        it_c = it_commit_by_bin.get(bname, [])
        pt_c = pt_commit_by_bin.get(bname, []) if pt_results else []
        color = bin_colors_map[bname]
        alpha_base = 0.55 - col_idx * 0.1

        if it_c:
            ax_b.hist(it_c, bins=bins_hist, density=True, alpha=alpha_base,
                      color=color, histtype="stepfilled",
                      label=f"IT {bname} med={np.median(it_c):.0f}")
            ax_b.axvline(np.median(it_c), color=color, lw=2, ls="-")
        if pt_c:
            ax_b.hist(pt_c, bins=bins_hist, density=True, alpha=alpha_base * 0.5,
                      color=color, histtype="step", lw=1.8, ls="--",
                      label=f"PT {bname} med={np.median(pt_c):.0f}")
            ax_b.axvline(np.median(pt_c), color=color, lw=1.5, ls="--", alpha=0.7)

    ax_b.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_b.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_b.set_xlabel("Commitment layer")
    ax_b.set_ylabel("Density")
    ax_b.set_title(
        f"Panel B — Commitment distribution per confidence bin\n"
        f"IT filled / PT outline  |  Same bin, does IT still commit later?"
    )
    ax_b.legend(fontsize=7, ncol=2)
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel C: layer_delta_norm by confidence bin ────────────────────────────
    ax_c.axvspan(_BOUNDARY, N_LAYERS, alpha=0.04, color="grey")
    ax_c.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_c.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)

    for bname, lo, hi, color, _ in _BINS:
        if bname in it_delta_by_bin:
            m, s = it_delta_by_bin[bname]
            ax_c.plot(layers, m, lw=2, color=color, ls="-",
                      label=f"IT {bname}")
            ax_c.fill_between(layers, np.maximum(0, m-s), m+s, alpha=0.12, color=color)
        if pt_results and bname in pt_delta_by_bin:
            pm, ps = pt_delta_by_bin[bname]
            ax_c.plot(layers, pm, lw=1.5, color=color, ls=_PT_LS, alpha=_PT_ALPHA)

    ax_c.set_xlabel("Transformer Layer")
    ax_c.set_ylabel("Mean |Δresidual| (±SEM)")
    ax_c.set_title("Panel C — Layer delta norm by confidence bin\n"
                   "Does corrective stage scale with token uncertainty?")
    ax_c.legend(fontsize=8, ncol=2)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: summary bar chart — median commitment layer ──────────────────
    x = np.arange(len(bin_names))
    w = 0.35

    def _median_iqr(commits: list[int]) -> tuple[float, float, float]:
        if not commits:
            return float("nan"), float("nan"), float("nan")
        arr = np.array(commits)
        return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))

    it_meds, it_q25, it_q75 = zip(*[_median_iqr(it_commit_by_bin.get(b, [])) for b in bin_names])
    it_err_lo = [m - q25 for m, q25 in zip(it_meds, it_q25)]
    it_err_hi = [q75 - m  for m, q75 in zip(it_meds, it_q75)]

    ax_d.bar(x - w/2, it_meds, w,
             color=[bin_colors_map[b] for b in bin_names],
             alpha=0.85, label="IT median")
    ax_d.errorbar(x - w/2, it_meds, yerr=[it_err_lo, it_err_hi],
                  fmt="none", color="black", capsize=5, lw=1.5)

    if pt_results:
        pt_meds, pt_q25, pt_q75 = zip(*[_median_iqr(pt_commit_by_bin.get(b, [])) for b in bin_names])
        pt_err_lo = [m - q25 for m, q25 in zip(pt_meds, pt_q25)]
        pt_err_hi = [q75 - m  for m, q75 in zip(pt_meds, pt_q75)]
        ax_d.bar(x + w/2, pt_meds, w,
                 color=[bin_colors_map[b] for b in bin_names],
                 alpha=0.40, hatch="//", label="PT median")
        ax_d.errorbar(x + w/2, pt_meds, yerr=[pt_err_lo, pt_err_hi],
                      fmt="none", color="black", capsize=5, lw=1.2, alpha=0.7)

    # Annotate IT-PT delta for each bin
    for xi, bname in enumerate(bin_names):
        if pt_results and not math.isnan(float(it_meds[xi])):
            pt_med = pt_meds[xi] if pt_results else float("nan")
            if not math.isnan(float(pt_med)):
                delta = float(it_meds[xi]) - float(pt_med)
                ax_d.text(xi, max(float(it_meds[xi]), float(pt_med)) + 0.5,
                          f"Δ={delta:+.1f}L", ha="center", va="bottom", fontsize=9,
                          fontweight="bold",
                          color="green" if delta > 0 else "red")

    ax_d.set_xticks(x)
    ax_d.set_xticklabels([
        f"{b}\n({lo:.0%}–{hi:.0%})" for b, lo, hi, _, _ in _BINS
    ], fontsize=9)
    ax_d.set_ylabel("Median commitment layer  (error bars = IQR)")
    ax_d.set_title("Panel D — Median commitment layer by confidence × variant\n"
                   "Δ = IT median − PT median per bin  |  Positive → IT delays more")
    ax_d.legend(fontsize=9)
    ax_d.grid(axis="y", alpha=0.25)
    ax_d.axhline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4,
                 label=f"corrective boundary (L{_BOUNDARY})")

    fig.suptitle(
        "E3.5 — Confidence-Stratified Commitment\n"
        "Controls 'IT generates harder tokens' confound at the probability level\n"
        f"Bins: HIGH >90%, MED 50–90%, LOW <50%  |  threshold={_KL_THRESHOLD} nats",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot_e3_5_confidence_stratified.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.5 saved → {out_path}")
