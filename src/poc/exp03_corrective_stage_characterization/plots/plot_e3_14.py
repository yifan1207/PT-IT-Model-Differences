"""
E3.14 — Quantitative Sniper Feature Analysis (IT vs PT).

Replaces LLM-as-judge characterization with reproducible distributional metrics
that directly test the "sniper vs. shotgun" hypothesis:

  IT's corrective-stage features are sparser, higher-magnitude, and more
  concentrated in the final layers than PT's, consistent with a specialized
  intervention mechanism rather than diffuse processing.

Metrics computed (all from feature_importance_summary.npz — no GPU needed):

  1. Gini coefficient of activation mass across features per layer
     - Higher Gini → fewer features carry most of the mass → "sniper"
     - Lower Gini → mass spread across many features → "shotgun"

  2. Firing rate distributions
     - How many features fire at all (L0 count) per layer
     - What fraction of total features are "active" (count > 0)
     - Active feature count ratio IT/PT

  3. Activation magnitude distributions
     - Mean activation per active feature (sum/count)
     - Median, 90th percentile, max activation
     - Heavy-tail index (ratio of p90/median)

  4. Layer concentration
     - What fraction of total corrective mass is in L32-33 vs L20-31
     - Comparison IT vs PT

  5. Feature overlap between IT and PT
     - For top-K features by activation mass, what fraction are shared?
     - Jaccard similarity at various K

  6. Feature sparsity (effective dimensionality)
     - Number of features needed to reach 50%/90% of total mass per layer
     - Ratio IT/PT

Panels:
  A: Gini coefficient per layer, IT vs PT (line plot)
  B: Activation magnitude distribution IT vs PT corrective layers (violin/box)
  C: Layer concentration heatmap (fraction of total mass per layer)
  D: Feature overlap (Jaccard) at varying top-K thresholds
  E: Cumulative mass concentration curves (features sorted by mass)
  F: Summary statistics table

Usage:
  python -m src.poc.exp03_corrective_stage_characterization.plots.plot_e3_14_sniper_quantitative
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

_N_LAYERS = 34
_CORRECTIVE_START = 20
_N_FEATURES = 16_384  # width_16k transcoder

# Default paths (match Exp3Config.run_dir structure)
_IT_SUMMARY = "results/exp3/it_16k_l0_big_affine_t512/feature_importance_summary.npz"
_PT_SUMMARY = "results/exp3/pt_16k_l0_big_affine_t512/feature_importance_summary.npz"


# ── Data Loading ─────────────────────────────────────────────────────────────

@dataclass
class FeatureStats:
    """Per-layer feature statistics for one model variant."""
    # Arrays shape [n_layers] or [n_layers, n_features]
    sums: dict[int, np.ndarray]     # layer → [n_features] total activation
    counts: dict[int, np.ndarray]   # layer → [n_features] firing count
    variant: str                     # "it" or "pt"

    @classmethod
    def from_npz(cls, path: str, variant: str) -> "FeatureStats":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Feature summary not found: {p}")
        raw = np.load(p, allow_pickle=True)
        sums: dict[int, np.ndarray] = {}
        counts: dict[int, np.ndarray] = {}
        for key in raw.files:
            prefix, _, layer_str = key.partition("_l")
            if not layer_str:
                continue
            layer = int(layer_str)
            if prefix == "sum":
                sums[layer] = raw[key].astype(float)
            elif prefix == "count":
                counts[layer] = raw[key].astype(float)
        return cls(sums=sums, counts=counts, variant=variant)


def _get_active_mask(counts: np.ndarray, min_count: int = 1) -> np.ndarray:
    """Boolean mask: features that fired at least min_count times."""
    return counts >= min_count


# ── Metric Computation ───────────────────────────────────────────────────────

def gini_coefficient(values: np.ndarray) -> float:
    """Compute Gini coefficient for a 1D non-negative array.

    Gini = 0 → perfect equality (all features equal mass)
    Gini = 1 → perfect inequality (one feature has all mass)

    Uses the standard formula: G = (2 * Σ_i (i * x_i)) / (n * Σ x_i) - (n+1)/n
    where x is sorted ascending.
    """
    v = np.sort(values[values > 0])  # only consider active features
    n = len(v)
    if n <= 1 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v)) / (n * np.sum(v)) - (n + 1) / n)


def cumulative_mass_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fraction_of_features, fraction_of_mass) for cumulative mass curve.

    Features sorted descending by mass. x-axis = fraction of active features used,
    y-axis = fraction of total mass captured.
    """
    active = values[values > 0]
    if len(active) == 0:
        return np.array([0, 1]), np.array([0, 0])
    sorted_desc = np.sort(active)[::-1]
    cumsum = np.cumsum(sorted_desc)
    total = cumsum[-1]
    n = len(sorted_desc)
    x = np.arange(1, n + 1) / n
    y = cumsum / total
    return x, y


def n_for_mass_fraction(values: np.ndarray, fraction: float) -> int:
    """Number of top features needed to capture `fraction` of total mass."""
    active = values[values > 0]
    if len(active) == 0:
        return 0
    sorted_desc = np.sort(active)[::-1]
    cumsum = np.cumsum(sorted_desc)
    total = cumsum[-1]
    idx = np.searchsorted(cumsum, fraction * total)
    return int(min(idx + 1, len(active)))


def overlap_jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def top_k_feature_ids(sums: np.ndarray, k: int) -> set[int]:
    """Return set of feature indices for top-K by activation sum."""
    if len(sums) == 0:
        return set()
    k = min(k, len(sums))
    top_idx = np.argpartition(sums, -k)[-k:]
    # Filter out zero-mass features
    return {int(i) for i in top_idx if sums[i] > 0}


# ── Per-Layer Metrics ────────────────────────────────────────────────────────

@dataclass
class LayerMetrics:
    layer: int
    gini: float
    n_active: int
    total_mass: float
    mean_activation: float      # mean of (sum/count) for active features
    median_activation: float
    p90_activation: float
    max_activation: float
    heavy_tail_ratio: float     # p90 / median
    n50: int                    # features for 50% mass
    n90: int                    # features for 90% mass
    frac_active: float          # n_active / n_features


def compute_layer_metrics(sums: np.ndarray, counts: np.ndarray, layer: int) -> LayerMetrics:
    """Compute all metrics for a single layer."""
    active_mask = _get_active_mask(counts)
    n_active = int(active_mask.sum())
    total_mass = float(sums.sum())

    if n_active == 0:
        return LayerMetrics(
            layer=layer, gini=0.0, n_active=0, total_mass=0.0,
            mean_activation=0.0, median_activation=0.0, p90_activation=0.0,
            max_activation=0.0, heavy_tail_ratio=0.0, n50=0, n90=0,
            frac_active=0.0,
        )

    # Mean activation = sum/count for each active feature
    mean_per_feat = np.zeros_like(sums)
    mean_per_feat[active_mask] = sums[active_mask] / counts[active_mask]
    active_means = mean_per_feat[active_mask]

    gini = gini_coefficient(sums)
    n50 = n_for_mass_fraction(sums, 0.5)
    n90 = n_for_mass_fraction(sums, 0.9)

    return LayerMetrics(
        layer=layer,
        gini=gini,
        n_active=n_active,
        total_mass=total_mass,
        mean_activation=float(np.mean(active_means)),
        median_activation=float(np.median(active_means)),
        p90_activation=float(np.percentile(active_means, 90)),
        max_activation=float(np.max(active_means)),
        heavy_tail_ratio=float(np.percentile(active_means, 90) / (np.median(active_means) + 1e-10)),
        n50=n50,
        n90=n90,
        frac_active=n_active / len(sums),
    )


def compute_all_metrics(stats: FeatureStats) -> dict[int, LayerMetrics]:
    """Compute metrics for all available layers."""
    metrics = {}
    for layer in sorted(stats.sums.keys()):
        if layer in stats.counts:
            metrics[layer] = compute_layer_metrics(
                stats.sums[layer], stats.counts[layer], layer
            )
    return metrics


# ── Aggregate Corrective-Stage Metrics ───────────────────────────────────────

@dataclass
class AggregateMetrics:
    """Summary statistics across corrective layers for one variant."""
    variant: str
    mean_gini: float
    mean_gini_corrective: float
    total_corrective_mass: float
    frac_mass_l32_33: float      # fraction of corrective mass in final 2 layers
    mean_heavy_tail: float       # mean heavy_tail_ratio across corrective layers
    median_n90: int
    total_active_corrective: int
    mean_activation_corrective: float


def compute_aggregate(
    metrics: dict[int, LayerMetrics],
    variant: str,
) -> AggregateMetrics:
    corrective = [m for l, m in metrics.items() if l >= _CORRECTIVE_START]
    all_layers = list(metrics.values())

    corr_mass = sum(m.total_mass for m in corrective)
    l32_33_mass = sum(m.total_mass for m in corrective if m.layer >= 32)

    return AggregateMetrics(
        variant=variant,
        mean_gini=float(np.mean([m.gini for m in all_layers])) if all_layers else 0.0,
        mean_gini_corrective=float(np.mean([m.gini for m in corrective])) if corrective else 0.0,
        total_corrective_mass=corr_mass,
        frac_mass_l32_33=l32_33_mass / (corr_mass + 1e-10),
        mean_heavy_tail=float(np.mean([m.heavy_tail_ratio for m in corrective])) if corrective else 0.0,
        median_n90=int(np.median([m.n90 for m in corrective])) if corrective else 0,
        total_active_corrective=sum(m.n_active for m in corrective),
        mean_activation_corrective=float(np.mean([m.mean_activation for m in corrective])) if corrective else 0.0,
    )


# ── Overlap Analysis ────────────────────────────────────────────────────────

def compute_overlap_curve(
    it_stats: FeatureStats,
    pt_stats: FeatureStats,
    layer_range: range,
    k_values: list[int] | None = None,
) -> dict[int, list[float]]:
    """For each K, compute mean Jaccard overlap across layers in layer_range.

    Returns: {k: [jaccard_per_layer...]} for plotting.
    """
    if k_values is None:
        k_values = [10, 25, 50, 100, 200, 500, 1000]

    result: dict[int, list[float]] = {k: [] for k in k_values}
    layers_used = []

    for layer in layer_range:
        if layer not in it_stats.sums or layer not in pt_stats.sums:
            continue
        layers_used.append(layer)
        it_sums = it_stats.sums[layer]
        pt_sums = pt_stats.sums[layer]
        for k in k_values:
            it_top = top_k_feature_ids(it_sums, k)
            pt_top = top_k_feature_ids(pt_sums, k)
            result[k].append(overlap_jaccard(it_top, pt_top))

    return result


# ── Plotting ─────────────────────────────────────────────────────────────────

_IT_COLOR = "#E65100"
_PT_COLOR = "#1565C0"


def make_plot(
    it_summary_path: str = _IT_SUMMARY,
    pt_summary_path: str = _PT_SUMMARY,
    output_dir: str = "results/exp3/it_16k_l0_big_affine_t512/plots",
) -> None:
    """Generate the 6-panel quantitative sniper feature analysis figure."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────
    print("  Loading IT feature summary ...")
    try:
        it_stats = FeatureStats.from_npz(it_summary_path, "it")
    except FileNotFoundError as e:
        print(f"  E3.14 skipped — {e}")
        return
    print("  Loading PT feature summary ...")
    try:
        pt_stats = FeatureStats.from_npz(pt_summary_path, "pt")
    except FileNotFoundError as e:
        print(f"  E3.14 skipped — {e}")
        return

    # ── Compute metrics ───────────────────────────────────────────────────
    print("  Computing per-layer metrics ...")
    it_metrics = compute_all_metrics(it_stats)
    pt_metrics = compute_all_metrics(pt_stats)

    it_agg = compute_aggregate(it_metrics, "it")
    pt_agg = compute_aggregate(pt_metrics, "pt")

    print("  Computing feature overlap ...")
    k_values = [10, 25, 50, 100, 200, 500, 1000]
    overlap = compute_overlap_curve(
        it_stats, pt_stats,
        range(_CORRECTIVE_START, _N_LAYERS),
        k_values=k_values,
    )

    # Layers present in both
    layers = sorted(set(it_metrics.keys()) & set(pt_metrics.keys()))

    # ── Create figure ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Gini coefficient per layer ───────────────────────────────
    ax_a = fig.add_subplot(gs[0, 0])
    it_gini = [it_metrics[l].gini for l in layers]
    pt_gini = [pt_metrics[l].gini for l in layers]
    ax_a.axvspan(_CORRECTIVE_START - 0.5, _N_LAYERS - 0.5,
                 color="#FFF3E0", alpha=0.5, label="Corrective stage")
    ax_a.plot(layers, it_gini, color=_IT_COLOR, lw=2, marker="o", ms=4, label="IT")
    ax_a.plot(layers, pt_gini, color=_PT_COLOR, lw=2, marker="s", ms=4, label="PT")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Gini coefficient")
    ax_a.set_title(
        "Panel A — Activation Mass Inequality (Gini)\n"
        "Higher = fewer features carry more mass ('sniper')"
    )
    ax_a.legend(fontsize=9)
    ax_a.grid(alpha=0.25)

    # Add mean values as text
    it_corr_gini = np.mean([it_metrics[l].gini for l in layers if l >= _CORRECTIVE_START])
    pt_corr_gini = np.mean([pt_metrics[l].gini for l in layers if l >= _CORRECTIVE_START])
    ax_a.text(0.02, 0.98,
              f"Corrective mean:\n  IT: {it_corr_gini:.3f}\n  PT: {pt_corr_gini:.3f}",
              transform=ax_a.transAxes, va="top", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    # ── Panel B: Activation magnitude distributions (corrective layers) ──
    ax_b = fig.add_subplot(gs[0, 1])

    # Collect mean activations (sum/count) for all active features in corrective layers
    it_activations = []
    pt_activations = []
    for layer in range(_CORRECTIVE_START, _N_LAYERS):
        if layer in it_stats.sums and layer in it_stats.counts:
            s, c = it_stats.sums[layer], it_stats.counts[layer]
            mask = c > 0
            if mask.any():
                it_activations.extend((s[mask] / c[mask]).tolist())
        if layer in pt_stats.sums and layer in pt_stats.counts:
            s, c = pt_stats.sums[layer], pt_stats.counts[layer]
            mask = c > 0
            if mask.any():
                pt_activations.extend((s[mask] / c[mask]).tolist())

    it_act = np.array(it_activations)
    pt_act = np.array(pt_activations)

    # Log-scale violin plots
    it_log = np.log10(it_act[it_act > 0] + 1e-10)
    pt_log = np.log10(pt_act[pt_act > 0] + 1e-10)

    parts = ax_b.violinplot([it_log, pt_log], positions=[1, 2], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor([_IT_COLOR, _PT_COLOR][i])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")

    ax_b.set_xticks([1, 2])
    ax_b.set_xticklabels(["IT", "PT"])
    ax_b.set_ylabel("log₁₀(mean activation per feature)")
    ax_b.set_title(
        "Panel B — Activation Magnitude Distribution\n"
        f"Corrective layers {_CORRECTIVE_START}–{_N_LAYERS-1} (active features only)"
    )
    ax_b.grid(axis="y", alpha=0.25)

    # Add statistics
    it_med = np.median(it_act[it_act > 0])
    pt_med = np.median(pt_act[pt_act > 0])
    it_p90 = np.percentile(it_act[it_act > 0], 90)
    pt_p90 = np.percentile(pt_act[pt_act > 0], 90)
    ax_b.text(0.02, 0.98,
              f"Median: IT={it_med:.0f}, PT={pt_med:.0f} ({it_med/pt_med:.1f}×)\n"
              f"P90: IT={it_p90:.0f}, PT={pt_p90:.0f} ({it_p90/pt_p90:.1f}×)\n"
              f"Active feats: IT={len(it_act)}, PT={len(pt_act)}",
              transform=ax_b.transAxes, va="top", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    # ── Panel C: Layer concentration (fraction of total mass per layer) ──
    ax_c = fig.add_subplot(gs[1, 0])
    it_mass_per_layer = np.array([it_metrics[l].total_mass if l in it_metrics else 0 for l in range(_N_LAYERS)])
    pt_mass_per_layer = np.array([pt_metrics[l].total_mass if l in pt_metrics else 0 for l in range(_N_LAYERS)])

    # Normalize to fraction of corrective mass only
    it_corr_total = it_mass_per_layer[_CORRECTIVE_START:].sum()
    pt_corr_total = pt_mass_per_layer[_CORRECTIVE_START:].sum()
    it_frac = it_mass_per_layer / (it_corr_total + 1e-10)
    pt_frac = pt_mass_per_layer / (pt_corr_total + 1e-10)

    width = 0.35
    x = np.arange(_CORRECTIVE_START, _N_LAYERS)
    ax_c.bar(x - width/2, it_frac[_CORRECTIVE_START:], width,
             color=_IT_COLOR, alpha=0.85, label="IT")
    ax_c.bar(x + width/2, pt_frac[_CORRECTIVE_START:], width,
             color=_PT_COLOR, alpha=0.85, label="PT")
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel("Fraction of corrective-stage mass")
    ax_c.set_title(
        "Panel C — Layer Concentration of Activation Mass\n"
        "Where does the corrective computation concentrate?"
    )
    ax_c.legend(fontsize=9)
    ax_c.grid(axis="y", alpha=0.25)

    it_l32_33 = it_frac[32:34].sum()
    pt_l32_33 = pt_frac[32:34].sum()
    ax_c.text(0.02, 0.98,
              f"L32-33 share:\n  IT: {it_l32_33:.1%}\n  PT: {pt_l32_33:.1%}",
              transform=ax_c.transAxes, va="top", fontsize=8,
              bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    # ── Panel D: Feature overlap (Jaccard) at varying K ──────────────────
    ax_d = fig.add_subplot(gs[1, 1])
    mean_jaccard = []
    for k in k_values:
        vals = overlap[k]
        mean_jaccard.append(np.mean(vals) if vals else 0.0)

    ax_d.plot(k_values, mean_jaccard, color="#4CAF50", lw=2, marker="D", ms=5)
    ax_d.set_xlabel("Top-K features (by activation mass)")
    ax_d.set_ylabel("Mean Jaccard overlap (IT ∩ PT)")
    ax_d.set_xscale("log")
    ax_d.set_title(
        "Panel D — IT vs PT Feature Overlap\n"
        f"Mean Jaccard across corrective layers {_CORRECTIVE_START}–{_N_LAYERS-1}"
    )
    ax_d.grid(alpha=0.25)
    ax_d.set_ylim(0, 1)

    # Annotate key points
    for k, j in zip(k_values, mean_jaccard):
        if k in [50, 100, 500]:
            ax_d.annotate(f"K={k}: {j:.2f}", (k, j),
                         textcoords="offset points", xytext=(8, 8), fontsize=8)

    # ── Panel E: Cumulative mass concentration curves ─────────────────────
    ax_e = fig.add_subplot(gs[2, 0])

    # Average across corrective layers
    it_cum_interp = []
    pt_cum_interp = []
    x_grid = np.linspace(0, 1, 500)

    for layer in range(_CORRECTIVE_START, _N_LAYERS):
        if layer in it_stats.sums:
            x, y = cumulative_mass_curve(it_stats.sums[layer])
            it_cum_interp.append(np.interp(x_grid, x, y))
        if layer in pt_stats.sums:
            x, y = cumulative_mass_curve(pt_stats.sums[layer])
            pt_cum_interp.append(np.interp(x_grid, x, y))

    if it_cum_interp:
        it_mean_cum = np.mean(it_cum_interp, axis=0)
        it_std_cum = np.std(it_cum_interp, axis=0)
        ax_e.plot(x_grid * 100, it_mean_cum, color=_IT_COLOR, lw=2, label="IT mean")
        ax_e.fill_between(x_grid * 100, it_mean_cum - it_std_cum,
                         it_mean_cum + it_std_cum, color=_IT_COLOR, alpha=0.15)
    if pt_cum_interp:
        pt_mean_cum = np.mean(pt_cum_interp, axis=0)
        pt_std_cum = np.std(pt_cum_interp, axis=0)
        ax_e.plot(x_grid * 100, pt_mean_cum, color=_PT_COLOR, lw=2, label="PT mean")
        ax_e.fill_between(x_grid * 100, pt_mean_cum - pt_std_cum,
                         pt_mean_cum + pt_std_cum, color=_PT_COLOR, alpha=0.15)

    ax_e.plot([0, 100], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect equality")
    ax_e.set_xlabel("% of active features (sorted by mass, descending)")
    ax_e.set_ylabel("Cumulative fraction of total mass")
    ax_e.set_title(
        "Panel E — Cumulative Mass Concentration\n"
        "Steeper curve = more concentrated ('sniper')"
    )
    ax_e.legend(fontsize=9)
    ax_e.grid(alpha=0.25)
    ax_e.set_xlim(0, 50)  # Zoom into the interesting region

    # Mark 50% and 90% mass thresholds
    if it_cum_interp:
        it_n50_pct = x_grid[np.searchsorted(it_mean_cum, 0.5)] * 100
        it_n90_pct = x_grid[np.searchsorted(it_mean_cum, 0.9)] * 100
        ax_e.axhline(0.5, color="grey", ls=":", lw=0.8)
        ax_e.axhline(0.9, color="grey", ls=":", lw=0.8)

        stats_text = (
            f"50% mass at:\n"
            f"  IT: {it_n50_pct:.1f}% of features\n"
        )
        if pt_cum_interp:
            pt_n50_pct = x_grid[np.searchsorted(pt_mean_cum, 0.5)] * 100
            pt_n90_pct = x_grid[np.searchsorted(pt_mean_cum, 0.9)] * 100
            stats_text += f"  PT: {pt_n50_pct:.1f}% of features"

        ax_e.text(0.98, 0.02, stats_text,
                  transform=ax_e.transAxes, va="bottom", ha="right", fontsize=8,
                  bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey"))

    # ── Panel F: Summary statistics table ─────────────────────────────────
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.axis("off")

    # Compute N50/N90 for corrective layers
    it_n50s = [it_metrics[l].n50 for l in range(_CORRECTIVE_START, _N_LAYERS) if l in it_metrics]
    pt_n50s = [pt_metrics[l].n50 for l in range(_CORRECTIVE_START, _N_LAYERS) if l in pt_metrics]
    it_n90s = [it_metrics[l].n90 for l in range(_CORRECTIVE_START, _N_LAYERS) if l in it_metrics]
    pt_n90s = [pt_metrics[l].n90 for l in range(_CORRECTIVE_START, _N_LAYERS) if l in pt_metrics]

    # Feature firing rates
    it_active_counts = [it_metrics[l].n_active for l in range(_CORRECTIVE_START, _N_LAYERS) if l in it_metrics]
    pt_active_counts = [pt_metrics[l].n_active for l in range(_CORRECTIVE_START, _N_LAYERS) if l in pt_metrics]

    # Jaccard at K=100
    j100 = np.mean(overlap.get(100, [0.0]))

    table_data = [
        ["Metric", "IT", "PT", "Ratio"],
        ["", "", "", ""],
        ["Gini (corrective mean)", f"{it_agg.mean_gini_corrective:.3f}", f"{pt_agg.mean_gini_corrective:.3f}",
         f"{it_agg.mean_gini_corrective / (pt_agg.mean_gini_corrective + 1e-10):.2f}×"],
        ["Median activation", f"{it_med:.0f}", f"{pt_med:.0f}", f"{it_med / (pt_med + 1e-10):.1f}×"],
        ["P90 activation", f"{it_p90:.0f}", f"{pt_p90:.0f}", f"{it_p90 / (pt_p90 + 1e-10):.1f}×"],
        ["Heavy-tail ratio (P90/med)", f"{it_agg.mean_heavy_tail:.1f}", f"{pt_agg.mean_heavy_tail:.1f}",
         f"{it_agg.mean_heavy_tail / (pt_agg.mean_heavy_tail + 1e-10):.2f}×"],
        ["Mass in L32-33", f"{it_l32_33:.1%}", f"{pt_l32_33:.1%}", "—"],
        ["Median N₅₀ (corrective)", f"{int(np.median(it_n50s))}", f"{int(np.median(pt_n50s))}",
         f"{np.median(it_n50s) / (np.median(pt_n50s) + 1e-10):.2f}×"],
        ["Median N₉₀ (corrective)", f"{int(np.median(it_n90s))}", f"{int(np.median(pt_n90s))}",
         f"{np.median(it_n90s) / (np.median(pt_n90s) + 1e-10):.2f}×"],
        ["Mean active features/layer", f"{np.mean(it_active_counts):.0f}", f"{np.mean(pt_active_counts):.0f}",
         f"{np.mean(it_active_counts) / (np.mean(pt_active_counts) + 1e-10):.2f}×"],
        ["Jaccard overlap (K=100)", "", "", f"{j100:.3f}"],
    ]

    # Draw table
    table = ax_f.table(
        cellText=[row for row in table_data],
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.18, 0.18, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Style header row
    for j in range(4):
        table[0, j].set_facecolor("#E0E0E0")
        table[0, j].set_text_props(fontweight="bold")
        table[1, j].set_height(0.01)
        table[1, j].set_facecolor("white")
        table[1, j].set_edgecolor("white")

    # Alternate row colors
    for i in range(2, len(table_data)):
        color = "#FAFAFA" if i % 2 == 0 else "white"
        for j in range(4):
            table[i, j].set_facecolor(color)

    ax_f.set_title(
        "Panel F — Summary Statistics\n"
        "Quantitative comparison of feature activation distributions",
        fontsize=10,
    )

    # ── Suptitle and save ─────────────────────────────────────────────────
    fig.suptitle(
        "E3.14 — Quantitative Sniper Feature Analysis\n"
        "IT's corrective features are sparser, higher-magnitude, and more layer-concentrated than PT's",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    out_path = Path(output_dir) / "plot_e3_14_sniper_quantitative.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  E3.14 saved → {out_path}")

    # ── Also save numeric results as JSON for paper ──────────────────────
    import json
    stats_out = {
        "it": {
            "gini_corrective_mean": round(it_agg.mean_gini_corrective, 4),
            "median_activation": round(float(it_med), 1),
            "p90_activation": round(float(it_p90), 1),
            "heavy_tail_ratio": round(it_agg.mean_heavy_tail, 2),
            "frac_mass_l32_33": round(float(it_l32_33), 4),
            "median_n50": int(np.median(it_n50s)),
            "median_n90": int(np.median(it_n90s)),
            "mean_active_per_layer": round(float(np.mean(it_active_counts)), 0),
        },
        "pt": {
            "gini_corrective_mean": round(pt_agg.mean_gini_corrective, 4),
            "median_activation": round(float(pt_med), 1),
            "p90_activation": round(float(pt_p90), 1),
            "heavy_tail_ratio": round(pt_agg.mean_heavy_tail, 2),
            "frac_mass_l32_33": round(float(pt_l32_33), 4),
            "median_n50": int(np.median(pt_n50s)),
            "median_n90": int(np.median(pt_n90s)),
            "mean_active_per_layer": round(float(np.mean(pt_active_counts)), 0),
        },
        "overlap_jaccard_k100_mean": round(float(j100), 4),
        "overlap_by_k": {str(k): round(np.mean(v), 4) for k, v in overlap.items()},
    }
    json_path = Path(output_dir) / "e3_14_sniper_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"  E3.14 stats → {json_path}")

    # ── Print summary to console ─────────────────────────────────────────
    print("\n  ═══ E3.14 Sniper Feature Summary ═══")
    print(f"  Gini (corrective):  IT={it_agg.mean_gini_corrective:.3f}  PT={pt_agg.mean_gini_corrective:.3f}")
    print(f"  Median activation:  IT={it_med:.0f}  PT={pt_med:.0f}  ({it_med/pt_med:.1f}×)")
    print(f"  P90 activation:     IT={it_p90:.0f}  PT={pt_p90:.0f}  ({it_p90/pt_p90:.1f}×)")
    print(f"  Mass in L32-33:     IT={it_l32_33:.1%}  PT={pt_l32_33:.1%}")
    print(f"  Median N₅₀:        IT={int(np.median(it_n50s))}  PT={int(np.median(pt_n50s))}")
    print(f"  Median N₉₀:        IT={int(np.median(it_n90s))}  PT={int(np.median(pt_n90s))}")
    print(f"  Overlap (K=100):    Jaccard={j100:.3f}")
    print(f"  Active feats/layer: IT={np.mean(it_active_counts):.0f}  PT={np.mean(pt_active_counts):.0f}")
    print()

    return stats_out


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="E3.14 Quantitative Sniper Feature Analysis")
    parser.add_argument("--it-summary", default=_IT_SUMMARY)
    parser.add_argument("--pt-summary", default=_PT_SUMMARY)
    parser.add_argument("--output-dir", default="results/exp3/it_16k_l0_big_affine_t512/plots")
    args = parser.parse_args()
    make_plot(args.it_summary, args.pt_summary, args.output_dir)
