"""
Continuous correlation analysis: broadness (N₉₀) vs. attribution magnitude.

The research hypothesis (Claim 1 — Attribution Inversion):
  Broad features (high N₉₀) get systematically HIGHER attribution than specific features
  (low N₉₀), even though the latter do harder computation.

We test this as a regression:
  x = log(N₉₀(f))      — unsigned broadness; high = spreads contribution across many tokens
  y = log(attribution)  = log(|activation(f) × logit_target(f)|)

If the slope is POSITIVE, broad features have higher attribution → bias confirmed.

Note: N₉₀ is used instead of signed specificity (logit_target / logit_norm) because
specificity conflates two orthogonal dimensions — broadness (specificity ≈ 0) and
suppression direction (specificity < 0). A strong suppressor scores low on signed
specificity but is not broad; using it as a proxy for broadness confounds the test.
N₉₀ directly measures distributional breadth regardless of promotion/suppression direction.

Limitation: the same (layer, feature_idx) can appear across multiple prompts. Pooled
p-values treat each occurrence as independent, which overstates significance. Treat
p-values as indicative, not exact, for this POC.
"""
import json
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr, linregress


def build_result(prompt: str, prompt_id: str, correct_token: str,
                 records: list, elapsed: float) -> dict:
    return {
        "prompt": prompt,
        "prompt_id": prompt_id,
        "correct_token": correct_token,
        "elapsed_s": round(elapsed, 1),
        "n_features": len(records),
        "features": [asdict(r) for r in records],
    }


def run_regression(all_results: list[dict]) -> tuple[dict, np.ndarray, np.ndarray, list[str]]:
    """
    Pool all (log_n90, log_attribution) pairs across all prompts and run correlation analysis.
    Skips features with n90 <= 0 (sentinel) or attribution <= 0 (can't log-transform).
    Returns a stats dict and prints a summary.
    """
    xs, ys, group_labels = [], [], []
    for r in all_results:
        for f in r["features"]:
            n90 = f["n90"]
            attr = f["attribution"]
            if n90 <= 0 or attr <= 0:
                continue
            xs.append(math.log(n90))
            ys.append(math.log(attr))
            group_labels.append(r["prompt_id"])

    if len(xs) < 2:
        raise ValueError(
            f"Need ≥2 features for regression, got {len(xs)}. "
            "Check that attribution ran successfully and features were collected."
        )

    xs = np.array(xs)
    ys = np.array(ys)

    pearson_r, pearson_p = pearsonr(xs, ys)
    spearman_r, spearman_p = spearmanr(xs, ys)
    slope, intercept, r_value, p_value, std_err = linregress(xs, ys)

    # Replace NaN stats (e.g. constant xs/ys) with None for valid JSON serialization
    def _safe(v: float) -> float | None:
        return None if (v is None or math.isnan(v) or math.isinf(v)) else v

    stats = {
        "n_features_total": len(xs),
        "n_prompts": len(all_results),
        "pearson_r": _safe(round(float(pearson_r), 4)),
        "pearson_p": _safe(float(pearson_p)),
        "spearman_r": _safe(round(float(spearman_r), 4)),
        "spearman_p": _safe(float(spearman_p)),
        "ols_slope": _safe(round(float(slope), 6)),
        "ols_intercept": _safe(round(float(intercept), 6)),
        "ols_r_squared": _safe(round(float(r_value ** 2), 4)),
        "ols_p_value": _safe(float(p_value)),
        "log_n90_mean": round(float(xs.mean()), 4),
        "log_n90_std": round(float(xs.std()), 4),
        "log_attribution_mean": round(float(ys.mean()), 6),
        "log_attribution_std": round(float(ys.std()), 6),
    }

    _print_regression_summary(stats)
    return stats, xs, ys, group_labels


def _print_regression_summary(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("REGRESSION: log(N₉₀)  →  log(attribution magnitude)")
    print("=" * 60)
    print(f"  N features pooled    : {stats['n_features_total']} across {stats['n_prompts']} prompts")
    print(f"  log(N₉₀) range      : mean={stats['log_n90_mean']:.4f}  std={stats['log_n90_std']:.4f}")
    print(f"  log(attr) range     : mean={stats['log_attribution_mean']:.6f}  std={stats['log_attribution_std']:.6f}")
    print()
    print(f"  Pearson  r = {stats['pearson_r']:+.4f}   p = {stats['pearson_p']:.2e}")
    print(f"  Spearman r = {stats['spearman_r']:+.4f}   p = {stats['spearman_p']:.2e}")
    print(f"  OLS slope  = {stats['ols_slope']:+.6f}   R² = {stats['ols_r_squared']:.4f}   p = {stats['ols_p_value']:.2e}")
    print()

    if stats["pearson_r"] is not None:
        if stats["pearson_r"] > 0.1 and stats["pearson_p"] < 0.05:
            print("  ✓ ATTRIBUTION BIAS CONFIRMED: broad features (high N₉₀) have higher attribution.")
            print("    High-entropy features dominate attribution; specific computation is invisible.")
        elif stats["pearson_r"] > 0:
            print("  ~ Positive trend present but weak or not significant.")
        else:
            print("  ✗ No positive correlation. Hypothesis not supported (or thresholds need checking).")
    else:
        print("  ~ Regression produced NaN (constant input?). Check data.")
    print("=" * 60)


def save_scatter_plot(xs: np.ndarray, ys: np.ndarray, group_labels: list[str],
                      stats: dict, path: str) -> None:
    """Scatter plot of log(N₉₀) vs log(attribution), colored by prompt group (M/R)."""
    group_colors = {"M": "#e74c3c", "R": "#3498db"}
    default_color = "#95a5a6"

    fig, ax = plt.subplots(figsize=(8, 5))

    # One vectorized scatter call per group (not one per point — that would be 29k calls)
    labels_arr = np.array([lbl[0] if lbl else "?" for lbl in group_labels])
    for group, color in group_colors.items():
        mask = labels_arr == group
        if mask.any():
            ax.scatter(xs[mask], ys[mask], color=color, alpha=0.35, s=12,
                       linewidths=0, rasterized=True, label=f"Group {group}")
    other = ~np.isin(labels_arr, list(group_colors))
    if other.any():
        ax.scatter(xs[other], ys[other], color=default_color, alpha=0.35, s=12,
                   linewidths=0, rasterized=True)

    # OLS regression line
    x_line = np.linspace(xs.min(), xs.max(), 200)
    y_line = stats["ols_slope"] * x_line + stats["ols_intercept"]
    ax.plot(x_line, y_line, color="black", linewidth=1.5, label=(
        f"OLS  slope={stats['ols_slope']:+.4f}  "
        f"R²={stats['ols_r_squared']:.3f}  "
        f"Pearson r={stats['pearson_r']:+.3f}"
    ))

    # (legend labels already set in scatter calls above)

    ax.set_xlabel("log(N₉₀)  [tokens for 90% of |contribution| mass]", fontsize=11)
    ax.set_ylabel("log(attribution)  [log |activation × logit_target|]", fontsize=11)
    ax.set_title("Attribution Inversion: do broad features dominate attribution?", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Scatter plot saved → {path}")


def save_results(all_results: list[dict], stats: dict, output_path: str) -> None:
    out = {"regression_stats": stats, "prompts": all_results}
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(out, indent=2))
    print(f"  Results saved → {output_path}")
