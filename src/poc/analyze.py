"""
Continuous correlation analysis: specificity vs. attribution magnitude.

The research hypothesis (Claim 1 — Attribution Inversion):
  High-entropy (low-specificity) features get systematically HIGHER attribution than
  low-entropy (high-specificity) features, even though the latter do harder computation.

We test this as a regression:
  x = specificity(f) = logit_target / ||logit_vec||  (~-1 = suppresses target, ~0 = broad, ~+1 = promotes target)
  y = attribution(f) = |activation(f) × logit_target|

If the slope is NEGATIVE, high-specificity features have smaller attribution → bias confirmed.
"""
import json
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


def run_regression(all_results: list[dict]) -> dict:
    """
    Pool all (specificity, attribution) pairs across all prompts and run correlation analysis.
    Returns a stats dict and prints a summary.
    """
    xs, ys, group_labels = [], [], []
    for r in all_results:
        for f in r["features"]:
            xs.append(f["specificity"])
            ys.append(f["attribution"])
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

    stats = {
        "n_features_total": len(xs),
        "n_prompts": len(all_results),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": float(pearson_p),
        "spearman_r": round(float(spearman_r), 4),
        "spearman_p": float(spearman_p),
        "ols_slope": round(float(slope), 6),
        "ols_intercept": round(float(intercept), 6),
        "ols_r_squared": round(float(r_value ** 2), 4),
        "ols_p_value": float(p_value),
        "specificity_mean": round(float(xs.mean()), 4),
        "specificity_std": round(float(xs.std()), 4),
        "attribution_mean": round(float(ys.mean()), 6),
        "attribution_std": round(float(ys.std()), 6),
    }

    _print_regression_summary(stats)
    return stats, xs, ys, group_labels


def _print_regression_summary(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("REGRESSION: specificity  →  attribution magnitude")
    print("=" * 60)
    print(f"  N features pooled    : {stats['n_features_total']} across {stats['n_prompts']} prompts")
    print(f"  Specificity range    : mean={stats['specificity_mean']:.4f}  std={stats['specificity_std']:.4f}")
    print(f"  Attribution range    : mean={stats['attribution_mean']:.6f}  std={stats['attribution_std']:.6f}")
    print()
    print(f"  Pearson  r = {stats['pearson_r']:+.4f}   p = {stats['pearson_p']:.2e}")
    print(f"  Spearman r = {stats['spearman_r']:+.4f}   p = {stats['spearman_p']:.2e}")
    print(f"  OLS slope  = {stats['ols_slope']:+.6f}   R² = {stats['ols_r_squared']:.4f}   p = {stats['ols_p_value']:.2e}")
    print()

    if stats["pearson_r"] < -0.1 and stats["pearson_p"] < 0.05:
        print("  ✓ ATTRIBUTION BIAS CONFIRMED: specificity negatively correlates with attribution.")
        print("    High-entropy (broad) features dominate attribution; hard computation is invisible.")
    elif stats["pearson_r"] < 0:
        print("  ~ Negative trend present but weak or not significant.")
    else:
        print("  ✗ No negative correlation. Hypothesis not supported (or thresholds need checking).")
    print("=" * 60)


def save_scatter_plot(xs: np.ndarray, ys: np.ndarray, group_labels: list[str],
                      stats: dict, path: str) -> None:
    """Scatter plot of specificity vs attribution, colored by prompt group (M/R)."""
    group_colors = {"M": "#e74c3c", "R": "#3498db"}
    default_color = "#95a5a6"

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, x, y in zip(group_labels, xs, ys):
        group = label[0] if label else "?"
        color = group_colors.get(group, default_color)
        ax.scatter(x, y, color=color, alpha=0.35, s=12)

    # OLS regression line
    x_line = np.linspace(xs.min(), xs.max(), 200)
    y_line = stats["ols_slope"] * x_line + stats["ols_intercept"]
    ax.plot(x_line, y_line, color="black", linewidth=1.5, label=(
        f"OLS  slope={stats['ols_slope']:+.4f}  "
        f"R²={stats['ols_r_squared']:.3f}  "
        f"Pearson r={stats['pearson_r']:+.3f}"
    ))

    # Legend for groups
    for group, color in group_colors.items():
        ax.scatter([], [], color=color, alpha=0.7, s=30,
                   label=f"Group {group}")

    ax.set_xlabel("Specificity  (logit_target / ||logit_vec||)", fontsize=11)
    ax.set_ylabel("Attribution magnitude  |activation × logit_target|", fontsize=11)
    ax.set_title("Attribution Inversion: does specificity predict attribution?", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
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
