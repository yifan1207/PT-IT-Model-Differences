"""
Plot 1: log(N₉₀) vs log(attribution) — the core attribution bias test.

For each feature f:
  c(f) = activation(f) × (W_dec[f] @ W_U)           [vocab_size]
  N₉₀(f) = fewest tokens whose |c(f)| values sum to ≥90% of Σ|c(f)|
  attribution(f) = |activation(f) × (W_dec[f] @ W_U)[target]|

Hypothesis: if broad features (high N₉₀) get systematically higher attribution,
the regression slope is positive → attribution bias is real.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, linregress

GROUP_COLORS = {"M": "#e74c3c", "R": "#3498db"}
FALLBACK_COLOR = "#95a5a6"


def make_plot(results: list[dict], output_path: str) -> None:
    xs, ys, colors = [], [], []

    for r in results:
        color = GROUP_COLORS.get(r["prompt_id"][0], FALLBACK_COLOR)
        for f in r["features"]:
            n90 = f["n90"]
            attr = f["attribution"]
            if n90 <= 0 or attr <= 0:
                continue  # skip zero-attribution features (can't log-transform)
            xs.append(np.log(n90))
            ys.append(np.log(attr))
            colors.append(color)

    if len(xs) < 2:
        print("  Plot 1: not enough data points, skipping.")
        return

    xs = np.array(xs)
    ys = np.array(ys)

    r_val, p_val = pearsonr(xs, ys)
    slope, intercept, *_ = linregress(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.3, s=12, linewidths=0)

    x_line = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", lw=1.5,
            label=f"OLS  slope={slope:+.3f}  r={r_val:+.3f}  p={p_val:.1e}")

    for g, c in GROUP_COLORS.items():
        ax.scatter([], [], c=c, alpha=0.8, s=30, label=f"Group {g}")

    ax.set_xlabel("log(N₉₀)  [tokens for 90% of |contribution| mass]", fontsize=11)
    ax.set_ylabel("log(attribution)  [log |activation × logit_target|]", fontsize=11)
    ax.set_title("Plot 1: Do broad features get higher attribution? (core bias test)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 1 saved → {output_path}")
