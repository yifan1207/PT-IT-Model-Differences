"""
Plot 8: log(N₉₀) vs target_efficiency — do broad features waste their effect?

target_efficiency(f) = |logit_target| / logit_norm
                     = |contribution[target]| / ||contribution||₂

This is the fraction of a feature's total distributional force that lands on the
correct answer token. A value near 1 means the feature is almost entirely focused
on the target; near 0 means it spreads its force across many tokens.

Combined reading with Plot 1:
  Plot 1 shows: broad features (high N₉₀) get HIGH attribution.
  Plot 8 shows: broad features (high N₉₀) have LOW target efficiency.
  → Together: broad features dominate attribution NOT because they push the
    correct token well, but because they fire strongly and push everything.
    They "win" the attribution metric despite being inefficient at the task.

Hypothesis: strong negative slope → confirmed.
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
            logit_norm = f["logit_norm"]
            if n90 <= 0 or logit_norm <= 0:
                continue
            target_eff = abs(f["logit_target"]) / logit_norm
            xs.append(np.log(n90))
            ys.append(target_eff)
            colors.append(color)

    if len(xs) < 2:
        print("  Plot 8: not enough data points, skipping.")
        return

    xs = np.array(xs)
    ys = np.array(ys)

    r_val, p_val = pearsonr(xs, ys)
    slope, intercept, *_ = linregress(xs, ys)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs, ys, c=colors, alpha=0.3, s=12, linewidths=0, rasterized=True)

    x_line = np.linspace(xs.min(), xs.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="black", lw=1.5,
            label=f"OLS  slope={slope:+.3f}  r={r_val:+.3f}  p={p_val:.1e}")

    for g, c in GROUP_COLORS.items():
        ax.scatter([], [], c=c, alpha=0.8, s=30, label=f"Group {g}")

    ax.set_xlabel("log(N₉₀)  [broadness — tokens for 90% of |contribution| mass]", fontsize=11)
    ax.set_ylabel("target_efficiency  =  |logit_target| / logit_norm", fontsize=11)
    ax.set_title(
        "Plot 8: Do broad features waste their effect on non-target tokens?\n"
        "(negative slope → broad features are inefficient at promoting the correct answer)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 8 saved → {output_path}")
