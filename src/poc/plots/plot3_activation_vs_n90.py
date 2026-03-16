"""
Plot 3: |activation| vs log(N₉₀) — does the model activate broad features more?

Separates two explanations for attribution bias:
  Explanation A: broad features get high attribution because the model fires them more
                 strongly. If so, plot3 shows a positive correlation (high N₉₀ → high
                 activation). Attribution bias is the model's inherent preference.
  Explanation B: broad features get high attribution even at equal activation due to
                 the math of the attribution formula. If so, plot3 shows no correlation.
                 Attribution bias is a measurement artifact.

Both are scientifically interesting but imply very different interpretations.
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
            act = abs(f["activation"])
            if n90 <= 0 or act == 0:
                continue
            xs.append(np.log(n90))
            ys.append(act)
            colors.append(color)

    if len(xs) < 2:
        print("  Plot 3: not enough data points, skipping.")
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

    ax.set_xlabel("log(N₉₀)  [broadness of feature's contribution]", fontsize=11)
    ax.set_ylabel("|activation|  [how strongly the feature fired]", fontsize=11)
    ax.set_title("Plot 3: Does the model fire broad features more strongly?", fontsize=12)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 3 saved → {output_path}")
