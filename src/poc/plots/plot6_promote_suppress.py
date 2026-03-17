"""
Plot 6: promote_ratio vs N₉₀ — how do features narrow the distribution?

Features narrow the output distribution in two ways:
  1. Promotion: pushing probability mass onto specific tokens (promote_ratio ≈ 1.0)
  2. Suppression: removing probability mass from many tokens (promote_ratio ≈ 0.0)

promote_ratio(f) = Σmax(c(f),0) / Σ|c(f)|
  = 1.0 → pure promotion  (all contribution is positive: "vote FOR these tokens")
  = 0.0 → pure suppression (all contribution is negative: "vote AGAINST these tokens")
  = 0.5 → mixed

x-axis: promote_ratio
y-axis: log(N₉₀)        [broadness of the contribution]
color:  attribution magnitude

Reveals: do suppression features tend to be broader or narrower than promotion features?
Are there natural clusters (suppressor vs promoter vs mixed)?
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path


def make_plot(results: list[dict], output_path: str) -> None:
    xs, ys, attribs = [], [], []

    for r in results:
        for f in r["features"]:
            n90 = f["n90"]
            attr = f["attribution"]
            if n90 <= 0 or attr <= 0:
                continue
            xs.append(f["promote_ratio"])
            ys.append(np.log(n90))
            attribs.append(attr)

    if len(xs) < 2:
        print("  Plot 6: not enough data points, skipping.")
        return

    xs = np.array(xs)
    ys = np.array(ys)
    attribs = np.array(attribs)

    vmin, vmax = attribs.min(), attribs.max()
    norm = LogNorm(vmin=vmin, vmax=vmax) if vmin < vmax else None

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(xs, ys, c=attribs, cmap="plasma", alpha=0.4, s=15, linewidths=0,
                    norm=norm, rasterized=True)
    plt.colorbar(sc, ax=ax, label="attribution  |activation × logit_target|")

    ax.set_xlabel("promote_ratio  [Σmax(c,0) / Σ|c|]  — 0=suppress, 1=promote", fontsize=11)
    ax.set_ylabel("log(N₉₀)  [broadness of contribution]", fontsize=11)
    ax.set_title("Plot 6: How do features narrow — promotion vs suppression?", fontsize=12)
    ax.axvline(0.5, color="gray", lw=0.8, linestyle="--", alpha=0.5, label="mixed (0.5)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 6 saved → {output_path}")
