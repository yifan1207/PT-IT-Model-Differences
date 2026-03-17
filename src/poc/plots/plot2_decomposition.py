"""
Plot 2: Total impact vs target efficiency — attribution decomposition.

attribution(f) = activation(f) × (W_dec[f] @ W_U)[target]
              = total_impact(f) × target_efficiency(f)

where:
  total_impact(f)     = |activation(f)| × ||W_dec[f] @ W_U||₂
                        (how hard does the feature push the full distribution?)
  target_efficiency(f) = |(W_dec[f] @ W_U)[target]| / ||W_dec[f] @ W_U||₂
                        (what fraction of that push lands on the target token?)

Both are computable from stored FeatureRecord fields:
  total_impact     = abs(activation) × logit_norm
  target_efficiency = abs(logit_target) / logit_norm

Quadrant interpretation:
  top-right:   strong AND precise → high attribution (easy to see)
  bottom-right: strong but imprecise → high attribution (broad narrowing features)
  top-left:    precise but weak → LOW attribution (hidden reasoning features)
  bottom-left: weak and imprecise → low attribution
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path


def make_plot(results: list[dict], output_path: str) -> None:
    xs, ys, attribs = [], [], []

    for r in results:
        for f in r["features"]:
            logit_norm = f["logit_norm"]
            if logit_norm <= 0:
                continue
            total_impact = abs(f["activation"]) * logit_norm
            target_eff = abs(f["logit_target"]) / logit_norm
            attr = f["attribution"]
            if total_impact <= 0 or attr <= 0:
                continue
            xs.append(total_impact)
            ys.append(target_eff)
            attribs.append(attr)

    if len(xs) < 2:
        print("  Plot 2: not enough data points, skipping.")
        return

    xs = np.array(xs)
    ys = np.array(ys)
    attribs = np.array(attribs)

    vmin, vmax = attribs.min(), attribs.max()
    norm = LogNorm(vmin=vmin, vmax=vmax) if vmin < vmax else None

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=attribs, cmap="viridis", alpha=0.4, s=15, linewidths=0,
                    norm=norm, rasterized=True)
    plt.colorbar(sc, ax=ax, label="attribution  |activation × logit_target|")

    ax.set_xscale("log")
    ax.set_xlabel("total_impact  =  |activation| × ||W_dec[f] @ W_U||₂", fontsize=11)
    ax.set_ylabel("target_efficiency  =  |logit_target| / ||W_dec[f] @ W_U||₂", fontsize=11)
    ax.set_title("Plot 2: Decomposing attribution into impact × efficiency", fontsize=12)
    ax.text(0.02, 0.98,
            "top-left: precise but weak\n"
            "top-right: precise AND strong\n"
            "bottom-right: strong but imprecise",
            transform=ax.transAxes, fontsize=8, va="top", alpha=0.65)
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 2 saved → {output_path}")
