"""
Plot 7: Token position × layer, colored by log(N₉₀), sized by attribution.

Each dot is a feature:
  x     = token position in the prompt (0 = first token)
  y     = layer (0 to n_layers-1)
  color = log(N₉₀)  — cool = narrow/specific, warm = broad/distributional
  size  = attribution magnitude (normalized to a visible range)

Tests the temporal-spatial hypothesis from the logit-lens literature:
  → Early/middle layers: broad features (high N₉₀, warm color) — "what type of answer?"
  → Late layers: specific features (low N₉₀, cool color)       — "which specific answer?"

Also tests positional structure:
  → Do broad features activate at earlier token positions (operator tokens)?
  → Do specific features activate near the answer-generating position (last token)?
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def make_plot(results: list[dict], output_path: str) -> None:
    positions, layers, log_n90s, attribs = [], [], [], []

    for r in results:
        for f in r["features"]:
            n90 = f["n90"]
            attr = f["attribution"]
            if n90 <= 0 or attr <= 0:
                continue
            positions.append(f["position"])
            layers.append(f["layer"])
            log_n90s.append(np.log(n90))
            attribs.append(attr)

    if not positions:
        print("  Plot 7: no data, skipping.")
        return

    positions = np.array(positions)
    layers = np.array(layers)
    log_n90s = np.array(log_n90s)
    attribs = np.array(attribs)

    # Normalize attribution to dot sizes in [20, 200]
    a_min, a_max = attribs.min(), attribs.max()
    if a_max > a_min:
        sizes = 20 + 180 * (attribs - a_min) / (a_max - a_min)
    else:
        sizes = np.full_like(attribs, 50.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(positions, layers, c=log_n90s, cmap="coolwarm_r",
                    s=sizes, alpha=0.5, linewidths=0)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("log(N₉₀)  — cool=narrow/specific, warm=broad/distributional", fontsize=9)

    ax.set_xlabel("Token position in prompt  (0 = first token)", fontsize=11)
    ax.set_ylabel("Layer", fontsize=11)
    ax.set_title("Plot 7: Where in the computation does narrowing happen?\n"
                 "(dot size = attribution magnitude)", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Legend for dot sizes
    for label, frac in [("small attr", 0.0), ("large attr", 1.0)]:
        size = 20 + 180 * frac
        ax.scatter([], [], c="gray", s=size, alpha=0.6, label=label)
    ax.legend(fontsize=8, loc="upper right", title="attribution")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 7 saved → {output_path}")
