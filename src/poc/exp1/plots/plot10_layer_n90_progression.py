"""
Plot 10: Layer-wise N₉₀ progression at the answer position.

Filters to features active at the final token position (where the answer is generated),
then shows how feature broadness (N₉₀) changes across layers.

Hypothesis (hierarchical narrowing):
  Early layers → broad features (high N₉₀): "what kind of answer?"
  Late layers  → specific features (low N₉₀): "which exact answer?"
  → Downward slope from layer 0 to layer 25 confirms progressive specialisation.

Each layer's bar = mean N₉₀ of all answer-position features from that layer,
pooled across all prompts.  Per-group lines overlay the same stat split by A/B/C/D.

"Answer position" = the highest token position seen across any feature in that prompt
(the last token the model attends to before generating the next token).
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

GROUP_COLORS = {"M": "#e74c3c", "R": "#3498db"}
FALLBACK_COLOR = "#95a5a6"


def make_plot(results: list[dict], output_path: str) -> None:
    # layer → list of N₉₀ values (pooled across all prompts, answer-position only)
    layer_n90_all: dict[int, list[float]] = defaultdict(list)
    # group → layer → list of N₉₀ values
    layer_n90_group: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))

    for r in results:
        if not r["features"]:
            continue
        answer_pos = max(f["position"] for f in r["features"])
        group = r["prompt_id"][0]

        for f in r["features"]:
            if f["position"] != answer_pos or f["n90"] <= 0:
                continue
            layer_n90_all[f["layer"]].append(f["n90"])
            layer_n90_group[group][f["layer"]].append(f["n90"])

    if not layer_n90_all:
        print("  Plot 10: no answer-position features found, skipping.")
        return

    layers_sorted = sorted(layer_n90_all.keys())
    means = [np.mean(layer_n90_all[l]) for l in layers_sorted]
    stds  = [np.std(layer_n90_all[l])  for l in layers_sorted]
    means = np.array(means)
    stds  = np.array(stds)

    fig, ax = plt.subplots(figsize=(11, 5))

    # Shaded band ±1 std (pooled)
    ax.fill_between(layers_sorted, means - stds, means + stds,
                    alpha=0.15, color="black", label="±1 std (all groups)")

    # Main pooled line
    ax.plot(layers_sorted, means, color="black", lw=2.5, zorder=5, label="mean (all groups)")

    # Per-group lines
    for group in sorted(layer_n90_group.keys()):
        gdata = layer_n90_group[group]
        g_layers = sorted(gdata.keys())
        if len(g_layers) < 2:
            continue
        g_means = [np.mean(gdata[l]) for l in g_layers]
        color = GROUP_COLORS.get(group, FALLBACK_COLOR)
        ax.plot(g_layers, g_means, color=color, lw=1.4, linestyle="--",
                alpha=0.8, label=f"Group {group}")

    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean N₉₀  (tokens for 90% of |contribution| mass)", fontsize=11)
    ax.set_title(
        "Plot 10: Does the model narrow its prediction layer by layer?\n"
        "(answer-position features only — lower N₉₀ in later layers = progressive specialisation)",
        fontsize=11,
    )
    ax.set_xticks(layers_sorted)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Annotate n (feature count) per layer along x-axis
    for l, m in zip(layers_sorted, means):
        n = len(layer_n90_all[l])
        ax.text(l, ax.get_ylim()[0], f"n={n}", ha="center", va="bottom",
                fontsize=6, color="gray", rotation=90)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 10 saved → {output_path}")
