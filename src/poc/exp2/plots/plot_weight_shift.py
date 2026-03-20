"""
Plot: PT → IT weight shift for Gemma 3 4B.

Four panels in a 2 × 2 layout:

  Panel A (top-left)
      Horizontal bar chart of the top-40 most-shifted parameters by Δ_rel.
      Bars are coloured by component type (Q/K/V/O proj, MLP gate/up/down,
      layer-norms, embeddings).  Gives an immediate ranked view of "what
      moved most".

  Panel B (top-right)
      Per-layer mean Δ_rel across all matrices that belong to that layer.
      Reveals which transformer depths were most affected by RLHF fine-tuning.

  Panel C (bottom-left)
      Box plot of Δ_rel grouped by component type.  Shows the distribution
      within each weight class — e.g. whether ALL MLP matrices shifted or
      only a few outliers.

  Panel D (bottom-right)
      Scatter of Δ_rel vs d_cos for every parameter, coloured by component
      type.  The two metrics are complementary: a parameter can have large
      Δ_rel with small d_cos (scaled, same direction) or small Δ_rel with
      large d_cos (rotated but same norm).
"""
import re
import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict


# ── component taxonomy ────────────────────────────────────────────────────────

# Ordered list of (substring, label) — first match wins.
_COMPONENT_PATTERNS: list[tuple[str, str]] = [
    ("self_attn.q_proj",              "Q proj"),
    ("self_attn.k_proj",              "K proj"),
    ("self_attn.v_proj",              "V proj"),
    ("self_attn.o_proj",              "O proj"),
    ("mlp.gate_proj",                 "MLP gate"),
    ("mlp.up_proj",                   "MLP up"),
    ("mlp.down_proj",                 "MLP down"),
    ("input_layernorm",               "LN input"),
    ("pre_feedforward_layernorm",     "LN pre-MLP"),
    ("post_feedforward_layernorm",    "LN post-MLP"),
    ("post_attention_layernorm",      "LN post-attn"),
    ("embed_tokens",                  "Embedding"),
    ("lm_head",                       "LM head"),
    ("norm",                          "Final LN"),
]

# Colour per component — consistent across all panels.
_COMPONENT_COLORS: dict[str, str] = {
    "Q proj":        "#1565C0",
    "K proj":        "#1E88E5",
    "V proj":        "#42A5F5",
    "O proj":        "#90CAF9",
    "MLP gate":      "#E65100",
    "MLP up":        "#FB8C00",
    "MLP down":      "#FFA726",
    "LN input":      "#6A1B9A",
    "LN pre-MLP":    "#AB47BC",
    "LN post-MLP":   "#CE93D8",
    "LN post-attn":  "#E1BEE7",
    "Embedding":     "#2E7D32",
    "LM head":       "#66BB6A",
    "Final LN":      "#A5D6A7",
    "other":         "#9E9E9E",
}

# Display order in the legend / box plot.
_COMPONENT_ORDER: list[str] = [
    "Q proj", "K proj", "V proj", "O proj",
    "MLP gate", "MLP up", "MLP down",
    "LN input", "LN pre-MLP", "LN post-MLP", "LN post-attn",
    "Embedding", "LM head", "Final LN", "other",
]


def _component_type(name: str) -> str:
    for pattern, label in _COMPONENT_PATTERNS:
        if pattern in name:
            return label
    return "other"


def _layer_index(name: str) -> int | None:
    """Return transformer layer index from a parameter name, or None."""
    m = re.search(r"\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ── panel helpers ─────────────────────────────────────────────────────────────

def _panel_top_n_bar(ax, results: list[dict], top_n: int = 40) -> None:
    """Panel A: horizontal bar chart of top-N parameters by Δ_rel."""
    valid = [r for r in results if not math.isnan(r["frob_shift"])]
    top = valid[:top_n]

    labels = [r["name"].split(".")[-2] + "." + r["name"].split(".")[-1] for r in top]
    frob   = [r["frob_shift"] for r in top]
    colors = [_COMPONENT_COLORS.get(_component_type(r["name"]), "#9E9E9E") for r in top]

    y_pos = np.arange(len(top))
    ax.barh(y_pos, frob, color=colors, edgecolor="none", height=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(np.mean([r["frob_shift"] for r in valid]),
               color="black", ls="--", lw=1.0, label="mean (all params)")
    ax.set_xlabel("Δ_rel  =  ‖W_it − W_pt‖_F / ‖W_pt‖_F", fontsize=9)
    ax.set_title(f"A: Top {top_n} parameters by Δ_rel", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)


def _panel_per_layer(ax, results: list[dict]) -> None:
    """Panel B: mean Δ_rel and mean d_cos per transformer layer."""
    layer_frob: dict[int, list[float]] = defaultdict(list)
    layer_cos:  dict[int, list[float]] = defaultdict(list)

    for r in results:
        li = _layer_index(r["name"])
        if li is None:
            continue
        if not math.isnan(r["frob_shift"]):
            layer_frob[li].append(r["frob_shift"])
        if not math.isnan(r["cos_dist"]):
            layer_cos[li].append(r["cos_dist"])

    if not layer_frob:
        ax.set_visible(False)
        return

    layers_sorted = sorted(layer_frob.keys())
    mean_frob = [np.mean(layer_frob[li]) for li in layers_sorted]
    mean_cos  = [np.mean(layer_cos[li])  if layer_cos.get(li) else float("nan")
                 for li in layers_sorted]

    ax2 = ax.twinx()

    ax.plot(layers_sorted, mean_frob, color="#1565C0", lw=2, marker="o", ms=4,
            label="Δ_rel (left axis)")
    ax2.plot(layers_sorted, mean_cos, color="#E65100", lw=2, marker="s", ms=4,
             ls="--", label="d_cos (right axis)")

    ax.set_xlabel("Transformer layer", fontsize=9)
    ax.set_ylabel("Mean Δ_rel", fontsize=9, color="#1565C0")
    ax2.set_ylabel("Mean d_cos", fontsize=9, color="#E65100")
    ax.tick_params(axis="y", labelcolor="#1565C0")
    ax2.tick_params(axis="y", labelcolor="#E65100")
    ax.set_title("B: Mean shift per transformer layer", fontsize=10, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")


def _panel_box_by_component(ax, results: list[dict]) -> None:
    """Panel C: box plot of Δ_rel distribution per component type."""
    by_comp: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if not math.isnan(r["frob_shift"]):
            by_comp[_component_type(r["name"])].append(r["frob_shift"])

    present = [c for c in _COMPONENT_ORDER if c in by_comp]
    data    = [by_comp[c] for c in present]
    colors  = [_COMPONENT_COLORS[c] for c in present]

    bp = ax.boxplot(data, vert=True, patch_artist=True, notch=False,
                    medianprops={"color": "black", "lw": 1.5},
                    flierprops={"marker": ".", "ms": 3, "alpha": 0.4})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=7.5)
    ax.set_ylabel("Δ_rel", fontsize=9)
    ax.set_title("C: Δ_rel distribution by component type", fontsize=10, fontweight="bold")


def _panel_scatter(ax, results: list[dict]) -> None:
    """Panel D: scatter of Δ_rel vs d_cos, coloured by component type."""
    by_comp: dict[str, tuple[list, list]] = defaultdict(lambda: ([], []))
    for r in results:
        if math.isnan(r["frob_shift"]) or math.isnan(r["cos_dist"]):
            continue
        comp = _component_type(r["name"])
        by_comp[comp][0].append(r["frob_shift"])
        by_comp[comp][1].append(r["cos_dist"])

    for comp in _COMPONENT_ORDER:
        if comp not in by_comp:
            continue
        xs, ys = by_comp[comp]
        ax.scatter(xs, ys, s=12, alpha=0.55,
                   color=_COMPONENT_COLORS[comp], label=comp)

    ax.set_xlabel("Δ_rel  (magnitude change)", fontsize=9)
    ax.set_ylabel("d_cos  (directional change)", fontsize=9)
    ax.set_title("D: Δ_rel vs d_cos per parameter", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="upper right")


# ── public entry point (matches convention of all other exp2 plot modules) ────

def make_plot(results: list[dict], output_dir: str, top_n: int = 40) -> None:
    """Render the 4-panel weight-shift figure and save to output_dir.

    Parameters
    ----------
    results : list[dict]
        Output of weight_shift.compute_weight_shift() — one dict per parameter.
    output_dir : str
        Directory where plot_weight_shift.png is written.
    top_n : int
        Number of top parameters shown in Panel A bar chart (default 40).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    valid = [r for r in results if not math.isnan(r["frob_shift"])]
    if not valid:
        print("  Weight-shift plot skipped — no valid results.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(20, max(12, top_n * 0.28)))

    _panel_top_n_bar(axes[0, 0], results, top_n=top_n)
    _panel_per_layer(axes[0, 1], results)
    _panel_box_by_component(axes[1, 0], results)
    _panel_scatter(axes[1, 1], results)

    # Legend patches for the colour scheme used in panels A, C, D.
    legend_patches = [
        mpatches.Patch(color=_COMPONENT_COLORS[c], label=c)
        for c in _COMPONENT_ORDER
        if any(_component_type(r["name"]) == c for r in results)
    ]
    fig.legend(
        handles=legend_patches,
        title="Component type",
        loc="lower center",
        ncol=min(len(legend_patches), 8),
        fontsize=8,
        title_fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        "PT → IT weight shift: google/gemma-3-4b\n"
        "Δ_rel = ‖W_it − W_pt‖_F / ‖W_pt‖_F  ·  d_cos = 1 − cos(vec(W_pt), vec(W_it))",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot_weight_shift.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Weight-shift plot saved → {out_path}")
