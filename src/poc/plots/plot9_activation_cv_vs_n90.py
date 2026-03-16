"""
Plot 9: Activation CV vs mean N₉₀ — consistent-firing vs context-specific features.

For each feature (layer, feature_idx) that appears in ≥2 prompts' attribution graphs:
  CV(f)      = std(|activation|) / mean(|activation|)   — consistency across prompts
  mean_N₉₀  = mean(N₉₀) across all appearances of f

Interpretation:
  low CV  → fires consistently regardless of specific prompt content
            → likely structural/format feature ("this is a math context", "answer follows")
  high CV → fires strongly on some prompts, weakly on others
            → likely content-specific ("this answer is 6", "this is about France")

Combined with mean_N₉₀:
  low CV  + high N₉₀ → consistent AND broad  → format/context features
  high CV + low N₉₀  → variable AND specific → computation/lookup features

Dot color = number of prompts the feature appears in (darker = more universal)
Dot size  = mean |activation| (bigger fires harder on average)
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path


def make_plot(results: list[dict], output_path: str) -> None:
    # Accumulate per-feature stats keyed by (layer, feature_idx)
    feature_data: dict[tuple, dict] = defaultdict(lambda: {"activations": [], "n90s": [], "prompt_ids": []})

    for r in results:
        for f in r["features"]:
            key = (f["layer"], f["feature_idx"])
            feature_data[key]["activations"].append(abs(f["activation"]))
            feature_data[key]["n90s"].append(f["n90"])
            feature_data[key]["prompt_ids"].append(r["prompt_id"])

    # Only keep features that appear in ≥2 prompts
    cvs, mean_n90s, counts, mean_acts = [], [], [], []
    for key, d in feature_data.items():
        if len(d["activations"]) < 2:
            continue
        acts = np.array(d["activations"])
        n90s = np.array([v for v in d["n90s"] if v > 0])  # drop sentinel -1 values
        mean_act = acts.mean()
        if mean_act < 1e-10 or len(n90s) == 0:
            continue
        cvs.append(acts.std() / mean_act)
        mean_n90s.append(float(n90s.mean()))
        counts.append(len(d["activations"]))
        mean_acts.append(mean_act)

    if len(cvs) < 2:
        print("  Plot 9: not enough cross-prompt features, skipping.")
        return

    cvs = np.array(cvs)
    mean_n90s = np.array(mean_n90s)
    counts = np.array(counts, dtype=float)
    mean_acts = np.array(mean_acts)

    # Normalize dot sizes to [20, 150] by mean activation
    a_min, a_max = mean_acts.min(), mean_acts.max()
    sizes = 20 + 130 * (mean_acts - a_min) / (a_max - a_min + 1e-10)

    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(cvs, np.log(mean_n90s), c=counts, cmap="Blues",
                    s=sizes, alpha=0.65, linewidths=0.4, edgecolors="gray")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("# prompts feature appears in", fontsize=9)

    ax.set_xlabel("CV of |activation|  =  std / mean  across prompts", fontsize=11)
    ax.set_ylabel("log(mean N₉₀)  [average broadness across appearances]", fontsize=11)
    ax.set_title(
        "Plot 9: Consistent-firing (low CV) vs context-specific (high CV) features\n"
        "(dot size = mean |activation|,  color = how many prompts it appears in)",
        fontsize=11,
    )

    # Quadrant annotations
    xm = float(np.median(cvs))
    ym = float(np.median(np.log(mean_n90s)))
    kw = dict(fontsize=8, alpha=0.55, ha="center")
    ax.text(xm * 0.35, ym * 1.12, "consistent\n+ broad\n(format/context)", **kw)
    ax.text(xm * 1.7,  ym * 1.12, "variable\n+ broad\n(rare format?)",    **kw)
    ax.text(xm * 0.35, ym * 0.88, "consistent\n+ specific\n(syntax?)",    **kw)
    ax.text(xm * 1.7,  ym * 0.88, "variable\n+ specific\n(computation)",  **kw)
    ax.axvline(xm, color="gray", lw=0.8, linestyle="--", alpha=0.4)
    ax.axhline(ym, color="gray", lw=0.8, linestyle="--", alpha=0.4)

    # Size legend
    for label, frac in [("small act", 0.0), ("large act", 1.0)]:
        s = 20 + 130 * frac
        ax.scatter([], [], c="steelblue", s=s, alpha=0.7, label=label)
    ax.legend(fontsize=8, loc="lower right", title="mean |activation|")

    ax.grid(True, alpha=0.25)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 9 saved → {output_path}  ({len(cvs)} cross-prompt features)")
