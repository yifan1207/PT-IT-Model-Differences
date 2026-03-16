"""
Plot 4: Histogram of log(N₉₀) — is feature broadness bimodal or continuous?

If the histogram is bimodal (a peak near N₉₀ ≈ 1-5 and another near N₉₀ ≈ 100+),
there are two natural classes of features and discrete categorization is justified.

If unimodal or uniform, broadness is a continuous spectrum and any discrete
classification (e.g. "format vs computation") is an imposed, not discovered, boundary.

Overlays separate distributions per prompt group (A/B/C/D) to reveal whether
task type changes the distribution of feature broadness.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

GROUP_COLORS = {"M": "#e74c3c", "R": "#3498db"}
FALLBACK_COLOR = "#95a5a6"


def make_plot(results: list[dict], output_path: str) -> None:
    # Collect log(N₉₀) values, grouped by prompt group
    all_log_n90: list[float] = []
    by_group: dict[str, list[float]] = defaultdict(list)

    for r in results:
        group = r["prompt_id"][0]
        for f in r["features"]:
            n90 = f["n90"]
            if n90 <= 0:
                continue
            val = np.log(n90)
            all_log_n90.append(val)
            by_group[group].append(val)

    if not all_log_n90:
        print("  Plot 4: no data, skipping.")
        return

    all_arr = np.array(all_log_n90)
    bins = np.linspace(all_arr.min(), all_arr.max(), 40)

    fig, (ax_all, ax_groups) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: pooled histogram
    ax_all.hist(all_arr, bins=bins, color="#2c3e50", alpha=0.75, edgecolor="white", lw=0.4)
    ax_all.set_xlabel("log(N₉₀)", fontsize=11)
    ax_all.set_ylabel("count", fontsize=11)
    ax_all.set_title("All features pooled", fontsize=11)
    ax_all.grid(True, alpha=0.3, axis="y")

    # Right: per-group overlay
    for group in sorted(by_group):
        arr = np.array(by_group[group])
        color = GROUP_COLORS.get(group, FALLBACK_COLOR)
        ax_groups.hist(arr, bins=bins, color=color, alpha=0.5, edgecolor="white", lw=0.3,
                       label=f"Group {group}  (n={len(arr)})")

    ax_groups.set_xlabel("log(N₉₀)", fontsize=11)
    ax_groups.set_ylabel("count", fontsize=11)
    ax_groups.set_title("By prompt group (M=memorization, R=reasoning)", fontsize=11)
    ax_groups.legend(fontsize=9)
    ax_groups.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Plot 4: Distribution of feature broadness (N₉₀)", fontsize=13)
    fig.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 4 saved → {output_path}")
