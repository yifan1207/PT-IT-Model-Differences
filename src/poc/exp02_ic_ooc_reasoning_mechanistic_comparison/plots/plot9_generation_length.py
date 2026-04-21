"""
Plot 9: Generation length distribution and L0 vs generation step, by category.

Two panels:
  Panel A (top): Histogram of how many tokens each prompt generated.
    - IC may produce short answers (confident, converges quickly).
    - OOC may produce longer hedging/confabulation.
    - R may produce multi-step chains of variable length.

  Panel B (bottom): Total L0 (summed over layers) vs generation step.
    - Shows whether feature engagement increases/decreases as generation progresses.
    - A rising L0 for R vs flat for IC would suggest more complex computation
      per step in reasoning tasks.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


CATEGORY_STYLE = {
    "in_context":     {"color": "#2196F3", "label": "IC (in-context)"},
    "out_of_context": {"color": "#FF9800", "label": "OOC (out-of-context)"},
    "reasoning":      {"color": "#4CAF50", "label": "R (reasoning)"},
}
CATEGORY_ORDER = ["in_context", "out_of_context", "reasoning"]


def _gen_lengths(results: list[dict]) -> dict[str, list[int]]:
    lengths: dict[str, list] = defaultdict(list)
    for r in results:
        lengths[r["category"]].append(len(r["generated_tokens"]))
    return lengths


def _total_l0_by_step(results: list[dict]) -> dict[str, np.ndarray]:
    """category → 2D array [n_prompts, max_steps] of total L0 (sum over layers), NaN-padded."""
    max_steps: dict[str, int] = defaultdict(int)
    for r in results:
        max_steps[r["category"]] = max(max_steps[r["category"]], len(r["l0"]))

    arrays: dict[str, list] = defaultdict(list)
    for r in results:
        cat = r["category"]
        ms = max_steps[cat]
        row = np.full(ms, np.nan)
        for step, layer_vals in enumerate(r["l0"]):
            row[step] = sum(layer_vals)
        arrays[cat].append(row)

    return {cat: np.array(v) for cat, v in arrays.items()}


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lengths = _gen_lengths(results)
    l0_by_step = _total_l0_by_step(results)

    fig, (ax_hist, ax_l0) = plt.subplots(2, 1, figsize=(11, 9))

    # ── Panel A: generation length histogram ────────────────────────────────
    max_len = max((max(v) for v in lengths.values() if v), default=50)
    bins = np.arange(0, max_len + 2) - 0.5

    for cat in CATEGORY_ORDER:
        style = CATEGORY_STYLE[cat]
        lens = lengths.get(cat, [])
        if not lens:
            continue
        ax_hist.hist(lens, bins=bins, alpha=0.55, color=style["color"],
                     label=f"{style['label']}  (n={len(lens)}, median={int(np.median(lens))})")

    ax_hist.set_xlabel("Number of generated tokens", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("Panel A: Generation length distribution by category", fontsize=10)
    ax_hist.legend(fontsize=9)

    # ── Panel B: total L0 vs generation step ────────────────────────────────
    for cat in CATEGORY_ORDER:
        style = CATEGORY_STYLE[cat]
        arr = l0_by_step.get(cat)
        if arr is None or arr.size == 0:
            continue
        steps = np.arange(arr.shape[1])
        mean = np.nanmean(arr, axis=0)
        count = np.sum(~np.isnan(arr), axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(count, 1))
        valid = count >= 5
        s = steps[valid]
        ax_l0.plot(s, mean[valid], color=style["color"],
                   label=f"{style['label']}  (n={arr.shape[0]})", lw=2)
        ax_l0.fill_between(s, (mean - sem)[valid], (mean + sem)[valid],
                           alpha=0.2, color=style["color"])

    ax_l0.set_xlabel("Generation step", fontsize=11)
    ax_l0.set_ylabel("Mean total L0 (sum over all layers)", fontsize=11)
    ax_l0.set_title("Panel B: Total active features vs generation step by category", fontsize=10)
    ax_l0.legend(fontsize=9)

    fig.tight_layout()

    out_path = Path(output_dir) / "plot9_generation_length.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 9 saved → {out_path}")
