"""
Plot 6: Output entropy over generation steps, by category.

Output entropy = H(softmax(logits_real)) at each step, where logits_real
restricts to real vocabulary tokens (filters <unusedXXXX> placeholders).

Expected pattern under null hypothesis (no special reasoning mode):
  - All three categories show similar entropy trajectories.

Alternative: IC has low entropy from step 1 (model is confident about
memorised continuations), OOC stays high (model is uncertain), R starts
higher and may decrease as the chain-of-thought resolves.

  X-axis : generation step (0 = first new token)
  Y-axis : mean output entropy (nats) across all prompts in that category
  Lines  : IC, OOC, R with ±1 SEM ribbon

Also shows a secondary panel of entropy variance per step (spread across
prompts within a category).
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


def _build_entropy_by_step(results: list[dict]) -> dict[str, np.ndarray]:
    """Returns category → 2-D array [n_prompts, max_steps] with NaN padding."""
    max_steps: dict[str, int] = defaultdict(int)
    for r in results:
        max_steps[r["category"]] = max(max_steps[r["category"]], len(r["output_entropy"]))

    arrays: dict[str, list] = defaultdict(list)
    for r in results:
        cat = r["category"]
        row = np.full(max_steps[cat], np.nan)
        vals = r["output_entropy"]
        row[: len(vals)] = vals
        arrays[cat].append(row)

    return {cat: np.array(v) for cat, v in arrays.items()}


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cat_arrays = _build_entropy_by_step(results)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)
    ax_mean, ax_var = axes

    for cat, style in CATEGORY_STYLE.items():
        arr = cat_arrays.get(cat)
        if arr is None or arr.size == 0:
            continue
        n_steps = arr.shape[1]
        steps = np.arange(n_steps)

        # nanmean / nanstd to handle NaN-padded shorter sequences
        mean = np.nanmean(arr, axis=0)
        count = np.sum(~np.isnan(arr), axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(np.maximum(count, 1))
        var = np.nanvar(arr, axis=0)

        # Mask positions where fewer than 5 prompts had that step
        valid = count >= 5
        s = steps[valid]

        label = f"{style['label']}  (n={arr.shape[0]})"
        ax_mean.plot(s, mean[valid], color=style["color"], label=label, lw=2)
        ax_mean.fill_between(s, (mean - sem)[valid], (mean + sem)[valid],
                             alpha=0.2, color=style["color"])

        ax_var.plot(s, var[valid], color=style["color"], label=label, lw=2, ls="--")

    ax_mean.set_ylabel("Mean output entropy (nats)", fontsize=11)
    ax_mean.set_title(
        "Plot 6: Output entropy over generation steps by category\n"
        "entropy = H(softmax(final logits over real tokens))",
        fontsize=10,
    )
    ax_mean.legend(fontsize=9)

    ax_var.set_xlabel("Generation step", fontsize=11)
    ax_var.set_ylabel("Entropy variance across prompts", fontsize=11)
    ax_var.set_title("Entropy variance within each category per step", fontsize=9)
    ax_var.legend(fontsize=9)

    fig.tight_layout()

    out_path = Path(output_dir) / "plot6_output_entropy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 6 saved → {out_path}")
