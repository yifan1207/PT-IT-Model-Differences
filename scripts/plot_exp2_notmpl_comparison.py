#!/usr/bin/env python3
"""Side-by-side comparison of exp2 δ-cosine plots: with vs without chat template.

Generates comparison versions of plot10 (cosine profile) and plot11 (cosine heatmap)
showing IT model results with chat template (left) and without (right).

Expected: if δ-cosine patterns (especially the negative cosine at corrective layers)
persist without chat template → weight-intrinsic. If they disappear → template-gated.

Output:
    results/exp2/plots_notmpl_comparison/plot10_notmpl_vs_template.png
    results/exp2/plots_notmpl_comparison/plot11_notmpl_vs_template.png
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.poc.exp2.plots.plot10_cosine_similarity import (
    _mean_cos_profile, CATEGORY_STYLE, CATEGORY_ORDER,
)
from src.poc.exp2.plots.plot11_cosine_heatmap import _build_cos_heatmaps, CATEGORY_LABEL


def _load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"exp2 results not found: {path}")
    with open(path) as f:
        return json.load(f)


def plot10_comparison(results_tmpl: list[dict], results_notmpl: list[dict],
                      out_path: Path) -> None:
    """Side-by-side mean cosine profiles: left=with template, right=no template."""
    n_layers = len(results_tmpl[0]["layer_delta_cosine"][0])
    layers   = np.arange(n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Exp2 δ-cosine Profile: IT with template vs IT no-template\n"
        "cos(delta_i, h_{i-1}) per layer — averaged over all prompts and steps",
        fontsize=11, fontweight="bold",
    )

    for ax, results, label in [
        (axes[0], results_tmpl,   "IT  with chat template"),
        (axes[1], results_notmpl, "IT  no chat template"),
    ]:
        for cat in CATEGORY_ORDER:
            profile = _mean_cos_profile(results, cat)
            if profile is None:
                continue
            style = CATEGORY_STYLE[cat]
            ax.plot(layers, profile, color=style["color"], linestyle=style["ls"],
                    linewidth=2, label=style["label"])
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer", fontsize=10)
        ax.set_ylabel("Mean cos(delta_i, h_{i-1})", fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, n_layers - 1)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"plot10 comparison saved → {out_path}")


def plot11_comparison(results_tmpl: list[dict], results_notmpl: list[dict],
                      out_path: Path) -> None:
    """Side-by-side heatmaps [layer x step]: left=with template, right=no template."""
    hm_t  = _build_cos_heatmaps(results_tmpl)
    hm_nt = _build_cos_heatmaps(results_notmpl)

    cats = [c for c in CATEGORY_ORDER if c in hm_t or c in hm_nt]
    n_cats = len(cats)
    fig, axes = plt.subplots(2, n_cats, figsize=(6 * n_cats, 10))
    fig.suptitle(
        "Exp2 δ-cosine Heatmap [layer × step]: IT with template vs IT no-template\n"
        "Top row: with template  |  Bottom row: no template",
        fontsize=11, fontweight="bold",
    )

    # Shared colour scale across all panels
    all_vals = []
    for hm in (hm_t, hm_nt):
        for arr in hm.values():
            finite = arr[np.isfinite(arr)]
            if finite.size > 0:
                all_vals.extend(finite.tolist())
    vmax = np.percentile(np.abs(all_vals), 98) if all_vals else 0.5
    vmin = -vmax

    for col, cat in enumerate(cats):
        for row, (hm, label) in enumerate([(hm_t, "with template"), (hm_nt, "no template")]):
            ax = axes[row][col] if n_cats > 1 else axes[row]
            arr = hm.get(cat)
            if arr is None:
                ax.set_visible(False)
                continue
            im = ax.imshow(arr.T, aspect="auto", origin="upper",
                           cmap="RdBu_r", vmin=vmin, vmax=vmax,
                           interpolation="nearest")
            ax.set_title(f"{CATEGORY_LABEL.get(cat, cat)}  [{label}]", fontsize=9)
            ax.set_xlabel("Layer" if row == 1 else "")
            ax.set_ylabel("Generation step" if col == 0 else "")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"plot11 comparison saved → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tmpl-dir",   default="results/exp2/it_16k_l0_big_affine_t512",
                   help="Path to exp2 IT results WITH chat template")
    p.add_argument("--notmpl-dir", default="results/exp2/it_16k_l0_big_affine_t512_notmpl",
                   help="Path to exp2 IT results WITHOUT chat template")
    p.add_argument("--out-dir",    default="results/exp2/plots_notmpl_comparison",
                   help="Output directory for comparison plots")
    args = p.parse_args()

    tmpl_dir   = Path(args.tmpl_dir)
    notmpl_dir = Path(args.notmpl_dir)
    out_dir    = Path(args.out_dir)

    print(f"Loading with-template results from: {tmpl_dir}")
    results_tmpl   = _load_results(tmpl_dir / "exp2_results.json")
    print(f"Loading no-template results from:   {notmpl_dir}")
    results_notmpl = _load_results(notmpl_dir / "exp2_results.json")

    print(f"Generating comparison plots → {out_dir}/")
    plot10_comparison(results_tmpl, results_notmpl,
                      out_dir / "plot10_notmpl_vs_template.png")
    plot11_comparison(results_tmpl, results_notmpl,
                      out_dir / "plot11_notmpl_vs_template.png")

    print("Done.")


if __name__ == "__main__":
    main()
