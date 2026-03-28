"""Cross-model tuned-lens commitment delay plots.

Generates three figures:
  1. L2_commitment_tuned_lens.png  — 6-panel histogram overlay (raw vs tuned)
  2. tuned_lens_scatter.png        — raw vs tuned commitment scatter (6 panels)
  3. tuned_lens_summary_table.json — machine-readable summary

Usage:
  uv run python scripts/plot_tuned_lens_commitment.py
  uv run python scripts/plot_tuned_lens_commitment.py --models gemma3_4b llama31_8b
  uv run python scripts/plot_tuned_lens_commitment.py --threshold 0.1
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE_RESULTS = Path("results/cross_model")
PLOT_DIR = BASE_RESULTS / "plots"

MODEL_LABELS = {
    "gemma3_4b":        "Gemma 3 4B",
    "llama31_8b":       "Llama 3.1 8B",
    "qwen3_4b":         "Qwen 3 4B",
    "mistral_7b":       "Mistral 7B v0.3",
    "deepseek_v2_lite": "DeepSeek-V2-Lite",
    "olmo2_7b":         "OLMo 2 7B",
}


def _save_fig(fig: plt.Figure, filename: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out)
    plt.close(fig)


def _load_commitment_data(
    model_name: str, variant: str, threshold: float,
) -> dict[str, list[int]] | None:
    """Load raw and tuned commitment layers from JSONL.

    Returns {"raw": [...], "tuned": [...], "top1_tuned": [...]} or None.
    """
    tuned_path = (
        BASE_RESULTS / model_name / "tuned_lens" / "commitment"
        / f"tuned_lens_commitment_{variant}.jsonl"
    )
    if not tuned_path.exists():
        return None

    raw_commits: list[int] = []
    tuned_commits: list[int] = []
    top1_tuned_commits: list[int] = []

    tuned_key = f"commitment_layer_tuned_{threshold}"

    with open(tuned_path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw_commits.extend(rec.get("commitment_layer_raw", []))
            tuned_commits.extend(rec.get(tuned_key, []))
            top1_tuned_commits.extend(rec.get("commitment_layer_top1_tuned", []))

    if not raw_commits:
        return None

    return {
        "raw": raw_commits,
        "tuned": tuned_commits,
        "top1_tuned": top1_tuned_commits,
    }


def _load_l1l2_commitment(model_name: str, variant: str) -> list[int] | None:
    """Load raw commitment from existing L1L2 results (for comparison)."""
    path = BASE_RESULTS / model_name / variant / "L1L2_results.jsonl"
    if not path.exists():
        return None
    commits: list[int] = []
    with open(path) as f:
        for line in f:
            if line.strip():
                try:
                    r = json.loads(line)
                    commits.extend(c for c in r.get("commitment_layer", []) if c is not None)
                except (KeyError, json.JSONDecodeError):
                    pass
    return commits if commits else None


# ── Figure 1: 6-panel commitment histogram overlay ─────────────────────────

def plot_commitment_histogram(models: list[str], threshold: float) -> None:
    """6-panel histogram: PT vs IT, raw vs tuned."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 4.5), squeeze=False)

    any_data = False

    for col, model_name in enumerate(models):
        ax = axes[0][col]
        spec = get_spec(model_name)
        bins = np.arange(0, spec.n_layers + 2) - 0.5

        plotted = False

        # Define histogram layers: (variant, method, data_key, color, label, histtype, alpha, lw, ls)
        hist_configs = [
            ("pt", "raw",   "#9E9E9E", "PT raw",     "stepfilled", 0.25, 0.0, "--"),
            ("it", "raw",   "#1565C0", "IT raw",     "step",       1.0,  1.5, "-"),
            ("pt", "tuned", "#E53935", "PT tuned",   "step",       0.8,  1.5, "--"),
            ("it", "tuned", "#FF9800", "IT tuned",   "step",       1.0,  2.0, "-"),
        ]

        for variant, method, color, label, histtype, alpha, lw, ls in hist_configs:
            data = _load_commitment_data(model_name, variant, threshold)
            if data is None:
                # Try L1L2 data for raw
                if method == "raw":
                    l1l2 = _load_l1l2_commitment(model_name, variant)
                    if l1l2:
                        ax.hist(l1l2, bins=bins, density=True, alpha=alpha,
                                color=color, label=label, histtype=histtype, linewidth=lw)
                        med = np.median(l1l2)
                        ax.axvline(med, color=color, lw=1.2, ls=ls, zorder=3)
                        plotted = True
                continue

            vals = data[method] if method != "tuned" else data.get("tuned", [])
            if not vals:
                continue

            ax.hist(vals, bins=bins, density=True, alpha=alpha,
                    color=color, label=label, histtype=histtype, linewidth=lw)
            med = np.median(vals)
            ax.axvline(med, color=color, lw=1.2, ls=ls, zorder=3)
            plotted = True

        # Corrective onset reference line
        ax.axvline(spec.corrective_onset, color="green", lw=1.2, ls=":",
                   alpha=0.7, label="Corrective onset")

        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=9, fontweight="bold")
        ax.set_xlabel("Commitment layer", fontsize=8)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=6, loc="upper left")

        if not plotted:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        else:
            any_data = True

    fig.suptitle(
        f"Commitment Delay: Raw vs Tuned Lens (KL threshold={threshold})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if any_data:
        _save_fig(fig, "L2_commitment_tuned_lens.png")
    else:
        log.warning("No data found for histogram plot.")
        plt.close(fig)


# ── Figure 2: Raw vs tuned scatter plot ────────────────────────────────────

def plot_scatter(models: list[str], threshold: float) -> None:
    """6-panel scatter: raw commitment (x) vs tuned commitment (y) per token."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 3.5), squeeze=False)

    any_data = False
    variant_colors = {"pt": "#9E9E9E", "it": "#1565C0"}
    tuned_key = f"commitment_layer_tuned_{threshold}"

    for col, model_name in enumerate(models):
        ax = axes[0][col]
        spec = get_spec(model_name)

        for variant in ("pt", "it"):
            tuned_path = (
                BASE_RESULTS / model_name / "tuned_lens" / "commitment"
                / f"tuned_lens_commitment_{variant}.jsonl"
            )
            if not tuned_path.exists():
                continue

            raw_vals: list[int] = []
            tuned_vals: list[int] = []

            with open(tuned_path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    raw_list = rec.get("commitment_layer_raw", [])
                    tuned_list = rec.get(tuned_key, [])
                    n = min(len(raw_list), len(tuned_list))
                    raw_vals.extend(raw_list[:n])
                    tuned_vals.extend(tuned_list[:n])

            if not raw_vals:
                continue

            # Subsample if too many points
            n_pts = len(raw_vals)
            if n_pts > 5000:
                rng = np.random.default_rng(42)
                idx = rng.choice(n_pts, size=5000, replace=False)
                raw_arr = np.array(raw_vals)[idx]
                tuned_arr = np.array(tuned_vals)[idx]
            else:
                raw_arr = np.array(raw_vals)
                tuned_arr = np.array(tuned_vals)

            ax.scatter(
                raw_arr, tuned_arr,
                c=variant_colors[variant],
                s=3, alpha=0.15,
                label=f"{variant.upper()} (n={n_pts})",
                rasterized=True,
            )

            # Pearson r
            if len(raw_arr) > 2:
                r = np.corrcoef(raw_arr, tuned_arr)[0, 1]
                label_txt = f"{variant.upper()} r={r:.2f}"
                ax.scatter([], [], c=variant_colors[variant], s=10, label=label_txt)

            any_data = True

        # Diagonal
        ax.plot([0, spec.n_layers], [0, spec.n_layers], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(-0.5, spec.n_layers + 0.5)
        ax.set_ylim(-0.5, spec.n_layers + 0.5)
        ax.set_aspect("equal")
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=9, fontweight="bold")
        ax.set_xlabel("Raw commitment layer", fontsize=8)
        if col == 0:
            ax.set_ylabel("Tuned commitment layer", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, markerscale=2)

    fig.suptitle(
        f"Raw vs Tuned Lens Commitment (KL threshold={threshold})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    if any_data:
        _save_fig(fig, "tuned_lens_scatter.png")
    else:
        log.warning("No data found for scatter plot.")
        plt.close(fig)


# ── Figure 3: Summary table ───────────────────────────────────────────────

def build_summary_table(models: list[str], threshold: float) -> list[dict]:
    """Build summary: model, method, PT median, IT median, delay."""
    rows: list[dict] = []

    for model_name in models:
        for method_label, data_key in [("raw", "raw"), ("tuned", "tuned")]:
            pt_data = _load_commitment_data(model_name, "pt", threshold)
            it_data = _load_commitment_data(model_name, "it", threshold)

            pt_vals = pt_data[data_key] if pt_data and data_key in pt_data else None
            it_vals = it_data[data_key] if it_data and data_key in it_data else None

            # Fallback to L1L2 for raw
            if method_label == "raw":
                if pt_vals is None:
                    pt_vals = _load_l1l2_commitment(model_name, "pt")
                if it_vals is None:
                    it_vals = _load_l1l2_commitment(model_name, "it")

            pt_med = float(np.median(pt_vals)) if pt_vals else None
            it_med = float(np.median(it_vals)) if it_vals else None
            delay = (it_med - pt_med) if (pt_med is not None and it_med is not None) else None

            rows.append({
                "model": model_name,
                "method": method_label,
                "threshold": threshold if method_label == "tuned" else None,
                "pt_median": pt_med,
                "it_median": it_med,
                "delay": delay,
                "delay_direction": (
                    "IT commits later" if delay and delay > 0
                    else "PT commits later" if delay and delay < 0
                    else None
                ),
                "n_pt": len(pt_vals) if pt_vals else 0,
                "n_it": len(it_vals) if it_vals else 0,
            })

    return rows


def save_summary(models: list[str], threshold: float) -> None:
    """Save summary table as JSON."""
    rows = build_summary_table(models, threshold)

    # Also check all 3 thresholds for sensitivity
    all_threshold_rows: list[dict] = []
    for t in [0.05, 0.1, 0.2]:
        all_threshold_rows.extend(build_summary_table(models, t))

    out = {
        "primary_threshold": threshold,
        "primary_table": rows,
        "all_thresholds": all_threshold_rows,
    }

    # Add transfer test results
    transfer_results: dict = {}
    for model_name in models:
        transfer_path = (
            BASE_RESULTS / model_name / "tuned_lens" / "commitment" / "transfer_test.json"
        )
        if transfer_path.exists():
            with open(transfer_path) as f:
                transfer_results[model_name] = json.load(f)
    if transfer_results:
        out["transfer_tests"] = transfer_results

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOT_DIR / "tuned_lens_summary_table.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Summary table → %s", out_path)

    # Print summary
    print("\n=== Tuned-Lens Commitment Delay Summary ===\n")
    print(f"{'Model':<20} {'Method':<8} {'PT med':>8} {'IT med':>8} {'Delay':>8} {'Direction'}")
    print("-" * 75)
    for row in rows:
        pt = f"{row['pt_median']:.1f}" if row["pt_median"] is not None else "N/A"
        it = f"{row['it_median']:.1f}" if row["it_median"] is not None else "N/A"
        dl = f"{row['delay']:.1f}" if row["delay"] is not None else "N/A"
        dr = row["delay_direction"] or ""
        print(f"{row['model']:<20} {row['method']:<8} {pt:>8} {it:>8} {dl:>8} {dr}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Plot tuned-lens commitment results")
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                   choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--threshold", type=float, default=0.1,
                   help="Primary KL threshold for tuned-lens commitment (default: 0.1)")
    args = p.parse_args()

    print(f"Models: {args.models}")
    print(f"KL threshold: {args.threshold}")

    plot_commitment_histogram(args.models, args.threshold)
    plot_scatter(args.models, args.threshold)
    save_summary(args.models, args.threshold)


if __name__ == "__main__":
    main()
