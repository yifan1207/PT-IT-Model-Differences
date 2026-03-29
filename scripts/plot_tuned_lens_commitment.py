"""Cross-model tuned-lens commitment delay plots (0G Phase 2).

Generates:
  0G_commitment_top1.png      — 6-panel: raw vs tuned top-1 commitment (PT/IT)
  0G_commitment_kl.png        — 6-panel: raw vs tuned KL commitment (PT/IT)
  0G_commitment_summary.png   — bar chart: delay by method across models
  tuned_lens_scatter.png      — raw vs tuned scatter (6 panels)
  tuned_lens_summary_table.json — machine-readable summary

Usage:
  uv run python scripts/plot_tuned_lens_commitment.py
  uv run python scripts/plot_tuned_lens_commitment.py --models gemma3_4b llama31_8b
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
PLOT_DIR_CROSS = BASE_RESULTS / "plots"
PLOT_DIR_EXP7 = Path("results/exp7/plots")
DATA_DIR = PLOT_DIR_EXP7 / "data"

MODEL_LABELS = {
    "gemma3_4b":        "Gemma 3 4B",
    "llama31_8b":       "Llama 3.1 8B",
    "qwen3_4b":         "Qwen 3 4B",
    "mistral_7b":       "Mistral 7B v0.3",
    "deepseek_v2_lite": "DeepSeek-V2-Lite",
    "olmo2_7b":         "OLMo 2 7B",
}


def _save_fig(fig: plt.Figure, filename: str) -> None:
    for d in [PLOT_DIR_CROSS, PLOT_DIR_EXP7]:
        d.mkdir(parents=True, exist_ok=True)
        out = d / filename
        fig.savefig(out, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", out)
    plt.close(fig)


def load_commitment(model: str, variant: str) -> dict[str, list[int]] | None:
    """Load commitment data from JSONL, flatten per-prompt arrays."""
    path = (BASE_RESULTS / model / "tuned_lens" / "commitment"
            / f"tuned_lens_commitment_{variant}.jsonl")
    if not path.exists():
        return None
    accum: dict[str, list[int]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for key, val in rec.items():
                if key.startswith("commitment_layer") and isinstance(val, list):
                    accum.setdefault(key, []).extend(val)
    return accum if accum else None


# ── 6-panel histogram ─────────────────────────────────────────────────────

def plot_histogram(
    models: list[str],
    series: list[tuple[str, str, str, str, float, float, str]],
    title: str,
    filename: str,
) -> dict[str, dict]:
    """
    6-panel histogram for given commitment series.

    series: list of (variant, data_key, color, label, alpha, lw, histtype)
    """
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4.5), squeeze=False)
    summary = {}

    for col, model_name in enumerate(models):
        ax = axes[0, col]
        spec = get_spec(model_name)
        bins = np.arange(0, spec.n_layers + 2) - 0.5
        medians = {}

        for variant, data_key, color, label, alpha, lw, histtype in series:
            data = load_commitment(model_name, variant)
            if data is None or data_key not in data:
                continue
            vals = [v for v in data[data_key] if v is not None]
            if not vals:
                continue

            ax.hist(vals, bins=bins, density=True, histtype=histtype,
                    color=color, alpha=alpha, linewidth=lw, label=label, zorder=2)

            med = np.median(vals)
            ls = "--" if "PT" in label else "-"
            ax.axvline(med, color=color, lw=1.5, ls=ls, zorder=3)
            medians[label] = float(med)

        # Corrective onset
        ax.axvline(spec.corrective_onset, color="green", lw=1.2, ls=":",
                   alpha=0.7, label="Corrective onset")

        ax.set_title(MODEL_LABELS.get(model_name, model_name),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Commitment layer", fontsize=8)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=5.5, loc="upper left")
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.5, spec.n_layers + 0.5)

        summary[model_name] = medians

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, filename)
    return summary


def plot_commitment_top1(models: list[str]) -> dict:
    """Raw vs Tuned top-1 commitment."""
    return plot_histogram(
        models,
        series=[
            ("pt", "commitment_layer_raw",         "#9E9E9E", "PT raw",   0.3, 0, "stepfilled"),
            ("it", "commitment_layer_raw",         "#1565C0", "IT raw",   0.9, 2, "step"),
            ("pt", "commitment_layer_top1_tuned",  "#E53935", "PT tuned", 0.7, 1.5, "step"),
            ("it", "commitment_layer_top1_tuned",  "#FF9800", "IT tuned", 0.9, 2.0, "step"),
        ],
        title="Top-1 Commitment: Raw Logit-Lens vs Tuned Lens",
        filename="0G_commitment_top1.png",
    )


def plot_commitment_kl(models: list[str], threshold: float = 0.1) -> dict:
    """Raw vs Tuned KL commitment."""
    raw_key = f"commitment_layer_raw_kl_{threshold}"
    tuned_key = f"commitment_layer_tuned_{threshold}"
    return plot_histogram(
        models,
        series=[
            ("pt", raw_key,   "#9E9E9E", "PT raw KL",   0.3, 0, "stepfilled"),
            ("it", raw_key,   "#1565C0", "IT raw KL",   0.9, 2, "step"),
            ("pt", tuned_key, "#E53935", "PT tuned KL", 0.7, 1.5, "step"),
            ("it", tuned_key, "#FF9800", "IT tuned KL", 0.9, 2.0, "step"),
        ],
        title=f"KL Commitment (< {threshold} nats): Raw vs Tuned Lens",
        filename="0G_commitment_kl.png",
    )


# ── Summary bar chart ────────────────────────────────────────────────────

def plot_summary_bar(models: list[str], threshold: float = 0.1) -> dict:
    """Bar chart: delay (IT median - PT median) by method."""
    methods = [
        ("Raw Top-1",   "commitment_layer_raw",        "#9E9E9E"),
        ("Tuned Top-1", "commitment_layer_top1_tuned", "#1565C0"),
        ("Raw KL",      f"commitment_layer_raw_kl_{threshold}", "#E65100"),
        ("Tuned KL",    f"commitment_layer_tuned_{threshold}", "#1B5E20"),
    ]

    results = {}  # method -> {model -> delay}
    for method_label, key, _ in methods:
        results[method_label] = {}
        for model in models:
            pt = load_commitment(model, "pt")
            it = load_commitment(model, "it")
            if pt and key in pt and it and key in it:
                pt_vals = [v for v in pt[key] if v is not None]
                it_vals = [v for v in it[key] if v is not None]
                if pt_vals and it_vals:
                    results[method_label][model] = {
                        "pt_median": float(np.median(pt_vals)),
                        "it_median": float(np.median(it_vals)),
                        "delay": float(np.median(it_vals) - np.median(pt_vals)),
                    }

    # Find models with at least some data
    models_with_data = [m for m in models
                        if any(m in results[ml] for ml, _, _ in methods)]
    if not models_with_data:
        log.warning("No data for summary bar chart")
        return results

    n_methods = len(methods)
    n_models = len(models_with_data)
    x = np.arange(n_models)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.8), 5))

    for i, (method_label, _, color) in enumerate(methods):
        delays = [results[method_label].get(m, {}).get("delay", 0)
                  for m in models_with_data]
        offset = i * width - (n_methods - 1) * width / 2
        bars = ax.bar(x + offset, delays, width, label=method_label,
                      color=color, alpha=0.8)
        # Value labels
        for bar, d in zip(bars, delays):
            if d != 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{d:+.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_with_data],
                       fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Commitment Delay (IT - PT median layers)", fontsize=10)
    ax.axhline(0, color="black", lw=0.5, ls="--")
    ax.legend(fontsize=8)
    ax.set_title("Commitment Delay Across Methods and Models",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "0G_commitment_summary.png")

    return results


# ── Scatter plot ─────────────────────────────────────────────────────────

def plot_scatter(models: list[str]) -> None:
    """Raw vs tuned top-1 commitment scatter."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), squeeze=False)
    any_data = False

    for col, model_name in enumerate(models):
        ax = axes[0, col]
        spec = get_spec(model_name)

        for variant, color in [("pt", "#9E9E9E"), ("it", "#1565C0")]:
            data = load_commitment(model_name, variant)
            if data is None:
                continue
            raw = data.get("commitment_layer_raw", [])
            tuned = data.get("commitment_layer_top1_tuned", [])
            n_pts = min(len(raw), len(tuned))
            if n_pts == 0:
                continue

            raw_arr = np.array(raw[:n_pts])
            tuned_arr = np.array(tuned[:n_pts])

            # Subsample
            if n_pts > 5000:
                idx = np.random.default_rng(42).choice(n_pts, 5000, replace=False)
                raw_arr, tuned_arr = raw_arr[idx], tuned_arr[idx]

            ax.scatter(raw_arr, tuned_arr, c=color, s=3, alpha=0.15,
                       rasterized=True)
            r = np.corrcoef(raw_arr, tuned_arr)[0, 1] if len(raw_arr) > 2 else 0
            ax.scatter([], [], c=color, s=10,
                       label=f"{variant.upper()} r={r:.2f} (n={n_pts})")
            any_data = True

        ax.plot([0, spec.n_layers], [0, spec.n_layers], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(-0.5, spec.n_layers + 0.5)
        ax.set_ylim(-0.5, spec.n_layers + 0.5)
        ax.set_aspect("equal")
        ax.set_title(MODEL_LABELS.get(model_name, model_name),
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Raw commitment", fontsize=8)
        if col == 0:
            ax.set_ylabel("Tuned commitment", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, markerscale=2)

    fig.suptitle("Raw vs Tuned Top-1 Commitment", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if any_data:
        _save_fig(fig, "0G_commitment_scatter.png")
    else:
        plt.close(fig)


# ── Summary table ────────────────────────────────────────────────────────

def save_summary(models: list[str], threshold: float, bar_results: dict) -> None:
    """Save machine-readable summary."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    out = {
        "threshold": threshold,
        "models": {},
    }

    for model in models:
        model_data = {}
        for method_name, method_results in bar_results.items():
            if model in method_results:
                model_data[method_name] = method_results[model]
        if model_data:
            out["models"][model] = model_data

    out_path = DATA_DIR / "0G_commitment_summary.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Data → %s", out_path)

    # Also save to cross_model/plots
    cm_path = PLOT_DIR_CROSS / "tuned_lens_summary_table.json"
    PLOT_DIR_CROSS.mkdir(parents=True, exist_ok=True)
    with open(cm_path, "w") as f:
        json.dump(out, f, indent=2)

    # Print table
    print("\n=== Commitment Delay Summary ===\n")
    print(f"{'Model':<22} {'Method':<12} {'PT med':>7} {'IT med':>7} {'Delay':>7}")
    print("-" * 60)
    for model in models:
        for method_name in bar_results:
            d = bar_results[method_name].get(model)
            if d:
                pt = f"{d['pt_median']:.1f}"
                it = f"{d['it_median']:.1f}"
                dl = f"{d['delay']:+.1f}"
                print(f"{MODEL_LABELS.get(model, model):<22} {method_name:<12} {pt:>7} {it:>7} {dl:>7}")


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()))
    p.add_argument("--threshold", type=float, default=0.1)
    args = p.parse_args()

    # Filter to models with data
    available = [m for m in args.models
                 if (BASE_RESULTS / m / "tuned_lens" / "commitment").exists()
                 and list((BASE_RESULTS / m / "tuned_lens" / "commitment").glob("*.jsonl"))]
    log.info("Models with data: %s", available)

    if not available:
        log.warning("No commitment data found.")
        return

    plot_commitment_top1(available)
    plot_commitment_kl(available, args.threshold)
    bar_results = plot_summary_bar(available, args.threshold)
    plot_scatter(available)
    save_summary(available, args.threshold, bar_results)


if __name__ == "__main__":
    main()
