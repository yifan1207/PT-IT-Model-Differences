"""Exp10 multi-model steering — generate all comparison figures.

Plots generated:
  dose_response_governance.png  — 6-panel: G2 vs alpha, d_mean vs d_conv per model
  dose_response_panel.png       — 6x4 panel: G1, G2, structural_token_ratio, MMLU
  safety_panel.png              — 6-panel: S1 refuse/comply/incoherent rates vs alpha
  cross_model_summary.png       — All models on one plot: normalized G2 vs alpha
  d_mean_vs_d_conv_delta.png    — Bar chart: G2 drop at alpha=0 for d_mean vs d_conv

Usage:
  uv run python scripts/plot_exp10_steering.py
  uv run python scripts/plot_exp10_steering.py --models gemma3_4b llama31_8b
  uv run python scripts/plot_exp10_steering.py --results-dir results/exp10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

ALL_MODELS = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"]
DIRECTIONS = ["d_mean", "d_conv"]

MODEL_LABELS = {
    "gemma3_4b":        "Gemma 3 4B",
    "llama31_8b":       "LLaMA 3.1 8B",
    "qwen3_4b":         "Qwen 3 4B",
    "mistral_7b":       "Mistral 7B",
    "deepseek_v2_lite": "DeepSeek-V2-Lite",
    "olmo2_7b":         "OLMo 2 7B",
}

MODEL_COLORS = {
    "gemma3_4b":        "#1565C0",
    "llama31_8b":       "#B71C1C",
    "qwen3_4b":         "#1B5E20",
    "mistral_7b":       "#E65100",
    "deepseek_v2_lite": "#4A148C",
    "olmo2_7b":         "#006064",
}

# Alpha values in canonical sort order
ALPHA_ORDER = [-5, -3, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]

# Matplotlib style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file, returning empty list if missing."""
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_alpha(condition: str) -> float | None:
    """Extract numeric alpha from condition string. Returns None for non-alpha conditions."""
    if condition == "A1_baseline":
        return 1.0
    if condition.startswith("A1_alpha_"):
        try:
            return float(condition[len("A1_alpha_"):])
        except ValueError:
            return None
    return None


def _steering_dir(results_dir: Path, model: str, direction: str) -> Path:
    return results_dir / model / "steering" / f"A1_{direction}_{model}"


def load_scores(results_dir: Path, model: str, direction: str) -> dict[str, dict[float, float]]:
    """Load programmatic scores.

    Returns {benchmark: {alpha: value}}.
    For duplicate (condition, benchmark) pairs, takes the last value.
    """
    path = _steering_dir(results_dir, model, direction) / "scores.jsonl"
    rows = _load_jsonl(path)
    result: dict[str, dict[float, float]] = defaultdict(dict)
    for row in rows:
        alpha = _parse_alpha(row["condition"])
        if alpha is None:
            continue
        result[row["benchmark"]][alpha] = row["value"]
    return dict(result)


def load_judge_scores(results_dir: Path, model: str, direction: str) -> list[dict]:
    """Load raw LLM judge scores."""
    path = _steering_dir(results_dir, model, direction) / "llm_judge_v2_scores.jsonl"
    return _load_jsonl(path)


def _judge_by_task_alpha(rows: list[dict], task: str) -> dict[float, list]:
    """Group judge rows by alpha for a given task.

    Returns {alpha: [score_values]}.
    """
    grouped: dict[float, list] = defaultdict(list)
    for row in rows:
        if row.get("task") != task:
            continue
        alpha = _parse_alpha(row["condition"])
        if alpha is None:
            continue
        grouped[alpha].append(row)
    return dict(grouped)


def _bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float, float]:
    """Bootstrap confidence interval. Returns (mean, ci_low, ci_high)."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(42)
    means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    mean = np.mean(values)
    tail = (1 - ci) / 2
    lo = np.percentile(means, 100 * tail)
    hi = np.percentile(means, 100 * (1 - tail))
    return float(mean), float(lo), float(hi)


def _compute_g_series(judge_rows: list[dict], task: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute mean + 95% CI for a governance task (g1/g2) across alphas.

    Returns (alphas, means, ci_lo, ci_hi) sorted by alpha.
    """
    by_alpha = _judge_by_task_alpha(judge_rows, task)
    alphas_present = sorted(a for a in by_alpha if a in ALPHA_ORDER)
    alphas_out, means, lo, hi = [], [], [], []
    for a in alphas_present:
        scores = np.array([r["score"] for r in by_alpha[a]], dtype=float)
        m, cl, ch = _bootstrap_ci(scores)
        alphas_out.append(a)
        means.append(m)
        lo.append(cl)
        hi.append(ch)
    return np.array(alphas_out), np.array(means), np.array(lo), np.array(hi)


def _compute_s1_rates(judge_rows: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute S1 class rates (REFUSE, COMPLY, INCOHERENT) per alpha.

    Returns (alphas, {class_name: rate_array}).
    """
    by_alpha = _judge_by_task_alpha(judge_rows, "s1")
    alphas_present = sorted(a for a in by_alpha if a in ALPHA_ORDER)
    classes = ["REFUSE", "COMPLY", "INCOHERENT"]
    rates = {c: [] for c in classes}
    alphas_out = []
    for a in alphas_present:
        rows = by_alpha[a]
        n = len(rows)
        if n == 0:
            continue
        alphas_out.append(a)
        counts = defaultdict(int)
        for r in rows:
            cls = r.get("s1_class", "INCOHERENT")
            counts[cls] += 1
        for c in classes:
            rates[c].append(counts[c] / n)
    return np.array(alphas_out), {c: np.array(v) for c, v in rates.items()}


def _save_fig(fig: plt.Figure, plot_dir: Path, filename: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Saved -> %s", out)
    plt.close(fig)


# ── Plot 1: Dose-response governance (G2 money figure) ──────────────────────

def plot_dose_response_governance(results_dir: Path, models: list[str], plot_dir: Path) -> None:
    """6-panel figure: G2 mean vs alpha, d_mean (blue) vs d_conv (gray dashed)."""
    n = len(models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]

        for direction, color, ls, lw, label in [
            ("d_mean", MODEL_COLORS.get(model, "#1565C0"), "-", 2.0, r"$d_{\mathrm{mean}}$"),
            ("d_conv", "#9E9E9E", "--", 1.5, r"$d_{\mathrm{conv}}$ (convergence-gap)"),
        ]:
            judge = load_judge_scores(results_dir, model, direction)
            if not judge:
                continue
            alphas, means, lo, hi = _compute_g_series(judge, "g2")
            if len(alphas) == 0:
                continue
            ax.plot(alphas, means, color=color, ls=ls, lw=lw, label=label, marker="o", ms=3, zorder=3)
            ax.fill_between(alphas, lo, hi, color=color, alpha=0.15, zorder=1)

        ax.axvline(1.0, color="black", ls=":", lw=0.8, alpha=0.5, zorder=0)
        ax.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("G2 score (1-5)")
        ax.set_ylim(0.5, 5.5)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="lower left", framealpha=0.9)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Dose-Response: Conversational Register (G2) Under Steering", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, plot_dir, "dose_response_governance.png")


# ── Plot 2: Dose-response panel (6x4) ───────────────────────────────────────

def plot_dose_response_panel(results_dir: Path, models: list[str], plot_dir: Path) -> None:
    """6-row x 4-col panel: G1, G2, structural_token_ratio, MMLU."""
    metrics = [
        ("G1", "judge", "g1"),
        ("G2", "judge", "g2"),
        ("Structural Token Ratio", "prog", "structural_token_ratio"),
        ("MMLU", "prog", "mmlu_forced_choice"),
    ]
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 4, figsize=(20, 3.5 * n_models), squeeze=False)

    for row, model in enumerate(models):
        for col, (label, source, key) in enumerate(metrics):
            ax = axes[row][col]

            for direction, color, ls, lw, dir_label in [
                ("d_mean", MODEL_COLORS.get(model, "#1565C0"), "-", 2.0, r"$d_{\mathrm{mean}}$"),
                ("d_conv", "#9E9E9E", "--", 1.5, r"$d_{\mathrm{conv}}$"),
            ]:
                if source == "judge":
                    judge = load_judge_scores(results_dir, model, direction)
                    if not judge:
                        continue
                    alphas, means, lo, hi = _compute_g_series(judge, key)
                    if len(alphas) == 0:
                        continue
                    ax.plot(alphas, means, color=color, ls=ls, lw=lw, marker="o", ms=2, label=dir_label)
                    ax.fill_between(alphas, lo, hi, color=color, alpha=0.12)
                else:
                    scores = load_scores(results_dir, model, direction)
                    bench_data = scores.get(key, {})
                    if not bench_data:
                        continue
                    sorted_alphas = sorted(a for a in bench_data if a in ALPHA_ORDER)
                    vals = [bench_data[a] for a in sorted_alphas]
                    ax.plot(sorted_alphas, vals, color=color, ls=ls, lw=lw, marker="o", ms=2, label=dir_label)

            ax.axvline(1.0, color="black", ls=":", lw=0.8, alpha=0.5)
            if row == 0:
                ax.set_title(label, fontweight="bold")
            if col == 0:
                ax.set_ylabel(MODEL_LABELS.get(model, model), fontsize=9, fontweight="bold")
            if row == n_models - 1:
                ax.set_xlabel(r"$\alpha$")
            if row == 0 and col == 0:
                ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle("Multi-Model Dose-Response: Governance vs Content Under Steering",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, plot_dir, "dose_response_panel.png")


# ── Plot 3: Safety panel (S1 stacked area) ──────────────────────────────────

def plot_safety_panel(results_dir: Path, models: list[str], plot_dir: Path) -> None:
    """6-panel: S1 refuse/comply/incoherent stacked area vs alpha for d_mean."""
    n = len(models)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    class_colors = {"REFUSE": "#4CAF50", "COMPLY": "#E53935", "INCOHERENT": "#9E9E9E"}
    class_order = ["REFUSE", "COMPLY", "INCOHERENT"]

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        judge = load_judge_scores(results_dir, model, "d_mean")
        if not judge:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(MODEL_LABELS.get(model, model))
            continue

        alphas, rates = _compute_s1_rates(judge)
        if len(alphas) == 0:
            ax.text(0.5, 0.5, "No S1 data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(MODEL_LABELS.get(model, model))
            continue

        # Stacked area
        bottom = np.zeros(len(alphas))
        for cls in class_order:
            vals = rates.get(cls, np.zeros(len(alphas)))
            ax.fill_between(alphas, bottom, bottom + vals, color=class_colors[cls],
                            alpha=0.7, label=cls.capitalize(), zorder=2)
            bottom = bottom + vals

        ax.axvline(1.0, color="black", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(MODEL_LABELS.get(model, model), fontweight="bold")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("S1 rate")
        ax.set_ylim(0, 1.05)
        if idx == 0:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Safety Alignment (S1) Under Steering", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, plot_dir, "safety_panel.png")


# ── Plot 4: Cross-model summary (normalized G2) ─────────────────────────────

def plot_cross_model_summary(results_dir: Path, models: list[str], plot_dir: Path) -> None:
    """Single figure: normalized G2 (divided by baseline) vs alpha, one line per model."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model in models:
        judge = load_judge_scores(results_dir, model, "d_mean")
        if not judge:
            continue
        alphas, means, lo, hi = _compute_g_series(judge, "g2")
        if len(alphas) == 0:
            continue

        # Normalize by baseline (alpha=1.0)
        baseline_mask = alphas == 1.0
        if not np.any(baseline_mask):
            log.warning("No baseline (alpha=1.0) for %s, skipping normalization", model)
            continue
        baseline_val = means[baseline_mask][0]
        if baseline_val == 0:
            continue

        norm_means = means / baseline_val
        norm_lo = lo / baseline_val
        norm_hi = hi / baseline_val

        color = MODEL_COLORS.get(model, "#333333")
        ax.plot(alphas, norm_means, color=color, lw=2, marker="o", ms=4,
                label=MODEL_LABELS.get(model, model), zorder=3)
        ax.fill_between(alphas, norm_lo, norm_hi, color=color, alpha=0.10, zorder=1)

    ax.axvline(1.0, color="black", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(1.0, color="black", ls="-", lw=0.5, alpha=0.3)
    ax.set_xlabel(r"$\alpha$", fontsize=12)
    ax.set_ylabel("G2 / G2(baseline)", fontsize=12)
    ax.set_title("Cross-Model: Normalized Governance Under Steering", fontweight="bold", fontsize=13)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    _save_fig(fig, plot_dir, "cross_model_summary.png")


# ── Plot 5: d_mean vs d_conv delta bar chart ────────────────────────────────

def plot_d_mean_vs_d_conv_delta(results_dir: Path, models: list[str], plot_dir: Path) -> None:
    """Bar chart: G2 drop from baseline to alpha=0, d_mean vs d_conv per model."""
    model_names = []
    d_mean_deltas = []
    d_conv_deltas = []

    for model in models:
        deltas = {}
        for direction in ["d_mean", "d_conv"]:
            judge = load_judge_scores(results_dir, model, direction)
            if not judge:
                deltas[direction] = np.nan
                continue
            by_alpha = _judge_by_task_alpha(judge, "g2")
            baseline_scores = by_alpha.get(1.0, [])
            zero_scores = by_alpha.get(0.0, [])
            if not baseline_scores or not zero_scores:
                deltas[direction] = np.nan
                continue
            baseline_mean = np.mean([r["score"] for r in baseline_scores])
            zero_mean = np.mean([r["score"] for r in zero_scores])
            deltas[direction] = baseline_mean - zero_mean  # positive = drop

        if np.isnan(deltas.get("d_mean", np.nan)) and np.isnan(deltas.get("d_conv", np.nan)):
            continue
        model_names.append(MODEL_LABELS.get(model, model))
        d_mean_deltas.append(deltas.get("d_mean", np.nan))
        d_conv_deltas.append(deltas.get("d_conv", np.nan))

    if not model_names:
        log.warning("No data for d_mean vs d_conv delta plot.")
        return

    x = np.arange(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width / 2, d_mean_deltas, width, label=r"$d_{\mathrm{mean}}$",
                   color="#1565C0", alpha=0.85, zorder=3)
    bars2 = ax.bar(x + width / 2, d_conv_deltas, width, label=r"$d_{\mathrm{conv}}$ (convergence-gap)",
                   color="#9E9E9E", alpha=0.85, zorder=3)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02,
                        f"{height:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("G2 drop (baseline - alpha=0)", fontsize=11)
    ax.set_title(r"Causal Specificity: $d_{\mathrm{mean}}$ vs $d_{\mathrm{conv}}$ at $\alpha=0$",
                 fontweight="bold", fontsize=13)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.axhline(0, color="black", lw=0.5, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, plot_dir, "d_mean_vs_d_conv_delta.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exp10 multi-model steering plots")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Subset of models to plot (default: all available)")
    parser.add_argument("--results-dir", type=str, default="results/exp10",
                        help="Root results directory (default: results/exp10)")
    parser.add_argument("--plots", nargs="+", default=None,
                        choices=["governance", "panel", "safety", "summary", "delta"],
                        help="Subset of plots to generate (default: all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plot_dir = results_dir / "plots"

    # Determine available models
    if args.models:
        models = args.models
    else:
        models = []
        for m in ALL_MODELS:
            # Check if at least d_mean steering dir exists
            if _steering_dir(results_dir, m, "d_mean").exists():
                models.append(m)
            else:
                log.info("Skipping %s (no steering results found)", m)
    if not models:
        log.error("No models with steering results found in %s", results_dir)
        sys.exit(1)

    log.info("Models: %s", models)
    log.info("Output: %s", plot_dir)

    plot_funcs = {
        "governance": ("dose_response_governance.png", plot_dose_response_governance),
        "panel":      ("dose_response_panel.png",      plot_dose_response_panel),
        "safety":     ("safety_panel.png",              plot_safety_panel),
        "summary":    ("cross_model_summary.png",       plot_cross_model_summary),
        "delta":      ("d_mean_vs_d_conv_delta.png",    plot_d_mean_vs_d_conv_delta),
    }

    selected = args.plots or list(plot_funcs.keys())
    for key in selected:
        fname, func = plot_funcs[key]
        log.info("Generating %s ...", fname)
        try:
            func(results_dir, models, plot_dir)
        except Exception:
            log.exception("Failed to generate %s", fname)

    log.info("Done.")


if __name__ == "__main__":
    main()
