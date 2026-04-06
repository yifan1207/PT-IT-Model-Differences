"""
Exp9: Commitment delay plots — ALL 7 methods × raw/tuned logit-lens.

Each plot clearly labels what lens and method is being used.
Uses tuned_lens_commitment_{variant}.jsonl data.
"""
from __future__ import annotations
import json, logging
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "deepseek_v2_lite", "olmo2_7b"]
N_LAYERS = {"gemma3_4b":34, "llama31_8b":32, "qwen3_4b":36, "mistral_7b":32, "deepseek_v2_lite":27, "olmo2_7b":32}
LABELS = {"gemma3_4b":"Gemma 3 4B", "llama31_8b":"Llama 3.1 8B", "qwen3_4b":"Qwen 3 4B",
          "mistral_7b":"Mistral 7B v0.3", "deepseek_v2_lite":"DeepSeek-V2-Lite", "olmo2_7b":"OLMo 2 7B"}
BASE = Path("results/cross_model")
OUT = Path("results/exp9/plots")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    log.info("Saved %s", OUT / name)
    plt.close(fig)


def _load_commitment(model: str, variant: str) -> list[dict] | None:
    """Load all commitment records for a model-variant."""
    path = BASE / model / "tuned_lens" / "commitment" / f"tuned_lens_commitment_{variant}.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _extract_commits(records: list[dict], key: str) -> list[int]:
    """Flatten per-step commitment values across all prompts."""
    values = []
    for rec in records:
        if key in rec:
            values.extend(rec[key])
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# Commitment histogram — generic 6-panel
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_commitment_6panel(key_pt_it: str, title: str, filename: str, lens_label: str):
    """
    6-panel commitment histogram. PT vs IT for each model.
    key_pt_it: the commitment key in the JSONL (same key used for both variants).
    lens_label: "Raw Logit-Lens" or "Tuned Logit-Lens" etc.
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Commitment Delay: {title}\n[{lens_label}]", fontsize=13, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5

        for variant, color, label_v in [("pt", "#2196F3", "PT"), ("it", "#E53935", "IT")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            commits = _extract_commits(recs, key_pt_it)
            if not commits:
                continue
            med = np.median(commits)
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{label_v} (n={len(commits):,}, med={med:.0f})")
            ax.axvline(med, color=color, ls="--", lw=1.5, alpha=0.8)

        ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.4)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SET 1: Top-1 commitment (raw + tuned)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_top1():
    _plot_commitment_6panel(
        "commitment_layer_raw", "Top-1 No-Flip-Back",
        "L2_commitment_raw_top1.png", "Raw Logit-Lens")
    _plot_commitment_6panel(
        "commitment_layer_top1_tuned", "Top-1 No-Flip-Back",
        "L2_commitment_tuned_top1.png", "Tuned Logit-Lens")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SET 2: KL commitment at τ=0.1 (raw + tuned)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_kl():
    _plot_commitment_6panel(
        "commitment_layer_raw_kl_0.1", "KL(ℓ ‖ final) < 0.1 nats",
        "L2_commitment_raw_kl_0.1.png", "Raw Logit-Lens")
    _plot_commitment_6panel(
        "commitment_layer_tuned_0.1", "KL(ℓ ‖ final) < 0.1 nats",
        "L2_commitment_tuned_kl_0.1.png", "Tuned Logit-Lens")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SET 3: Cosine commitment at θ=0.95
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_cosine():
    _plot_commitment_6panel(
        "commitment_layer_cosine_0.95", "cos(h_ℓ, h_final) > 0.95",
        "L2_commitment_cosine_0.95.png", "Residual Stream (no logit-lens)")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SET 4: Entropy commitment at τ=0.2
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_entropy():
    _plot_commitment_6panel(
        "commitment_layer_entropy_0.2", "|H_ℓ - H_final| < 0.2 nats",
        "L2_commitment_entropy_0.2.png", "Raw Logit-Lens Entropy")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SET 5: Majority commitment at τ=0.1
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_majority():
    _plot_commitment_6panel(
        "commitment_layer_majority_0.1", "≥90% subsequent layers KL < 0.1",
        "L2_commitment_majority_0.1.png", "Tuned Logit-Lens (majority vote)")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SET 6: Qualified commitment — raw KL τ=0.5 qual 3x and 5x
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_qualified():
    _plot_commitment_6panel(
        "commitment_layer_raw_kl_qual_0.5_3x", "Raw KL < 0.5, qualified 3×",
        "L2_commitment_raw_kl_qual_0.5_3x.png", "Raw Logit-Lens (qualified)")
    _plot_commitment_6panel(
        "commitment_layer_tuned_kl_qual_0.5_3x", "Tuned KL < 0.5, qualified 3×",
        "L2_commitment_tuned_kl_qual_0.5_3x.png", "Tuned Logit-Lens (qualified)")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT: Summary bar chart — IT-PT delay × method × model
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_summary_bar():
    methods = [
        ("commitment_layer_raw",         "Raw top-1",      "#1565C0"),
        ("commitment_layer_top1_tuned",  "Tuned top-1",    "#1B5E20"),
        ("commitment_layer_raw_kl_0.1",  "Raw KL(0.1)",    "#E65100"),
        ("commitment_layer_tuned_0.1",   "Tuned KL(0.1)",  "#B71C1C"),
        ("commitment_layer_cosine_0.95", "Cosine(0.95)",   "#4A148C"),
        ("commitment_layer_entropy_0.2", "Entropy(0.2)",   "#006064"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Commitment Delay: IT − PT (median, in normalized depth)\nPositive = IT commits later",
                 fontsize=13, fontweight="bold")

    x = np.arange(len(MODELS))
    width = 0.12
    offsets = np.linspace(-width * 2.5, width * 2.5, len(methods))

    for mi, (key, label, color) in enumerate(methods):
        delays = []
        for model in MODELS:
            nl = N_LAYERS[model]
            pt_recs = _load_commitment(model, "pt")
            it_recs = _load_commitment(model, "it")
            if pt_recs and it_recs:
                pt_c = _extract_commits(pt_recs, key)
                it_c = _extract_commits(it_recs, key)
                if pt_c and it_c:
                    delay = (np.median(it_c) - np.median(pt_c)) / nl
                    delays.append(delay)
                else:
                    delays.append(0)
            else:
                delays.append(0)

        bars = ax.bar(x + offsets[mi], delays, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("Normalized delay (IT − PT median) / n_layers")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, "L2_commitment_summary_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT: CDF — commitment by normalized depth, all models
# ═══════════════════════════════════════════════════════════════════════════════
def plot_commitment_cdf():
    methods = [
        ("commitment_layer_raw",         "Raw Top-1"),
        ("commitment_layer_top1_tuned",  "Tuned Top-1"),
        ("commitment_layer_raw_kl_0.1",  "Raw KL(0.1)"),
        ("commitment_layer_tuned_0.1",   "Tuned KL(0.1)"),
    ]
    model_colors = {
        "gemma3_4b": "#1565C0", "llama31_8b": "#B71C1C", "qwen3_4b": "#1B5E20",
        "mistral_7b": "#E65100", "deepseek_v2_lite": "#4A148C", "olmo2_7b": "#006064",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Commitment CDF by Normalized Depth\nSolid = IT, Dashed = PT", fontsize=13, fontweight="bold")

    for mi, (key, method_label) in enumerate(methods):
        ax = axes[mi // 2, mi % 2]
        ax.set_title(method_label, fontsize=11)

        for model in MODELS:
            nl = N_LAYERS[model]
            color = model_colors[model]
            for variant, ls in [("pt", "--"), ("it", "-")]:
                recs = _load_commitment(model, variant)
                if recs is None:
                    continue
                commits = _extract_commits(recs, key)
                if not commits:
                    continue
                norm = np.array(commits) / nl
                sorted_norm = np.sort(norm)
                cdf = np.arange(1, len(sorted_norm) + 1) / len(sorted_norm)
                label = f"{LABELS[model]} {variant.upper()}" if ls == "-" else None
                ax.plot(sorted_norm, cdf, color=color, ls=ls, lw=1.2,
                        label=label, alpha=0.8)

        ax.set_xlabel("Normalized depth")
        ax.set_ylabel("Fraction committed")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "L2_commitment_cdf_4methods.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT: KL threshold sensitivity — median commitment vs τ
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kl_sensitivity():
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL Threshold Sensitivity: Median Commitment vs τ\n[Raw and Tuned Logit-Lens]",
                 fontsize=13, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]

        for variant, ls in [("pt", "--"), ("it", "-")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue

            # Raw KL
            raw_meds = []
            for t in thresholds:
                c = _extract_commits(recs, f"commitment_layer_raw_kl_{t}")
                raw_meds.append(np.median(c) / nl if c else np.nan)

            # Tuned KL
            tuned_meds = []
            for t in thresholds:
                c = _extract_commits(recs, f"commitment_layer_tuned_{t}")
                tuned_meds.append(np.median(c) / nl if c else np.nan)

            label_v = variant.upper()
            ax.plot(thresholds, raw_meds, "o-", color="#1565C0", ls=ls, lw=1.2, markersize=4,
                    label=f"Raw {label_v}" if idx == 0 else None)
            ax.plot(thresholds, tuned_meds, "s-", color="#E53935", ls=ls, lw=1.2, markersize=4,
                    label=f"Tuned {label_v}" if idx == 0 else None)

        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)")
        ax.set_ylabel("Median commitment (norm. depth)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)

    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "L2_kl_threshold_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT: Raw vs Tuned commitment scatter — per model
# ═══════════════════════════════════════════════════════════════════════════════
def plot_raw_vs_tuned_scatter():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Raw vs Tuned Top-1 Commitment (per generation step)\nBelow diagonal = tuned commits earlier",
                 fontsize=13, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]

        for variant, color, label_v in [("pt", "#2196F3", "PT"), ("it", "#E53935", "IT")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            raw = _extract_commits(recs, "commitment_layer_raw")
            tuned = _extract_commits(recs, "commitment_layer_top1_tuned")
            if raw and tuned and len(raw) == len(tuned):
                # Subsample for plotting speed
                n = min(len(raw), 20000)
                idx_sub = np.random.default_rng(42).choice(len(raw), n, replace=False)
                r = np.array(raw)[idx_sub]
                t = np.array(tuned)[idx_sub]
                ax.scatter(r, t, s=1, alpha=0.15, color=color, label=label_v, rasterized=True)

        ax.plot([0, nl], [0, nl], "k--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Raw top-1 commitment")
        ax.set_ylabel("Tuned top-1 commitment")
        ax.legend(fontsize=8, markerscale=5)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "L2_raw_vs_tuned_scatter.png")


if __name__ == "__main__":
    log.info("Generating exp9 commitment plots (all methods, raw + tuned)...")

    plot_commitment_top1()      # 2 plots: raw + tuned
    plot_commitment_kl()        # 2 plots: raw + tuned
    plot_commitment_cosine()    # 1 plot: residual stream
    plot_commitment_entropy()   # 1 plot: entropy-based
    plot_commitment_majority()  # 1 plot: majority vote
    plot_commitment_qualified() # 2 plots: qualified raw + tuned
    plot_commitment_summary_bar()  # 1 plot: IT-PT delay summary
    plot_commitment_cdf()       # 1 plot: 4-method CDF
    plot_kl_sensitivity()       # 1 plot: threshold sensitivity
    plot_raw_vs_tuned_scatter() # 1 plot: raw vs tuned scatter

    log.info("All commitment plots saved to %s", OUT)
