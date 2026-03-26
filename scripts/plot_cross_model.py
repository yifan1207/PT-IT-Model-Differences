"""
Cross-model replication study — generate all comparison figures.

Plots generated:
  L1_heatmaps.png         — 6×2 δ-cosine heatmaps (model × PT/IT)
  L1_summary.png          — cross-model summary: normalized depth vs mean δ-cosine
  L3_weight_diff.png      — per-model per-layer RMS weight diff (attn/mlp/norm)
  L8_id_profiles.png      — intrinsic dimensionality PT vs IT per model
  L9_attn_entropy.png     — attention entropy divergence (IT-PT) per model
  L9_summary.png          — cross-model entropy divergence (normalized depth)
  summary_panel.png       — 6-row × 4-column overview figure (all metrics)

Usage:
  uv run python scripts/plot_cross_model.py
  uv run python scripts/plot_cross_model.py --models gemma3_4b llama31_8b
  uv run python scripts/plot_cross_model.py --skip L8 L9   # skip missing data
"""
from __future__ import annotations

import json
import math
import argparse
import logging
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE_RESULTS = Path("results/cross_model")
PLOT_DIR = BASE_RESULTS / "plots"

# Model display names for plot labels
MODEL_LABELS = {
    "gemma3_4b":       "Gemma 3 4B",
    "llama31_8b":      "Llama 3.1 8B",
    "qwen3_4b":        "Qwen 3 4B",
    "mistral_7b":      "Mistral 7B v0.3",
    "deepseek_v2_lite": "DeepSeek-V2-Lite",
    "olmo2_7b":        "OLMo 2 7B",
}

# Color palette for models
MODEL_COLORS = {
    "gemma3_4b":       "#1565C0",
    "llama31_8b":      "#B71C1C",
    "qwen3_4b":        "#1B5E20",
    "mistral_7b":      "#E65100",
    "deepseek_v2_lite": "#4A148C",
    "olmo2_7b":        "#006064",
}


# ── data loaders ──────────────────────────────────────────────────────────────

def _load_mean_cosine(model_name: str, variant: str) -> np.ndarray | None:
    path = BASE_RESULTS / model_name / variant / "L1L2_mean_cosine.npy"
    if not path.exists():
        return None
    return np.load(path)


def _load_weight_diff(model_name: str) -> dict | None:
    path = BASE_RESULTS / model_name / "weight_diff.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_id_profile(model_name: str, variant: str) -> dict | None:
    path = BASE_RESULTS / model_name / variant / "L8_id_profile.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_attn_entropy(model_name: str, variant: str) -> dict | None:
    path = BASE_RESULTS / model_name / variant / "L9_attn_entropy.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── L1: δ-cosine summary ─────────────────────────────────────────────────────

def plot_L1_summary(models: list[str]) -> None:
    """Cross-model δ-cosine profile vs normalized depth.
    PT = dashed; IT = solid. X = fractional depth (0-1).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False

    for model_name in models:
        spec = get_spec(model_name)
        color = MODEL_COLORS[model_name]
        label = MODEL_LABELS[model_name]
        depth = np.linspace(0, 1, spec.n_layers)

        for variant, ls, lw, alpha in [("pt", "--", 1.5, 0.7), ("it", "-", 2.0, 1.0)]:
            mc = _load_mean_cosine(model_name, variant)
            if mc is None:
                continue
            lbl = f"{label} IT" if variant == "it" else None
            ax.plot(depth, mc, color=color, linestyle=ls, linewidth=lw,
                    alpha=alpha, label=lbl)
            any_data = True

        # Predicted corrective onset line
        onset = spec.corrective_onset / spec.n_layers
        ax.axvline(onset, color=color, linestyle=":", linewidth=0.8, alpha=0.4)

    if not any_data:
        log.warning("No L1 data available for summary plot.")
        plt.close()
        return

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-", alpha=0.3)
    ax.axvline(0.60, color="grey", linewidth=1.0, linestyle="--", alpha=0.5,
               label="Predicted corrective onset (0.60)")
    ax.set_xlabel("Normalized depth (layer / n_layers)")
    ax.set_ylabel("Mean δ-cosine (steps 10–100)")
    ax.set_title("L1: δ-Cosine Profile Across 6 Model Families")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "L1_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", out)


# ── L3: weight diff ───────────────────────────────────────────────────────────

def plot_L3_weight_diff(models: list[str]) -> None:
    """Per-model per-layer RMS weight diff, decomposed into attn/mlp/norm."""
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        spec = get_spec(model_name)
        data = _load_weight_diff(model_name)
        if data is None:
            ax.text(0.5, 0.5, f"{model_name}: no data", transform=ax.transAxes, ha="center")
            continue

        layers = np.arange(spec.n_layers)
        delta_attn = np.array(data.get("delta_attn", [0] * spec.n_layers))
        delta_mlp  = np.array(data.get("delta_mlp",  [0] * spec.n_layers))
        delta_norm = np.array(data.get("delta_norm", [0] * spec.n_layers))

        # Stacked bar
        ax.bar(layers, delta_attn, label="Attention", color="#1565C0", alpha=0.8)
        ax.bar(layers, delta_mlp,  bottom=delta_attn, label="MLP", color="#E65100", alpha=0.8)
        ax.bar(layers, delta_norm, bottom=delta_attn + delta_mlp, label="Norm", color="#9E9E9E", alpha=0.8)

        # Shade corrective region
        onset = spec.corrective_onset
        ax.axvspan(onset - 0.5, spec.n_layers - 0.5, color="green", alpha=0.08, label="Corrective region")
        ax.axvspan(spec.phase_boundary - 0.5, spec.phase_boundary + 0.5, color="red", alpha=0.15, label="Phase boundary")

        ax.set_title(f"{MODEL_LABELS[model_name]}", fontsize=10)
        ax.set_ylabel("RMS diff")
        ax.set_xlabel("Layer")
        if ax is axes[0]:
            ax.legend(fontsize=7, ncol=4)

    fig.suptitle("L3: Per-Layer Weight Change Localization (PT → IT)", fontsize=12, y=1.01)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "L3_weight_diff.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", out)


# ── L8: intrinsic dimensionality ─────────────────────────────────────────────

def plot_L8_id_profiles(models: list[str]) -> None:
    """PT vs IT intrinsic dimensionality profiles per model."""
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        spec = get_spec(model_name)
        depth = np.linspace(0, 1, spec.n_layers)

        for variant, color, ls, label in [
            ("pt", "#1565C0", "--", "PT"),
            ("it", "#B71C1C", "-",  "IT"),
        ]:
            d = _load_id_profile(model_name, variant)
            if d is None:
                continue
            id_vals = np.array(d["id_twonn"])
            ci_low  = np.array(d["id_ci_low"])
            ci_high = np.array(d["id_ci_high"])
            ax.plot(depth, id_vals, color=color, linestyle=ls, linewidth=2, label=label)
            ax.fill_between(depth, ci_low, ci_high, color=color, alpha=0.15)

        ax.axvline(1/3, color="grey", linestyle=":", linewidth=1.0, alpha=0.7, label="1/3 depth")
        ax.set_title(MODEL_LABELS[model_name], fontsize=9)
        ax.set_xlabel("Normalized depth")
        if ax is axes[0]:
            ax.set_ylabel("Intrinsic dimension (TwoNN)")
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("L8: Intrinsic Dimensionality Profile", fontsize=12)
    fig.tight_layout()

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "L8_id_profiles.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", out)


# ── L9: attention entropy ─────────────────────────────────────────────────────

def plot_L9_attn_entropy(models: list[str]) -> None:
    """IT−PT attention entropy divergence per model, normalized depth."""
    fig, ax = plt.subplots(figsize=(10, 5))
    any_data = False

    for model_name in models:
        spec = get_spec(model_name)
        pt_data = _load_attn_entropy(model_name, "pt")
        it_data = _load_attn_entropy(model_name, "it")
        if pt_data is None or it_data is None:
            continue

        pt_mean = np.array(pt_data["mean_entropy"])   # [n_layers, n_heads]
        it_mean = np.array(it_data["mean_entropy"])

        # Per-layer mean over heads (mean entropy)
        pt_layer = np.nanmean(pt_mean, axis=1)
        it_layer = np.nanmean(it_mean, axis=1)
        divergence = it_layer - pt_layer

        depth = np.linspace(0, 1, spec.n_layers)
        ax.plot(depth, divergence, color=MODEL_COLORS[model_name], linewidth=2,
                label=MODEL_LABELS[model_name])
        any_data = True

    if not any_data:
        log.warning("No L9 data available.")
        plt.close()
        return

    ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
    ax.axvline(1/3, color="grey", linestyle="--", linewidth=1.0, alpha=0.5,
               label="Predicted phase boundary (0.33)")
    ax.set_xlabel("Normalized depth")
    ax.set_ylabel("Attention entropy divergence (IT − PT, nats)")
    ax.set_title("L9: Attention Entropy Divergence Across 6 Model Families")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / "L9_attn_entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", out)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate cross-model replication plots.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                        help="Models to include (default: all 6)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["L1", "L3", "L8", "L9"],
                        help="Experiments to skip (e.g., if data not yet available)")
    args = parser.parse_args()

    models = args.models
    skip = set(args.skip)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Generating cross-model plots for: %s", models)

    if "L1" not in skip:
        log.info("Plotting L1 summary...")
        plot_L1_summary(models)

    if "L3" not in skip:
        log.info("Plotting L3 weight diff...")
        plot_L3_weight_diff(models)

    if "L8" not in skip:
        log.info("Plotting L8 ID profiles...")
        plot_L8_id_profiles(models)

    if "L9" not in skip:
        log.info("Plotting L9 attention entropy...")
        plot_L9_attn_entropy(models)

    log.info("All plots saved to %s", PLOT_DIR)


if __name__ == "__main__":
    main()
