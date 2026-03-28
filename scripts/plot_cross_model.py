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


def _save_fig(fig: plt.Figure, filename: str, models: list[str]) -> None:
    """Save figure to shared plots/ dir and to per-model subfolders."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    out = PLOT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    log.info("Saved → %s", out)
    # Also save to each model's own subfolder
    for model_name in models:
        model_dir = PLOT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(model_dir / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)

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
    # Try both naming conventions
    for name in (f"L1L2_heatmap_{variant}.npy", "L1L2_mean_cosine.npy"):
        path = BASE_RESULTS / model_name / variant / name
        if path.exists():
            arr = np.load(path)
            # heatmap shape: [n_steps, n_layers] → collapse to [n_layers]
            if arr.ndim == 2:
                arr = np.nanmean(arr[10:100], axis=0)
            return arr
    return None


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
    for name in ("L9_attn_entropy.json", "L9_summary.json"):
        path = BASE_RESULTS / model_name / variant / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


# ── L1: per-model δ-cosine heatmaps ──────────────────────────────────────────

def plot_L1_heatmaps(models: list[str]) -> None:
    """N×2 grid (only models with data): rows=models, cols=PT/IT."""
    # Collect (model, variant, array) triples
    data = []
    for m in models:
        for v in ("pt", "it"):
            for name in (f"L1L2_heatmap_{v}.npy", "L1L2_mean_cosine.npy"):
                path = BASE_RESULTS / m / v / name
                if path.exists():
                    arr = np.load(path)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]  # [1, n_layers]
                    data.append((m, v, arr))
                    break

    if not data:
        log.warning("No heatmap data available.")
        return

    # Only include models that have at least one heatmap
    models_with_data = [m for m in models if any(md == m for md, _, _ in data)]
    n_rows = len(models_with_data)

    # Use constrained_layout and reserve right margin for colorbar
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(13, 2.8 * n_rows),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.10, right=0.88, top=0.93, bottom=0.05,
                        hspace=0.45, wspace=0.25)

    global_vmax = 0.0
    im_ref = None

    for row, model_name in enumerate(models_with_data):
        spec = get_spec(model_name)
        for col, variant in enumerate(("pt", "it")):
            ax = axes[row][col]
            arr = next((a for m, v, a in data if m == model_name and v == variant), None)
            if arr is None:
                ax.set_visible(False)
                continue
            # Show steps 5-200 (skip prefill noise at step 0)
            show = arr[5:201] if arr.shape[0] > 200 else arr
            vmax = np.nanpercentile(np.abs(show), 95)
            global_vmax = max(global_vmax, vmax)
            im = ax.imshow(show, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, origin="upper",
                           extent=[0, spec.n_layers, show.shape[0], 0])
            ax.axvline(spec.corrective_onset, color="black", lw=1.2, ls="--", alpha=0.7)
            if row == 0:
                ax.set_title(f"{'PT (base)' if variant == 'pt' else 'IT (instruct)'}", fontsize=10)
            ax.set_ylabel(MODEL_LABELS[model_name], fontsize=8, rotation=90, labelpad=4)
            ax.set_xlabel("Layer", fontsize=7)
            ax.tick_params(labelsize=7)
            im_ref = im

    fig.suptitle("L1: δ-Cosine Heatmap (generation step × layer)", fontsize=12, fontweight="bold")
    # Single colorbar in the reserved right margin
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.70])
    fig.colorbar(im_ref, cax=cax, label="δ-cosine")
    _save_fig(fig, "L1_heatmaps.png", models_with_data)


def plot_L2_commitment(models: list[str]) -> None:
    """Commitment delay: distribution of commitment layer per model, PT vs IT."""
    fig, axes = plt.subplots(1, len(models), figsize=(3.5 * len(models), 4), squeeze=False)

    for col, model_name in enumerate(models):
        ax = axes[0][col]
        spec = get_spec(model_name)
        plotted = False
        # Collect both variants first, then draw PT behind IT
        variant_data = {}
        for variant in ("pt", "it"):
            path = BASE_RESULTS / model_name / variant / "L1L2_results.jsonl"
            if not path.exists():
                continue
            commit_layers = []
            with open(path) as f:
                for line in f:
                    if line.strip():
                        try:
                            r = json.loads(line)
                            commit_layers.extend([c for c in r["commitment_layer"] if c is not None])
                        except (KeyError, json.JSONDecodeError):
                            pass
            if commit_layers:
                variant_data[variant] = commit_layers

        bins = np.arange(0, spec.n_layers + 2) - 0.5
        # Draw PT as light fill, IT as step outline on top — always distinguishable
        for variant, color, label, histtype, alpha, lw, ls in [
            ("pt", "#9E9E9E", "PT", "stepfilled", 0.35, 0.0, "--"),
            ("it", "#1565C0", "IT", "step",       1.0,  2.0, "-"),
        ]:
            if variant not in variant_data:
                continue
            vals = variant_data[variant]
            ax.hist(vals, bins=bins, density=True, alpha=alpha,
                    color=color, label=label, histtype=histtype, linewidth=lw)
            med = np.median(vals)
            # Median lines: PT dashed behind, IT solid on top
            ax.axvline(med, color=color, lw=1.5, ls=ls,
                       zorder=3 if variant == "it" else 2)
            plotted = True
        ax.axvline(spec.corrective_onset, color="green", lw=1.2, ls=":", alpha=0.7, label="Corrective onset")
        ax.set_title(MODEL_LABELS[model_name], fontsize=9, fontweight="bold")
        ax.set_xlabel("Commitment layer", fontsize=8)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)
        if col == 0:
            ax.legend(fontsize=7)
        if not plotted:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")

    fig.suptitle("L2: Commitment Delay — First Layer Where Top-1 Token Locks In", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "L2_commitment.png", models)


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

    _save_fig(fig, "L1_summary.png", models)


# ── L3: weight diff ───────────────────────────────────────────────────────────

def plot_L3_weight_diff(models: list[str]) -> None:
    """Per-model per-layer RMS weight diff: MLP vs Attention lines (exp3 plot8 style)."""
    n = len(models)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        spec = get_spec(model_name)
        data = _load_weight_diff(model_name)
        if data is None:
            ax.text(0.5, 0.5, f"{model_name}: no data", transform=ax.transAxes, ha="center")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue

        layers = np.arange(spec.n_layers)
        delta_attn = np.array(data.get("delta_attn", [0] * spec.n_layers))
        delta_mlp  = np.array(data.get("delta_mlp",  [0] * spec.n_layers))

        # Normalise by each component's mean so all models share the same y scale
        attn_norm = delta_attn / (delta_attn.mean() + 1e-12)
        mlp_norm  = delta_mlp  / (delta_mlp.mean()  + 1e-12)

        # Line plots matching exp3 style
        ax.plot(layers, mlp_norm,  color="#E65100", lw=2, label="MLP", zorder=3)
        ax.plot(layers, attn_norm, color="#1565C0", lw=2, label="Attention", zorder=3)
        ax.fill_between(layers, mlp_norm,  alpha=0.15, color="#E65100")
        ax.fill_between(layers, attn_norm, alpha=0.15, color="#1565C0")
        ax.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.5)  # mean reference line

        # Phase boundary and corrective region shading
        onset = spec.corrective_onset
        ax.axvspan(onset - 0.5, spec.n_layers - 0.5, color="#2E7D32", alpha=0.07, label="Corrective region")
        ax.axvline(spec.phase_boundary, color="#B71C1C", lw=1.2, ls="--", alpha=0.7, label="Phase boundary")

        ax.set_title(MODEL_LABELS[model_name], fontsize=10, fontweight="bold")
        ax.set_ylabel("Relative weight shift\n(normalised to mean=1)", fontsize=8)
        ax.set_xlabel("Transformer layer", fontsize=8)
        ax.set_xlim(-0.5, spec.n_layers - 0.5)
        ax.tick_params(labelsize=8)
        if ax is axes[0]:
            ax.legend(fontsize=8, ncol=3, loc="upper left")

    fig.suptitle("L3: Per-Layer Weight Change Localization (PT → IT)", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save_fig(fig, "L3_weight_diff.png", models)


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
            id_vals = np.array(d["intrinsic_dim"])
            ci_low  = np.array(d["intrinsic_dim_ci_low"])
            ci_high = np.array(d["intrinsic_dim_ci_high"])
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

    _save_fig(fig, "L8_id_profiles.png", models)


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

    _save_fig(fig, "L9_attn_entropy.png", models)


def plot_L9_attn_entropy_panels(models: list[str]) -> None:
    """Separate panel per model showing IT-PT entropy divergence vs Gemma 3 4B baseline."""
    # Load Gemma reference first
    gemma_pt = _load_attn_entropy("gemma3_4b", "pt")
    gemma_it = _load_attn_entropy("gemma3_4b", "it")
    if gemma_pt is None or gemma_it is None:
        log.warning("No Gemma 3 4B L9 data — cannot plot panels.")
        return
    gemma_spec = get_spec("gemma3_4b")
    gemma_div = (np.nanmean(np.array(gemma_it["mean_entropy"]), axis=1) -
                 np.nanmean(np.array(gemma_pt["mean_entropy"]), axis=1))
    gemma_depth = np.linspace(0, 1, gemma_spec.n_layers)

    other_models = [m for m in models if m != "gemma3_4b"]
    if not other_models:
        log.warning("No non-Gemma models to compare.")
        return

    ncols = 2
    nrows = (len(other_models) + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()

    for idx, model_name in enumerate(other_models):
        ax = axes[idx]
        pt_data = _load_attn_entropy(model_name, "pt")
        it_data = _load_attn_entropy(model_name, "it")

        # Always plot Gemma reference
        ax.plot(gemma_depth, gemma_div, color=MODEL_COLORS["gemma3_4b"],
                linewidth=1.5, linestyle="--", alpha=0.6, label="Gemma 3 4B (ref)")

        if pt_data is None or it_data is None:
            ax.text(0.5, 0.5, f"{MODEL_LABELS.get(model_name, model_name)}: no data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
        else:
            spec = get_spec(model_name)
            pt_layer = np.nanmean(np.array(pt_data["mean_entropy"]), axis=1)
            it_layer = np.nanmean(np.array(it_data["mean_entropy"]), axis=1)
            divergence = it_layer - pt_layer
            depth = np.linspace(0, 1, spec.n_layers)
            ax.plot(depth, divergence, color=MODEL_COLORS[model_name],
                    linewidth=2.0, label=MODEL_LABELS[model_name])

        ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
        ax.axvline(1/3, color="grey", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_title(MODEL_LABELS.get(model_name, model_name), fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalized depth", fontsize=9)
        ax.set_ylabel("Entropy div. (IT−PT, nats)", fontsize=8)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.25)

    # Hide any unused axes
    for idx in range(len(other_models), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("L9: Attention Entropy Divergence — Each Model vs Gemma 3 4B\n"
                 "Dashed = Gemma 3 4B reference  |  Dashed vertical = predicted phase boundary (0.33)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "L9_attn_entropy_panels.png", models)


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
        log.info("Plotting L1 heatmaps...")
        plot_L1_heatmaps(models)
        log.info("Plotting L1 summary...")
        plot_L1_summary(models)

    if "L2" not in skip:
        log.info("Plotting L2 commitment delay...")
        plot_L2_commitment(models)

    if "L3" not in skip:
        log.info("Plotting L3 weight diff...")
        plot_L3_weight_diff(models)

    if "L8" not in skip:
        log.info("Plotting L8 ID profiles...")
        plot_L8_id_profiles(models)

    if "L9" not in skip:
        log.info("Plotting L9 attention entropy...")
        plot_L9_attn_entropy(models)
        plot_L9_attn_entropy_panels(models)

    log.info("All plots saved to %s", PLOT_DIR)


if __name__ == "__main__":
    main()
