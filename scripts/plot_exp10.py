"""
Exp10 plots: Contrastive Activation Patching results.

Generates 5 figures:
  1. R² by layer (6-panel) — convergence-gap probe fit
  2. Probe magnitude ‖w‖ by layer (6-panel)
  3. Cosine(d_conv, d_mean) by layer (6-panel) — THE key comparison
  4. Patching ΔKL by condition (6-panel) — causal effect on downstream KL
  5. PCA explained variance (6-panel) — rank diagnostic

All data read from results/exp10/{model}/probes/probe_summary.json
and results/exp10/{model}/patching/patching_summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

MODELS = [
    "gemma3_4b", "llama31_8b", "qwen3_4b",
    "mistral_7b", "deepseek_v2_lite", "olmo2_7b",
]

MODEL_LABELS = {
    "gemma3_4b": "Gemma 3 4B",
    "llama31_8b": "LLaMA 3.1 8B",
    "qwen3_4b": "Qwen 3 4B",
    "mistral_7b": "Mistral 7B v0.3",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
    "olmo2_7b": "OLMo 2 7B",
}

BASE = Path("results/exp10")
PLOT_DIR = BASE / "plots"
DATA_DIR = PLOT_DIR / "data"

CONDITION_LABELS = {
    "commit": r"$d_{\rm conv}$",
    "full": r"Full $\Delta h$",
    "random": "Random",
    "mean": r"$d_{\rm mean}$",
    "orthogonal": r"Orthogonal",
}

CONDITION_COLORS = {
    "commit": "#d62728",
    "full": "#2ca02c",
    "random": "#7f7f7f",
    "mean": "#1f77b4",
    "orthogonal": "#ff7f0e",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_probe_summaries() -> dict:
    """Load probe_summary.json for all models."""
    data = {}
    for m in MODELS:
        path = BASE / m / "probes" / "probe_summary.json"
        if path.exists():
            with open(path) as f:
                data[m] = json.load(f)
    return data


def load_patching_summaries() -> dict:
    """Load patching_summary.json for all models."""
    data = {}
    for m in MODELS:
        path = BASE / m / "patching" / "patching_summary.json"
        if path.exists():
            with open(path) as f:
                data[m] = json.load(f)
    return data


# ── Plot 1: R² by layer ──────────────────────────────────────────────────────

def plot_r2_by_layer(probe_data: dict):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in probe_data:
            ax.set_title(f"{MODEL_LABELS[m]} (no data)")
            continue

        layers = [r["layer"] for r in probe_data[m]["per_layer"]]
        r2 = [r["r2_test"] for r in probe_data[m]["per_layer"]]
        onset = probe_data[m].get("corrective_onset", 20)

        ax.bar(layers, r2, color="#2ca02c", alpha=0.7, width=0.8)
        ax.axvline(onset, color="red", ls="--", alpha=0.5, label=f"Onset={onset}")
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel(r"$R^2$ (test)")
        ax.legend(fontsize=7)

    fig.suptitle("Exp10: Convergence-Gap Probe R² by Layer", fontsize=14)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "r2_by_layer_6panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'r2_by_layer_6panel.png'}")


# ── Plot 2: Probe magnitude by layer ────────────────────────────────────────

def plot_probe_magnitude(probe_data: dict):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in probe_data:
            ax.set_title(f"{MODEL_LABELS[m]} (no data)")
            continue

        layers = [r["layer"] for r in probe_data[m]["per_layer"]]
        mag = [r["probe_magnitude"] for r in probe_data[m]["per_layer"]]
        onset = probe_data[m].get("corrective_onset", 20)

        ax.bar(layers, mag, color="#1f77b4", alpha=0.7, width=0.8)
        ax.axvline(onset, color="red", ls="--", alpha=0.5, label=f"Onset={onset}")
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel(r"$\|w\|$")
        ax.legend(fontsize=7)

    fig.suptitle("Exp10: Convergence-Gap Probe Weight Magnitude by Layer", fontsize=14)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "probe_magnitude_6panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'probe_magnitude_6panel.png'}")


# ── Plot 3: Cosine(d_commit, d_mean) by layer ───────────────────────────────

def plot_cosine_with_mean(probe_data: dict):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in probe_data:
            ax.set_title(f"{MODEL_LABELS[m]} (no data)")
            continue

        layers = [r["layer"] for r in probe_data[m]["per_layer"]]
        cos_mean = [r["cosine_with_mean_dir"] for r in probe_data[m]["per_layer"]]
        cos_grad = [r["cosine_with_kl_gradient"] for r in probe_data[m]["per_layer"]]
        onset = probe_data[m].get("corrective_onset", 20)

        ax.plot(layers, cos_mean, "o-", color="#d62728", ms=3, lw=1.2, label=r"cos($d_{\rm commit}$, $d_{\rm mean}$)")
        ax.plot(layers, cos_grad, "s-", color="#1f77b4", ms=3, lw=1.2, alpha=0.6, label=r"cos($d_{\rm commit}$, $d_{\rm grad}$)")
        ax.axvline(onset, color="red", ls="--", alpha=0.3)
        ax.axhline(0.95, color="gray", ls=":", alpha=0.5, label="Redundancy threshold")
        ax.axhline(0, color="black", ls="-", alpha=0.2)
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-0.3, 1.05)
        ax.legend(fontsize=6)

    fig.suptitle("Exp10: Cosine of Convergence-Gap Direction with Mean / Gradient Direction", fontsize=13)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "cosine_with_mean_6panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'cosine_with_mean_6panel.png'}")


# ── Plot 4: Patching Δc by condition ─────────────────────────────────────────

def plot_patching_effect(patching_data: dict):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    conditions = ["commit", "full", "random", "mean", "orthogonal"]

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in patching_data:
            ax.set_title(f"{MODEL_LABELS[m]} (no data)")
            continue

        pdata = patching_data[m]
        # Aggregate across layers
        cond_means = {}
        cond_stds = {}
        for cond in conditions:
            all_vals = []
            for layer_key, layer_data in pdata.items():
                if cond in layer_data:
                    mean_dc = layer_data[cond].get("mean_delta_kl_downstream",
                                                      layer_data[cond].get("mean_delta_c", 0))
                    n = layer_data[cond].get("n", 0)
                    if n > 0:
                        all_vals.append(mean_dc)
            cond_means[cond] = np.mean(all_vals) if all_vals else 0
            cond_stds[cond] = np.std(all_vals) if len(all_vals) > 1 else 0

        x = np.arange(len(conditions))
        bars = ax.bar(
            x,
            [cond_means[c] for c in conditions],
            yerr=[cond_stds[c] for c in conditions],
            color=[CONDITION_COLORS[c] for c in conditions],
            alpha=0.8, capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=7, rotation=15)
        ax.axhline(0, color="black", ls="-", alpha=0.3)
        ax.set_title(MODEL_LABELS[m])
        if idx % 3 == 0:
            ax.set_ylabel(r"Mean $\Delta$KL downstream (nats)")

    fig.suptitle("Exp10: Causal Patching Effect on KL Convergence (Mean Across Top Layers)", fontsize=13)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "patching_causal_effect.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'patching_causal_effect.png'}")


# ── Plot 5: PCA explained variance ──────────────────────────────────────────

def plot_pca_rank(probe_data: dict):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    axes = axes.flatten()

    for idx, m in enumerate(MODELS):
        ax = axes[idx]
        if m not in probe_data:
            ax.set_title(f"{MODEL_LABELS[m]} (no data)")
            continue

        # Average PCA explained variance across corrective layers
        onset = probe_data[m].get("corrective_onset", 20)
        n_layers = probe_data[m].get("n_layers", 34)
        corrective_pca = []
        for r in probe_data[m]["per_layer"]:
            if r["layer"] >= onset:
                pca = r.get("pca_explained_var", [0] * 10)
                corrective_pca.append(pca)

        if corrective_pca:
            avg_pca = np.mean(corrective_pca, axis=0)
        else:
            avg_pca = np.zeros(10)

        x = np.arange(1, len(avg_pca) + 1)
        ax.bar(x, avg_pca, color="#9467bd", alpha=0.8)
        ax.axhline(0.6, color="red", ls="--", alpha=0.5, label="Rank-1 threshold (60%)")
        ax.set_title(MODEL_LABELS[m])
        ax.set_xlabel("PC index")
        if idx % 3 == 0:
            ax.set_ylabel("Explained variance ratio")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=7)

    fig.suptitle("Exp10: PCA of KL-Excess-Weighted Δh (Corrective Layers Avg)", fontsize=13)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "pca_rank_6panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'pca_rank_6panel.png'}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    probe_data = load_probe_summaries()
    patching_data = load_patching_summaries()

    if not probe_data:
        print("No probe data found. Run exp10 first.")
        return

    print(f"Loaded probe data for: {list(probe_data.keys())}")
    print(f"Loaded patching data for: {list(patching_data.keys())}")

    # Export data for paper
    with open(DATA_DIR / "probe_summaries.json", "w") as f:
        json.dump(probe_data, f, indent=2)
    with open(DATA_DIR / "patching_summaries.json", "w") as f:
        json.dump(patching_data, f, indent=2)

    plot_r2_by_layer(probe_data)
    plot_probe_magnitude(probe_data)
    plot_cosine_with_mean(probe_data)
    if patching_data:
        plot_patching_effect(patching_data)
    plot_pca_rank(probe_data)

    # Print summary table
    print("\n" + "=" * 80)
    print("EXP10 SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Go/NoGo':<12} {'Mean|cos|':<12} {'Peak R²':<10} {'Peak Layer':<12}")
    print("-" * 80)
    for m in MODELS:
        if m not in probe_data:
            print(f"{MODEL_LABELS[m]:<20} {'N/A':<12}")
            continue
        d = probe_data[m]
        go = d.get("go_nogo", "?")
        mcos = d.get("mean_corrective_cosine_abs", 0)
        best_r2 = max(r["r2_test"] for r in d["per_layer"])
        best_layer = max(d["per_layer"], key=lambda r: r["r2_test"])["layer"]
        print(f"{MODEL_LABELS[m]:<20} {go:<12} {mcos:<12.3f} {best_r2:<10.4f} {best_layer:<12}")


if __name__ == "__main__":
    main()
