"""
Exp9: Cross-model replication — all plots.

Reads from:
  - results/cross_model/{model}/{variant}/L1L2_heatmap_{variant}.npy  (δ-cosine)
  - results/cross_model/{model}/{variant}/L1L2_results.jsonl          (commitment)
  - results/cross_model/{model}/weight_diff.json                      (L3)
  - results/cross_model/{model}/{variant}/L8_id_profile.json          (ID)
  - results/cross_model/{model}/{variant}/L9_summary.json             (entropy)
  - results/cross_model/{model}/tuned_lens/commitment/arrays/         (tuned lens)

Outputs to: results/exp09_cross_model_observational_replication/plots/
"""
from __future__ import annotations
import json, logging, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "deepseek_v2_lite", "olmo2_7b"]
N_LAYERS = {"gemma3_4b":34, "llama31_8b":32, "qwen3_4b":36, "mistral_7b":32, "deepseek_v2_lite":27, "olmo2_7b":32}
LABELS = {"gemma3_4b":"Gemma 3 4B", "llama31_8b":"Llama 3.1 8B", "qwen3_4b":"Qwen 3 4B",
          "mistral_7b":"Mistral 7B v0.3", "deepseek_v2_lite":"DeepSeek-V2-Lite", "olmo2_7b":"OLMo 2 7B"}
COLORS_PT = "#2196F3"
COLORS_IT = "#E53935"
BASE = Path("results/cross_model")
OUT = Path("results/exp09_cross_model_observational_replication/plots")
DATA_OUT = Path("results/exp09_cross_model_observational_replication/data")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    log.info("Saved %s", OUT / name)
    plt.close(fig)


def _load_cosine_profile(model, variant):
    """Load heatmap and return mean profile [n_layers]."""
    for fname in [f"L1L2_heatmap_{variant}.npy", "L1L2_mean_cosine.npy"]:
        p = BASE / model / variant / fname
        if p.exists():
            arr = np.load(p)
            if arr.ndim == 2:
                end = min(100, arr.shape[0])
                start = min(10, end - 1)
                return np.nanmean(arr[start:end], axis=0)
            return arr
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 1: L1 δ-cosine — 6 panels, PT vs IT per model
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L1_6panel():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.suptitle("L1: δ-Cosine Profile — PT vs IT per Model", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        pt = _load_cosine_profile(model, "pt")
        it = _load_cosine_profile(model, "it")

        if pt is not None:
            ax.plot(x_norm, pt, color=COLORS_PT, linewidth=1.5, label="PT (base)", alpha=0.8)
        if it is not None:
            ax.plot(x_norm, it, color=COLORS_IT, linewidth=1.5, label="IT (instruct)", alpha=0.8)

        # Shade corrective region
        onset = 0.60
        ax.axvspan(onset, 1.0, alpha=0.08, color="gray")
        ax.axvline(onset, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axhline(0, color="black", ls="-", lw=0.5, alpha=0.3)

        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Mean δ-cosine")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, 1)

    fig.tight_layout()
    _save(fig, "L1_delta_cosine_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 2: L1 δ-cosine heatmaps — 6 rows × 2 cols (PT | IT)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L1_heatmaps():
    fig, axes = plt.subplots(6, 2, figsize=(14, 20))
    fig.suptitle("L1: δ-Cosine Heatmap (generation step × layer)", fontsize=14, fontweight="bold", y=0.995)

    for row, model in enumerate(MODELS):
        nl = N_LAYERS[model]
        for col, variant in enumerate(["pt", "it"]):
            ax = axes[row, col]
            for fname in [f"L1L2_heatmap_{variant}.npy", "L1L2_mean_cosine.npy"]:
                p = BASE / model / variant / fname
                if p.exists():
                    h = np.load(p)
                    break
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            if h.ndim == 1:
                h = h.reshape(1, -1)

            im = ax.imshow(h.T, aspect="auto", cmap="RdBu_r", vmin=-0.1, vmax=0.1,
                          origin="lower", interpolation="nearest")
            onset_layer = round(nl * 0.6)
            ax.axhline(onset_layer - 0.5, color="black", ls="--", lw=0.8, alpha=0.6)

            title = f"{LABELS[model]} — {'PT' if variant == 'pt' else 'IT'}"
            ax.set_title(title, fontsize=9)
            if col == 0:
                ax.set_ylabel("Layer")
            ax.set_xlabel("Generation step")

    fig.colorbar(im, ax=axes, shrink=0.6, label="δ-cosine", pad=0.02)
    fig.tight_layout()
    _save(fig, "L1_heatmaps_6x2.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 3: L2 commitment — 6 panels, PT vs IT histograms
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L2_commitment():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L2: Commitment Delay — First Layer Where Top-1 Token Locks In", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]

        for variant, color, label in [("pt", COLORS_PT, "PT"), ("it", COLORS_IT, "IT")]:
            path = BASE / model / variant / "L1L2_results.jsonl"
            if not path.exists():
                continue
            commits = []
            with open(path) as f:
                for line in f:
                    d = json.loads(line)
                    if d["n_steps"] > 0:
                        commits.extend(d["commitment_layer"])
            if commits:
                bins = np.arange(nl + 1) - 0.5
                ax.hist(commits, bins=bins, density=True, alpha=0.5, color=color, label=f"{label} (med={np.median(commits):.0f})")
                ax.axvline(np.median(commits), color=color, ls="--", lw=1.5, alpha=0.8)

        ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.5, label="Corrective onset")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "L2_commitment_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 4: L3 weight change — 6 panels, MLP vs Attention
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L3_weight_diff():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L3: Per-Layer Weight Change Localization (PT → IT)", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        path = BASE / model / "weight_diff.json"
        if not path.exists():
            continue
        d = json.load(open(path))
        nl = N_LAYERS[model]
        layers = np.arange(nl)

        ax.plot(layers, d["delta_mlp"], color="#E65100", lw=1.5, label="MLP")
        ax.plot(layers, d["delta_attn"], color="#1565C0", lw=1.5, label="Attention")
        if "delta_norm" in d:
            ax.plot(layers, d["delta_norm"], color="#4CAF50", lw=1, alpha=0.6, label="Norm")

        ax.axvspan(round(nl * 0.6), nl, alpha=0.08, color="gray")
        ax.axvline(round(nl * 0.33), color="red", ls="--", lw=0.8, alpha=0.5, label="Phase boundary")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel("RMS weight diff")
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "L3_weight_diff_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 5: L8 intrinsic dimensionality — 6 panels, PT vs IT
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L8_id():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L8: Intrinsic Dimensionality Profile (TwoNN)", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        for variant, color, label in [("pt", COLORS_PT, "PT"), ("it", COLORS_IT, "IT")]:
            path = BASE / model / variant / "L8_id_profile.json"
            if not path.exists():
                continue
            d = json.load(open(path))
            ids = np.array(d["intrinsic_dim"])
            ax.plot(x_norm, ids, color=color, lw=1.5, label=label)

        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5, label="1/3 depth")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Intrinsic dim")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "L8_id_profiles_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 6: L9 attention entropy divergence — 6 panels, IT-PT
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L9_entropy():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L9: Attention Entropy (IT − PT) per Model", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        pt_path = BASE / model / "pt" / "L9_summary.json"
        it_path = BASE / model / "it" / "L9_summary.json"
        if not pt_path.exists() or not it_path.exists():
            continue

        pt_ent = np.array(json.load(open(pt_path))["mean_entropy"])
        it_ent = np.array(json.load(open(it_path))["mean_entropy"])
        # Average across heads
        diff = np.mean(it_ent, axis=1) - np.mean(pt_ent, axis=1)

        ax.fill_between(x_norm, diff, 0, where=diff > 0, alpha=0.3, color=COLORS_IT, label="IT > PT")
        ax.fill_between(x_norm, diff, 0, where=diff < 0, alpha=0.3, color=COLORS_PT, label="PT > IT")
        ax.plot(x_norm, diff, color="black", lw=1.2)

        ax.axhline(0, color="black", ls="-", lw=0.5, alpha=0.3)
        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Entropy diff (IT − PT, nats)")
        ax.legend(fontsize=7, loc="lower left")

    fig.tight_layout()
    _save(fig, "L9_entropy_divergence_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 7: Entropy profiles (raw + tuned) from one-pass data — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def _load_tuned_lens_arrays(model):
    """Load tuned lens arrays for a model. Returns dict of arrays or None."""
    base = BASE / model / "tuned_lens" / "commitment" / "arrays"
    if not base.exists():
        return None
    result = {}
    for name in ["delta_cosine", "raw_entropy", "tuned_entropy", "raw_kl_final",
                 "tuned_kl_final", "raw_kl_adj", "tuned_kl_adj",
                 "raw_ntprob", "tuned_ntprob", "raw_ntrank", "tuned_ntrank",
                 "raw_top1", "tuned_top1", "generated_ids", "cosine_h_to_final"]:
        p = base / f"{name}.npy"
        if p.exists():
            result[name] = np.load(p)
    # Load step_index
    si_path = base / "step_index.jsonl"
    if si_path.exists():
        with open(si_path) as f:
            result["step_index"] = [json.loads(l) for l in f]
    return result


def plot_entropy_profiles():
    """Raw vs tuned logit-lens entropy by layer — from one-pass arrays."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Entropy Profiles: Raw vs Tuned Logit-Lens", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "raw_entropy" not in data:
            ax.text(0.5, 0.5, "No tuned lens data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        raw_ent = np.nanmean(data["raw_entropy"].astype(np.float32), axis=0)
        tuned_ent = np.nanmean(data["tuned_entropy"].astype(np.float32), axis=0)

        ax.plot(x_norm, raw_ent, color="#1565C0", lw=1.5, label="Raw logit-lens")
        ax.plot(x_norm, tuned_ent, color="#E53935", lw=1.5, label="Tuned-lens")

        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Mean entropy (nats)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "entropy_profiles_raw_vs_tuned_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 8: Adjacent-layer KL (three-phase) — raw + tuned — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def plot_adjacent_kl():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Adjacent-Layer KL Divergence: Raw vs Tuned Logit-Lens", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "raw_kl_adj" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        raw_adj = np.nanmean(data["raw_kl_adj"].astype(np.float32), axis=0)
        tuned_adj = np.nanmean(data["tuned_kl_adj"].astype(np.float32), axis=0)

        ax.plot(x_norm, raw_adj, color="#1565C0", lw=1.5, label="Raw KL(ℓ‖ℓ-1)")
        ax.plot(x_norm, tuned_adj, color="#E53935", lw=1.5, label="Tuned KL(ℓ‖ℓ-1)")

        # Shade three phases
        ax.axvspan(0, 0.33, alpha=0.05, color="green", label="Early")
        ax.axvspan(0.33, 0.60, alpha=0.05, color="yellow")
        ax.axvspan(0.60, 1.0, alpha=0.05, color="red")
        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0.60, color="gray", ls="--", lw=0.8, alpha=0.5)

        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("KL(layer ℓ ‖ layer ℓ-1)")
        ax.legend(fontsize=7)

    fig.tight_layout()
    _save(fig, "adjacent_kl_raw_vs_tuned_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 9: Token emergence — generated token rank across layers — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def plot_emergence():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Token Emergence: Rank of Generated Token Across Layers", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "raw_ntrank" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        # Median rank (lower = token is more prominent)
        raw_rank = np.nanmedian(data["raw_ntrank"].astype(np.float32), axis=0)
        tuned_rank = np.nanmedian(data["tuned_ntrank"].astype(np.float32), axis=0)

        ax.plot(x_norm, raw_rank, color="#1565C0", lw=1.5, label="Raw logit-lens")
        ax.plot(x_norm, tuned_rank, color="#E53935", lw=1.5, label="Tuned-lens")

        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0.60, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_yscale("log")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Median rank (log scale)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "emergence_rank_raw_vs_tuned_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 10: KL-to-final — how quickly does the distribution converge? — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kl_to_final():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL(layer ℓ ‖ final layer) — Prediction Convergence", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "raw_kl_final" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        raw_kl = np.nanmedian(data["raw_kl_final"].astype(np.float32), axis=0)
        tuned_kl = np.nanmedian(data["tuned_kl_final"].astype(np.float32), axis=0)

        ax.plot(x_norm, raw_kl, color="#1565C0", lw=1.5, label="Raw logit-lens")
        ax.plot(x_norm, tuned_kl, color="#E53935", lw=1.5, label="Tuned-lens")

        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0.60, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Median KL (nats)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "kl_to_final_raw_vs_tuned_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 11: Cosine-to-final h_ℓ · h_L — representational convergence — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cosine_to_final():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Cosine(h_ℓ, h_final) — Residual Stream Convergence", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "cosine_h_to_final" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        cos = data["cosine_h_to_final"].astype(np.float32)
        mean_cos = np.nanmean(cos, axis=0)
        p25 = np.nanpercentile(cos, 25, axis=0)
        p75 = np.nanpercentile(cos, 75, axis=0)

        ax.fill_between(x_norm, p25, p75, alpha=0.2, color="#1565C0")
        ax.plot(x_norm, mean_cos, color="#1565C0", lw=1.5, label="Mean (IQR band)")

        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0.60, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("cos(h_ℓ, h_final)")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)

    fig.tight_layout()
    _save(fig, "cosine_to_final_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 12: Next-token probability — how confident is each layer? — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def plot_next_token_prob():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Next-Token Probability by Layer (Raw vs Tuned Lens)", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "raw_ntprob" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        raw_prob = np.nanmean(data["raw_ntprob"].astype(np.float32), axis=0)
        tuned_prob = np.nanmean(data["tuned_ntprob"].astype(np.float32), axis=0)

        ax.plot(x_norm, raw_prob, color="#1565C0", lw=1.5, label="Raw logit-lens")
        ax.plot(x_norm, tuned_prob, color="#E53935", lw=1.5, label="Tuned-lens")

        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.axvline(0.60, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("P(generated token)")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "next_token_prob_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT 13: δ-cosine from tuned lens arrays (same prompt set) — 6 panels
# ═══════════════════════════════════════════════════════════════════════════════
def plot_delta_cosine_tuned_lens():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("δ-Cosine from Tuned-Lens Eval Pass (same prompts as lens data)", fontsize=14, fontweight="bold")

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "delta_cosine" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        nl = N_LAYERS[model]
        x_norm = np.arange(nl) / nl

        dc = data["delta_cosine"].astype(np.float32)
        mean_dc = np.nanmean(dc, axis=0)
        p25 = np.nanpercentile(dc, 25, axis=0)
        p75 = np.nanpercentile(dc, 75, axis=0)

        ax.fill_between(x_norm, p25, p75, alpha=0.2, color="#1565C0")
        ax.plot(x_norm, mean_dc, color="#1565C0", lw=1.5, label="Mean (IQR band)")

        ax.axhline(0, color="black", ls="-", lw=0.5, alpha=0.3)
        ax.axvline(0.60, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("δ-cosine")
        ax.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, "delta_cosine_tuned_lens_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# Export critical data to exp9/data
# ═══════════════════════════════════════════════════════════════════════════════
def export_data():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model in MODELS:
        nl = N_LAYERS[model]
        entry = {"n_layers": nl, "label": LABELS[model]}

        # L1 profiles
        for variant in ["pt", "it"]:
            prof = _load_cosine_profile(model, variant)
            if prof is not None:
                entry[f"L1_delta_cosine_{variant}"] = prof.tolist()

        # L3
        path = BASE / model / "weight_diff.json"
        if path.exists():
            d = json.load(open(path))
            entry["L3_delta_mlp"] = d["delta_mlp"]
            entry["L3_delta_attn"] = d["delta_attn"]

        # L8
        for variant in ["pt", "it"]:
            path = BASE / model / variant / "L8_id_profile.json"
            if path.exists():
                entry[f"L8_id_{variant}"] = json.load(open(path))["intrinsic_dim"]

        # L9
        for variant in ["pt", "it"]:
            path = BASE / model / variant / "L9_summary.json"
            if path.exists():
                ent = np.array(json.load(open(path))["mean_entropy"])
                entry[f"L9_mean_entropy_{variant}"] = np.mean(ent, axis=1).tolist()

        # Tuned lens summary stats
        data = _load_tuned_lens_arrays(model)
        if data is not None:
            for key in ["raw_entropy", "tuned_entropy", "raw_kl_adj", "tuned_kl_adj",
                        "raw_kl_final", "tuned_kl_final", "raw_ntprob", "tuned_ntprob",
                        "delta_cosine", "cosine_h_to_final"]:
                if key in data:
                    entry[f"tuned_lens_{key}_mean"] = np.nanmean(
                        data[key].astype(np.float32), axis=0).tolist()

        summary[model] = entry

    out_path = DATA_OUT / "exp9_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Exported data → %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    log.info("Generating exp9 plots...")

    plot_L1_6panel()
    plot_L1_heatmaps()
    plot_L2_commitment()
    plot_L3_weight_diff()
    plot_L8_id()
    plot_L9_entropy()
    plot_entropy_profiles()
    plot_adjacent_kl()
    plot_emergence()
    plot_kl_to_final()
    plot_cosine_to_final()
    plot_next_token_prob()
    plot_delta_cosine_tuned_lens()
    export_data()

    log.info("All exp9 plots saved to %s", OUT)
    log.info("Data exported to %s", DATA_OUT)
