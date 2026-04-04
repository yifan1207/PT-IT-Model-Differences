"""
Exp9 plot fixes — regenerate all plots with:
  1. Clear lens/data-source label on EVERY plot
  2. Gemma3 tuned lens marked as BROKEN
  3. DeepSeek tuned lens marked as MIXED
  4. Missing commitment data marked clearly
  5. KL-to-final raw vs tuned as key validation plot
  6. Remove misleading delta_cosine_tuned_lens (was single-variant, no PT vs IT)
"""
from __future__ import annotations
import json, logging, os
from pathlib import Path
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
COL_PT, COL_IT = "#2196F3", "#E53935"

# Tuned lens status per model
TUNED_LENS_STATUS = {
    "gemma3_4b": "BROKEN",       # probes diverge (embedding scaling bug)
    "llama31_8b": "OK",
    "qwen3_4b": "OK",
    "mistral_7b": "OK",
    "deepseek_v2_lite": "MIXED", # entropy worse, prob ok
    "olmo2_7b": "OK",
}

BASE = Path("results/cross_model")
OUT = Path("results/exp9/plots")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    log.info("Saved → %s", OUT / name)
    plt.close(fig)


def _mark_broken(ax, model):
    """Overlay warning on panel if tuned lens is broken for this model."""
    status = TUNED_LENS_STATUS.get(model, "OK")
    if status == "BROKEN":
        ax.text(0.5, 0.5, "TUNED LENS\nBROKEN", transform=ax.transAxes,
                ha="center", va="center", fontsize=16, color="red", alpha=0.4,
                fontweight="bold", rotation=30,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    elif status == "MIXED":
        ax.text(0.98, 0.02, "tuned lens\nquality mixed", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="orange", alpha=0.8,
                fontstyle="italic")


def _load_cosine_profile(model, variant):
    for fname in [f"L1L2_heatmap_{variant}.npy"]:
        p = BASE / model / variant / fname
        if p.exists():
            arr = np.load(p)
            if arr.ndim == 2:
                end = min(100, arr.shape[0])
                start = min(10, end - 1)
                return np.nanmean(arr[start:end], axis=0)
            return arr
    return None


def _load_tuned_lens_arrays(model):
    base = BASE / model / "tuned_lens" / "commitment" / "arrays"
    if not base.exists():
        return None
    result = {}
    for name in ["delta_cosine", "raw_entropy", "tuned_entropy", "raw_kl_final",
                 "tuned_kl_final", "raw_kl_adj", "tuned_kl_adj",
                 "raw_ntprob", "tuned_ntprob", "raw_ntrank", "tuned_ntrank",
                 "cosine_h_to_final"]:
        p = base / f"{name}.npy"
        if p.exists():
            result[name] = np.load(p)
    return result


def _load_commitment(model, variant):
    path = BASE / model / "tuned_lens" / "commitment" / f"tuned_lens_commitment_{variant}.jsonl"
    if not path.exists():
        return None
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _extract_commits(records, key):
    values = []
    for rec in records:
        if key in rec:
            values.extend(rec[key])
    return values


# ═══════════════════════════════════════════════════════════════════════════════
# 1. L1 δ-cosine — 6 panels, PT vs IT
#    Data: L1L2 collect (residual stream, NO logit lens)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L1():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=True)
    fig.suptitle("L1: δ-Cosine Profile — PT vs IT\n[Residual stream — no logit lens]",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        x = np.arange(nl) / nl
        pt = _load_cosine_profile(model, "pt")
        it = _load_cosine_profile(model, "it")
        if pt is not None:
            ax.plot(x, pt, color=COL_PT, lw=2, label="PT (base)")
        if it is not None:
            ax.plot(x, it, color=COL_IT, lw=2, label="IT (instruct)")
        ax.axvspan(0.6, 1.0, alpha=0.06, color="gray")
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Mean δ-cosine (steps 10-100)")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L1_delta_cosine_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. L3 weight diff — 6 panels
#    Data: weight_diff.json (parameter space, NO logit lens)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L3():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L3: Per-Layer Weight Change (PT → IT)\n[Parameter space — no logit lens]",
                 fontsize=14, fontweight="bold")
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
        ax.axvspan(round(nl * 0.6), nl, alpha=0.06, color="gray")
        ax.axvline(round(nl * 0.33), color="red", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Layer")
        if idx % 3 == 0:
            ax.set_ylabel("RMS weight diff")
        ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L3_weight_diff_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. L8 ID profiles — 6 panels, PT vs IT
#    Data: L8 collect (TwoNN on residual stream, NO logit lens)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L8():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L8: Intrinsic Dimensionality (TwoNN) — PT vs IT\n[Residual stream — no logit lens]",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        x = np.arange(nl) / nl
        for variant, color, label in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            path = BASE / model / variant / "L8_id_profile.json"
            if not path.exists():
                continue
            ids = np.array(json.load(open(path))["intrinsic_dim"])
            ax.plot(x, ids, color=color, lw=2, label=label)
        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Intrinsic dimensionality")
        ax.legend(fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L8_id_profiles_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. L9 attention entropy — 6 panels, IT−PT
#    Data: L9 collect (attention weights, NO logit lens)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_L9():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L9: Attention Entropy Divergence (IT − PT)\n[Attention weights — no logit lens]",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        x = np.arange(nl) / nl
        pt_path = BASE / model / "pt" / "L9_summary.json"
        it_path = BASE / model / "it" / "L9_summary.json"
        if not pt_path.exists() or not it_path.exists():
            continue
        pt_ent = np.mean(np.array(json.load(open(pt_path))["mean_entropy"]), axis=1)
        it_ent = np.mean(np.array(json.load(open(it_path))["mean_entropy"]), axis=1)
        diff = it_ent - pt_ent
        ax.fill_between(x, diff, 0, where=diff > 0, alpha=0.3, color=COL_IT)
        ax.fill_between(x, diff, 0, where=diff < 0, alpha=0.3, color=COL_PT)
        ax.plot(x, diff, color="black", lw=1.2)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Entropy diff (IT − PT, nats)")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L9_entropy_divergence_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. KEY TUNED LENS VALIDATION: KL(ℓ ‖ final) — raw vs tuned logit-lens
#    Data: tuned lens NPY arrays (ALL PT variant)
#    Purpose: Shows tuned lens predictions converge to final faster than raw
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kl_to_final_validation():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Tuned Lens Validation: KL(layer ℓ ‖ final layer)\n"
                 "[All models: PT variant — lower = better prediction of final output]",
                 fontsize=13, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "raw_kl_final" not in data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        nl = N_LAYERS[model]
        x = np.arange(nl) / nl
        raw_kl = np.nanmedian(data["raw_kl_final"].astype(np.float32), axis=0)
        tuned_kl = np.nanmedian(data["tuned_kl_final"].astype(np.float32), axis=0)
        ax.plot(x, raw_kl, color=COL_PT, lw=2, label="Raw logit-lens")
        ax.plot(x, tuned_kl, color=COL_IT, lw=2, label="Tuned logit-lens")
        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("Median KL to final (nats)")
        ax.legend(fontsize=9)
        _mark_broken(ax, model)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "tuned_lens_validation_kl_to_final.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Cosine-to-final — residual stream convergence
#    Data: tuned lens NPY arrays (PT variant, but lens-independent metric)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cosine_to_final():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Cosine(h_ℓ, h_final) — Residual Stream Convergence\n"
                 "[PT variant — residual stream, no logit lens]",
                 fontsize=13, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        data = _load_tuned_lens_arrays(model)
        if data is None or "cosine_h_to_final" not in data:
            continue
        nl = N_LAYERS[model]
        x = np.arange(nl) / nl
        cos = data["cosine_h_to_final"].astype(np.float32)
        ax.fill_between(x, np.nanpercentile(cos, 25, axis=0),
                        np.nanpercentile(cos, 75, axis=0), alpha=0.2, color=COL_PT)
        ax.plot(x, np.nanmean(cos, axis=0), color=COL_PT, lw=2, label="Mean (IQR band)")
        ax.axvline(0.33, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Normalized depth")
        if idx % 3 == 0:
            ax.set_ylabel("cos(h_ℓ, h_final)")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "cosine_to_final_6panel.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 7-9. Commitment histograms — generic function with proper data handling
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_commitment(key, title, filename, lens_label):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Commitment Delay: {title}\n[{lens_label}]",
                 fontsize=13, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        has_data = False

        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                ax.text(0.5, 0.85 if variant == "it" else 0.78,
                        f"{lbl}: NO DATA", transform=ax.transAxes,
                        ha="center", fontsize=9, color=color, alpha=0.7)
                continue
            commits = _extract_commits(recs, key)
            if not commits:
                ax.text(0.5, 0.85 if variant == "it" else 0.78,
                        f"{lbl}: no '{key}' field", transform=ax.transAxes,
                        ha="center", fontsize=8, color=color, alpha=0.7)
                continue
            has_data = True
            n = len(commits)
            med = np.median(commits)
            partial = "" if n > 100000 else f" PARTIAL"
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, med={med:.0f}){partial}")
            ax.axvline(med, color=color, ls="--", lw=2, alpha=0.9)

        if has_data:
            ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

        # Mark broken tuned lens if this is a tuned-lens method
        if "tuned" in key.lower() or "majority" in key.lower():
            _mark_broken(ax, model)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, filename)


def plot_all_commitment():
    # === Raw logit-lens methods ===
    _plot_commitment("commitment_layer_raw", "Top-1 No-Flip-Back",
                     "L2_commitment_raw_top1.png", "Raw Logit-Lens")
    _plot_commitment("commitment_layer_raw_kl_0.1", "KL(ℓ ‖ final) < 0.1 nats",
                     "L2_commitment_raw_kl_0.1.png", "Raw Logit-Lens")

    # === Tuned logit-lens methods ===
    _plot_commitment("commitment_layer_top1_tuned", "Top-1 No-Flip-Back",
                     "L2_commitment_tuned_top1.png", "Tuned Logit-Lens")
    _plot_commitment("commitment_layer_tuned_0.1", "KL(ℓ ‖ final) < 0.1 nats",
                     "L2_commitment_tuned_kl_0.1.png", "Tuned Logit-Lens")

    # === Residual-stream method (no logit lens) ===
    _plot_commitment("commitment_layer_cosine_0.95", "cos(h_ℓ, h_final) > 0.95",
                     "L2_commitment_cosine_0.95.png", "Residual Stream — no logit lens")

    # === Entropy method ===
    _plot_commitment("commitment_layer_entropy_0.2", "|H_ℓ - H_final| < 0.2 nats",
                     "L2_commitment_entropy_0.2.png", "Raw Logit-Lens Entropy")

    # === Majority vote ===
    _plot_commitment("commitment_layer_majority_0.1", "≥90% subsequent layers KL < 0.1",
                     "L2_commitment_majority_0.1.png", "Tuned Logit-Lens (majority)")

    # === Qualified ===
    _plot_commitment("commitment_layer_raw_kl_qual_0.5_3x", "KL < 0.5, holds 3 layers",
                     "L2_commitment_raw_kl_qual_0.5_3x.png", "Raw Logit-Lens (qualified)")
    _plot_commitment("commitment_layer_tuned_kl_qual_0.5_3x", "KL < 0.5, holds 3 layers",
                     "L2_commitment_tuned_kl_qual_0.5_3x.png", "Tuned Logit-Lens (qualified)")


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Summary bar — IT−PT delay across methods
# ═══════════════════════════════════════════════════════════════════════════════
def plot_summary_bar():
    methods = [
        ("commitment_layer_raw",         "Raw top-1",      "#1565C0"),
        ("commitment_layer_top1_tuned",  "Tuned top-1",    "#1B5E20"),
        ("commitment_layer_raw_kl_0.1",  "Raw KL(0.1)",    "#E65100"),
        ("commitment_layer_tuned_0.1",   "Tuned KL(0.1)",  "#B71C1C"),
        ("commitment_layer_cosine_0.95", "Cosine(0.95)",   "#4A148C"),
        ("commitment_layer_entropy_0.2", "Entropy(0.2)",   "#006064"),
    ]
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("Commitment Delay: IT − PT (median, normalized by depth)\n"
                 "Positive = IT commits later | Gray = insufficient data",
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
                    delays.append((np.median(it_c) - np.median(pt_c)) / nl)
                else:
                    delays.append(np.nan)
            else:
                delays.append(np.nan)
        vals = [d if not np.isnan(d) else 0 for d in delays]
        colors = [color if not np.isnan(d) else "#CCCCCC" for d in delays]
        for i, (v, c) in enumerate(zip(vals, colors)):
            ax.bar(x[i] + offsets[mi], v, width, color=c,
                   alpha=0.8 if c != "#CCCCCC" else 0.3,
                   label=label if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in MODELS], fontsize=9)
    ax.set_ylabel("Normalized delay (IT − PT median) / n_layers")
    ax.axhline(0, color="black", lw=0.5)
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "L2_commitment_summary_bar.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 11. CDF — 4 methods
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cdf():
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
    fig.suptitle("Commitment CDF by Normalized Depth\nSolid = IT, Dashed = PT",
                 fontsize=13, fontweight="bold")
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
                norm = np.sort(np.array(commits) / nl)
                cdf = np.arange(1, len(norm) + 1) / len(norm)
                lbl = f"{LABELS[model]} {variant.upper()}" if ls == "-" else None
                ax.plot(norm, cdf, color=color, ls=ls, lw=1.2, label=lbl, alpha=0.8)
        ax.set_xlabel("Normalized depth")
        ax.set_ylabel("Fraction committed")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L2_commitment_cdf_4methods.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. KL threshold sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kl_sensitivity():
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL Threshold Sensitivity: Median Commitment vs τ\n"
                 "[Blue = raw logit-lens, Red = tuned logit-lens | Solid = IT, Dashed = PT]",
                 fontsize=13, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            raw_meds, tuned_meds = [], []
            for t in thresholds:
                c = _extract_commits(recs, f"commitment_layer_raw_kl_{t}")
                raw_meds.append(np.median(c) / nl if c else np.nan)
                c = _extract_commits(recs, f"commitment_layer_tuned_{t}")
                tuned_meds.append(np.median(c) / nl if c else np.nan)
            lbl_v = variant.upper()
            ax.plot(thresholds, raw_meds, marker=marker, color=COL_PT, ls=ls, lw=1.2,
                    markersize=4, label=f"Raw {lbl_v}")
            ax.plot(thresholds, tuned_meds, marker=marker, color=COL_IT, ls=ls, lw=1.2,
                    markersize=4, label=f"Tuned {lbl_v}")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)")
        ax.set_ylabel("Median commitment (norm. depth)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)
        _mark_broken(ax, model)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "L2_kl_threshold_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Raw vs Tuned scatter
# ═══════════════════════════════════════════════════════════════════════════════
def plot_scatter():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Raw vs Tuned Top-1 Commitment (per step)\n"
                 "Below diagonal = tuned commits earlier",
                 fontsize=13, fontweight="bold")
    rng = np.random.default_rng(42)
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            raw = _extract_commits(recs, "commitment_layer_raw")
            tuned = _extract_commits(recs, "commitment_layer_top1_tuned")
            if raw and tuned and len(raw) == len(tuned):
                n = min(len(raw), 20000)
                sub = rng.choice(len(raw), n, replace=False)
                ax.scatter(np.array(raw)[sub], np.array(tuned)[sub],
                           s=1, alpha=0.15, color=color, label=lbl, rasterized=True)
        ax.plot([0, nl], [0, nl], "k--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Raw top-1 commitment")
        ax.set_ylabel("Tuned top-1 commitment")
        ax.legend(fontsize=8, markerscale=5)
        _mark_broken(ax, model)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L2_raw_vs_tuned_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — delete old misleading plot, regenerate everything
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Remove misleading single-variant delta_cosine plot
    old = OUT / "delta_cosine_tuned_lens_6panel.png"
    if old.exists():
        old.unlink()
        log.info("DELETED misleading %s", old)

    log.info("=== Regenerating ALL exp9 plots with fixes ===")

    # Group A: Residual/architecture (no logit lens)
    log.info("--- Group A: Residual stream / architecture ---")
    plot_L1()
    plot_L3()
    plot_L8()
    plot_L9()
    plot_cosine_to_final()

    # Group B: Tuned lens validation (raw vs tuned comparison)
    log.info("--- Group B: Tuned lens validation ---")
    plot_kl_to_final_validation()

    # Group C: Commitment (all methods, both lenses)
    log.info("--- Group C: Commitment plots (all 7 methods) ---")
    plot_all_commitment()
    plot_summary_bar()
    plot_cdf()
    plot_kl_sensitivity()
    plot_scatter()

    # Final inventory
    plots = sorted(OUT.glob("*.png"))
    log.info("=== DONE: %d plots in %s ===", len(plots), OUT)
    for p in plots:
        log.info("  %s", p.name)
