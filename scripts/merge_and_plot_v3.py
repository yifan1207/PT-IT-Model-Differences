"""
Merge worker arrays for gemma3_4b/pt and mistral_7b/pt, then regenerate all 12 exp9 plots.

Run after modal_eval_parallel.py completes:
    uv run python scripts/merge_and_plot_v3.py
"""
import json, logging, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("merge_plot")

SRC = Path("/tmp/modal_v3")
OUT = Path("results/exp9/plots")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "deepseek_v2_lite", "olmo2_7b"]
N_LAYERS = {"gemma3_4b":34, "llama31_8b":32, "qwen3_4b":36, "mistral_7b":32, "deepseek_v2_lite":27, "olmo2_7b":32}
LABELS = {"gemma3_4b":"Gemma 3 4B", "llama31_8b":"Llama 3.1 8B", "qwen3_4b":"Qwen 3 4B",
          "mistral_7b":"Mistral 7B v0.3", "deepseek_v2_lite":"DeepSeek-V2-Lite", "olmo2_7b":"OLMo 2 7B"}
COL_PT, COL_IT = "#2196F3", "#E53935"
KL_THRESHOLDS = [0.05, 0.1, 0.2, 0.5, 1.0]

# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Merge worker arrays
# ═══════════════════════════════════════════════════════════════════════════
def merge_workers(model, variant, n_workers=5):
    """Merge arrays_pt_w{i}/ into arrays_pt/."""
    merged_dir = SRC / model / f"arrays_{variant}"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Collect all worker dirs
    worker_dirs = []
    for wi in range(n_workers):
        wd = SRC / model / f"arrays_{variant}_w{wi}"
        if wd.exists() and (wd / "raw_kl_final.npy").exists():
            worker_dirs.append(wd)

    if not worker_dirs:
        log.warning("No worker arrays for %s/%s", model, variant)
        return False

    log.info("Merging %d workers for %s/%s", len(worker_dirs), model, variant)

    # Merge NPY arrays by concatenation along axis 0 (steps)
    array_files = [
        "raw_kl_final.npy", "tuned_kl_final.npy",
        "raw_entropy.npy", "tuned_entropy.npy",
        "delta_cosine.npy", "cosine_h_to_final.npy",
        "raw_top1.npy", "tuned_top1.npy", "generated_ids.npy",
        "raw_kl_adj.npy", "tuned_kl_adj.npy",
        "raw_ntprob.npy", "tuned_ntprob.npy",
        "raw_ntrank.npy", "tuned_ntrank.npy",
    ]

    for arr_name in array_files:
        parts = []
        for wd in worker_dirs:
            f = wd / arr_name
            if f.exists():
                parts.append(np.load(f))
        if parts:
            merged = np.concatenate(parts, axis=0)
            np.save(merged_dir / arr_name, merged)
            log.info("  %s: %s", arr_name, merged.shape)

    # Merge step_index.jsonl (update start/end offsets)
    all_entries = []
    offset = 0
    for wd in worker_dirs:
        si = wd / "step_index.jsonl"
        if si.exists():
            with open(si) as f:
                for line in f:
                    entry = json.loads(line)
                    n = entry["n_steps"]
                    entry["start_step"] = offset
                    entry["end_step"] = offset + n
                    all_entries.append(entry)
                    offset += n
    with open(merged_dir / "step_index.jsonl", "w") as f:
        for e in all_entries:
            f.write(json.dumps(e) + "\n")
    log.info("  step_index: %d prompts, %d total steps", len(all_entries), offset)

    # Merge JSONL commitment files
    merged_jsonl = SRC / model / f"tuned_lens_commitment_{variant}.jsonl"
    with open(merged_jsonl, "w") as fout:
        for wi in range(n_workers):
            wf = SRC / model / f"tuned_lens_commitment_{variant}_w{wi}.jsonl"
            if wf.exists():
                with open(wf) as fin:
                    for line in fin:
                        fout.write(line)
    n_merged = sum(1 for _ in open(merged_jsonl))
    log.info("  JSONL: %d prompts merged", n_merged)

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Helpers
# ═══════════════════════════════════════════════════════════════════════════
def load_jsonl(model, variant):
    f = SRC / model / f"tuned_lens_commitment_{variant}.jsonl"
    if not f.exists():
        return None
    with open(f) as fh:
        return [json.loads(l) for l in fh]

def extract(recs, key):
    vals = []
    for r in recs:
        if key in r:
            vals.extend(r[key])
    return vals

def _save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", OUT / name)


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: All 12 plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_commitment(key, title, filename, lens_label):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Commitment Delay: {title}\n[{lens_label} — v3, IT uses chat template]",
                 fontsize=13, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        mean_lines = {}
        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            recs = load_jsonl(model, variant)
            if not recs: continue
            commits = extract(recs, key)
            if not commits: continue
            n, mn = len(commits), np.mean(commits)
            mean_lines[variant] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")
        if mean_lines:
            vals = list(mean_lines.values())
            close = len(vals) == 2 and abs(vals[0] - vals[1]) < 0.8
            for vi, (v, mn) in enumerate(mean_lines.items()):
                c = COL_PT if v == "pt" else COL_IT
                off = (-0.25 if vi == 0 else 0.25) if close else 0
                ax.axvline(mn + off, color=c, ls="--", lw=2, alpha=0.9)
        ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0: ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, filename)


def plot_all():
    # 1. L2_commitment_raw_kl_0.1.png
    plot_commitment("commitment_layer_raw_kl_0.1", "KL(ℓ ‖ final) < 0.1 nats, no-flip-back",
                    "L2_commitment_raw_kl_0.1.png", "Raw Logit-Lens")

    # 2. L2_commitment_tuned_kl_0.1.png
    plot_commitment("commitment_layer_tuned_0.1", "KL(ℓ ‖ final) < 0.1 nats, no-flip-back",
                    "L2_commitment_tuned_kl_0.1.png", "Tuned Logit-Lens")

    # 3. L2_commitment_raw_top1.png
    plot_commitment("commitment_layer_raw", "Top-1 No-Flip-Back",
                    "L2_commitment_raw_top1.png", "Raw Logit-Lens")

    # 4. L2_commitment_tuned_top1.png
    plot_commitment("commitment_layer_top1_tuned", "Top-1 No-Flip-Back",
                    "L2_commitment_tuned_top1.png", "Tuned Logit-Lens")

    # 5. L2_pure_kl_threshold_sensitivity.png
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL Threshold Sensitivity: Mean Commitment vs τ\n"
                 "[First layer ℓ where KL(ℓ ‖ final) < τ and stays below | "
                 "Blue = raw, Red = tuned | Solid = IT, Dashed = PT]\n"
                 "[v3, IT uses chat template]", fontsize=11, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            recs = load_jsonl(model, variant)
            if not recs: continue
            lv = variant.upper()
            raw_v, tuned_v = [], []
            for t in KL_THRESHOLDS:
                c = extract(recs, f"commitment_layer_raw_kl_{t}")
                raw_v.append(np.mean(c) / nl if c else np.nan)
                c = extract(recs, f"commitment_layer_tuned_{t}")
                tuned_v.append(np.mean(c) / nl if c else np.nan)
            ax.plot(KL_THRESHOLDS, raw_v, marker=marker, color=COL_PT, ls=ls, lw=1.2, markersize=4, label=f"Raw {lv}")
            ax.plot(KL_THRESHOLDS, tuned_v, marker=marker, color=COL_IT, ls=ls, lw=1.2, markersize=4, label=f"Tuned {lv}")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)"); ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05); ax.legend(fontsize=6)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    _save(fig, "L2_pure_kl_threshold_sensitivity.png")

    # 6 & 7. L2_mean_kl_per_layer_raw.png / tuned.png
    for lens, ll in [("raw", "Raw Logit-Lens"), ("tuned", "Tuned Logit-Lens")]:
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle(f"Mean KL(layer ℓ ‖ final) per Layer — PT vs IT\n[{ll} — v3, IT uses chat template]",
                     fontsize=13, fontweight="bold")
        for idx, model in enumerate(MODELS):
            ax = axes[idx // 3, idx % 3]
            nl = N_LAYERS[model]; x = np.arange(nl)
            for variant, color, ls in [("pt", COL_PT, "--"), ("it", COL_IT, "-")]:
                f = SRC / model / f"arrays_{variant}" / f"{lens}_kl_final.npy"
                if not f.exists(): continue
                arr = np.load(f).astype(np.float32)
                ax.plot(x, np.nanmean(arr, axis=0), color=color, ls=ls, lw=2,
                        label=f"{variant.upper()} (n={arr.shape[0]:,})")
            ax.set_title(LABELS[model], fontsize=11); ax.set_xlabel("Layer")
            if idx % 3 == 0: ax.set_ylabel("Mean KL to final (nats)")
            ax.legend(fontsize=7); ax.set_xlim(0, nl-1); ax.set_ylim(bottom=0)
        fig.tight_layout(rect=[0, 0, 1, 0.89])
        _save(fig, f"L2_mean_kl_per_layer_{lens}.png")

    # 8. cosine_to_final_6panel.png
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Cosine(h_ℓ, h_final) — Residual Stream Convergence, PT vs IT\n[v3, IT uses chat template]",
                 fontsize=13, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]; x = np.arange(nl) / nl
        for variant, color, ls, lbl in [("pt", COL_PT, "--", "PT"), ("it", COL_IT, "-", "IT")]:
            f = SRC / model / f"arrays_{variant}" / "cosine_h_to_final.npy"
            if not f.exists(): continue
            arr = np.load(f).astype(np.float32)
            ax.fill_between(x, np.nanpercentile(arr, 25, axis=0), np.nanpercentile(arr, 75, axis=0), alpha=0.15, color=color)
            ax.plot(x, np.nanmean(arr, axis=0), color=color, ls=ls, lw=2, label=f"{lbl} (mean ± IQR)")
        ax.axvline(0.6, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_title(LABELS[model], fontsize=11); ax.set_xlabel("Normalized depth")
        if idx % 3 == 0: ax.set_ylabel("cos(h_ℓ, h_final)")
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "cosine_to_final_6panel.png")

    # 9. L1_delta_cosine_6panel.png
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("L1: δ-Cosine Profile — PT vs IT\n[cos(Δh_ℓ, h_{ℓ-1}) — v3, IT uses chat template]",
                 fontsize=14, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]; x = np.arange(nl) / nl
        for variant, color, ls, lbl in [("pt", COL_PT, "--", "PT (base)"), ("it", COL_IT, "-", "IT (instruct)")]:
            f = SRC / model / f"arrays_{variant}" / "delta_cosine.npy"
            if not f.exists(): continue
            ax.plot(x, np.nanmean(np.load(f).astype(np.float32), axis=0), color=color, ls=ls, lw=2, label=lbl)
        ax.axvspan(0.6, 1.0, alpha=0.06, color="gray"); ax.axhline(0, color="black", lw=0.5, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11); ax.set_xlabel("Normalized depth")
        if idx % 3 == 0: ax.set_ylabel("Mean δ-cosine")
        ax.legend(fontsize=9); ax.set_xlim(0, 1)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, "L1_delta_cosine_6panel.png")

    # 10. L1_residual_opposition_6panel.png
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Residual Stream Opposition: IT − PT δ-Cosine per Layer\n"
                 "[Negative = IT opposes residual MORE than PT | v3, IT uses chat template]",
                 fontsize=12, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]; x = np.arange(nl) / nl
        pt_f = SRC / model / "arrays_pt" / "delta_cosine.npy"
        it_f = SRC / model / "arrays_it" / "delta_cosine.npy"
        if not pt_f.exists() or not it_f.exists(): continue
        diff = np.nanmean(np.load(it_f).astype(np.float32), axis=0) - np.nanmean(np.load(pt_f).astype(np.float32), axis=0)
        ax.fill_between(x, diff, 0, where=diff < 0, alpha=0.3, color=COL_IT, label="IT opposes more")
        ax.fill_between(x, diff, 0, where=diff >= 0, alpha=0.3, color=COL_PT, label="PT opposes more")
        ax.plot(x, diff, color="black", lw=1.2)
        ax.axhline(0, color="black", lw=0.5, alpha=0.3); ax.axvspan(0.6, 1.0, alpha=0.06, color="gray")
        ax.set_title(LABELS[model], fontsize=11); ax.set_xlabel("Normalized depth")
        if idx % 3 == 0: ax.set_ylabel("δ-cosine diff (IT − PT)")
        ax.legend(fontsize=8); ax.set_xlim(0, 1)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L1_residual_opposition_6panel.png")

    # 11. L1_opposition_pt_vs_it_panels.png
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), sharey=True)
    fig.suptitle("Residual Stream: Reinforcing vs Opposing Layers\n"
                 "[δ-cosine = cos(Δh_ℓ, h_{ℓ-1}) | Green = reinforcing, Red = opposing | v3]",
                 fontsize=12, fontweight="bold")
    for idx, model in enumerate(MODELS):
        nl = N_LAYERS[model]; x = np.arange(nl) / nl
        for vi, (variant, title_s) in enumerate([("pt", "PT"), ("it", "IT")]):
            ax = axes[vi, idx]
            f = SRC / model / f"arrays_{variant}" / "delta_cosine.npy"
            if not f.exists(): continue
            dc = np.nanmean(np.load(f).astype(np.float32), axis=0)
            ax.fill_between(x, dc, 0, where=dc >= 0, alpha=0.4, color="#4CAF50", label="Reinforcing")
            ax.fill_between(x, dc, 0, where=dc < 0, alpha=0.4, color="#E53935", label="Opposing")
            ax.plot(x, dc, color="black", lw=1.2)
            ax.axhline(0, color="black", lw=0.5, alpha=0.3); ax.axvline(0.6, color="gray", ls="--", lw=0.8, alpha=0.3)
            if vi == 0: ax.set_title(LABELS[model], fontsize=10)
            if idx == 0: ax.set_ylabel(f"{title_s}\nMean δ-cosine", fontsize=10)
            if vi == 1: ax.set_xlabel("Norm. depth", fontsize=8)
            ax.set_xlim(0, 1); ax.set_ylim(-0.8, 1.0)
            if idx == 0 and vi == 0: ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L1_opposition_pt_vs_it_panels.png")

    # 12. L1_heatmaps_6x2.png — δ-cosine heatmap [step × layer]
    fig, axes = plt.subplots(2, 6, figsize=(20, 8))
    fig.suptitle("L1: δ-Cosine Heatmap [generation step × layer]\n"
                 "[Top = PT, Bottom = IT | v3, IT uses chat template]",
                 fontsize=12, fontweight="bold")
    for idx, model in enumerate(MODELS):
        nl = N_LAYERS[model]
        for vi, (variant, title_s) in enumerate([("pt", "PT"), ("it", "IT")]):
            ax = axes[vi, idx]
            f = SRC / model / f"arrays_{variant}" / "delta_cosine.npy"
            if not f.exists(): continue
            arr = np.load(f).astype(np.float32)
            # Subsample to max 200 steps for visualization
            if arr.shape[0] > 200:
                idx_sub = np.linspace(0, arr.shape[0]-1, 200, dtype=int)
                arr = arr[idx_sub]
            im = ax.imshow(arr.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                          origin="lower", interpolation="nearest")
            if vi == 0: ax.set_title(LABELS[model], fontsize=9)
            if idx == 0: ax.set_ylabel(f"{title_s}\nLayer")
            if vi == 1: ax.set_xlabel("Step (subsampled)")
            ax.set_yticks([0, nl//2, nl-1])
    fig.tight_layout(rect=[0, 0, 0.95, 0.89])
    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.6])
    fig.colorbar(im, cax=cbar_ax, label="δ-cosine")
    _save(fig, "L1_heatmaps_6x2.png")


if __name__ == "__main__":
    # Step 1: Merge workers for gemma and mistral PT
    log.info("=== Step 1: Merge worker arrays ===")
    for model in ["gemma3_4b", "mistral_7b"]:
        merge_workers(model, "pt", n_workers=5)

    # Step 2: Generate all 12 plots
    log.info("=== Step 2: Generate all 12 plots ===")
    plot_all()

    log.info("=== ALL DONE ===")
