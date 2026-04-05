"""
Exp9 plot fixes — regenerate all plots with:
  1. Clear lens/data-source label on EVERY plot
  2. Missing commitment data marked clearly
  3. KL-to-final raw vs tuned as key validation plot
  4. Remove misleading delta_cosine_tuned_lens (was single-variant, no PT vs IT)
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

BASE = Path("results/cross_model")
OUT = Path("results/exp9/plots")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    log.info("Saved → %s", OUT / name)
    plt.close(fig)


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


def _majority_commit_from_kl(kl_arr, threshold, frac=0.9):
    """Compute majority-vote commitment for each row of a [steps, layers] KL array.

    Vectorized: for each layer ℓ, compute fraction of layers ℓ..L-1 below threshold.
    Returns numpy array of commitment layers (one per step).
    """
    kl = kl_arr.astype(np.float32)
    n_steps, n_layers = kl.shape
    below = (kl < threshold)  # [steps, layers] bool

    # For each layer ℓ, fraction of layers ℓ..L-1 that are below threshold
    # Use reverse cumsum: cumsum of below from right
    rev_cumsum = np.cumsum(below[:, ::-1], axis=1)[:, ::-1]  # [steps, layers]
    remaining = np.arange(n_layers, 0, -1)  # [layers]: n_layers, n_layers-1, ..., 1
    frac_below = rev_cumsum / remaining  # [steps, layers]

    # Commitment = first layer where frac_below >= frac
    committed = frac_below >= frac  # [steps, layers] bool
    # argmax on bool gives first True; if none True, gives 0 (wrong), so handle that
    has_any = committed.any(axis=1)
    result = np.where(has_any, committed.argmax(axis=1), n_layers - 1)
    return result.tolist()


def _load_raw_kl_majority(model, variant, threshold=0.1):
    """Compute raw-KL majority commitment from NPY arrays, split by variant.

    Uses (prompt_id, n_steps) matching between step_index and JSONL to
    determine which array rows belong to which variant, since prompt_ids
    are shared between PT and IT (same eval dataset) but n_steps differ.
    """
    arr_dir = BASE / model / "tuned_lens" / "commitment" / "arrays"
    arr_path = arr_dir / "raw_kl_final.npy"
    idx_path = arr_dir / "step_index.jsonl"
    if not arr_path.exists() or not idx_path.exists():
        return None

    kl_arr = np.load(arr_path)
    with open(idx_path) as f:
        step_index = [json.loads(line) for line in f]

    # Build (prompt_id, n_steps) set from this variant's JSONL
    recs = _load_commitment(model, variant)
    if recs is None:
        return None
    variant_keys = {(rec["prompt_id"], rec["n_steps"]) for rec in recs}

    # Collect row ranges from step_index that match this variant
    total_rows = kl_arr.shape[0]
    row_ranges = []
    for entry in step_index:
        key = (entry["prompt_id"], entry["n_steps"])
        if key in variant_keys and entry["end_step"] <= total_rows:
            row_ranges.append((entry["start_step"], entry["end_step"]))

    if not row_ranges:
        return None

    # Build index array from ranges (avoids per-element list extension)
    row_indices = np.concatenate([np.arange(s, e) for s, e in row_ranges])
    sub = kl_arr[row_indices]
    return _majority_commit_from_kl(sub, threshold)


def _load_kl_for_variant(model, variant, lens="raw"):
    """Load per-step per-layer KL-to-final values for a specific variant.

    Args:
        lens: "raw" or "tuned"
    Returns [n_steps, n_layers] float32 array or None.
    """
    arr_dir = BASE / model / "tuned_lens" / "commitment" / "arrays"
    arr_path = arr_dir / f"{lens}_kl_final.npy"
    idx_path = arr_dir / "step_index.jsonl"
    if not arr_path.exists() or not idx_path.exists():
        return None

    kl_arr = np.load(arr_path)
    with open(idx_path) as f:
        step_index = [json.loads(line) for line in f]

    recs = _load_commitment(model, variant)
    if recs is None:
        return None
    variant_keys = {(rec["prompt_id"], rec["n_steps"]) for rec in recs}

    total_rows = kl_arr.shape[0]
    row_ranges = []
    for entry in step_index:
        key = (entry["prompt_id"], entry["n_steps"])
        if key in variant_keys and entry["end_step"] <= total_rows:
            row_ranges.append((entry["start_step"], entry["end_step"]))

    if not row_ranges:
        return None

    row_indices = np.concatenate([np.arange(s, e) for s, e in row_ranges])
    return kl_arr[row_indices].astype(np.float32)


def _mean_kl_commit(kl_arr, threshold):
    """Commitment: earliest layer ℓ where mean(KL[ℓ:]) < threshold.

    Vectorized via reverse cumsum.
    Returns numpy int array of commitment layers (one per step).
    """
    n_steps, n_layers = kl_arr.shape
    rev_cumsum = np.cumsum(kl_arr[:, ::-1], axis=1)[:, ::-1]
    remaining = np.arange(n_layers, 0, -1, dtype=np.float32)
    mean_kl = rev_cumsum / remaining
    committed = mean_kl < threshold
    has_any = committed.any(axis=1)
    return np.where(has_any, committed.argmax(axis=1), n_layers - 1)


def _median_kl_commit(kl_arr, threshold):
    """Commitment: earliest layer ℓ where median(KL[ℓ:]) < threshold.

    Iterates over layers; computes median of KL[ℓ:] for all steps at once.
    Returns numpy int array of commitment layers (one per step).
    """
    n_steps, n_layers = kl_arr.shape
    result = np.full(n_steps, n_layers - 1, dtype=np.int32)
    uncommitted = np.ones(n_steps, dtype=bool)
    for li in range(n_layers):
        if not uncommitted.any():
            break
        med = np.median(kl_arr[uncommitted, li:], axis=1)
        newly = med < threshold
        # Map back to full index
        idx = np.where(uncommitted)[0][newly]
        result[idx] = li
        uncommitted[idx] = False
    return result


def _adaptive_majority_commit(kl_arr, threshold):
    """Commitment: earliest layer ℓ where KL[ℓ] < τ and subsequent layers mostly below τ.

    Rule:
      - If ≥10 subsequent layers (including ℓ): ≥90% must have KL < τ
      - If <10 subsequent layers: at most 1 may exceed τ

    Vectorized. Returns numpy int array of commitment layers (one per step).
    """
    n_steps, n_layers = kl_arr.shape
    below = (kl_arr < threshold)  # [steps, layers] bool

    # Reverse cumsum: count of below-threshold layers from ℓ to end
    rev_cumsum = np.cumsum(below[:, ::-1], axis=1)[:, ::-1]  # [steps, layers]
    remaining = np.arange(n_layers, 0, -1)  # layers remaining from ℓ

    # Build committed mask per layer
    committed = np.zeros((n_steps, n_layers), dtype=bool)
    for li in range(n_layers):
        rem = remaining[li]  # n_layers - li
        n_below = rev_cumsum[:, li]
        n_above = rem - n_below
        if rem >= 10:
            # ≥90% below threshold
            committed[:, li] = (n_below / rem) >= 0.9
        else:
            # At most 1 exceeds threshold
            committed[:, li] = n_above <= 1

    # Must also have KL[ℓ] < threshold at the commitment layer itself
    committed = committed & below

    has_any = committed.any(axis=1)
    return np.where(has_any, committed.argmax(axis=1), n_layers - 1)


def _kl_plus_mean_commit(kl_arr, threshold):
    """Commitment: earliest layer ℓ where KL[ℓ] < τ AND mean(KL[ℓ:]) < τ.

    The layer itself must be below threshold (candidate),
    AND the average of it and all subsequent layers must also be below.
    Vectorized. Returns numpy int array.
    """
    n_steps, n_layers = kl_arr.shape
    below_thresh = kl_arr < threshold  # candidate layers
    rev_cumsum = np.cumsum(kl_arr[:, ::-1], axis=1)[:, ::-1]
    remaining = np.arange(n_layers, 0, -1, dtype=np.float32)
    mean_kl = rev_cumsum / remaining
    mean_ok = mean_kl < threshold
    committed = below_thresh & mean_ok
    has_any = committed.any(axis=1)
    return np.where(has_any, committed.argmax(axis=1), n_layers - 1)


def _mean_and_median_kl_commit(kl_arr, threshold):
    """Commitment: earliest layer ℓ where BOTH mean(KL[ℓ:]) < τ AND median(KL[ℓ:]) < τ.

    Returns numpy int array of commitment layers (one per step).
    """
    mean_commits = _mean_kl_commit(kl_arr, threshold)
    median_commits = _median_kl_commit(kl_arr, threshold)
    # Commitment = max of the two (whichever is later / more conservative)
    return np.maximum(mean_commits, median_commits)


def _plot_meanmedian_commitment(lens, threshold=0.1):
    """6-panel histogram: commitment where both mean & median KL[ℓ:] < τ."""
    lens_label = "Raw Logit-Lens" if lens == "raw" else "Tuned Logit-Lens"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Commitment Delay: mean(KL[ℓ:]) < {threshold} AND median(KL[ℓ:]) < {threshold}\n"
        f"[{lens_label}]",
        fontsize=13, fontweight="bold",
    )
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        has_data = False
        mean_lines = {}

        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            kl = _load_kl_for_variant(model, variant, lens=lens)
            if kl is None:
                ax.text(0.5, 0.85 if variant == "it" else 0.78,
                        f"{lbl}: NO DATA", transform=ax.transAxes,
                        ha="center", fontsize=9, color=color, alpha=0.7)
                continue
            commits = _mean_and_median_kl_commit(kl, threshold)
            has_data = True
            n = len(commits)
            mn = float(np.mean(commits))
            mean_lines[variant] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")

        if mean_lines:
            vals = list(mean_lines.values())
            close = len(vals) == 2 and abs(vals[0] - vals[1]) < 0.8
            for vi, (variant, mn) in enumerate(mean_lines.items()):
                color = COL_PT if variant == "pt" else COL_IT
                offset = (-0.25 if vi == 0 else 0.25) if close else 0
                ax.axvline(mn + offset, color=color, ls="--", lw=2, alpha=0.9)

        if has_data:
            ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, f"L2_commitment_{lens}_meanmedian_kl_{threshold}.png")


def _plot_raw_vs_tuned_meankl(threshold=0.1):
    """6-panel histogram: raw vs tuned commitment on same axes, PT and IT combined."""
    COL_RAW, COL_TUNED = "#2196F3", "#E53935"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Commitment Delay: mean(KL[ℓ:]) < {threshold} nats\n"
        f"[Raw (blue) vs Tuned (red) Logit-Lens — PT+IT combined]",
        fontsize=13, fontweight="bold",
    )
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        mean_lines = {}

        for lens, color, lbl in [("raw", COL_RAW, "Raw"), ("tuned", COL_TUNED, "Tuned")]:
            all_commits = []
            for variant in ["pt", "it"]:
                kl = _load_kl_for_variant(model, variant, lens=lens)
                if kl is not None:
                    all_commits.append(_mean_kl_commit(kl, threshold))
            if not all_commits:
                ax.text(0.5, 0.5, f"{lbl}: NO DATA", transform=ax.transAxes,
                        ha="center", fontsize=9, color=color, alpha=0.7)
                continue
            commits = np.concatenate(all_commits)
            n = len(commits)
            mn = float(np.mean(commits))
            mean_lines[lens] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.4, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")

        if mean_lines:
            for lens, mn in mean_lines.items():
                color = COL_RAW if lens == "raw" else COL_TUNED
                ax.axvline(mn, color=color, ls="--", lw=2, alpha=0.9)

        ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, f"L2_commitment_raw_vs_tuned_meankl_{threshold}.png")


def _plot_mean_commitment(lens, threshold=0.1):
    """6-panel histogram: commitment where mean(KL[ℓ:]) < τ."""
    lens_label = "Raw Logit-Lens" if lens == "raw" else "Tuned Logit-Lens"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Commitment Delay: mean(KL[ℓ:]) < {threshold} nats\n"
        f"[{lens_label}]",
        fontsize=13, fontweight="bold",
    )
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        has_data = False
        mean_lines = {}

        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            kl = _load_kl_for_variant(model, variant, lens=lens)
            if kl is None:
                ax.text(0.5, 0.85 if variant == "it" else 0.78,
                        f"{lbl}: NO DATA", transform=ax.transAxes,
                        ha="center", fontsize=9, color=color, alpha=0.7)
                continue
            commits = _mean_kl_commit(kl, threshold)
            has_data = True
            n = len(commits)
            mn = float(np.mean(commits))
            mean_lines[variant] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")

        if mean_lines:
            vals = list(mean_lines.values())
            close = len(vals) == 2 and abs(vals[0] - vals[1]) < 0.8
            for vi, (variant, mn) in enumerate(mean_lines.items()):
                color = COL_PT if variant == "pt" else COL_IT
                offset = (-0.25 if vi == 0 else 0.25) if close else 0
                ax.axvline(mn + offset, color=color, ls="--", lw=2, alpha=0.9)

        if has_data:
            ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, f"L2_commitment_{lens}_meankl_{threshold}.png")


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
        mean_lines = {}  # variant -> mean value, for offset when overlapping

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
            mn = np.mean(commits)
            mean_lines[variant] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")

        # Draw mean lines with offset when they overlap
        if mean_lines:
            vals = list(mean_lines.values())
            close = len(vals) == 2 and abs(vals[0] - vals[1]) < 0.8
            for vi, (variant, mn) in enumerate(mean_lines.items()):
                color = COL_PT if variant == "pt" else COL_IT
                offset = (-0.25 if vi == 0 else 0.25) if close else 0
                ax.axvline(mn + offset, color=color, ls="--", lw=2, alpha=0.9)

        if has_data:
            ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, filename)


def _plot_adaptive_majority(lens, threshold=0.1):
    """Plot adaptive majority commitment from NPY arrays for a given lens type."""
    lens_label = "Raw Logit-Lens" if lens == "raw" else "Tuned Logit-Lens"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Commitment Delay: adaptive majority KL < {threshold}\n"
        f"[{lens_label} | ≥10 remaining: ≥90% below | <10 remaining: ≤1 exceeds]",
        fontsize=12, fontweight="bold",
    )
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        has_data = False
        mean_lines = {}

        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            kl = _load_kl_for_variant(model, variant, lens=lens)
            if kl is None:
                ax.text(0.5, 0.85 if variant == "it" else 0.78,
                        f"{lbl}: NO DATA", transform=ax.transAxes,
                        ha="center", fontsize=9, color=color, alpha=0.7)
                continue
            commits = _adaptive_majority_commit(kl, threshold)
            has_data = True
            n = len(commits)
            mn = float(np.mean(commits))
            mean_lines[variant] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")

        if mean_lines:
            vals = list(mean_lines.values())
            close = len(vals) == 2 and abs(vals[0] - vals[1]) < 0.8
            for vi, (variant, mn) in enumerate(mean_lines.items()):
                color = COL_PT if variant == "pt" else COL_IT
                offset = (-0.25 if vi == 0 else 0.25) if close else 0
                ax.axvline(mn + offset, color=color, ls="--", lw=2, alpha=0.9)

        if has_data:
            ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, f"L2_commitment_{lens}_majority_0.1.png")


def plot_all_commitment():
    # === Raw logit-lens methods ===
    _plot_commitment("commitment_layer_raw", "Top-1 No-Flip-Back",
                     "L2_commitment_raw_top1.png", "Raw Logit-Lens")
    _plot_commitment("commitment_layer_raw_kl_0.1", "KL(ℓ ‖ final) < 0.1 nats",
                     "L2_commitment_raw_kl_0.1.png", "Raw Logit-Lens")
    _plot_commitment("commitment_layer_raw_kl_0.5", "KL(ℓ ‖ final) < 0.5 nats",
                     "L2_commitment_raw_kl_0.5.png", "Raw Logit-Lens")

    # === Tuned logit-lens methods ===
    _plot_commitment("commitment_layer_top1_tuned", "Top-1 No-Flip-Back",
                     "L2_commitment_tuned_top1.png", "Tuned Logit-Lens")
    _plot_commitment("commitment_layer_tuned_0.1", "KL(ℓ ‖ final) < 0.1 nats",
                     "L2_commitment_tuned_kl_0.1.png", "Tuned Logit-Lens")
    _plot_commitment("commitment_layer_tuned_0.5", "KL(ℓ ‖ final) < 0.5 nats",
                     "L2_commitment_tuned_kl_0.5.png", "Tuned Logit-Lens")

    # === Residual-stream method (no logit lens) ===
    _plot_commitment("commitment_layer_cosine_0.95", "cos(h_ℓ, h_final) > 0.95",
                     "L2_commitment_cosine_0.95.png", "Residual Stream — no logit lens")

    # === Entropy method ===
    _plot_commitment("commitment_layer_entropy_0.2", "|H_ℓ - H_final| < 0.2 nats",
                     "L2_commitment_entropy_0.2.png", "Raw Logit-Lens Entropy")

    # === Majority vote ===
    # Tuned: use JSONL (complete data for both variants)
    _plot_commitment("commitment_layer_majority_0.1",
                     "≥90% subsequent layers KL < 0.1",
                     "L2_commitment_tuned_majority_0.1.png",
                     "Tuned Logit-Lens | ≥10 remaining: ≥90% below | <10 remaining: ≤1 exceeds")
    # Raw: compute from NPY arrays (no pre-computed raw majority in JSONL)
    _plot_adaptive_majority("raw", threshold=0.1)

    # === Qualified ===
    _plot_commitment("commitment_layer_raw_kl_qual_0.5_3x", "KL < 0.5, holds 3 layers",
                     "L2_commitment_raw_kl_qual_0.5_3x.png", "Raw Logit-Lens (qualified)")
    _plot_commitment("commitment_layer_tuned_kl_qual_0.5_3x", "KL < 0.5, holds 3 layers",
                     "L2_commitment_tuned_kl_qual_0.5_3x.png", "Tuned Logit-Lens (qualified)")

    # === Mean+Median KL commitment (from NPY arrays) ===
    _plot_meanmedian_commitment("raw", threshold=0.1)
    _plot_meanmedian_commitment("tuned", threshold=0.1)

    # === Mean-only KL commitment (from NPY arrays) ===
    _plot_mean_commitment("raw", threshold=0.1)
    _plot_mean_commitment("tuned", threshold=0.1)


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
#     Tuned: JSONL majority (complete data)
#     Raw: NPY adaptive majority (partial but best available)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_kl_sensitivity():
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL Threshold Sensitivity: Mean Commitment vs τ\n"
                 "[≥90% of layers ℓ..end have KL < τ (incl. ℓ itself) | "
                 "Blue = raw, Red = tuned | Solid = IT, Dashed = PT]",
                 fontsize=11, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            lbl_v = variant.upper()

            # Raw: from NPY arrays (adaptive majority)
            raw_kl = _load_kl_for_variant(model, variant, lens="raw")
            if raw_kl is not None:
                raw_vals = []
                for t in thresholds:
                    commits = _adaptive_majority_commit(raw_kl, t)
                    raw_vals.append(np.mean(commits) / nl)
                ax.plot(thresholds, raw_vals, marker=marker, color=COL_PT, ls=ls,
                        lw=1.2, markersize=4, label=f"Raw {lbl_v}")

            # Tuned: from JSONL (complete data, pre-computed majority)
            recs = _load_commitment(model, variant)
            if recs is not None:
                tuned_vals = []
                for t in thresholds:
                    c = _extract_commits(recs, f"commitment_layer_majority_{t}")
                    tuned_vals.append(np.mean(c) / nl if c else np.nan)
                ax.plot(thresholds, tuned_vals, marker=marker, color=COL_IT, ls=ls,
                        lw=1.2, markersize=4, label=f"Tuned {lbl_v}")

        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)")
        ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L2_kl_threshold_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12b-i. Entropy threshold sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_kl_plus_mean_commitment(lens, threshold=0.1):
    """6-panel histogram: KL[ℓ] < τ AND mean(KL[ℓ:]) < τ."""
    lens_label = "Raw Logit-Lens" if lens == "raw" else "Tuned Logit-Lens"
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Commitment Delay: KL[ℓ] < {threshold} AND mean(KL[ℓ:]) < {threshold}\n"
        f"[{lens_label}]",
        fontsize=13, fontweight="bold",
    )
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        bins = np.arange(nl + 1) - 0.5
        has_data = False
        mean_lines = {}

        for variant, color, lbl in [("pt", COL_PT, "PT"), ("it", COL_IT, "IT")]:
            kl = _load_kl_for_variant(model, variant, lens=lens)
            if kl is None:
                ax.text(0.5, 0.85 if variant == "it" else 0.78,
                        f"{lbl}: NO DATA", transform=ax.transAxes,
                        ha="center", fontsize=9, color=color, alpha=0.7)
                continue
            commits = _kl_plus_mean_commit(kl, threshold)
            has_data = True
            n = len(commits)
            mn = float(np.mean(commits))
            mean_lines[variant] = mn
            ax.hist(commits, bins=bins, density=True, alpha=0.45, color=color,
                    label=f"{lbl} (n={n:,}, μ={mn:.1f})")

        if mean_lines:
            vals = list(mean_lines.values())
            close = len(vals) == 2 and abs(vals[0] - vals[1]) < 0.8
            for vi, (variant, mn) in enumerate(mean_lines.items()):
                color = COL_PT if variant == "pt" else COL_IT
                offset = (-0.25 if vi == 0 else 0.25) if close else 0
                ax.axvline(mn + offset, color=color, ls="--", lw=2, alpha=0.9)

        if has_data:
            ax.axvline(round(nl * 0.6), color="gray", ls=":", lw=1, alpha=0.3)
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Commitment layer")
        if idx % 3 == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=7, loc="upper left")

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, f"L2_commitment_{lens}_kl_plus_mean_{threshold}.png")


def plot_kl_plus_mean_sensitivity():
    """Sensitivity: KL[ℓ] < τ AND mean(KL[ℓ:]) < τ, varying τ."""
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "KL Threshold Sensitivity: Mean Commitment vs τ\n"
        "[KL[ℓ] < τ AND mean(KL[ℓ:]) < τ | "
        "Blue = raw, Red = tuned | Solid = IT, Dashed = PT]",
        fontsize=11, fontweight="bold",
    )
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            lbl_v = variant.upper()
            for lens, color, lens_lbl in [("raw", COL_PT, "Raw"), ("tuned", COL_IT, "Tuned")]:
                kl = _load_kl_for_variant(model, variant, lens=lens)
                if kl is None:
                    continue
                vals = []
                for t in thresholds:
                    commits = _kl_plus_mean_commit(kl, t)
                    vals.append(np.mean(commits) / nl)
                ax.plot(thresholds, vals, marker=marker, color=color, ls=ls,
                        lw=1.2, markersize=4, label=f"{lens_lbl} {lbl_v}")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)")
        ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L2_kl_plus_mean_sensitivity.png")


def plot_entropy_sensitivity():
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Entropy Threshold Sensitivity: Mean Commitment vs τ\n"
                 "[|H_ℓ − H_final| < τ, stays below | "
                 "Solid = IT, Dashed = PT]",
                 fontsize=12, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            lbl_v = variant.upper()
            vals = []
            for t in thresholds:
                c = _extract_commits(recs, f"commitment_layer_entropy_{t}")
                vals.append(np.mean(c) / nl if c else np.nan)
            ax.plot(thresholds, vals, marker=marker, color=COL_IT, ls=ls,
                    lw=1.5, markersize=5, label=f"Entropy {lbl_v}")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Entropy threshold τ (nats)")
        ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L2_entropy_threshold_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12b-ii. Cosine threshold sensitivity
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cosine_sensitivity():
    thresholds = [0.8, 0.9, 0.95, 0.99]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Cosine Threshold Sensitivity: Mean Commitment vs θ\n"
                 "[cos(h_ℓ, h_final) > θ, stays above | "
                 "Solid = IT, Dashed = PT]",
                 fontsize=12, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            lbl_v = variant.upper()
            vals = []
            for t in thresholds:
                c = _extract_commits(recs, f"commitment_layer_cosine_{t}")
                vals.append(np.mean(c) / nl if c else np.nan)
            ax.plot(thresholds, vals, marker=marker, color="#4A148C", ls=ls,
                    lw=1.5, markersize=5, label=f"Cosine {lbl_v}")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("Cosine threshold θ")
        ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L2_cosine_threshold_sensitivity.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 12b-iii. Pure KL threshold sensitivity (no-flip-back, no majority, JSONL)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_pure_kl_sensitivity():
    thresholds = [0.05, 0.1, 0.2, 0.5, 1.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("KL Threshold Sensitivity: Mean Commitment vs τ\n"
                 "[First layer ℓ where KL(ℓ ‖ final) < τ and stays below | "
                 "Blue = raw, Red = tuned | Solid = IT, Dashed = PT]",
                 fontsize=11, fontweight="bold")
    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]
        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            recs = _load_commitment(model, variant)
            if recs is None:
                continue
            lbl_v = variant.upper()
            raw_vals, tuned_vals = [], []
            for t in thresholds:
                c = _extract_commits(recs, f"commitment_layer_raw_kl_{t}")
                raw_vals.append(np.mean(c) / nl if c else np.nan)
                c = _extract_commits(recs, f"commitment_layer_tuned_{t}")
                tuned_vals.append(np.mean(c) / nl if c else np.nan)
            ax.plot(thresholds, raw_vals, marker=marker, color=COL_PT, ls=ls,
                    lw=1.2, markersize=4, label=f"Raw {lbl_v}")
            ax.plot(thresholds, tuned_vals, marker=marker, color=COL_IT, ls=ls,
                    lw=1.2, markersize=4, label=f"Tuned {lbl_v}")
        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)")
        ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_xscale("log"); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, "L2_pure_kl_threshold_sensitivity.png")


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

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "L2_raw_vs_tuned_scatter.png")


# ═══════════════════════════════════════════════════════════════════════════════
# 14-15. Mean/Median KL commitment sensitivity
#    Commitment at layer ℓ iff mean/median(KL[ℓ:]) < τ
#    Computed from NPY arrays (raw_kl_final, tuned_kl_final)
# ═══════════════════════════════════════════════════════════════════════════════
def _plot_avgkl_sensitivity(agg_name, commit_fn, filename):
    """6-panel sensitivity: x = KL threshold, y = mean commitment (norm. depth).

    agg_name: "mean" or "median" (the aggregation used in the criterion)
    commit_fn: function(kl_arr, threshold) -> int array of commitment layers
    """
    thresholds = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"KL Threshold Sensitivity: Mean Commitment vs τ\n"
        f"[Criterion: {agg_name}(KL[ℓ:]) < τ | "
        f"Blue = raw logit-lens, Red = tuned logit-lens | Solid = IT, Dashed = PT]",
        fontsize=12, fontweight="bold",
    )

    for idx, model in enumerate(MODELS):
        ax = axes[idx // 3, idx % 3]
        nl = N_LAYERS[model]

        for variant, ls, marker in [("pt", "--", "o"), ("it", "-", "s")]:
            # Load KL arrays once per variant
            raw_kl = _load_kl_for_variant(model, variant, lens="raw")
            tuned_kl = _load_kl_for_variant(model, variant, lens="tuned")
            lbl_v = variant.upper()

            if raw_kl is not None:
                raw_means = []
                for t in thresholds:
                    commits = commit_fn(raw_kl, t)
                    raw_means.append(np.mean(commits) / nl)
                ax.plot(thresholds, raw_means, marker=marker, color=COL_PT, ls=ls,
                        lw=1.2, markersize=4, label=f"Raw {lbl_v}")

            if tuned_kl is not None:
                tuned_means = []
                for t in thresholds:
                    commits = commit_fn(tuned_kl, t)
                    tuned_means.append(np.mean(commits) / nl)
                ax.plot(thresholds, tuned_means, marker=marker, color=COL_IT, ls=ls,
                        lw=1.2, markersize=4, label=f"Tuned {lbl_v}")

        ax.set_title(LABELS[model], fontsize=11)
        ax.set_xlabel("KL threshold τ (nats)")
        ax.set_ylabel("Mean commitment (norm. depth)")
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.89])
    _save(fig, filename)


def plot_mean_kl_sensitivity():
    _plot_avgkl_sensitivity("mean", _mean_kl_commit,
                            "L2_kl_sensitivity_mean_criterion.png")


def plot_median_kl_sensitivity():
    _plot_avgkl_sensitivity("median", _median_kl_commit,
                            "L2_kl_sensitivity_median_criterion.png")


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
    plot_pure_kl_sensitivity()
    plot_scatter()

    # Group D: Mean/Median KL commitment sensitivity
    log.info("--- Group D: Mean/Median KL commitment sensitivity ---")
    plot_mean_kl_sensitivity()
    plot_median_kl_sensitivity()

    # Final inventory
    plots = sorted(OUT.glob("*.png"))
    log.info("=== DONE: %d plots in %s ===", len(plots), OUT)
    for p in plots:
        log.info("  %s", p.name)
