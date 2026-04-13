#!/usr/bin/env python3
"""
Compare commitment delay: Pipeline A (PT) vs Pipeline B (PT+IT MLP) vs Pure IT.
Uses exp11 overview metrics + exp7 0G commitment data.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path("results/exp11/plots/exp11_extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ──────────────────────────────────────────────────────────────
with open("results/exp11/data/exp11_exp3_all2936_tunedlens_v10/gemma3_4b/trajectory_summary.json") as f:
    _ = json.load(f)  # just to verify

# Exp11 overview metrics (commitment layers from KL threshold 0.1)
with open("results/exp11/plots/exp11_exp3_all2936_tunedlens_v10/overview_metrics.json") as f:
    exp11_overview = {d["model"]: d for d in json.load(f)}

# Exp7 0G commitment summary (pure IT vs PT)
with open("results/exp7/data/0G_commitment_summary.json") as f:
    commitment_0g = json.load(f)

# Exp9 convergence gap values (per-layer KL for IT and PT)
with open("results/exp9/data/convergence_gap_values.json") as f:
    exp9 = json.load(f)

MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
LABELS = {
    "gemma3_4b": "Gemma 3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "qwen3_4b": "Qwen 3 4B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo 2 7B",
}
COLORS = {
    "gemma3_4b": "#E24A33",
    "llama31_8b": "#348ABD",
    "qwen3_4b": "#988ED5",
    "mistral_7b": "#FBC15E",
    "olmo2_7b": "#8EBA42",
}

# ═══════════════════════════════════════════════════════════════════════════
# PLOT: Commitment layer comparison — A vs B vs IT
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(MODELS))
width = 0.25

# Exp11 Pipeline A (pure PT)
a_vals = [exp11_overview[m]["pipeline_a_mean_commitment_layer_kl_0.1"] for m in MODELS]

# Exp11 Pipeline B (PT + IT MLP graft)
b_vals = [exp11_overview[m]["pipeline_b_mean_commitment_layer_kl_0.1"] for m in MODELS]

# Pure IT from 0G (Raw KL median) — closest to exp11's KL-based metric
# Also get PT from 0G for reference
it_raw_kl = [commitment_0g["models"][m]["Raw KL"]["it_median"] for m in MODELS]
pt_raw_kl = [commitment_0g["models"][m]["Raw KL"]["pt_median"] for m in MODELS]

bars_a = ax.bar(x - width, a_vals, width, label="Pipeline A (Pure PT, exp11)",
                color="steelblue", alpha=0.7)
bars_b = ax.bar(x, b_vals, width, label="Pipeline B (PT + IT MLP, exp11)",
                color="#E24A33", alpha=0.8)
bars_it = ax.bar(x + width, it_raw_kl, width, label="Pure IT (0G Raw KL median)",
                 color="#2ca02c", alpha=0.7)

# Add value labels
for bars in [bars_a, bars_b, bars_it]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{bar.get_height():.1f}", ha="center", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in MODELS])
ax.set_ylabel("Mean Commitment Layer")
ax.set_title("Commitment Layer: MLP Graft (B) vs Pure PT (A) vs Full IT", fontsize=13)
ax.legend(loc="upper left", fontsize=9)

fig.tight_layout()
fig.savefig(OUT_DIR / "commitment_a_b_it_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'commitment_a_b_it_comparison.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# Print quantitative comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("COMMITMENT LAYER COMPARISON: A vs B vs IT")
print("="*70)

for m in MODELS:
    a = exp11_overview[m]["pipeline_a_mean_commitment_layer_kl_0.1"]
    b = exp11_overview[m]["pipeline_b_mean_commitment_layer_kl_0.1"]
    it_rk = commitment_0g["models"][m]["Raw KL"]["it_median"]
    pt_rk = commitment_0g["models"][m]["Raw KL"]["pt_median"]
    n_layers = len(json.load(open(f"results/exp11/data/exp11_exp3_all2936_tunedlens_v10/{m}/trajectory_summary.json"))["A"]["delta_cosine"])

    delay_ba = b - a
    delay_itpt = it_rk - pt_rk

    # How much of IT-PT gap does B close?
    if abs(delay_itpt) > 0.1:
        pct = delay_ba / delay_itpt * 100
    else:
        pct = float("nan")

    print(f"\n{LABELS[m]} ({n_layers} layers):")
    print(f"  Pipeline A (PT):       commitment = {a:.2f}")
    print(f"  Pipeline B (PT+MLP):   commitment = {b:.2f}  (Δ = +{delay_ba:.2f})")
    print(f"  Pure IT (0G Raw KL):   commitment = {it_rk:.1f}")
    print(f"  Pure PT (0G Raw KL):   commitment = {pt_rk:.1f}  (IT-PT delay = {delay_itpt:.1f})")
    if not np.isnan(pct):
        print(f"  MLP graft closes {pct:.0f}% of IT-PT commitment gap")
    else:
        print(f"  IT-PT gap ≈ 0, cannot compute %")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT: KL trajectory shape comparison (normalized)
# Normalize each curve to [0,1] range for shape comparison
# ═══════════════════════════════════════════════════════════════════════════
EXP11_DATA = Path("results/exp11/data/exp11_exp3_all2936_tunedlens_v10")

fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
fig.suptitle("KL-to-Final Normalized Shape: MLP Graft vs Pure IT vs Pure PT", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    # exp11
    with open(EXP11_DATA / m / "trajectory_summary.json") as f:
        traj = json.load(f)
    kl_a = np.array(traj["A"]["kl_to_own_final"])
    kl_b = np.array(traj["B"]["kl_to_own_final"])
    n11 = len(kl_a)

    # exp9 (tuned lens)
    kl_pt = np.array(exp9["tuned"][m]["mean_kl_pt_per_layer"])
    kl_it = np.array(exp9["tuned"][m]["mean_kl_it_per_layer"])
    n9 = len(kl_pt)

    # Normalize to fractional depth [0, 1] and KL to [0, 1]
    def norm_kl(kl):
        mx = np.max(kl)
        return kl / mx if mx > 0 else kl

    frac11 = np.linspace(0, 1, n11)
    frac9 = np.linspace(0, 1, n9)

    ax.plot(frac9, norm_kl(kl_pt), "b--", alpha=0.5, lw=1.5, label="Pure PT")
    ax.plot(frac9, norm_kl(kl_it), "r--", alpha=0.5, lw=1.5, label="Pure IT")
    ax.plot(frac11, norm_kl(kl_a), "b-", alpha=0.8, lw=2, label="A (PT)")
    ax.plot(frac11, norm_kl(kl_b), "r-", alpha=0.8, lw=2, label="B (PT+IT MLP)")

    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Fractional Depth")
    if ax == axes[0]:
        ax.set_ylabel("Normalized KL-to-final")
    ax.legend(fontsize=6.5, loc="upper right")

fig.tight_layout()
fig.savefig(OUT_DIR / "kl_shape_normalized.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_DIR / 'kl_shape_normalized.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT: Convergence gap — how much extra KL does B add vs A at each layer
# compared to how much extra KL IT adds vs PT
# Both measured as fraction of total KL range
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
fig.suptitle("Convergence Gap: Fractional KL Excess at 2nd-Half Layers", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    with open(EXP11_DATA / m / "trajectory_summary.json") as f:
        traj = json.load(f)
    kl_a = np.array(traj["A"]["kl_to_own_final"])
    kl_b = np.array(traj["B"]["kl_to_own_final"])
    n11 = len(kl_a)

    kl_pt = np.array(exp9["tuned"][m]["mean_kl_pt_per_layer"])
    kl_it = np.array(exp9["tuned"][m]["mean_kl_it_per_layer"])
    n9 = len(kl_pt)

    # Compute fractional gap at second half
    half11 = n11 // 2
    half9 = n9 // 2

    # For each layer in 2nd half, compute (B-A)/A and (IT-PT)/PT
    gap_ba = (kl_b[half11:] - kl_a[half11:])
    safe_a = np.maximum(kl_a[half11:], 0.01)
    frac_ba = gap_ba / safe_a

    gap_itpt = (kl_it[half9:] - kl_pt[half9:])
    safe_pt = np.maximum(kl_pt[half9:], 0.01)
    frac_itpt = gap_itpt / safe_pt

    layers_ba = np.arange(half11, n11)
    layers_itpt = np.arange(half9, n9)

    ax.bar(np.arange(len(frac_ba)), frac_ba, alpha=0.6, color=COLORS[m],
           label=f"(B−A)/A, mean={np.mean(frac_ba):.2f}")
    # Overlay IT-PT as line (may have different length)
    x_itpt = np.linspace(0, len(frac_ba)-1, len(frac_itpt))
    ax.plot(x_itpt, frac_itpt, "k--", lw=1.5, alpha=0.6,
            label=f"(IT−PT)/PT, mean={np.mean(frac_itpt):.2f}")

    ax.axhline(0, color="gray", lw=0.5)
    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Layer (2nd half only)")
    if ax == axes[0]:
        ax.set_ylabel("Fractional KL excess")
    ax.legend(fontsize=7, loc="upper left")

fig.tight_layout()
fig.savefig(OUT_DIR / "convergence_gap_fractional.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'convergence_gap_fractional.png'}")
