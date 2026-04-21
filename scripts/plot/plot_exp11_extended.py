#!/usr/bin/env python3
"""
Extended exp11 plots:
1. Residual divergence (||h_B - h_A|| / ||h_A||) per layer — shows norm divergence
2. B-vs-IT KL comparison: how far does MLP graft (B) get toward pure IT?
3. δ-cosine difference (B - A) per layer — cleaner than overlaid lines
4. Entropy difference (B - A) per layer
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
EXP11_DATA = Path("results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_all2936_tunedlens_v10")
EXP9_DATA  = Path("results/exp09_cross_model_observational_replication/data/convergence_gap_values.json")
OUT_DIR    = Path("results/exp11_matched_prefix_mlp_graft/plots/exp11_extended")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

def _load_exp11(model):
    p = EXP11_DATA / model / "trajectory_summary.json"
    with open(p) as f:
        return json.load(f)

def _load_exp9():
    with open(EXP9_DATA) as f:
        return json.load(f)

# ── load all data ──────────────────────────────────────────────────────────
exp11 = {m: _load_exp11(m) for m in MODELS}
exp9 = _load_exp9()

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1: Residual divergence per layer (all models)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
fig.suptitle("Residual Divergence: ||h_B − h_A|| / ||h_A|| per Layer", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    rd = np.array(exp11[m]["B"]["residual_divergence"])
    n = len(rd)
    layers = np.arange(n)
    onset = int(n * 0.6)

    ax.bar(layers[:onset], rd[:onset], color="gray", alpha=0.3, width=0.8, label="Pre-onset")
    ax.bar(layers[onset:], rd[onset:], color=COLORS[m], alpha=0.7, width=0.8, label="Post-onset (graft)")
    ax.axvline(onset - 0.5, color="black", ls="--", alpha=0.5, lw=1)
    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Layer")
    if ax == axes[0]:
        ax.set_ylabel("Relative norm divergence")
    ax.legend(fontsize=7, loc="upper right")

fig.tight_layout()
fig.savefig(OUT_DIR / "residual_divergence_all.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'residual_divergence_all.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2: B pipeline vs pure IT — KL-to-final comparison
# Shows: PT (exp9), IT (exp9), Pipeline A (exp11), Pipeline B (exp11)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(24, 5))
fig.suptitle("KL-to-Final: How Far Does MLP Graft (B) Get Toward Pure IT?", fontsize=14, y=1.02)

contribution_pcts = {}
for ax, m in zip(axes, MODELS):
    # exp11 data
    kl_a = np.array(exp11[m]["A"]["kl_to_own_final"])
    kl_b = np.array(exp11[m]["B"]["kl_to_own_final"])
    n11 = len(kl_a)
    layers11 = np.arange(n11)

    # exp9 data (tuned lens)
    kl_pt_exp9 = np.array(exp9["tuned"][m]["mean_kl_pt_per_layer"])
    kl_it_exp9 = np.array(exp9["tuned"][m]["mean_kl_it_per_layer"])
    n9 = len(kl_pt_exp9)
    layers9 = np.arange(n9)

    # Plot all four curves
    ax.plot(layers9, kl_pt_exp9, "b--", alpha=0.5, lw=1.5, label="Pure PT (exp9)")
    ax.plot(layers9, kl_it_exp9, "r--", alpha=0.5, lw=1.5, label="Pure IT (exp9)")
    ax.plot(layers11, kl_a, "b-", alpha=0.8, lw=2, label="Pipeline A (PT, exp11)")
    ax.plot(layers11, kl_b, "r-", alpha=0.8, lw=2, label="Pipeline B (PT+IT MLP)")

    # Shade the gap B creates beyond A
    ax.fill_between(layers11, kl_a, kl_b, alpha=0.15, color="red",
                     label="MLP graft effect")

    onset = int(n11 * 0.6)
    ax.axvline(onset - 0.5, color="black", ls=":", alpha=0.4, lw=1)

    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Layer")
    if ax == axes[0]:
        ax.set_ylabel("KL(layer → final) [nats]")
    ax.legend(fontsize=6.5, loc="upper right")

    # Compute contribution %: how much of (IT - PT) gap does B close?
    # Use second half of layers where the gap matters
    half = n11 // 2
    # Mean KL in second half for each
    mean_a_2nd = np.mean(kl_a[half:])
    mean_b_2nd = np.mean(kl_b[half:])
    # For exp9 IT and PT, use same layer range
    if n9 == n11:
        mean_pt_2nd = np.mean(kl_pt_exp9[half:])
        mean_it_2nd = np.mean(kl_it_exp9[half:])
    else:
        # Different layer counts: use proportional range
        half9 = n9 // 2
        mean_pt_2nd = np.mean(kl_pt_exp9[half9:])
        mean_it_2nd = np.mean(kl_it_exp9[half9:])

    it_pt_gap = mean_it_2nd - mean_pt_2nd
    b_a_gap = mean_b_2nd - mean_a_2nd
    pct = (b_a_gap / it_pt_gap * 100) if abs(it_pt_gap) > 0.01 else float("nan")
    contribution_pcts[m] = pct

    ax.text(0.05, 0.05, f"MLP→{pct:.0f}% of IT gap",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig.tight_layout()
fig.savefig(OUT_DIR / "b_vs_it_kl_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'b_vs_it_kl_comparison.png'}")
print(f"MLP contribution percentages: {contribution_pcts}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3: δ-cosine difference (B - A) per layer — the negative shift
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
fig.suptitle("δ-Cosine Shift: Pipeline B − Pipeline A (negative = more opposition)", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    dc_a = np.array(exp11[m]["A"]["delta_cosine"])
    dc_b = np.array(exp11[m]["B"]["delta_cosine"])
    diff = dc_b - dc_a
    n = len(diff)
    layers = np.arange(n)
    onset = int(n * 0.6)

    colors_bar = ["#E24A33" if d < 0 else "#348ABD" for d in diff]
    ax.bar(layers, diff, color=colors_bar, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(onset - 0.5, color="black", ls="--", alpha=0.5, lw=1)

    # Count negative in post-onset
    n_neg_post = np.sum(diff[onset:] < 0)
    n_post = len(diff[onset:])
    mean_post = np.mean(diff[onset:])

    ax.set_title(f"{LABELS[m]}\n{n_neg_post}/{n_post} neg, mean={mean_post:.4f}", fontsize=10)
    ax.set_xlabel("Layer")
    if ax == axes[0]:
        ax.set_ylabel("δ-cos(B) − δ-cos(A)")

fig.tight_layout()
fig.savefig(OUT_DIR / "delta_cosine_diff_ba.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'delta_cosine_diff_ba.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4: Entropy difference (B - A) per layer — "deliberate then decide"
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=False)
fig.suptitle("Entropy Shift: Pipeline B − Pipeline A (positive = more uncertain)", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    ent_a = np.array(exp11[m]["A"]["entropy"])
    ent_b = np.array(exp11[m]["B"]["entropy"])
    diff = ent_b - ent_a
    n = len(diff)
    layers = np.arange(n)
    onset = int(n * 0.6)

    colors_bar = ["#E24A33" if d > 0 else "#348ABD" for d in diff]
    ax.bar(layers, diff, color=colors_bar, alpha=0.7, width=0.8)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(onset - 0.5, color="black", ls="--", alpha=0.5, lw=1)

    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Layer")
    if ax == axes[0]:
        ax.set_ylabel("Entropy(B) − Entropy(A) [nats]")

fig.tight_layout()
fig.savefig(OUT_DIR / "entropy_diff_ba.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'entropy_diff_ba.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5: Residual cosine similarity (cross-pipeline alignment)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
fig.suptitle("Cross-Pipeline Residual Cosine Similarity per Layer", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    rc = np.array(exp11[m]["B"]["residual_cosine"])
    n = len(rc)
    layers = np.arange(n)
    onset = int(n * 0.6)

    ax.plot(layers, rc, color=COLORS[m], lw=2)
    ax.axvline(onset - 0.5, color="black", ls="--", alpha=0.5, lw=1)
    ax.fill_between(layers, rc, 1.0, alpha=0.1, color=COLORS[m])

    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Layer")
    if ax == axes[0]:
        ax.set_ylabel("cos(h_B, h_A)")
    ax.set_ylim(0.3, 1.05)

fig.tight_layout()
fig.savefig(OUT_DIR / "residual_cosine_alignment.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'residual_cosine_alignment.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 6: Combined summary — MLP contribution bar chart
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_title("MLP Contribution: % of IT-PT Convergence Gap Explained by IT MLP Weights", fontsize=12)

models_sorted = sorted(contribution_pcts.keys(), key=lambda m: contribution_pcts[m], reverse=True)
x = np.arange(len(models_sorted))
vals = [contribution_pcts[m] for m in models_sorted]
bars = ax.bar(x, vals, color=[COLORS[m] for m in models_sorted], alpha=0.8, width=0.6)

for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{v:.0f}%", ha="center", fontsize=11, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in models_sorted])
ax.set_ylabel("% of IT–PT KL gap (2nd half layers)")
ax.axhline(100, color="gray", ls="--", alpha=0.5, lw=1)
ax.text(len(x)-0.5, 101, "Full IT", fontsize=9, color="gray", ha="right")

fig.tight_layout()
fig.savefig(OUT_DIR / "mlp_contribution_bar.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'mlp_contribution_bar.png'}")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 7: KL difference curves (B-A) vs (IT-PT) overlay — per model
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 5, figsize=(24, 4.5))
fig.suptitle("KL Shift: MLP Graft Effect (B−A) vs Full IT Effect (IT−PT)", fontsize=14, y=1.02)

for ax, m in zip(axes, MODELS):
    kl_a = np.array(exp11[m]["A"]["kl_to_own_final"])
    kl_b = np.array(exp11[m]["B"]["kl_to_own_final"])
    n11 = len(kl_a)

    kl_pt_exp9 = np.array(exp9["tuned"][m]["mean_kl_pt_per_layer"])
    kl_it_exp9 = np.array(exp9["tuned"][m]["mean_kl_it_per_layer"])
    n9 = len(kl_pt_exp9)

    # B - A difference
    diff_ba = kl_b - kl_a
    layers11 = np.arange(n11)

    # IT - PT difference (from exp9)
    diff_itpt = kl_it_exp9 - kl_pt_exp9
    layers9 = np.arange(n9)

    ax.plot(layers11, diff_ba, color=COLORS[m], lw=2.5, label="B − A (MLP graft)")
    ax.plot(layers9, diff_itpt, color="black", lw=1.5, ls="--", alpha=0.7, label="IT − PT (full)")
    ax.fill_between(layers11, 0, diff_ba, alpha=0.15, color=COLORS[m])
    ax.axhline(0, color="gray", lw=0.5)

    onset = int(n11 * 0.6)
    ax.axvline(onset - 0.5, color="black", ls=":", alpha=0.4, lw=1)

    ax.set_title(LABELS[m], fontsize=11)
    ax.set_xlabel("Layer")
    if ax == axes[0]:
        ax.set_ylabel("Δ KL-to-final [nats]")
    ax.legend(fontsize=7, loc="upper left")

fig.tight_layout()
fig.savefig(OUT_DIR / "kl_diff_ba_vs_itpt.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_DIR / 'kl_diff_ba_vs_itpt.png'}")

# ── Print summary statistics ───────────────────────────────────────────────
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

for m in MODELS:
    dc_a = np.array(exp11[m]["A"]["delta_cosine"])
    dc_b = np.array(exp11[m]["B"]["delta_cosine"])
    diff = dc_b - dc_a
    n = len(diff)
    onset = int(n * 0.6)

    rd = np.array(exp11[m]["B"]["residual_divergence"])
    rc = np.array(exp11[m]["B"]["residual_cosine"])

    ent_a = np.array(exp11[m]["A"]["entropy"])
    ent_b = np.array(exp11[m]["B"]["entropy"])
    ent_diff = ent_b - ent_a

    kl_a = np.array(exp11[m]["A"]["kl_to_own_final"])
    kl_b = np.array(exp11[m]["B"]["kl_to_own_final"])
    kl_diff = kl_b - kl_a

    print(f"\n{LABELS[m]} ({n} layers, onset={onset}):")
    print(f"  δ-cos diff (post-onset): mean={np.mean(diff[onset:]):.4f}, "
          f"neg={np.sum(diff[onset:]<0)}/{len(diff[onset:])}")
    print(f"  Residual divergence (post-onset): mean={np.mean(rd[onset:]):.4f}, "
          f"range=[{np.min(rd[onset:]):.4f}, {np.max(rd[onset:]):.4f}]")
    print(f"  Residual cosine (post-onset): mean={np.mean(rc[onset:]):.4f}")
    print(f"  Entropy diff (post-onset): mean={np.mean(ent_diff[onset:]):.4f}")
    print(f"  KL diff (post-onset): mean={np.mean(kl_diff[onset:]):.4f}")
    print(f"  MLP contribution: {contribution_pcts[m]:.1f}%")

print("\nDone!")
