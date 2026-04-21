#!/usr/bin/env python3
"""Plot B-experiment results: B1 (feature clamping), B2 (wdec inject),
B3 (control features), B4 (layer specificity).

Usage:
    uv run python scripts/plot_feature_steering.py
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/exp06_corrective_direction_steering")
OUT_DIR = RESULTS_DIR / "plots_B"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOVERNANCE_BENCH = ["coherent_assistant_rate", "format_compliance", "exp3_alignment_behavior"]
CAPABILITY_BENCH = ["mmlu_accuracy", "exp3_reasoning_em", "structural_token_ratio"]

GOV_COLORS = {"coherent_assistant_rate": "#e74c3c", "format_compliance": "#e67e22",
              "exp3_alignment_behavior": "#9b59b6"}
CAP_COLORS = {"mmlu_accuracy": "#2980b9", "exp3_reasoning_em": "#27ae60",
              "structural_token_ratio": "#16a085"}
ALL_COLORS = {**GOV_COLORS, **CAP_COLORS}

BENCH_LABELS = {
    "coherent_assistant_rate": "Coherent Asst Rate",
    "format_compliance": "Format Compliance",
    "exp3_alignment_behavior": "Alignment Behavior",
    "mmlu_accuracy": "MMLU Accuracy",
    "exp3_reasoning_em": "Reasoning EM",
    "structural_token_ratio": "Structural Token Ratio",
}


def load_scores(merged_dir: Path) -> dict[str, dict[str, float]]:
    """Returns {condition: {benchmark: value}}"""
    out: dict[str, dict[str, float]] = {}
    for row in csv.DictReader(open(merged_dir / "scores.csv")):
        v = row["value"]
        if v in ("", "None"):
            continue
        out.setdefault(row["condition"], {})[row["benchmark"]] = float(v)
    return out


def load_first_baseline(*merged_dirs: Path) -> dict[str, float]:
    """Return the first available `*_baseline` score row found in the given merged dirs."""
    for merged_dir in merged_dirs:
        if not (merged_dir / "scores.csv").exists():
            continue
        scores = load_scores(merged_dir)
        for condition, bench_scores in scores.items():
            if condition.endswith("_baseline"):
                return bench_scores
    return {}


# ── B1: Feature clamping dose-response ───────────────────────────────────────

def plot_B1() -> None:
    scores = load_scores(RESULTS_DIR / "merged_B1_it")

    FEATURE_SETS = [
        ("method12_top10",  "M12 top-10",  "#1abc9c"),
        ("method12_top50",  "M12 top-50",  "#3498db"),
        ("method12_top100", "M12 top-100", "#e74c3c"),
        ("method12_top500", "M12 top-500", "#e67e22"),
        ("method12_all",    "M12 all",     "#9b59b6"),
        ("method3_it_amplified_top100", "M3 IT-amp top-100", "#2ecc71"),
    ]
    GAMMA_RE = re.compile(r"_g([\d.]+)$")

    baseline = {b: scores.get("B1_baseline", {}).get(b, float("nan"))
                for b in GOVERNANCE_BENCH + CAPABILITY_BENCH}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax_idx, bench in enumerate(GOVERNANCE_BENCH + CAPABILITY_BENCH):
        ax = axes[ax_idx]
        bl_val = baseline.get(bench, float("nan"))
        ax.axhline(bl_val, color="gray", linestyle="--", alpha=0.6, label=f"Baseline {bl_val:.3f}")

        for fset_key, fset_label, color in FEATURE_SETS:
            gammas, vals = [], []
            for cond, bdict in sorted(scores.items()):
                if fset_key not in cond:
                    continue
                m = GAMMA_RE.search(cond)
                if m and bench in bdict:
                    gammas.append(float(m.group(1)))
                    vals.append(bdict[bench])
            if gammas:
                order = np.argsort(gammas)
                ax.plot([gammas[i] for i in order], [vals[i] for i in order],
                        marker="o", markersize=4, label=fset_label, color=color, linewidth=1.5)

        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10)
        ax.set_xlabel("γ (clamping strength)")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_ylim(-0.05, 1.05)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("B1: Feature Clamping — γ Sweep × Feature Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "B1_dose_response.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── B2: W_dec governance direction injection ──────────────────────────────────

def plot_B2() -> None:
    pt_scores = load_scores(RESULTS_DIR / "merged_B2_pt")
    it_scores = load_scores(RESULTS_DIR / "merged_B2_it")

    BETA_RE = re.compile(r"_b([-\d.]+)$")
    # M12 and M3 use the same precomputed governance direction → results are identical.
    # Use method12_top100 as the canonical fset; both fsets produce the same curve.
    FSET_KEY = "method12_top100"

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax_idx, bench in enumerate(GOVERNANCE_BENCH + CAPABILITY_BENCH):
        ax = axes[ax_idx]

        # B2a: PT inject (positive beta → adds governance structure)
        betas, vals = [], []
        for cond, bdict in sorted(pt_scores.items()):
            if "B2a" not in cond or FSET_KEY not in cond:
                continue
            m = BETA_RE.search(cond)
            if m and bench in bdict:
                betas.append(float(m.group(1)))
                pt_bl = pt_scores.get("B2_baseline", {}).get(bench, float("nan"))
                vals.append(bdict[bench] - pt_bl)
        if betas:
            order = np.argsort(betas)
            ax.plot([betas[i] for i in order], [vals[i] for i in order],
                    marker="s", markersize=5, label="PT inject: Δ vs PT baseline",
                    color="#e74c3c", linewidth=2.0, linestyle="-")

        # B2b: IT subtract (negative beta → removes governance structure from IT)
        betas, vals = [], []
        for cond, bdict in sorted(it_scores.items()):
            if "B2b" not in cond or FSET_KEY not in cond:
                continue
            m = BETA_RE.search(cond)
            if m and bench in bdict:
                betas.append(float(m.group(1)))
                it_bl = it_scores.get("B2_baseline", {}).get(bench, float("nan"))
                vals.append(bdict[bench] - it_bl)
        if betas:
            order = np.argsort(betas)
            ax.plot([betas[i] for i in order], [vals[i] for i in order],
                    marker="^", markersize=5, label="IT subtract: Δ vs IT baseline",
                    color="#3498db", linewidth=2.0, linestyle="--")

        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10)
        ax.set_xlabel("β (direction strength)")
        ax.set_ylabel("Δ score vs own baseline")
        ax.axhline(0, color="black", linestyle=":", alpha=0.8, linewidth=1.2)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.4)
        ax.grid(axis="y", alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="best")

    fig.suptitle("B2: W_dec Direction Injection (PT inject / IT subtract)\n"
                 "Y-axis is change relative to each model's own baseline (0 = no effect)\n"
                 "Note: M12 and M3 use the same precomputed direction, so one canonical curve is shown",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "B2_wdec_inject.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── B3: Control feature sets comparison ──────────────────────────────────────

def plot_B3() -> None:
    scores = load_scores(RESULTS_DIR / "merged_B3_it")
    baseline = load_first_baseline(RESULTS_DIR / "merged_B1_it", RESULTS_DIR / "merged_B2_it")

    FSETS_ORDER = ["random_100", "method12_content_top100", "method3_it_suppressed_top100"]
    FSET_LABELS = {
        "random_100": "Random control",
        "method12_content_top100": "Content-feature control",
        "method3_it_suppressed_top100": "IT-suppressed-feature control",
    }
    GAMMAS = [0.0, 1.0, 5.0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for panel_idx, bench_group in enumerate([GOVERNANCE_BENCH, CAPABILITY_BENCH]):
        ax = axes[panel_idx]
        group_title = "Governance benchmarks" if panel_idx == 0 else "Capability benchmarks"

        x = np.arange(len(FSETS_ORDER))
        width = 0.25
        gamma_colors = {0.0: "#95a5a6", 1.0: "#f39c12", 5.0: "#e74c3c"}

        for g_idx, gamma in enumerate(GAMMAS):
            vals = []
            for fset in FSETS_ORDER:
                cond = f"B3_{fset}_g{gamma:g}"
                deltas = [
                    scores.get(cond, {}).get(bench, float("nan")) - baseline.get(bench, float("nan"))
                    for bench in bench_group
                ]
                deltas = [d for d in deltas if not np.isnan(d)]
                vals.append(float(np.mean(deltas)) if deltas else float("nan"))
            ax.bar(x + g_idx * width - width, vals, width, alpha=0.85,
                   color=gamma_colors[gamma], label=f"γ={gamma:g}")

        ax.axhline(0, color="black", linestyle=":", alpha=0.8, linewidth=1.2)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([FSET_LABELS[f] for f in FSETS_ORDER], fontsize=8, rotation=10)
        ax.set_ylabel("Mean Δ vs IT baseline")
        ax.set_title(group_title, fontsize=11)
        if panel_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("B3: Control Feature Sets Summary\n"
                 "Bars show mean change relative to the IT baseline across each benchmark group",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "B3_control_features.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── B3 simple version: line plot per benchmark ───────────────────────────────

def plot_B3_lines() -> None:
    scores = load_scores(RESULTS_DIR / "merged_B3_it")
    baseline = load_first_baseline(RESULTS_DIR / "merged_B1_it", RESULTS_DIR / "merged_B2_it")

    FSET_COLORS = {
        "random_100": "#95a5a6",
        "method12_content_top100": "#3498db",
        "method3_it_suppressed_top100": "#e74c3c",
    }
    FSET_LABELS = {
        "random_100": "Random (ctrl)",
        "method12_content_top100": "M12 Content (ctrl)",
        "method3_it_suppressed_top100": "M3 IT-suppressed (ctrl)",
    }
    GAMMAS = [0.0, 1.0, 5.0]
    ALL_BENCH = GOVERNANCE_BENCH + CAPABILITY_BENCH

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax_idx, bench in enumerate(ALL_BENCH):
        ax = axes[ax_idx]
        for fset, color in FSET_COLORS.items():
            vals = [
                scores.get(f"B3_{fset}_g{g:g}", {}).get(bench, float("nan")) - baseline.get(bench, float("nan"))
                for g in GAMMAS
            ]
            ax.plot(GAMMAS, vals, marker="o", color=color, label=FSET_LABELS[fset], linewidth=1.5)
        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10)
        ax.set_xlabel("γ (feature clamp strength)")
        ax.set_ylabel("Δ vs IT baseline")
        ax.axhline(0, color="black", linestyle=":", alpha=0.8, linewidth=1.2)
        ax.set_xticks(GAMMAS)
        ax.grid(axis="y", alpha=0.2)
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("B3: Control Feature Sets — γ Sweep\n"
                 "Each line shows how a control feature set changes the metric relative to the IT baseline",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "B3_control_lines.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── B4: Layer specificity ─────────────────────────────────────────────────────

def plot_B4() -> None:
    scores = load_scores(RESULTS_DIR / "merged_B4_it")
    baseline = load_first_baseline(RESULTS_DIR / "merged_B1_it", RESULTS_DIR / "merged_B2_it")

    LAYER_RANGES = ["early_20_25", "mid_26_29", "late_30_33"]
    LR_LABELS = {"early_20_25": "Early (20-25)", "mid_26_29": "Mid (26-29)", "late_30_33": "Late (30-33)"}
    GAMMAS = [0.0, 5.0]
    ALL_BENCH = GOVERNANCE_BENCH + CAPABILITY_BENCH

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax_idx, bench in enumerate(ALL_BENCH):
        ax = axes[ax_idx]
        x = np.arange(len(LAYER_RANGES))
        width = 0.35
        gamma_colors = {0.0: "#bdc3c7", 5.0: "#e74c3c"}

        for g_idx, gamma in enumerate(GAMMAS):
            vals = [scores.get(f"B4_{lr}_g{gamma:g}", {}).get(bench, float("nan")) - baseline.get(bench, float("nan"))
                    for lr in LAYER_RANGES]
            ax.bar(x + g_idx * width - width / 2, vals, width,
                   color=gamma_colors[gamma], alpha=0.85,
                   label=f"γ={gamma:g}")

        ax.axhline(0, color="black", linestyle=":", alpha=0.8, linewidth=1.2)
        ax.grid(axis="y", alpha=0.2)
        ax.set_xticks(x)
        ax.set_xticklabels([LR_LABELS[lr] for lr in LAYER_RANGES], fontsize=9)
        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10)
        ax.set_ylabel("Δ vs IT baseline")
        if ax_idx == 0:
            ax.legend(fontsize=9)

    fig.suptitle("B4: Layer Specificity — Which Corrective Subrange Matters?\n"
                 "Bars show change relative to the IT baseline for the same governance feature set",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "B4_layer_specificity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── B1 summary: best feature set per gamma ────────────────────────────────────

def plot_B1_summary() -> None:
    """Single plot: coherent_assistant_rate vs gamma for all feature sets."""
    scores = load_scores(RESULTS_DIR / "merged_B1_it")

    FEATURE_SETS = [
        ("method12_top10",  "M12 top-10",  "#1abc9c", "o"),
        ("method12_top50",  "M12 top-50",  "#3498db", "s"),
        ("method12_top100", "M12 top-100", "#e74c3c", "^"),
        ("method12_top500", "M12 top-500", "#e67e22", "D"),
        ("method12_all",    "M12 all",     "#9b59b6", "v"),
        ("method3_it_amplified_top100", "M3 IT-amp", "#2ecc71", "P"),
    ]
    GAMMA_RE = re.compile(r"_g([\d.]+)$")
    BENCH = "coherent_assistant_rate"

    fig, ax = plt.subplots(figsize=(10, 6))
    bl = scores.get("B1_baseline", {}).get(BENCH, float("nan"))
    ax.axhline(bl, color="black", linestyle="--", linewidth=1.5, label=f"IT baseline ({bl:.3f})")

    for fset_key, fset_label, color, marker in FEATURE_SETS:
        gammas, vals = [], []
        for cond, bdict in sorted(scores.items()):
            if fset_key not in cond:
                continue
            m = GAMMA_RE.search(cond)
            if m and BENCH in bdict:
                gammas.append(float(m.group(1)))
                vals.append(bdict[BENCH])
        if gammas:
            order = np.argsort(gammas)
            xs = [gammas[i] for i in order]
            ys = [vals[i] for i in order]
            ax.plot(xs, ys, marker=marker, markersize=6, label=fset_label,
                    color=color, linewidth=2)

    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("γ (feature clamping strength)", fontsize=12)
    ax.set_ylabel("Coherent Assistant Rate", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("B1: Feature Clamping — Coherent Assistant Rate vs γ", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)

    fig.tight_layout()
    out = OUT_DIR / "B1_summary_coherent.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── B5: Full γ sweep × layer range ────────────────────────────────────────────

def plot_B5() -> None:
    """B5: Full γ sweep (including negative) × layer sub-range, method12_top100."""
    merged = RESULTS_DIR / "merged_B5_it"
    if not (merged / "scores.csv").exists():
        print("B5 results not yet available — skipping plot_B5()")
        return
    scores = load_scores(merged)

    GAMMA_RE = re.compile(r"_g([-\d.]+)$")
    LAYER_RANGES = [
        ("all",          "All 20-33", "#2c3e50"),
        ("early_20_25",  "Early 20-25", "#e74c3c"),
        ("mid_26_29",    "Mid 26-29",   "#3498db"),
        ("late_30_33",   "Late 30-33",  "#2ecc71"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for ax_idx, bench in enumerate(GOVERNANCE_BENCH + CAPABILITY_BENCH):
        ax = axes[ax_idx]

        bl = scores.get("B5_baseline", {}).get(bench, float("nan"))
        ax.axhline(bl, color="gray", linestyle="--", alpha=0.7,
                   linewidth=1.5, label=f"Baseline {bl:.3f}")
        ax.axvline(1.0, color="lightgray", linestyle=":", alpha=0.6)  # γ=1 = neutral

        for lr_key, lr_label, color in LAYER_RANGES:
            gammas, vals = [], []
            for cond, bdict in sorted(scores.items()):
                if f"B5_{lr_key}_g" not in cond:
                    continue
                m = GAMMA_RE.search(cond)
                if m and bench in bdict:
                    gammas.append(float(m.group(1)))
                    vals.append(bdict[bench])
            if gammas:
                order = np.argsort(gammas)
                ax.plot([gammas[i] for i in order], [vals[i] for i in order],
                        marker="o", markersize=4, label=lr_label,
                        color=color, linewidth=1.8)

        ax.set_title(BENCH_LABELS.get(bench, bench), fontsize=10)
        ax.set_xlabel("γ (feature clamp strength)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle("B5: Layer-Specificity × Full γ Sweep (method12_top100)\n"
                 "Negative γ = below-mean suppression; γ=0 = zero-out; γ=1 = neutral",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / "B5_layer_gamma_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    print("Generating B-experiment plots...")
    plot_B1_summary()
    plot_B1()
    plot_B2()
    plot_B3_lines()
    plot_B3()
    plot_B4()
    plot_B5()
    print(f"\nAll plots saved to {OUT_DIR}/")
