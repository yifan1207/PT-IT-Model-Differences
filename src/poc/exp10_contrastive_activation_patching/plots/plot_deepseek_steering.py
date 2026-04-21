"""Plot DeepSeek V2 Lite steering results: d_conv vs d_mean across alpha.

Two-panel figure:
  Left:  Governance (structural_token_ratio) — shows d_conv effect
  Right: Content (alignment_behavior) — shows d_conv preserves content while d_mean degrades it
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_BASE = Path("results/exp10/data")
OUT_DIR = Path("results/exp10/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_scores(direction: str) -> list[dict]:
    path = RESULTS_BASE / f"deepseek_{direction}_scores.json"
    with open(path) as f:
        return json.load(f)


def extract_alpha_series(scores: list[dict], benchmark: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract (alphas, values) for directional_remove conditions + baseline value."""
    baseline_val = None
    alpha_vals = {}

    for s in scores:
        if s["benchmark"] != benchmark or s["value"] is None:
            continue
        if s["method"] == "none":
            baseline_val = s["value"]
        elif s["method"] == "directional_remove":
            a = s["alpha"]
            # Deduplicate (some baselines repeated)
            alpha_vals[a] = s["value"]

    alphas = sorted(alpha_vals.keys())
    values = [alpha_vals[a] for a in alphas]
    return np.array(alphas), np.array(values), baseline_val


def main():
    scores_conv = load_scores("d_conv")
    scores_mean = load_scores("d_mean")

    benchmarks = [
        ("structural_token_ratio", "Structural Token Ratio\n(Governance)", "higher = more formatting"),
        ("exp3_alignment_behavior", "Alignment Behavior Match\n(Content Quality)", "higher = better content"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (bm, title, ylabel_note) in zip(axes, benchmarks):
        # d_conv
        a_conv, v_conv, bl_conv = extract_alpha_series(scores_conv, bm)
        # d_mean
        a_mean, v_mean, bl_mean = extract_alpha_series(scores_mean, bm)

        # Use the average baseline (should be identical across directions)
        bl = bl_conv if bl_conv is not None else bl_mean

        ax.plot(a_conv, v_conv, "o-", color="#2196F3", linewidth=2, markersize=5,
                label=r"$d_{\mathrm{conv}}$ (convergence-gap)", zorder=3)
        ax.plot(a_mean, v_mean, "s--", color="#FF5722", linewidth=2, markersize=5,
                label=r"$d_{\mathrm{mean}}$ (IT−PT mean)", zorder=3)

        if bl is not None:
            ax.axhline(bl, color="#666", linestyle=":", linewidth=1.5, alpha=0.7,
                       label=f"baseline ({bl:.3f})")

        ax.axvline(0, color="#ccc", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"Steering strength $\alpha$" + "\n" +
                      r"($\alpha > 0$: remove direction, $\alpha < 0$: amplify)",
                      fontsize=10)
        ax.set_ylabel(ylabel_note, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("DeepSeek V2 Lite — Causal Steering: Convergence-Gap vs Mean Direction",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "deepseek_steering_d_conv_vs_d_mean.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT_DIR / "deepseek_steering_d_conv_vs_d_mean.pdf", bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/deepseek_steering_d_conv_vs_d_mean.png")
    plt.close()

    # Also make a 4-panel version including MMLU and reasoning
    benchmarks_4 = [
        ("structural_token_ratio", "Structural Token Ratio\n(Governance)"),
        ("exp3_alignment_behavior", "Alignment Behavior\n(Content)"),
        ("mmlu_forced_choice", "MMLU Accuracy\n(Factual Knowledge)"),
        ("exp3_reasoning_em", "Reasoning EM\n(Reasoning)"),
    ]

    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
    axes2 = axes2.flatten()

    for ax, (bm, title) in zip(axes2, benchmarks_4):
        a_conv, v_conv, bl_conv = extract_alpha_series(scores_conv, bm)
        a_mean, v_mean, bl_mean = extract_alpha_series(scores_mean, bm)
        bl = bl_conv if bl_conv is not None else bl_mean

        # Get sample sizes
        n_vals = [s["n"] for s in scores_conv if s["benchmark"] == bm and s["value"] is not None]
        n = n_vals[0] if n_vals else 0

        ax.plot(a_conv, v_conv, "o-", color="#2196F3", linewidth=2, markersize=4,
                label=r"$d_{\mathrm{conv}}$", zorder=3)
        ax.plot(a_mean, v_mean, "s--", color="#FF5722", linewidth=2, markersize=4,
                label=r"$d_{\mathrm{mean}}$", zorder=3)

        if bl is not None:
            ax.axhline(bl, color="#666", linestyle=":", linewidth=1.5, alpha=0.7)

        ax.axvline(0, color="#ccc", linestyle="-", linewidth=0.8, alpha=0.5)
        ax.set_xlabel(r"$\alpha$", fontsize=10)
        ax.set_title(f"{title}  (n={n})", fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle("DeepSeek V2 Lite — All Benchmarks under Causal Steering",
                  fontsize=13, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(OUT_DIR / "deepseek_steering_all_benchmarks.png", dpi=200, bbox_inches="tight")
    fig2.savefig(OUT_DIR / "deepseek_steering_all_benchmarks.pdf", bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/deepseek_steering_all_benchmarks.png")
    plt.close()


if __name__ == "__main__":
    main()
