#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RAW_CATEGORIES = ["CONTENT", "FUNCTION", "DISCOURSE", "STRUCTURAL", "PUNCTUATION", "OTHER"]
COLLAPSED = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]
WINDOW_LABELS = {
    "B_early_raw": "Early",
    "B_mid_raw": "Mid",
    "B_late_raw": "Late",
}


def _fraction(block: dict[str, dict[str, float | int | None]], key: str) -> float:
    frac = block.get(key, {}).get("fraction")
    return float(frac) if frac is not None else 0.0


def _mean_gain(block: dict[str, dict[str, float | int | None]], key: str) -> float:
    val = block.get(key, {}).get("mean_rank_gain")
    return float(val) if val is not None else 0.0


def _make_main_figure(summary: dict, out_dir: Path) -> None:
    pooled = summary["pooled"]
    tf = pooled["dense5_teacherforced_400"]
    depth = pooled["dense5_depth_ablation_600"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ax_a, ax_b, ax_c = axes

    x = np.arange(len(RAW_CATEGORIES))
    w = 0.38
    suppressed = [_fraction(tf["suppressed"], cat) for cat in RAW_CATEGORIES]
    supported = [_fraction(tf["supported"], cat) for cat in RAW_CATEGORIES]
    ax_a.bar(x - w / 2, suppressed, width=w, color="#C44E52", label="Suppressed in A'")
    ax_a.bar(x + w / 2, supported, width=w, color="#4C72B0", label="Supported in B_late")
    ax_a.set_xticks(x, RAW_CATEGORIES, rotation=25, ha="right")
    ax_a.set_ylabel("Fraction of displaced top-1 tokens")
    ax_a.set_title("Panel A — What late graft redirects away from and toward")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend()

    x = np.arange(len(COLLAPSED))
    gains = [_mean_gain(tf["teacher_rank_gain"], cat) for cat in COLLAPSED]
    ax_b.bar(x, gains, color=["#2A9D8F", "#457B9D", "#8D99AE"])
    ax_b.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax_b.set_xticks(x, COLLAPSED)
    ax_b.set_ylabel("Mean teacher-token rank gain\n(A' rank - B_late rank)")
    ax_b.set_title("Panel B — Which teacher-token types late graft helps")
    ax_b.grid(axis="y", alpha=0.25)

    x = np.arange(len(COLLAPSED))
    w = 0.24
    for idx, window in enumerate(["B_early_raw", "B_mid_raw", "B_late_raw"]):
        ax_c.bar(
            x + (idx - 1) * w,
            [_mean_gain(depth[window], cat) for cat in COLLAPSED],
            width=w,
            label=WINDOW_LABELS[window],
        )
    ax_c.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax_c.set_xticks(x, COLLAPSED)
    ax_c.set_ylabel("Mean teacher-token rank gain\n(A' rank - B_window rank)")
    ax_c.set_title("Panel C — Depth specificity of teacher-token support")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend()

    fig.suptitle(
        "Exp13A-lite — What the late corrective stage suppresses and supports",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "exp13a_lite_paper_main.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_appendix_figure(summary: dict, out_dir: Path) -> None:
    pooled = summary["pooled"]
    tf = pooled["dense5_teacherforced_400"]
    deepseek = pooled.get("deepseek_separate")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    x = np.arange(len(COLLAPSED))
    entries = [_fraction(summary["models"]["gemma3_4b"]["teacherforced_400"]["candidate_reshuffling"]["entries_collapsed"], cat) for cat in COLLAPSED]
    exits = [_fraction(summary["models"]["gemma3_4b"]["teacherforced_400"]["candidate_reshuffling"]["exits_collapsed"], cat) for cat in COLLAPSED]
    w = 0.38
    ax_a.bar(x - w / 2, exits, width=w, color="#E76F51", label="Exit A' top-20")
    ax_a.bar(x + w / 2, entries, width=w, color="#264653", label="Enter B_late top-20")
    ax_a.set_xticks(x, COLLAPSED)
    ax_a.set_ylabel("Fraction")
    ax_a.set_title("Panel A — Gemma candidate-set reshuffling")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend()

    for label, color in [("A_prime", "#C44E52"), ("B_late_raw", "#4C72B0"), ("C_it_chat", "#2A9D8F")]:
        ax_b.bar(
            np.arange(len(COLLAPSED)),
            [
                _fraction(tf["teacher_rank_gain"], cat)
                if False
                else _fraction(summary["models"]["gemma3_4b"]["teacherforced_400"]["mind_change"][label]["late_win_types_collapsed"], cat)
                for cat in COLLAPSED
            ],
            alpha=0.55,
            label=label,
            color=color,
        )
    ax_b.set_xticks(np.arange(len(COLLAPSED)), COLLAPSED)
    ax_b.set_ylabel("Fraction of late wins")
    ax_b.set_title("Panel B — Gemma late wins by token type")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend()

    models = [m for m in ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"] if m in summary["models"]]
    x = np.arange(len(models))
    if models:
        ax_c.bar(x - 0.12, [_mean_gain(summary["models"][m]["depth_ablation_600"]["teacher_rank_gain_by_window"]["B_mid_raw"], "FORMAT") for m in models], width=0.24, label="Mid")
        ax_c.bar(x + 0.12, [_mean_gain(summary["models"][m]["depth_ablation_600"]["teacher_rank_gain_by_window"]["B_late_raw"], "FORMAT") for m in models], width=0.24, label="Late")
        ax_c.set_xticks(x, [m.replace("_", " ").replace("3 4b", "3-4B") for m in models], rotation=20)
        ax_c.set_ylabel("Mean rank gain on FORMAT targets")
        ax_c.set_title("Panel C — Dense-5 per-model depth effect on FORMAT targets")
        ax_c.grid(axis="y", alpha=0.25)
        ax_c.legend()
    else:
        ax_c.axis("off")

    if deepseek is not None:
        vals = deepseek["depth_ablation_600"]["teacher_rank_gain_by_window"]
        ax_d.bar(
            np.arange(3),
            [_mean_gain(vals[window], "FORMAT") for window in ["B_early_raw", "B_mid_raw", "B_late_raw"]],
            color=["#8D99AE", "#457B9D", "#2A9D8F"],
        )
        ax_d.set_xticks(np.arange(3), ["Early", "Mid", "Late"])
        ax_d.set_ylabel("Mean rank gain on FORMAT targets")
        ax_d.set_title("Panel D — DeepSeek separate case")
        ax_d.grid(axis="y", alpha=0.25)
    else:
        ax_d.axis("off")

    fig.suptitle("Exp13A-lite appendix diagnostics", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "exp13a_lite_appendix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_per_model_panels(summary: dict, out_dir: Path) -> None:
    for model, payload in summary["models"].items():
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
        ax_a, ax_b = axes
        tf = payload["teacherforced_400"]
        x = np.arange(len(COLLAPSED))
        w = 0.38
        ax_a.bar(x - w / 2, [_fraction(tf["displacement"]["suppressed_collapsed"], cat) for cat in COLLAPSED], width=w, label="Suppressed", color="#C44E52")
        ax_a.bar(x + w / 2, [_fraction(tf["displacement"]["supported_collapsed"], cat) for cat in COLLAPSED], width=w, label="Supported", color="#4C72B0")
        ax_a.set_xticks(x, COLLAPSED)
        ax_a.set_ylabel("Fraction")
        ax_a.set_title("Late displacement categories")
        ax_a.grid(axis="y", alpha=0.25)
        ax_a.legend()

        depth = payload["depth_ablation_600"]["teacher_rank_gain_by_window"]
        for idx, window in enumerate(["B_early_raw", "B_mid_raw", "B_late_raw"]):
            ax_b.bar(
                x + (idx - 1) * 0.24,
                [_mean_gain(depth[window], cat) for cat in COLLAPSED],
                width=0.24,
                label=WINDOW_LABELS[window],
            )
        ax_b.axhline(0.0, color="black", lw=0.8, alpha=0.5)
        ax_b.set_xticks(x, COLLAPSED)
        ax_b.set_ylabel("Mean rank gain")
        ax_b.set_title("Depth-specific teacher-token support")
        ax_b.grid(axis="y", alpha=0.25)
        ax_b.legend()

        fig.suptitle(f"Exp13A-lite — {model}", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig(out_dir / f"{model}_panel.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exp13A-lite summary outputs.")
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "results/exp13_late_stage_token_support_analysis/"
            "exp13A_lite_20260415/exp13a_lite_summary.json"
        ),
        help="Summary JSON produced by analyze_exp13a_lite.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots; defaults to summary parent.",
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    out_dir = args.out_dir or args.summary.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_main_figure(summary, out_dir)
    _make_appendix_figure(summary, out_dir)
    _make_per_model_panels(summary, out_dir)
    print(f"[exp13A-lite] wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
