#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3 4B",
    "qwen3_4b": "Qwen3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo2 7B",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
}
MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
PT_PIPELINES = ["B_early_raw", "B_mid_raw", "B_late_raw"]
IT_PIPELINES = ["D_early_ptswap", "D_mid_ptswap", "D_late_ptswap"]
PIPELINE_LABEL = {
    "B_early_raw": "Early",
    "B_mid_raw": "Mid",
    "B_late_raw": "Late",
    "D_early_ptswap": "Early",
    "D_mid_ptswap": "Mid",
    "D_late_ptswap": "Late",
}
PIPELINE_COLOR = {
    "B_early_raw": "#1f77b4",
    "B_mid_raw": "#ff7f0e",
    "B_late_raw": "#d62728",
    "D_early_ptswap": "#1f77b4",
    "D_mid_ptswap": "#ff7f0e",
    "D_late_ptswap": "#d62728",
}
MECH_PIPELINES = ["A_prime_raw", "B_late_raw", "C_it_chat", "D_late_ptswap"]
MECH_LABEL = {
    "A_prime_raw": "A' raw",
    "B_late_raw": "B late",
    "C_it_chat": "C IT",
    "D_late_ptswap": "D late swap",
}
MECH_COLOR = {
    "A_prime_raw": "#7f7f7f",
    "B_late_raw": "#d62728",
    "C_it_chat": "#2ca02c",
    "D_late_ptswap": "#9467bd",
}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _plot_main(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    models = [m for m in MODEL_ORDER if m in summary["models"]]
    x = np.arange(len(models))
    width = 0.22

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_pt, ax_it, ax_mech = axes

    for i, pipeline in enumerate(PT_PIPELINES):
        offsets = x + (i - 1) * width
        vals = [
            summary["models"][model]["pt_side"][pipeline]["regions"]["final_20pct"]["kl_to_own_final"]["delta"] or 0.0
            for model in models
        ]
        ax_pt.bar(offsets, vals, width, color=PIPELINE_COLOR[pipeline], label=PIPELINE_LABEL[pipeline])
    ax_pt.set_title("PT-side sufficiency\nB_window - A' on late KL-to-own-final")
    ax_pt.set_ylabel("Positive = later / less committed")

    for i, pipeline in enumerate(IT_PIPELINES):
        offsets = x + (i - 1) * width
        vals = [
            summary["models"][model]["it_side"][pipeline]["regions"]["final_20pct"]["kl_to_own_final"]["delta"] or 0.0
            for model in models
        ]
        ax_it.bar(offsets, vals, width, color=PIPELINE_COLOR[pipeline], label=PIPELINE_LABEL[pipeline])
    ax_it.set_title("IT-side necessity\nD_window - C on late KL-to-own-final")
    ax_it.set_ylabel("Negative = collapses IT late delay")

    dense_mech = summary.get("dense_family_means", {}).get("mechanism_predictive_correlation", {})
    corr_metrics = ["delta_cosine", "anti_top1_proj", "support_teacher_proj", "anti_kl_final_proj"]
    corr_labels = ["delta-cos", "anti-top1", "support-teacher", "anti-KL-final"]
    corr_vals = [dense_mech.get(metric) or 0.0 for metric in corr_metrics]
    ax_mech.bar(np.arange(len(corr_metrics)), corr_vals, color=["#888888", "#1f77b4", "#2ca02c", "#d62728"])
    ax_mech.set_xticks(np.arange(len(corr_metrics)))
    ax_mech.set_xticklabels(corr_labels, rotation=15, ha="right")
    ax_mech.set_title("Dense-5 predictive correlation with late KL")
    ax_mech.set_ylabel("Mean Pearson r")

    for ax in (ax_pt, ax_it):
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(model, model) for model in models], rotation=20, ha="right")
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
    ax_mech.axhline(0, color="#888888", linewidth=0.8)
    ax_mech.grid(axis="y", alpha=0.2)

    handles, labels = ax_pt.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Exp13 full + Exp14 combined causal run\n"
        "Symmetric late-window sufficiency/necessity with output-relevant late-stage mechanism summaries",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_appendix(summary: dict, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    dense_pipeline_means = summary.get("dense_family_means", {}).get("mechanism_late_mean_by_pipeline", {})
    metrics = ["support_teacher_proj", "anti_top1_proj", "teacher_token_rank_gain"]
    titles = {
        "support_teacher_proj": "Support-teacher projection",
        "anti_top1_proj": "Anti-top1 projection",
        "teacher_token_rank_gain": "Teacher-token rank gain",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    x = np.arange(len(MECH_PIPELINES))
    for ax, metric in zip(axes, metrics, strict=True):
        vals = [
            (dense_pipeline_means.get(pipeline, {}) or {}).get(metric) or 0.0
            for pipeline in MECH_PIPELINES
        ]
        colors = [MECH_COLOR[pipeline] for pipeline in MECH_PIPELINES]
        ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([MECH_LABEL[p] for p in MECH_PIPELINES], rotation=20, ha="right")
        ax.set_title(titles[metric])
        ax.axhline(0, color="#888888", linewidth=0.8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Exp13 full appendix: dense-5 late-stage mechanism summaries", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exp13 full + exp14 summary outputs.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = _read_json(args.summary)
    out_dir = args.out_dir or args.summary.parent
    _plot_main(summary, out_dir / "exp13_full_causal_main.png")
    _plot_appendix(summary, out_dir / "exp13_full_causal_appendix.png")
    print(f"[exp13-full] wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
