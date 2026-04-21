from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]

MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3 4B",
    "qwen3_4b": "Qwen3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo2 7B",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
}

COND_ORDER = ["A_pt_raw", "B_graft_raw", "C_it_chat"]
COND_DISPLAY = {
    "A_pt_raw": "A  PT raw",
    "B_graft_raw": "B  PT+IT graft",
    "C_it_chat": "C  IT chat",
}
COND_COLORS = {
    "A_pt_raw": "#4C78A8",
    "B_graft_raw": "#E45756",
    "C_it_chat": "#54A24B",
}

TASK_CONFIG = {
    "g1": {"title": "G1 Structure / Presentation", "ylim": (1.0, 5.0), "higher_better": True},
    "g2": {"title": "G2 Assistant Register", "ylim": (1.0, 5.0), "higher_better": True},
    "s2": {"title": "S2 False Refusal Rate", "ylim": (0.0, 1.0), "higher_better": False},
    "s1_refuse_rate": {"title": "S1 Harmful-Prompt Refusal Rate", "ylim": (0.0, 1.0), "higher_better": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot exp12 A/B/C judge comparisons.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("results/exp12_free_running_abc_graft/data"),
        help="Root dir containing exp12 merged run dirs.",
    )
    parser.add_argument(
        "--run-prefix",
        default="exp12_eval_v1_20260413",
        help="Run prefix under run-root.",
    )
    parser.add_argument(
        "--judge-file",
        default="llm_judge_v3_scores.jsonl",
        help="Judge score file name inside each merged run dir.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp12_free_running_abc_graft/plots/exp12_eval_v1_20260413_v3"),
        help="Output dir for plots and summary JSON.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Bootstrap resamples for 95% CIs.",
    )
    return parser.parse_args()


def _bootstrap_mean(vals: list[float], n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.asarray(vals, dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, mean
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def _bootstrap_rate(classes: list[str], target: str, n_boot: int, rng: np.random.Generator) -> tuple[float, float, float]:
    arr = np.asarray([1.0 if x == target else 0.0 for x in classes], dtype=float)
    mean = float(arr.mean())
    if len(arr) <= 1:
        return mean, mean, mean
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def load_metrics(run_root: Path, run_prefix: str, judge_file: str, n_boot: int) -> dict:
    rng = np.random.default_rng(0)
    out: dict[str, dict] = {}
    for model in MODEL_ORDER:
        merged_dir = run_root / f"{run_prefix}_{model}" / f"{run_prefix}_{model}"
        score_path = merged_dir / judge_file
        rows = [json.loads(line) for line in score_path.read_text().splitlines() if line.strip()]
        model_data = {"merged_dir": str(merged_dir), "metrics": {}, "s1_breakdown": {}}
        for cond in COND_ORDER:
            for task in ["g1", "g2", "s2"]:
                vals = [float(r["score"]) for r in rows if r["condition"] == cond and r["task"] == task]
                mean, lo, hi = _bootstrap_mean(vals, n_boot=n_boot, rng=rng)
                model_data["metrics"][f"{cond}__{task}"] = {"n": len(vals), "mean": mean, "ci95": [lo, hi]}
            s1 = [r["s1_class"] for r in rows if r["condition"] == cond and r["task"] == "s1"]
            refuse_mean, refuse_lo, refuse_hi = _bootstrap_rate(s1, "REFUSE", n_boot=n_boot, rng=rng)
            comply_mean, comply_lo, comply_hi = _bootstrap_rate(s1, "COMPLY", n_boot=n_boot, rng=rng)
            incoh_mean, incoh_lo, incoh_hi = _bootstrap_rate(s1, "INCOHERENT", n_boot=n_boot, rng=rng)
            model_data["metrics"][f"{cond}__s1_refuse_rate"] = {
                "n": len(s1),
                "mean": refuse_mean,
                "ci95": [refuse_lo, refuse_hi],
            }
            model_data["s1_breakdown"][cond] = {
                "REFUSE": refuse_mean,
                "COMPLY": comply_mean,
                "INCOHERENT": incoh_mean,
                "REFUSE_ci95": [refuse_lo, refuse_hi],
                "COMPLY_ci95": [comply_lo, comply_hi],
                "INCOHERENT_ci95": [incoh_lo, incoh_hi],
            }
        out[model] = model_data
    return out


def plot_overview(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    task_order = ["g1", "g2", "s2", "s1_refuse_rate"]
    for ax, task in zip(axes.flat, task_order):
        cfg = TASK_CONFIG[task]
        for idx, cond in enumerate(COND_ORDER):
            means = [metrics[m]["metrics"][f"{cond}__{task}"]["mean"] for m in MODEL_ORDER]
            ci = [metrics[m]["metrics"][f"{cond}__{task}"]["ci95"] for m in MODEL_ORDER]
            yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, ci)]).T
            ax.bar(
                x + (idx - 1) * width,
                means,
                width=width,
                color=COND_COLORS[cond],
                label=COND_DISPLAY[cond],
                yerr=yerr,
                capsize=3,
                alpha=0.9,
            )
        ax.set_title(cfg["title"])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylim(*cfg["ylim"])
        ax.grid(axis="y", alpha=0.25)
        if task == "s2":
            ax.set_ylabel("Lower is better")
        elif task == "s1_refuse_rate":
            ax.set_ylabel("Higher is better")
        else:
            ax.set_ylabel("Higher is better")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("exp12: Free-Running A/B/C Output Evaluation", y=1.06, fontsize=15)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_delta(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    task_order = ["g1", "g2", "s2", "s1_refuse_rate"]
    for ax, task in zip(axes.flat, task_order):
        cfg = TASK_CONFIG[task]
        sign = 1.0 if cfg["higher_better"] else -1.0
        b_minus_a = []
        c_minus_a = []
        for model in MODEL_ORDER:
            a = metrics[model]["metrics"][f"A_pt_raw__{task}"]["mean"]
            b = metrics[model]["metrics"][f"B_graft_raw__{task}"]["mean"]
            c = metrics[model]["metrics"][f"C_it_chat__{task}"]["mean"]
            b_minus_a.append(sign * (b - a))
            c_minus_a.append(sign * (c - a))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.bar(x - width / 2, b_minus_a, width=width, color="#E45756", label="B - A")
        ax.bar(x + width / 2, c_minus_a, width=width, color="#54A24B", label="C - A")
        ax.set_title(cfg["title"])
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if cfg["higher_better"]:
            ax.set_ylabel("Improvement over A")
        else:
            ax.set_ylabel("Reduction over A")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("exp12: Improvement Relative to PT Raw Baseline", y=1.06, fontsize=15)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_s1_breakdown(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), constrained_layout=True, sharey=True)
    labels = ["REFUSE", "COMPLY", "INCOHERENT"]
    colors = {"REFUSE": "#54A24B", "COMPLY": "#E45756", "INCOHERENT": "#9D9D9D"}
    x = np.arange(len(COND_ORDER))
    width = 0.58
    for ax, model in zip(axes.flat, MODEL_ORDER):
        bottom = np.zeros(len(COND_ORDER))
        for label in labels:
            vals = [metrics[model]["s1_breakdown"][cond][label] for cond in COND_ORDER]
            ax.bar(x, vals, width=width, bottom=bottom, color=colors[label], label=label)
            bottom += np.asarray(vals)
        ax.set_title(MODEL_DISPLAY[model])
        ax.set_xticks(x)
        ax.set_xticklabels([COND_DISPLAY[c] for c in COND_ORDER], rotation=15, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("exp12: S1 Harmful-Prompt Behavior Breakdown", y=1.06, fontsize=15)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_paper_main(metrics: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    panels = [
        ("g2", "Assistant Register (G2)", "Higher is better"),
        ("s2", "False Refusal on Benign Prompts (S2)", "Lower is better"),
    ]
    for ax, (task, title, ylabel) in zip(axes, panels):
        for idx, cond in enumerate(COND_ORDER):
            means = [metrics[m]["metrics"][f"{cond}__{task}"]["mean"] for m in MODEL_ORDER]
            ci = [metrics[m]["metrics"][f"{cond}__{task}"]["ci95"] for m in MODEL_ORDER]
            yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, ci)]).T
            ax.bar(
                x + (idx - 1) * width,
                means,
                width=width,
                color=COND_COLORS[cond],
                label=COND_DISPLAY[cond],
                yerr=yerr,
                capsize=3,
                alpha=0.9,
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel(ylabel)
        if task == "g2":
            ax.set_ylim(1.0, 5.0)
        else:
            ax.set_ylim(0.0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("exp12: Free-Running Output Effects of Late IT MLP Grafting", y=1.10, fontsize=15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(
        run_root=args.run_root,
        run_prefix=args.run_prefix,
        judge_file=args.judge_file,
        n_boot=args.bootstrap_samples,
    )
    (args.out_dir / "exp12_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_overview(metrics, args.out_dir / "exp12_scores_overview.png")
    plot_delta(metrics, args.out_dir / "exp12_delta_vs_a.png")
    plot_s1_breakdown(metrics, args.out_dir / "exp12_s1_breakdown.png")
    plot_paper_main(metrics, args.out_dir / "exp12_paper_main.png")
    print(f"Wrote exp12 plots to {args.out_dir}")


if __name__ == "__main__":
    main()
