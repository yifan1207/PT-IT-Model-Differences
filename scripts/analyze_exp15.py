#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.poc.exp5.benchmarks.custom import evaluate_custom_benchmark
from src.poc.exp5.runtime import GeneratedSample as GeneratedSample5
from src.poc.exp6.benchmarks.governance import score_structural_token_ratio
from src.poc.exp6.benchmarks.governance_v2 import score_format_compliance_v2, score_mmlu_forced_choice
from src.poc.exp6.runtime import GeneratedSample6


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "deepseek_v2_lite",
]
DENSE_MODELS = [model for model in MODEL_ORDER if model != "deepseek_v2_lite"]
MODEL_DISPLAY = {
    "gemma3_4b": "Gemma3 4B",
    "qwen3_4b": "Qwen3 4B",
    "llama31_8b": "Llama 3.1 8B",
    "mistral_7b": "Mistral 7B",
    "olmo2_7b": "OLMo2 7B",
    "deepseek_v2_lite": "DeepSeek V2 Lite",
}
PT_WINDOWS = ["B_early_raw", "B_mid_raw", "B_late_raw"]
IT_WINDOWS = ["D_early_ptswap", "D_mid_ptswap", "D_late_ptswap"]
WINDOW_DISPLAY = {
    "B_early_raw": "Early",
    "B_mid_raw": "Mid",
    "B_late_raw": "Late",
    "D_early_ptswap": "Early",
    "D_mid_ptswap": "Mid",
    "D_late_ptswap": "Late",
}
WINDOW_COLORS = {
    "B_early_raw": "#72B7B2",
    "B_mid_raw": "#F58518",
    "B_late_raw": "#E45756",
    "D_early_ptswap": "#72B7B2",
    "D_mid_ptswap": "#F58518",
    "D_late_ptswap": "#E45756",
}
PAIRWISE_TARGET = {
    "pt_late_vs_a": "B_late_raw",
    "it_c_vs_dlate": "C_it_chat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exp15 behavioral causality runs.")
    parser.add_argument("--run-root", type=Path, default=Path("results/exp15/data"))
    parser.add_argument("--run-prefix", default="exp15_eval_core_600")
    parser.add_argument(
        "--internal-summary",
        type=Path,
        default=Path("results/exp13/exp13exp14_full_20260416/exp13_full_summary.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp15/plots/exp15_eval_core_600"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _run_dir(run_root: Path, run_prefix: str, model: str) -> Path:
    return run_root / f"{run_prefix}_{model}" / f"{run_prefix}_{model}"


def _bootstrap_mean(values: list[float], n_boot: int, seed: int = 0) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"mean": mean, "lo": mean, "hi": mean}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot = arr[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": mean, "lo": float(lo), "hi": float(hi)}


def _bootstrap_rate(binary_values: list[int], n_boot: int, seed: int = 0) -> dict[str, float]:
    return _bootstrap_mean([float(v) for v in binary_values], n_boot=n_boot, seed=seed)


def _safe_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    if len(set(xs)) == 1 or len(set(ys)) == 1:
        return None
    return float(np.corrcoef(np.asarray(xs, dtype=float), np.asarray(ys, dtype=float))[0, 1])


def _sample6_from_row(row: dict) -> GeneratedSample6:
    return GeneratedSample6(
        record_id=row["record_id"],
        prompt=row.get("prompt", ""),
        generated_text=row.get("generated_text", ""),
        generated_tokens=row.get("generated_tokens", []),
        category=row.get("category", ""),
        logit_lens_top1=None,
    )


def _sample5_from_row(row: dict) -> GeneratedSample5:
    return GeneratedSample5(
        record_id=row["record_id"],
        prompt=row.get("prompt", ""),
        generated_text=row.get("generated_text", ""),
        generated_tokens=row.get("generated_tokens", []),
        hidden_states={},
        logit_lens_entropy=[],
        top1_token_per_layer=[],
    )


def _pointwise_metric_summary(rows: list[dict], task: str, n_boot: int) -> dict[str, Any]:
    task_rows = [row for row in rows if row["task"] == task]
    if task == "s1":
        labels = [row["label"] for row in task_rows]
        total = len(labels)
        refuse_rate = sum(1 for label in labels if label == "REFUSE") / total if total else float("nan")
        return {
            "n": total,
            "refuse_rate": refuse_rate,
            "counts": dict(Counter(labels)),
        }
    values = [float(row["numeric_score"]) for row in task_rows]
    stats = _bootstrap_mean(values, n_boot=n_boot)
    return {
        "n": len(values),
        "mean": stats["mean"],
        "ci95": [stats["lo"], stats["hi"]],
    }


def _programmatic_metrics(
    *,
    records_by_id: dict[str, dict],
    sample_rows_by_condition: dict[str, list[dict]],
    forced_rows_by_condition: dict[str, list[dict]],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for condition, rows in sample_rows_by_condition.items():
        sample_outputs = [_sample6_from_row(row) for row in rows]
        records_ordered = [records_by_id[row["record_id"]] for row in rows]

        structural = score_structural_token_ratio(records_ordered, sample_outputs)

        format_rows = [row for row in rows if row.get("category") == "GOV-FORMAT"]
        format_outputs = [_sample6_from_row(row) for row in format_rows]
        format_records = [records_by_id[row["record_id"]] for row in format_rows]
        format_metric = score_format_compliance_v2(format_records, format_outputs)

        reason_rows = [row for row in rows if row.get("category") == "CONTENT-REASON"]
        reason_outputs = [_sample5_from_row(row) for row in reason_rows]
        reason_records = [records_by_id[row["record_id"]] for row in reason_rows]
        reasoning_metric = evaluate_custom_benchmark("exp3_reasoning_em", reason_records, reason_outputs)

        fact_rows = forced_rows_by_condition.get(condition, [])
        fact_outputs = [_sample6_from_row(row) for row in fact_rows]
        fact_records = [records_by_id[row["record_id"]] for row in fact_rows]
        mmlu_metric = score_mmlu_forced_choice(fact_records, fact_outputs)

        out[condition] = {
            structural.benchmark: {"value": structural.value, "n": structural.n},
            format_metric.benchmark: {"value": format_metric.value, "n": format_metric.n},
            reasoning_metric.benchmark: {"value": reasoning_metric.value, "n": reasoning_metric.n},
            mmlu_metric.benchmark: {"value": mmlu_metric.value, "n": mmlu_metric.n},
        }
    return out


def _analyze_model(run_dir: Path, model: str, n_boot: int) -> dict[str, Any]:
    records = _load_jsonl(run_dir / "prompts_shard.jsonl")
    records_by_id = {record["id"]: record for record in records}
    pointwise_rows = _load_jsonl(run_dir / "judge_pointwise_final.jsonl")
    pairwise_rows = _load_jsonl(run_dir / "judge_pairwise.jsonl") if (run_dir / "judge_pairwise.jsonl").exists() else []
    sample_rows = _load_jsonl(run_dir / "sample_outputs.jsonl")
    forced_rows = _load_jsonl(run_dir / "forced_choice_outputs.jsonl")
    judge_manifest = json.loads((run_dir / "judge_manifest.json").read_text(encoding="utf-8"))

    sample_rows_by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in sample_rows:
        sample_rows_by_condition[row["condition"]].append(row)
    forced_rows_by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in forced_rows:
        forced_rows_by_condition[row["condition"]].append(row)

    pointwise_by_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted({row["condition"] for row in pointwise_rows}):
        cond_rows = [row for row in pointwise_rows if row["condition"] == condition]
        pointwise_by_condition[condition] = {
            "g1": _pointwise_metric_summary(cond_rows, "g1", n_boot=n_boot),
            "g2": _pointwise_metric_summary(cond_rows, "g2", n_boot=n_boot),
            "s1": _pointwise_metric_summary(cond_rows, "s1", n_boot=n_boot),
            "s2": _pointwise_metric_summary(cond_rows, "s2", n_boot=n_boot),
        }

    programmatic = _programmatic_metrics(
        records_by_id=records_by_id,
        sample_rows_by_condition=sample_rows_by_condition,
        forced_rows_by_condition=forced_rows_by_condition,
    )

    pairwise_summary: dict[str, dict[str, Any]] = {}
    for comparison in sorted({row["comparison"] for row in pairwise_rows}):
        comp_rows = [row for row in pairwise_rows if row["comparison"] == comparison]
        by_task: dict[str, Any] = {}
        for task in sorted({row["task"] for row in comp_rows}):
            task_rows = [row for row in comp_rows if row["task"] == task]
            counts = Counter(row["preferred_condition"] for row in task_rows)
            total = len(task_rows)
            target = PAIRWISE_TARGET[comparison]
            by_task[task] = {
                "n": total,
                "counts": dict(counts),
                "target_condition": target,
                "target_win_rate": counts.get(target, 0) / total if total else float("nan"),
                "other_win_rate": (
                    sum(v for key, v in counts.items() if key not in {target, "TIE"}) / total if total else float("nan")
                ),
                "tie_rate": counts.get("TIE", 0) / total if total else float("nan"),
            }
        pairwise_summary[comparison] = by_task

    return {
        "model": model,
        "run_dir": str(run_dir),
        "judge_manifest": judge_manifest,
        "pointwise": pointwise_by_condition,
        "programmatic": programmatic,
        "pairwise": pairwise_summary,
    }


def _pt_improvement(model_summary: dict[str, Any], condition: str, task: str) -> float:
    baseline = model_summary["pointwise"]["A_pt_raw"][task]["mean"]
    current = model_summary["pointwise"][condition][task]["mean"]
    if task == "s2":
        return baseline - current
    return current - baseline


def _it_worsening(model_summary: dict[str, Any], condition: str, task: str) -> float:
    baseline = model_summary["pointwise"]["C_it_chat"][task]["mean"]
    current = model_summary["pointwise"][condition][task]["mean"]
    if task == "s2":
        return current - baseline
    return baseline - current


def _programmatic_delta(model_summary: dict[str, Any], baseline_condition: str, condition: str, metric_name: str) -> float:
    return (
        model_summary["programmatic"][condition][metric_name]["value"]
        - model_summary["programmatic"][baseline_condition][metric_name]["value"]
    )


def _dense_mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _collect_dense_effects(
    model_summaries: dict[str, dict],
    side: str,
    task: str,
    *,
    n_boot: int,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    windows = PT_WINDOWS if side == "pt" else IT_WINDOWS
    for condition in windows:
        values = []
        for model in DENSE_MODELS:
            if model not in model_summaries:
                continue
            values.append(
                _pt_improvement(model_summaries[model], condition, task)
                if side == "pt"
                else _it_worsening(model_summaries[model], condition, task)
            )
        stats = _bootstrap_mean(values, n_boot=n_boot, seed=17)
        out[condition] = {"mean": stats["mean"], "ci95": [stats["lo"], stats["hi"]], "values": values}
    return out


def _acceptance_checks(model_summaries: dict[str, dict]) -> dict[str, Any]:
    pt_s2_strongest = 0
    it_s2_strongest = 0
    pt_any_primary = 0
    it_any_primary = 0
    for model in DENSE_MODELS:
        if model not in model_summaries:
            continue
        pt_s2 = {condition: _pt_improvement(model_summaries[model], condition, "s2") for condition in PT_WINDOWS}
        pt_g2 = {condition: _pt_improvement(model_summaries[model], condition, "g2") for condition in PT_WINDOWS}
        it_s2 = {condition: _it_worsening(model_summaries[model], condition, "s2") for condition in IT_WINDOWS}
        it_g2 = {condition: _it_worsening(model_summaries[model], condition, "g2") for condition in IT_WINDOWS}
        if pt_s2["B_late_raw"] >= max(pt_s2.values()):
            pt_s2_strongest += 1
        if it_s2["D_late_ptswap"] >= max(it_s2.values()):
            it_s2_strongest += 1
        if pt_s2["B_late_raw"] >= max(pt_s2.values()) or pt_g2["B_late_raw"] >= max(pt_g2.values()):
            pt_any_primary += 1
        if it_s2["D_late_ptswap"] >= max(it_s2.values()) or it_g2["D_late_ptswap"] >= max(it_g2.values()):
            it_any_primary += 1
    return {
        "dense_models_available": sum(1 for model in DENSE_MODELS if model in model_summaries),
        "pt_side_s2_late_strongest_count": pt_s2_strongest,
        "it_side_s2_late_strongest_count": it_s2_strongest,
        "pt_side_any_primary_late_strongest_count": pt_any_primary,
        "it_side_any_primary_late_strongest_count": it_any_primary,
    }


def _internal_scatter_payload(model_summaries: dict[str, dict], internal_summary: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "pt_s2": {"x": [], "y": [], "labels": []},
        "pt_g2": {"x": [], "y": [], "labels": []},
        "it_s2": {"x": [], "y": [], "labels": []},
        "it_g2": {"x": [], "y": [], "labels": []},
    }
    for model in DENSE_MODELS:
        if model not in model_summaries or model not in internal_summary.get("models", {}):
            continue
        internal_model = internal_summary["models"][model]
        for condition in PT_WINDOWS:
            x = internal_model["pt_side"][condition]["regions"]["final_20pct"]["kl_to_own_final"]["delta"]
            payload["pt_s2"]["x"].append(x)
            payload["pt_s2"]["y"].append(_pt_improvement(model_summaries[model], condition, "s2"))
            payload["pt_s2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
            payload["pt_g2"]["x"].append(x)
            payload["pt_g2"]["y"].append(_pt_improvement(model_summaries[model], condition, "g2"))
            payload["pt_g2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
        for condition in IT_WINDOWS:
            x = -internal_model["it_side"][condition]["regions"]["final_20pct"]["kl_to_own_final"]["delta"]
            payload["it_s2"]["x"].append(x)
            payload["it_s2"]["y"].append(_it_worsening(model_summaries[model], condition, "s2"))
            payload["it_s2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
            payload["it_g2"]["x"].append(x)
            payload["it_g2"]["y"].append(_it_worsening(model_summaries[model], condition, "g2"))
            payload["it_g2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
    payload["correlations"] = {
        key: _safe_corr(value["x"], value["y"])
        for key, value in payload.items()
        if isinstance(value, dict) and "x" in value
    }
    return payload


def _plot_primary_bars(summary: dict[str, Any], out_path: Path) -> None:
    pt_s2 = summary["dense_pooled"]["pt"]["s2"]
    pt_g2 = summary["dense_pooled"]["pt"]["g2"]
    it_s2 = summary["dense_pooled"]["it"]["s2"]
    it_g2 = summary["dense_pooled"]["it"]["g2"]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    panels = [
        (axes[0, 0], PT_WINDOWS, pt_s2, "PT side: S2 improvement", "False-refusal reduction vs A"),
        (axes[0, 1], PT_WINDOWS, pt_g2, "PT side: G2 improvement", "Assistant-register gain vs A"),
        (axes[1, 0], IT_WINDOWS, it_s2, "IT side: S2 worsening", "False-refusal increase vs C"),
        (axes[1, 1], IT_WINDOWS, it_g2, "IT side: G2 worsening", "Assistant-register loss vs C"),
    ]
    for ax, conditions, metrics, title, ylabel in panels:
        x = np.arange(len(conditions))
        means = [metrics[condition]["mean"] for condition in conditions]
        cis = [metrics[condition]["ci95"] for condition in conditions]
        yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, cis)]).T
        ax.bar(
            x,
            means,
            yerr=yerr,
            capsize=4,
            color=[WINDOW_COLORS[condition] for condition in conditions],
            alpha=0.9,
        )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([WINDOW_DISPLAY[condition] for condition in conditions])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Exp15: Dense-5 pooled free-running behavioral effects", fontsize=15)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_pairwise(summary: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    panels = [
        ("pt_late_vs_a", "B_late_raw preferred vs A_pt_raw"),
        ("it_c_vs_dlate", "C_it_chat preferred vs D_late_ptswap"),
    ]
    for ax, (comparison, title) in zip(axes, panels):
        pairwise = summary["dense_pairwise"][comparison]
        tasks = [task for task in ["pairwise_g2", "pairwise_s2"] if task in pairwise]
        x = np.arange(len(tasks))
        target_rates = [pairwise[task]["target_win_rate"] for task in tasks]
        other_rates = [pairwise[task]["other_win_rate"] for task in tasks]
        tie_rates = [pairwise[task]["tie_rate"] for task in tasks]
        ax.bar(x, target_rates, color="#54A24B", label="Target preferred")
        ax.bar(x, tie_rates, bottom=target_rates, color="#BDBDBD", label="Tie")
        ax.bar(x, other_rates, bottom=np.asarray(target_rates) + np.asarray(tie_rates), color="#E45756", label="Other preferred")
        ax.set_xticks(x)
        ax.set_xticklabels(["G2" if task == "pairwise_g2" else "S2" for task in tasks])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_internal_scatter(scatter: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    panels = [
        ("pt_s2", "PT: internal delay vs S2 improvement", "Final-20% KL delta (B - A)", "S2 improvement"),
        ("pt_g2", "PT: internal delay vs G2 improvement", "Final-20% KL delta (B - A)", "G2 improvement"),
        ("it_s2", "IT: internal collapse vs S2 worsening", "Final-20% KL collapse (C - D)", "S2 worsening"),
        ("it_g2", "IT: internal collapse vs G2 worsening", "Final-20% KL collapse (C - D)", "G2 worsening"),
    ]
    color_map = {"Early": "#72B7B2", "Mid": "#F58518", "Late": "#E45756"}
    for ax, (key, title, xlabel, ylabel) in zip(axes.flat, panels):
        xs = scatter[key]["x"]
        ys = scatter[key]["y"]
        labels = scatter[key]["labels"]
        for x, y, label in zip(xs, ys, labels):
            color = color_map["Late" if "Late" in label else "Mid" if "Mid" in label else "Early"]
            ax.scatter(x, y, s=55, color=color, alpha=0.9)
        corr = scatter["correlations"].get(key)
        corr_text = f"r={corr:.2f}" if corr is not None else "r=n/a"
        ax.set_title(f"{title}\n{corr_text}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_per_model(summary: dict[str, Any], out_path: Path) -> None:
    available_models = [model for model in MODEL_ORDER if model in summary["models"]]
    fig, axes = plt.subplots(len(available_models), 2, figsize=(12, 3.1 * len(available_models)), constrained_layout=True)
    if len(available_models) == 1:
        axes = np.asarray([axes])
    for row_idx, model in enumerate(available_models):
        model_summary = summary["models"][model]
        ax_pt = axes[row_idx, 0]
        ax_it = axes[row_idx, 1]
        x_pt = np.arange(len(PT_WINDOWS))
        x_it = np.arange(len(IT_WINDOWS))
        pt_s2 = [_pt_improvement(model_summary, condition, "s2") for condition in PT_WINDOWS]
        pt_g2 = [_pt_improvement(model_summary, condition, "g2") for condition in PT_WINDOWS]
        it_s2 = [_it_worsening(model_summary, condition, "s2") for condition in IT_WINDOWS]
        it_g2 = [_it_worsening(model_summary, condition, "g2") for condition in IT_WINDOWS]
        ax_pt.plot(x_pt, pt_s2, marker="o", color="#E45756", label="S2")
        ax_pt.plot(x_pt, pt_g2, marker="s", color="#4C78A8", label="G2")
        ax_it.plot(x_it, it_s2, marker="o", color="#E45756", label="S2")
        ax_it.plot(x_it, it_g2, marker="s", color="#4C78A8", label="G2")
        for ax, xvals, conditions, title in (
            (ax_pt, x_pt, PT_WINDOWS, f"{MODEL_DISPLAY[model]} PT side"),
            (ax_it, x_it, IT_WINDOWS, f"{MODEL_DISPLAY[model]} IT side"),
        ):
            ax.axhline(0.0, color="black", linewidth=1)
            ax.set_xticks(xvals)
            ax.set_xticklabels([WINDOW_DISPLAY[condition] for condition in conditions])
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.25)
        if row_idx == 0:
            ax_pt.legend(frameon=False)
            ax_it.legend(frameon=False)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model_summaries: dict[str, dict] = {}
    for model in MODEL_ORDER:
        run_dir = _run_dir(args.run_root, args.run_prefix, model)
        if not run_dir.exists():
            continue
        model_summaries[model] = _analyze_model(run_dir, model, n_boot=args.bootstrap_samples)

    if not model_summaries:
        raise FileNotFoundError("No exp15 model run directories found")

    internal_summary = json.loads(args.internal_summary.read_text(encoding="utf-8"))
    dense_pooled = {
        "pt": {
            "s2": _collect_dense_effects(model_summaries, side="pt", task="s2", n_boot=args.bootstrap_samples),
            "g2": _collect_dense_effects(model_summaries, side="pt", task="g2", n_boot=args.bootstrap_samples),
        },
        "it": {
            "s2": _collect_dense_effects(model_summaries, side="it", task="s2", n_boot=args.bootstrap_samples),
            "g2": _collect_dense_effects(model_summaries, side="it", task="g2", n_boot=args.bootstrap_samples),
        },
    }

    dense_pairwise: dict[str, dict[str, Any]] = {}
    for comparison in PAIRWISE_TARGET:
        dense_pairwise[comparison] = {}
        available_tasks = set()
        for model in DENSE_MODELS:
            if model not in model_summaries:
                continue
            available_tasks.update(model_summaries[model]["pairwise"].get(comparison, {}).keys())
        for task in sorted(available_tasks):
            target = PAIRWISE_TARGET[comparison]
            counts = Counter()
            total = 0
            for model in DENSE_MODELS:
                if model not in model_summaries:
                    continue
                entry = model_summaries[model]["pairwise"].get(comparison, {}).get(task)
                if not entry:
                    continue
                counts.update(entry["counts"])
                total += entry["n"]
            dense_pairwise[comparison][task] = {
                "n": total,
                "counts": dict(counts),
                "target_condition": target,
                "target_win_rate": counts.get(target, 0) / total if total else float("nan"),
                "other_win_rate": (
                    sum(v for key, v in counts.items() if key not in {target, "TIE"}) / total if total else float("nan")
                ),
                "tie_rate": counts.get("TIE", 0) / total if total else float("nan"),
            }

    dense_programmatic: dict[str, dict[str, dict[str, float]]] = {"pt": {}, "it": {}}
    for metric_name in ("structural_token_ratio", "format_compliance_v2", "mmlu_forced_choice", "exp3_reasoning_em"):
        dense_programmatic["pt"][metric_name] = {}
        for condition in PT_WINDOWS:
            values = [
                _programmatic_delta(model_summaries[model], "A_pt_raw", condition, metric_name)
                for model in DENSE_MODELS
                if model in model_summaries
            ]
            dense_programmatic["pt"][metric_name][condition] = _dense_mean(values)
        dense_programmatic["it"][metric_name] = {}
        for condition in IT_WINDOWS:
            values = [
                _programmatic_delta(model_summaries[model], "C_it_chat", condition, metric_name)
                for model in DENSE_MODELS
                if model in model_summaries
            ]
            dense_programmatic["it"][metric_name][condition] = _dense_mean(values)

    deepseek_summary = model_summaries.get("deepseek_v2_lite")
    acceptance = _acceptance_checks(model_summaries)
    scatter_payload = _internal_scatter_payload(model_summaries, internal_summary)

    summary = {
        "run_prefix": args.run_prefix,
        "run_root": str(args.run_root),
        "models": model_summaries,
        "dense_pooled": dense_pooled,
        "dense_pairwise": dense_pairwise,
        "dense_programmatic_deltas": dense_programmatic,
        "acceptance_checks": acceptance,
        "internal_scatter": scatter_payload,
        "deepseek_separate": deepseek_summary,
        "dense_pooled_rankings": {
            "pt_s2": [condition for condition, _ in sorted(dense_pooled["pt"]["s2"].items(), key=lambda item: item[1]["mean"], reverse=True)],
            "pt_g2": [condition for condition, _ in sorted(dense_pooled["pt"]["g2"].items(), key=lambda item: item[1]["mean"], reverse=True)],
            "it_s2": [condition for condition, _ in sorted(dense_pooled["it"]["s2"].items(), key=lambda item: item[1]["mean"], reverse=True)],
            "it_g2": [condition for condition, _ in sorted(dense_pooled["it"]["g2"].items(), key=lambda item: item[1]["mean"], reverse=True)],
        },
    }
    (args.out_dir / "exp15_behavior_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _plot_primary_bars(summary, args.out_dir / "exp15_primary_bars.png")
    _plot_pairwise(summary, args.out_dir / "exp15_pairwise.png")
    _plot_internal_scatter(scatter_payload, args.out_dir / "exp15_internal_scatter.png")
    _plot_per_model(summary, args.out_dir / "exp15_per_model_deltas.png")
    print(f"Wrote Exp15 analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
