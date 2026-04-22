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
from matplotlib.lines import Line2D

from src.poc.exp05_corrective_direction_ablation_cartography.benchmarks.custom import evaluate_custom_benchmark
from src.poc.exp05_corrective_direction_ablation_cartography.runtime import GeneratedSample as GeneratedSample5
from src.poc.exp06_corrective_direction_steering.benchmarks.governance import score_structural_token_ratio
from src.poc.exp06_corrective_direction_steering.benchmarks.governance_v2 import score_format_compliance_v2, score_mmlu_forced_choice
from src.poc.exp06_corrective_direction_steering.runtime import GeneratedSample6


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
MODEL_SHORT = {
    "gemma3_4b": "Ge",
    "qwen3_4b": "Qw",
    "llama31_8b": "Ll",
    "mistral_7b": "Mi",
    "olmo2_7b": "Ol",
    "deepseek_v2_lite": "Ds",
}
ASSISTANT_BUCKETS = [
    ("conv_source", "Conversational source"),
    ("gov_register", "GOV-REGISTER"),
    ("gov_conv_extra", "GOV-CONV"),
]
PROGRAMMATIC_METRIC_ORDER = [
    "format_compliance_v2",
    "structural_token_ratio",
    "exp3_reasoning_em",
    "mmlu_forced_choice",
]
PROGRAMMATIC_METRIC_DISPLAY = {
    "format_compliance_v2": "Format",
    "structural_token_ratio": "Structural",
    "exp3_reasoning_em": "Reasoning EM",
    "mmlu_forced_choice": "Forced-choice",
}
BASELINE_DISPLAY = {
    "A_pt_raw": "A (PT)",
    "C_it_chat": "C (IT)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze exp15 behavioral causality runs.")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("results/exp15_symmetric_behavioral_causality/data"),
    )
    parser.add_argument("--run-prefix", default="exp15_eval_core_600")
    parser.add_argument(
        "--internal-summary",
        type=Path,
        default=Path(
            "results/exp14_symmetric_matched_prefix_causality/"
            "exp13exp14_full_20260416/exp13_full_summary.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp15_symmetric_behavioral_causality/plots/exp15_eval_core_600"),
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


def _token_count(row: dict) -> int:
    tokens = row.get("generated_tokens", [])
    if isinstance(tokens, list):
        return len(tokens)
    if isinstance(tokens, int):
        return tokens
    try:
        return int(tokens)
    except Exception:
        return 0


def _assistant_bucket_keys(record: dict) -> set[str]:
    buckets: set[str] = set()
    if record.get("exp15_conversation_source"):
        buckets.add("conv_source")
    if record.get("category") == "GOV-REGISTER":
        buckets.add("gov_register")
    if record.get("category") == "GOV-CONV":
        buckets.add("gov_conv_extra")
    return buckets


def _assistant_g2_bucket_summary(pointwise_rows: list[dict], records_by_id: dict[str, dict], n_boot: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    conditions = sorted({row["condition"] for row in pointwise_rows})
    for condition in conditions:
        cond_rows = [row for row in pointwise_rows if row["condition"] == condition and row["task"] == "g2"]
        out[condition] = {}
        for bucket_key, _ in ASSISTANT_BUCKETS:
            values = [
                float(row["numeric_score"])
                for row in cond_rows
                if bucket_key in _assistant_bucket_keys(records_by_id[row["record_id"]])
            ]
            stats = _bootstrap_mean(values, n_boot=n_boot, seed=23)
            out[condition][bucket_key] = {
                "n": len(values),
                "mean": stats["mean"],
                "ci95": [stats["lo"], stats["hi"]],
            }
    return out


def _generation_diagnostics(
    *,
    records_by_id: dict[str, dict],
    sample_rows_by_condition: dict[str, list[dict]],
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for condition, rows in sample_rows_by_condition.items():
        out[condition] = {}
        subsets = {
            "all": rows,
            "assistant_facing": [row for row in rows if records_by_id[row["record_id"]].get("exp15_assistant_facing")],
        }
        for subset_name, subset_rows in subsets.items():
            token_counts = [_token_count(row) for row in subset_rows]
            caps = [1 if count >= 512 else 0 for count in token_counts]
            empty_count = sum(1 for row in subset_rows if not row.get("generated_text", "").strip())
            n = len(subset_rows)
            out[condition][subset_name] = {
                "n": n,
                "mean_tokens": float(sum(token_counts) / n) if n else float("nan"),
                "cap_rate": float(sum(caps) / n) if n else float("nan"),
                "empty_count": empty_count,
                "empty_rate": float(empty_count / n) if n else float("nan"),
            }
    return out


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
    assistant_g2_buckets = _assistant_g2_bucket_summary(pointwise_rows, records_by_id, n_boot=n_boot)
    generation_diagnostics = _generation_diagnostics(
        records_by_id=records_by_id,
        sample_rows_by_condition=sample_rows_by_condition,
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
        "assistant_g2_buckets": assistant_g2_buckets,
        "generation_diagnostics": generation_diagnostics,
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


def _collect_dense_programmatic_deltas(
    model_summaries: dict[str, dict],
    *,
    n_boot: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {"pt": {}, "it": {}}
    for metric_name in PROGRAMMATIC_METRIC_ORDER:
        out["pt"][metric_name] = {}
        for condition in PT_WINDOWS:
            values = [
                _programmatic_delta(model_summaries[model], "A_pt_raw", condition, metric_name)
                for model in DENSE_MODELS
                if model in model_summaries
            ]
            stats = _bootstrap_mean(values, n_boot=n_boot, seed=31)
            out["pt"][metric_name][condition] = {
                "mean": stats["mean"],
                "ci95": [stats["lo"], stats["hi"]],
                "values": values,
            }
        out["it"][metric_name] = {}
        for condition in IT_WINDOWS:
            values = [
                _programmatic_delta(model_summaries[model], "C_it_chat", condition, metric_name)
                for model in DENSE_MODELS
                if model in model_summaries
            ]
            stats = _bootstrap_mean(values, n_boot=n_boot, seed=37)
            out["it"][metric_name][condition] = {
                "mean": stats["mean"],
                "ci95": [stats["lo"], stats["hi"]],
                "values": values,
            }
    return out


def _collect_dense_bucket_g2_deltas(
    model_summaries: dict[str, dict],
    *,
    n_boot: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {"pt": {}, "it": {}}
    for bucket_key, _ in ASSISTANT_BUCKETS:
        out["pt"][bucket_key] = {}
        for condition in PT_WINDOWS:
            values = []
            for model in DENSE_MODELS:
                if model not in model_summaries:
                    continue
                baseline = model_summaries[model]["assistant_g2_buckets"]["A_pt_raw"][bucket_key]["mean"]
                current = model_summaries[model]["assistant_g2_buckets"][condition][bucket_key]["mean"]
                values.append(current - baseline)
            stats = _bootstrap_mean(values, n_boot=n_boot, seed=41)
            out["pt"][bucket_key][condition] = {
                "mean": stats["mean"],
                "ci95": [stats["lo"], stats["hi"]],
                "values": values,
            }
        out["it"][bucket_key] = {}
        for condition in IT_WINDOWS:
            values = []
            for model in DENSE_MODELS:
                if model not in model_summaries:
                    continue
                baseline = model_summaries[model]["assistant_g2_buckets"]["C_it_chat"][bucket_key]["mean"]
                current = model_summaries[model]["assistant_g2_buckets"][condition][bucket_key]["mean"]
                values.append(baseline - current)
            stats = _bootstrap_mean(values, n_boot=n_boot, seed=43)
            out["it"][bucket_key][condition] = {
                "mean": stats["mean"],
                "ci95": [stats["lo"], stats["hi"]],
                "values": values,
            }
    return out


def _collect_dense_generation_profiles(
    model_summaries: dict[str, dict],
    *,
    n_boot: int,
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {"pt": {}, "it": {}}
    for side, conditions in (
        ("pt", ["A_pt_raw", *PT_WINDOWS]),
        ("it", ["C_it_chat", *IT_WINDOWS]),
    ):
        out[side] = {"mean_tokens": {}, "cap_rate": {}, "empty_rate": {}}
        for metric_name in ("mean_tokens", "cap_rate", "empty_rate"):
            for condition in conditions:
                values = [
                    model_summaries[model]["generation_diagnostics"][condition]["assistant_facing"][metric_name]
                    for model in DENSE_MODELS
                    if model in model_summaries
                ]
                stats = _bootstrap_mean(values, n_boot=n_boot, seed=47)
                out[side][metric_name][condition] = {
                    "mean": stats["mean"],
                    "ci95": [stats["lo"], stats["hi"]],
                    "values": values,
                }
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
        "pt_s2": {"x": [], "y": [], "labels": [], "model_keys": [], "window_keys": []},
        "pt_g2": {"x": [], "y": [], "labels": [], "model_keys": [], "window_keys": []},
        "it_s2": {"x": [], "y": [], "labels": [], "model_keys": [], "window_keys": []},
        "it_g2": {"x": [], "y": [], "labels": [], "model_keys": [], "window_keys": []},
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
            payload["pt_s2"]["model_keys"].append(model)
            payload["pt_s2"]["window_keys"].append(WINDOW_DISPLAY[condition])
            payload["pt_g2"]["x"].append(x)
            payload["pt_g2"]["y"].append(_pt_improvement(model_summaries[model], condition, "g2"))
            payload["pt_g2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
            payload["pt_g2"]["model_keys"].append(model)
            payload["pt_g2"]["window_keys"].append(WINDOW_DISPLAY[condition])
        for condition in IT_WINDOWS:
            x = -internal_model["it_side"][condition]["regions"]["final_20pct"]["kl_to_own_final"]["delta"]
            payload["it_s2"]["x"].append(x)
            payload["it_s2"]["y"].append(_it_worsening(model_summaries[model], condition, "s2"))
            payload["it_s2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
            payload["it_s2"]["model_keys"].append(model)
            payload["it_s2"]["window_keys"].append(WINDOW_DISPLAY[condition])
            payload["it_g2"]["x"].append(x)
            payload["it_g2"]["y"].append(_it_worsening(model_summaries[model], condition, "g2"))
            payload["it_g2"]["labels"].append(f"{MODEL_DISPLAY[model]} {WINDOW_DISPLAY[condition]}")
            payload["it_g2"]["model_keys"].append(model)
            payload["it_g2"]["window_keys"].append(WINDOW_DISPLAY[condition])
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
    ranking_map = {
        "PT side: S2 improvement": summary["dense_pooled_rankings"]["pt_s2"],
        "PT side: G2 improvement": summary["dense_pooled_rankings"]["pt_g2"],
        "IT side: S2 worsening": summary["dense_pooled_rankings"]["it_s2"],
        "IT side: G2 worsening": summary["dense_pooled_rankings"]["it_g2"],
    }

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
        ordering = " > ".join(WINDOW_DISPLAY[condition] for condition in ranking_map[title])
        ax.text(
            0.02,
            0.97,
            f"Ordering: {ordering}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        y_range = max((max(means) - min(means)), 0.1)
        for xi, mean in enumerate(means):
            offset = 0.04 * y_range if mean >= 0 else -0.06 * y_range
            va = "bottom" if mean >= 0 else "top"
            ax.text(xi, mean + offset, f"{mean:+.2f}", ha="center", va=va, fontsize=9, fontweight="bold")
    fig.suptitle(
        "Exp15: Dense-5 pooled free-running behavioral effects\n"
        "Positive bars mean expected-direction movement: PT improves vs A_pt_raw, IT degrades vs C_it_chat.",
        fontsize=15,
    )
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
        ax.set_xticklabels(
            [
                f"{'G2' if task == 'pairwise_g2' else 'S2'}\n(n={pairwise[task]['n']})"
                for task in tasks
            ]
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for xi, (target_rate, tie_rate, other_rate) in enumerate(zip(target_rates, tie_rates, other_rates)):
            ax.text(xi, target_rate / 2, f"{target_rate:.0%}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            if tie_rate > 0.04:
                ax.text(xi, target_rate + tie_rate / 2, f"{tie_rate:.0%}", ha="center", va="center", fontsize=9)
            ax.text(
                xi,
                target_rate + tie_rate + other_rate / 2,
                f"{other_rate:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
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
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[window], markersize=8, label=window)
        for window in ("Early", "Mid", "Late")
    ]
    for ax, (key, title, xlabel, ylabel) in zip(axes.flat, panels):
        xs = scatter[key]["x"]
        ys = scatter[key]["y"]
        model_keys = scatter[key]["model_keys"]
        window_keys = scatter[key]["window_keys"]
        for x, y, model_key, window_key in zip(xs, ys, model_keys, window_keys):
            color = color_map[window_key]
            ax.scatter(x, y, s=55, color=color, alpha=0.9)
            ax.annotate(
                f"{MODEL_SHORT[model_key]}-{window_key[0]}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=7,
                color="#333333",
            )
        corr = scatter["correlations"].get(key)
        corr_text = f"r={corr:.2f}" if corr is not None else "r=n/a"
        ax.set_title(f"{title}\n{corr_text}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize=8)
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


def _plot_bucket_g2_deltas(summary: dict[str, Any], out_path: Path) -> None:
    bucket_deltas = summary["dense_g2_bucket_deltas"]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    panels = [
        ("pt", PT_WINDOWS, "PT side: assistant-register gain by prompt bucket", "G2 gain vs A_pt_raw"),
        ("it", IT_WINDOWS, "IT side: assistant-register loss by prompt bucket", "G2 loss vs C_it_chat"),
    ]
    width = 0.23
    x = np.arange(len(ASSISTANT_BUCKETS))
    for ax, (side, conditions, title, ylabel) in zip(axes, panels):
        for idx, condition in enumerate(conditions):
            means = [bucket_deltas[side][bucket_key][condition]["mean"] for bucket_key, _ in ASSISTANT_BUCKETS]
            cis = [bucket_deltas[side][bucket_key][condition]["ci95"] for bucket_key, _ in ASSISTANT_BUCKETS]
            yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, cis)]).T
            xpos = x + (idx - 1) * width
            ax.bar(
                xpos,
                means,
                width=width,
                color=WINDOW_COLORS[condition],
                yerr=yerr,
                capsize=3,
                label=WINDOW_DISPLAY[condition],
                alpha=0.92,
            )
            for xp, mean in zip(xpos, means):
                ax.text(xp, mean + 0.015, f"{mean:+.2f}", ha="center", va="bottom", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in ASSISTANT_BUCKETS])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=3, bbox_to_anchor=(0.02, 1.12), loc="upper left")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_generation_diagnostics(summary: dict[str, Any], out_path: Path) -> None:
    profiles = summary["dense_generation_profiles"]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)
    panels = [
        ("pt", "mean_tokens", ["A_pt_raw", *PT_WINDOWS], axes[0, 0], "PT side: assistant-facing mean output length", "Mean generated tokens"),
        ("pt", "cap_rate", ["A_pt_raw", *PT_WINDOWS], axes[0, 1], "PT side: 512-token cap rate", "Cap-hit rate"),
        ("it", "mean_tokens", ["C_it_chat", *IT_WINDOWS], axes[1, 0], "IT side: assistant-facing mean output length", "Mean generated tokens"),
        ("it", "cap_rate", ["C_it_chat", *IT_WINDOWS], axes[1, 1], "IT side: 512-token cap rate", "Cap-hit rate"),
    ]
    for side, metric_name, conditions, ax, title, ylabel in panels:
        values = [profiles[side][metric_name][condition]["mean"] for condition in conditions]
        cis = [profiles[side][metric_name][condition]["ci95"] for condition in conditions]
        x = np.arange(len(conditions))
        colors = ["#4C78A8"] + [WINDOW_COLORS[condition] for condition in conditions[1:]]
        labels = [BASELINE_DISPLAY.get(conditions[0], conditions[0])] + [WINDOW_DISPLAY[condition] for condition in conditions[1:]]
        yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(values, cis)]).T
        ax.bar(x, values, yerr=yerr, capsize=4, color=colors, alpha=0.92)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
        for xi, value in enumerate(values):
            text = f"{value:.0f}" if metric_name == "mean_tokens" else f"{value:.0%}"
            ax.text(xi, value + (8 if metric_name == "mean_tokens" else 0.02), text, ha="center", va="bottom", fontsize=9)
    fig.suptitle(
        "Exp15 generation diagnostics\n"
        "Mid often shortens outputs and lowers cap saturation, while late more often remains near the 512-token ceiling.",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_programmatic_deltas(summary: dict[str, Any], out_path: Path) -> None:
    dense_programmatic = summary["dense_programmatic_deltas"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    panels = [
        ("pt", PT_WINDOWS, "PT side: programmatic changes vs A_pt_raw"),
        ("it", IT_WINDOWS, "IT side: programmatic losses vs C_it_chat"),
    ]
    width = 0.23
    x = np.arange(len(PROGRAMMATIC_METRIC_ORDER))
    for ax, (side, conditions, title) in zip(axes, panels):
        for idx, condition in enumerate(conditions):
            means = [dense_programmatic[side][metric_name][condition]["mean"] for metric_name in PROGRAMMATIC_METRIC_ORDER]
            cis = [dense_programmatic[side][metric_name][condition]["ci95"] for metric_name in PROGRAMMATIC_METRIC_ORDER]
            yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, cis)]).T
            xpos = x + (idx - 1) * width
            ax.bar(
                xpos,
                means,
                width=width,
                color=WINDOW_COLORS[condition],
                yerr=yerr,
                capsize=3,
                label=WINDOW_DISPLAY[condition],
                alpha=0.92,
            )
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([PROGRAMMATIC_METRIC_DISPLAY[metric] for metric in PROGRAMMATIC_METRIC_ORDER])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Delta")
    axes[0].legend(frameon=False, ncol=3, bbox_to_anchor=(0.02, 1.12), loc="upper left")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_behavior_main(summary: dict[str, Any], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9), constrained_layout=True)

    # Panel A: IT-side depth ranking only, where the late-centered necessity story is strongest.
    ax = axes[0]
    tasks = [("g2", "G2"), ("s2", "S2")]
    x = np.arange(len(tasks))
    width = 0.23
    for idx, condition in enumerate(IT_WINDOWS):
        means = [summary["dense_pooled"]["it"][task][condition]["mean"] for task, _ in tasks]
        cis = [summary["dense_pooled"]["it"][task][condition]["ci95"] for task, _ in tasks]
        yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, cis)]).T
        xpos = x + (idx - 1) * width
        ax.bar(
            xpos,
            means,
            width=width,
            color=WINDOW_COLORS[condition],
            yerr=yerr,
            capsize=3,
            label=WINDOW_DISPLAY[condition],
            alpha=0.92,
        )
        for xp, mean in zip(xpos, means):
            ax.text(xp, mean + 0.018, f"{mean:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in tasks])
    ax.set_ylabel("Behavioral worsening vs C_it_chat")
    ax.set_title("IT-side necessity: late swap is strongest")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    # Panels B/C: pairwise late-vs-baseline comparisons, which are cleaner than full pointwise PT-side bars.
    pairwise_panels = [
        ("pt_late_vs_a", "PT late graft vs PT baseline"),
        ("it_c_vs_dlate", "IT baseline vs late PT swap"),
    ]
    for ax, (comparison, title) in zip(axes[1:], pairwise_panels):
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
        ax.set_xticklabels(
            [
                f"{'G2' if task == 'pairwise_g2' else 'S2'}\n(n={pairwise[task]['n']})"
                for task in tasks
            ]
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for xi, (target_rate, tie_rate, other_rate) in enumerate(zip(target_rates, tie_rates, other_rates)):
            ax.text(xi, target_rate / 2, f"{target_rate:.0%}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")
            if tie_rate > 0.04:
                ax.text(xi, target_rate + tie_rate / 2, f"{tie_rate:.0%}", ha="center", va="center", fontsize=9)
            ax.text(
                xi,
                target_rate + tie_rate + other_rate / 2,
                f"{other_rate:.0%}",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3, frameon=False)
    fig.suptitle(
        "Exp15 paper-facing behavioral summary\n"
        "Late is the clearest behavioral necessity window in IT, while late grafting into PT shows partial sufficiency under blind pairwise judging.",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_paper_it_targeting(summary: dict[str, Any], out_path: Path) -> None:
    bucket_deltas = summary["dense_g2_bucket_deltas"]["it"]
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8), constrained_layout=True)
    width = 0.23
    x = np.arange(len(ASSISTANT_BUCKETS))
    for idx, condition in enumerate(IT_WINDOWS):
        means = [bucket_deltas[bucket_key][condition]["mean"] for bucket_key, _ in ASSISTANT_BUCKETS]
        cis = [bucket_deltas[bucket_key][condition]["ci95"] for bucket_key, _ in ASSISTANT_BUCKETS]
        yerr = np.array([[mean - lo, hi - mean] for mean, (lo, hi) in zip(means, cis)]).T
        xpos = x + (idx - 1) * width
        ax.bar(
            xpos,
            means,
            width=width,
            color=WINDOW_COLORS[condition],
            yerr=yerr,
            capsize=3,
            label=WINDOW_DISPLAY[condition],
            alpha=0.92,
        )
        for xp, mean in zip(xpos, means):
            ax.text(xp, mean + 0.02, f"{mean:+.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in ASSISTANT_BUCKETS])
    ax.set_ylabel("G2 loss vs C_it_chat")
    ax.set_title("IT-side assistant-register degradation is targeted\nlate PT swap is strongest across assistant-facing prompt buckets")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_plot_notes(summary: dict[str, Any], out_dir: Path) -> None:
    acceptance = summary["acceptance_checks"]
    pt_pair = summary["dense_pairwise"]["pt_late_vs_a"]
    it_pair = summary["dense_pairwise"]["it_c_vs_dlate"]
    notes = f"""# Exp15 Plot Notes

## What This Folder Contains

- `exp15_primary_bars.png`: Dense-5 pooled pointwise effects on the two main behavioral endpoints.
- `exp15_pairwise.png`: Dense-5 pooled pairwise preferences for late-vs-baseline comparisons.
- `exp15_per_model_deltas.png`: Model-by-model heterogeneity across PT-side sufficiency and IT-side necessity.
- `exp15_internal_scatter.png`: Cross-condition link between matched-prefix internal deltas and free-running behavioral deltas.
- `exp15_g2_bucket_deltas.png`: Assistant-register deltas split by prompt subtype.
- `exp15_generation_diagnostics.png`: Assistant-facing output length and 512-token cap-rate diagnostics.
- `exp15_programmatic_deltas.png`: Programmatic cross-checks for structure/format/content.

## High-Level Read

- The cleanest behavioral result is on the IT-side necessity test: late PT-swaps hurt behavior most strongly on pooled `S2` and `G2`.
- PT-side late grafts are behaviorally real but not pointwise-maximal: `B_late_raw` beats `A_pt_raw` pairwise, yet `B_mid_raw` is strongest on pooled pointwise PT-side `S2` and `G2`.
- A likely reason is visible in `exp15_generation_diagnostics.png`: PT mid windows often shorten outputs and reduce cap saturation, while PT late windows frequently remain near the `512`-token cap.

## Plot-By-Plot Guide

### `exp15_primary_bars.png`

- Positive bars always mean movement in the expected causal direction.
- PT side: improvement relative to `A_pt_raw`.
- IT side: degradation relative to `C_it_chat`.
- Main message: late is strongest on the IT side, but mid is strongest on the PT-side pointwise bars.

### `exp15_pairwise.png`

- These plots compare only the late branch to its baseline under blind pairwise judging.
- PT side pooled late-vs-A preference:
  - `G2`: target preferred `{pt_pair['pairwise_g2']['target_win_rate']:.1%}`, other preferred `{pt_pair['pairwise_g2']['other_win_rate']:.1%}`, tie `{pt_pair['pairwise_g2']['tie_rate']:.1%}`
  - `S2`: target preferred `{pt_pair['pairwise_s2']['target_win_rate']:.1%}`, other preferred `{pt_pair['pairwise_s2']['other_win_rate']:.1%}`, tie `{pt_pair['pairwise_s2']['tie_rate']:.1%}`
- IT side pooled C-vs-Dlate preference:
  - `G2`: target preferred `{it_pair['pairwise_g2']['target_win_rate']:.1%}`
  - `S2`: target preferred `{it_pair['pairwise_s2']['target_win_rate']:.1%}`

### `exp15_g2_bucket_deltas.png`

- This explains where assistant-register effects come from.
- PT mid gains are concentrated especially on conversational-source prompts and extra conversational governance prompts.
- IT late losses are broader and remain strong on conversational and register-focused prompts.

### `exp15_generation_diagnostics.png`

- This is the main cautionary diagnostic for interpreting PT-side free-running sufficiency.
- Under capped `512`-token generation, a branch can look more assistant-like partly because it reaches a shorter, cleaner stopping point.
- The PT mid branch often reduces mean length and cap rate more than the PT late branch.

### `exp15_programmatic_deltas.png`

- These are non-judge cross-checks.
- They help separate structure/format changes from broad content collapse.
- In this run, PT mid is also stronger than PT late on several format-like programmatic views, while IT late remains the strongest broad necessity window behaviorally.

### `exp15_internal_scatter.png`

- Each point is a model-window condition.
- Colors are depth windows; labels use model abbreviations plus `E/M/L`.
- The IT-side correlations are materially cleaner than the PT-side ones:
  - PT `S2`: `r={summary['internal_scatter']['correlations']['pt_s2']:.2f}`
  - PT `G2`: `r={summary['internal_scatter']['correlations']['pt_g2']:.2f}`
  - IT `S2`: `r={summary['internal_scatter']['correlations']['it_s2']:.2f}`
  - IT `G2`: `r={summary['internal_scatter']['correlations']['it_g2']:.2f}`

## Acceptance Snapshot

- Dense models available: `{acceptance['dense_models_available']}`
- PT-side `S2` late-strongest count: `{acceptance['pt_side_s2_late_strongest_count']}/5`
- IT-side `S2` late-strongest count: `{acceptance['it_side_s2_late_strongest_count']}/5`
- PT-side any-primary late-strongest count: `{acceptance['pt_side_any_primary_late_strongest_count']}/5`
- IT-side any-primary late-strongest count: `{acceptance['it_side_any_primary_late_strongest_count']}/5`

## Recommended Paper Framing

- Use `exp15` primarily as a late-centered behavioral **necessity** result.
- Treat the PT-side behavioral sufficiency result as **real but partial**, with clear late pairwise gains but stronger mid pointwise gains under capped free-running decoding.
- Keep `exp11/13/14` as the main localization backbone, and use `exp15` to show that the same circuit matters under natural decoding rather than to claim a perfectly symmetric late-only behavioral story.
"""
    (out_dir / "README.md").write_text(notes, encoding="utf-8")


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

    dense_programmatic = _collect_dense_programmatic_deltas(model_summaries, n_boot=args.bootstrap_samples)
    dense_g2_bucket_deltas = _collect_dense_bucket_g2_deltas(model_summaries, n_boot=args.bootstrap_samples)
    dense_generation_profiles = _collect_dense_generation_profiles(model_summaries, n_boot=args.bootstrap_samples)

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
        "dense_g2_bucket_deltas": dense_g2_bucket_deltas,
        "dense_generation_profiles": dense_generation_profiles,
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
    _plot_bucket_g2_deltas(summary, args.out_dir / "exp15_g2_bucket_deltas.png")
    _plot_generation_diagnostics(summary, args.out_dir / "exp15_generation_diagnostics.png")
    _plot_programmatic_deltas(summary, args.out_dir / "exp15_programmatic_deltas.png")
    _plot_paper_behavior_main(summary, args.out_dir / "exp15_paper_behavior_main.png")
    _plot_paper_it_targeting(summary, args.out_dir / "exp15_paper_it_targeting.png")
    _write_plot_notes(summary, args.out_dir)
    print(f"Wrote Exp15 analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
