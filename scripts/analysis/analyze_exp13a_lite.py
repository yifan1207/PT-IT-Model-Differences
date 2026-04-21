#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import classify_generated_tokens_by_word


RAW_CATEGORIES = [
    "CONTENT",
    "FUNCTION",
    "DISCOURSE",
    "STRUCTURAL",
    "PUNCTUATION",
    "OTHER",
]
COLLAPSED_CATEGORIES = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]
DENSE5_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
DEPTH_WINDOWS = ["B_early_raw", "B_mid_raw", "B_late_raw"]

MATCHED_SOURCE_ROOT = Path("results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_400rand_v11_teacherforced")
DEPTH_SOURCE_ROOT = Path("results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_600rand_v11_depthablation")
DEPTH_FLAT_SOURCE_ROOT = Path("results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_600rand_v11_depthablation_flat")
EXP12_JUDGE_SUMMARY = Path("results/exp12_free_running_abc_graft/judge_summary_exp12_eval_v1_20260413_v3.json")


def _default_out_dir() -> Path:
    return Path(
        "results/exp13_late_stage_token_support_analysis"
    ) / f"exp13A_lite_{Path.cwd().stat().st_mtime_ns // 1_000_000_000}"


def _make_nested_counter() -> dict[str, int]:
    return {k: 0 for k in RAW_CATEGORIES}


def _make_collapsed_counter() -> dict[str, int]:
    return {k: 0 for k in COLLAPSED_CATEGORIES}


def _make_transition_matrix(categories: list[str]) -> dict[str, dict[str, int]]:
    return {src: {dst: 0 for dst in categories} for src in categories}


def _collapsed_category(raw: str) -> str:
    if raw in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}:
        return "FORMAT"
    if raw == "CONTENT":
        return "CONTENT"
    return "FUNCTION_OTHER"


def _increment(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[key] = counter.get(key, 0) + amount


def _safe_mean(total: float, count: int) -> float | None:
    if count <= 0:
        return None
    return total / count


def _fraction(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


def _layer_phase(layer_idx: int, n_layers: int, late_start: int) -> str:
    early_end = max(1, math.floor(0.4 * n_layers))
    if layer_idx < early_end:
        return "early"
    if layer_idx < late_start:
        return "mid"
    return "late"


@dataclass
class TeacherStep:
    token_id: int
    token_str: str
    raw_category: str
    collapsed_category: str


@dataclass
class ModelPaths:
    model: str
    matched_dir: Path
    depth_dir: Path


class DecodeCache:
    def __init__(self, fallback_maps: dict[str, dict[int, str]]) -> None:
        self._tokenizers: dict[str, Any] = {}
        self._tokenizer_failures: set[str] = set()
        self._fallback_maps = fallback_maps
        self._strings: dict[tuple[str, int], str] = {}
        self._cats: dict[tuple[str, int], str] = {}

    def decode(self, model_key: str, model_id: str, token_id: int) -> str:
        key = (model_key, token_id)
        if key not in self._strings:
            tok = None
            if model_id not in self._tokenizer_failures:
                if model_id not in self._tokenizers:
                    try:
                        self._tokenizers[model_id] = AutoTokenizer.from_pretrained(
                            model_id,
                            trust_remote_code=True,
                            local_files_only=True,
                        )
                    except Exception:
                        self._tokenizer_failures.add(model_id)
                tok = self._tokenizers.get(model_id)
            if tok is not None:
                self._strings[key] = tok.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
            else:
                self._strings[key] = self._fallback_maps.get(model_key, {}).get(token_id, f"<id:{token_id}>")
        return self._strings[key]

    def category(self, model_key: str, model_id: str, token_id: int) -> str:
        key = (model_key, token_id)
        if key not in self._cats:
            token_str = self.decode(model_key, model_id, token_id)
            cat = classify_generated_tokens_by_word([{"token_str": token_str}])[0]
            self._cats[key] = cat if cat in RAW_CATEGORIES else "OTHER"
        return self._cats[key]


def _paths_for_model(model: str) -> ModelPaths:
    matched_candidates = [
        MATCHED_SOURCE_ROOT / model / f"exp11_exp3_400rand_v11_teacherforced_{model}",
        MATCHED_SOURCE_ROOT / model / f"exp11_exp3_400rand_v11_teacherforced_{model}_recovered",
    ]
    matched_dir = None
    for candidate in matched_candidates:
        if (candidate / "config.json").exists():
            matched_dir = candidate
            break
    if matched_dir is None:
        raise FileNotFoundError(f"Could not resolve matched-prefix dir for {model}")
    depth_candidates = [
        DEPTH_SOURCE_ROOT / model,
        DEPTH_SOURCE_ROOT / model / f"exp11_exp3_600rand_v11_depthablation_{model}",
        DEPTH_FLAT_SOURCE_ROOT / model,
    ]
    depth_dir = None
    for candidate in depth_candidates:
        if (candidate / "config.json").exists():
            depth_dir = candidate
            break
    if depth_dir is None:
        raise FileNotFoundError(f"Could not resolve depth-ablation dir for {model}")
    return ModelPaths(model=model, matched_dir=matched_dir, depth_dir=depth_dir)


def _fallback_generated_paths(model: str) -> list[Path]:
    candidates = [
        _paths_for_model(model).matched_dir / "generated_texts.jsonl",
        _paths_for_model(model).depth_dir / "generated_texts.jsonl",
        Path(f"results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_all2936_tunedlens_v10/{model}/generated_texts.jsonl"),
    ]
    out: list[Path] = []
    seen = set()
    for path in candidates:
        if path.exists() and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _build_fallback_token_maps(models: list[str]) -> dict[str, dict[int, str]]:
    out: dict[str, dict[int, str]] = {}
    for model in models:
        token_map: dict[int, str] = {}
        for path in _fallback_generated_paths(model):
            for line in _iter_jsonl_text(path):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for tok in row.get("generated_tokens", []):
                    token_id = int(tok["token_id"])
                    token_str = str(tok.get("token_str", ""))
                    token_map.setdefault(token_id, token_str)
        out[model] = token_map
    return out


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_jsonl_text(path: Path):
    with path.open("rb") as f:
        for raw in f:
            yield raw.decode("utf-8", errors="ignore")


def _build_teacher_tokens(
    generated_path: Path,
    step_metrics_path: Path,
    teacher_pipeline: str,
    integrity: dict[str, Any],
) -> dict[str, list[TeacherStep]]:
    if not generated_path.exists():
        return _build_teacher_tokens_from_step_metrics(step_metrics_path, teacher_pipeline, integrity)
    outputs_by_prompt: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    invalid_lines = 0
    for line in _iter_jsonl_text(generated_path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        outputs_by_prompt[row["prompt_id"]][row["pipeline"]] = row["generated_tokens"]

    teacher_steps: dict[str, list[TeacherStep]] = {}
    mismatched_prompts = []
    for prompt_id, by_pipeline in outputs_by_prompt.items():
        if teacher_pipeline not in by_pipeline:
            raise ValueError(f"Missing teacher pipeline {teacher_pipeline} for prompt {prompt_id} in {generated_path}")
        teacher_tokens = by_pipeline[teacher_pipeline]
        teacher_categories = classify_generated_tokens_by_word(teacher_tokens)
        teacher_steps[prompt_id] = [
            TeacherStep(
                token_id=int(tok["token_id"]),
                token_str=str(tok.get("token_str", "")),
                raw_category=teacher_categories[idx],
                collapsed_category=_collapsed_category(teacher_categories[idx]),
            )
            for idx, tok in enumerate(teacher_tokens)
        ]
        teacher_ids = [int(tok["token_id"]) for tok in teacher_tokens]
        for pipeline, tokens in by_pipeline.items():
            candidate_ids = [int(tok["token_id"]) for tok in tokens]
            if candidate_ids != teacher_ids:
                mismatched_prompts.append({"prompt_id": prompt_id, "pipeline": pipeline})
                break

    integrity["teacher_forced_token_identity"] = {
        "prompts_checked": len(outputs_by_prompt),
        "mismatched_prompts": len(mismatched_prompts),
        "examples": mismatched_prompts[:5],
        "invalid_json_lines": invalid_lines,
    }
    return teacher_steps


def _build_teacher_tokens_from_step_metrics(
    step_metrics_path: Path,
    teacher_pipeline: str,
    integrity: dict[str, Any],
) -> dict[str, list[TeacherStep]]:
    outputs_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_lines = 0
    for line in _iter_jsonl_text(step_metrics_path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            invalid_lines += 1
            continue
        if row["pipeline"] != teacher_pipeline:
            continue
        outputs_by_prompt[row["prompt_id"]].append(
            {
                "step": int(row["step"]),
                "token_id": int(row["token_id"]),
                "token_str": str(row.get("token_str", "")),
            }
        )
    teacher_steps: dict[str, list[TeacherStep]] = {}
    for prompt_id, rows in outputs_by_prompt.items():
        rows.sort(key=lambda row: row["step"])
        generated_tokens = [{"token_id": row["token_id"], "token_str": row["token_str"]} for row in rows]
        teacher_categories = classify_generated_tokens_by_word(generated_tokens)
        teacher_steps[prompt_id] = [
            TeacherStep(
                token_id=row["token_id"],
                token_str=row["token_str"],
                raw_category=teacher_categories[idx],
                collapsed_category=_collapsed_category(teacher_categories[idx]),
            )
            for idx, row in enumerate(rows)
        ]
    integrity["teacher_forced_token_identity"] = {
        "prompts_checked": len(outputs_by_prompt),
        "mismatched_prompts": None,
        "examples": [],
        "source": "step_metrics_only",
        "invalid_json_lines": invalid_lines,
    }
    return teacher_steps


def _stream_prompt_step_rows(
    step_metrics_path: Path,
    pipelines: set[str],
    on_prompt: Any,
) -> dict[str, int]:
    duplicate_counts = Counter()
    invalid_lines = 0
    current_prompt_id: str | None = None
    current_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_steps: set[tuple[str, int]] = set()
    for line in _iter_jsonl_text(step_metrics_path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            pipeline = row["pipeline"]
            if pipeline not in pipelines:
                continue
            prompt_id = row["prompt_id"]
            if current_prompt_id is None:
                current_prompt_id = prompt_id
            if prompt_id != current_prompt_id:
                on_prompt(current_prompt_id, dict(current_rows))
                current_prompt_id = prompt_id
                current_rows = defaultdict(list)
                seen_steps = set()
            key = (pipeline, int(row["step"]))
            if key in seen_steps:
                duplicate_counts[pipeline] += 1
            else:
                seen_steps.add(key)
            current_rows[pipeline].append(row)
    if current_prompt_id is not None:
        on_prompt(current_prompt_id, dict(current_rows))
    if invalid_lines:
        duplicate_counts["__invalid_json_lines__"] = invalid_lines
    return dict(duplicate_counts)


def _make_rank_stats() -> dict[str, dict[str, float | int | None]]:
    return {
        cat: {"sum": 0.0, "count": 0, "improved": 0}
        for cat in RAW_CATEGORIES + COLLAPSED_CATEGORIES
    }


def _finalize_rank_stats(stats: dict[str, dict[str, float | int | None]]) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for cat, values in stats.items():
        count = int(values["count"])
        improved = int(values["improved"])
        total = float(values["sum"])
        out[cat] = {
            "count": count,
            "mean_rank_gain": _safe_mean(total, count),
            "fraction_improved": _fraction(improved, count),
        }
    return out


def _normalize_counts(counter: dict[str, int]) -> dict[str, float | int | None]:
    total = sum(counter.values())
    return {
        key: {
            "count": value,
            "fraction": _fraction(value, total),
        }
        for key, value in counter.items()
    }


def _add_rank_gain(stats: dict[str, dict[str, float | int | None]], raw_cat: str, gain: float) -> None:
    stats[raw_cat]["sum"] = float(stats[raw_cat]["sum"]) + gain
    stats[raw_cat]["count"] = int(stats[raw_cat]["count"]) + 1
    if gain > 0:
        stats[raw_cat]["improved"] = int(stats[raw_cat]["improved"]) + 1
    collapsed = _collapsed_category(raw_cat)
    stats[collapsed]["sum"] = float(stats[collapsed]["sum"]) + gain
    stats[collapsed]["count"] = int(stats[collapsed]["count"]) + 1
    if gain > 0:
        stats[collapsed]["improved"] = int(stats[collapsed]["improved"]) + 1


def _teacher_rank_gain_for_prompt(
    prompt_rows: dict[str, list[dict[str, Any]]],
    teacher_steps: list[TeacherStep],
    late_layers: range,
    baseline_pipeline: str,
    compare_pipeline: str,
) -> tuple[dict[str, dict[str, float | int | None]], list[dict[str, Any]]]:
    out = _make_rank_stats()
    samples = []
    baseline_rows = prompt_rows.get(baseline_pipeline, [])
    compare_rows = prompt_rows.get(compare_pipeline, [])
    by_step_a = {int(row["step"]): row for row in baseline_rows}
    by_step_b = {int(row["step"]): row for row in compare_rows}
    for step in sorted(set(by_step_a) & set(by_step_b) & set(range(len(teacher_steps)))):
        row_a = by_step_a[step]
        row_b = by_step_b[step]
        ranks_a = row_a["metrics"]["next_token_rank"]
        ranks_b = row_b["metrics"]["next_token_rank"]
        gains = [float(ranks_a[layer]) - float(ranks_b[layer]) for layer in late_layers]
        gain = sum(gains) / len(gains) if gains else 0.0
        teacher_cat = teacher_steps[step].raw_category
        _add_rank_gain(out, teacher_cat, gain)
        if len(samples) < 10:
            samples.append(
                {
                    "step": step,
                    "token_str": teacher_steps[step].token_str,
                    "teacher_category": teacher_cat,
                    "rank_gain": gain,
                    "ranks_a_first_last": [ranks_a[late_layers.start], ranks_a[late_layers.stop - 1]],
                    "ranks_b_first_last": [ranks_b[late_layers.start], ranks_b[late_layers.stop - 1]],
                }
            )
    return out, samples


def _process_matched_prompt(
    model_key: str,
    model_id: str,
    prompt_id: str,
    prompt_rows: dict[str, list[dict[str, Any]]],
    teacher_steps: list[TeacherStep],
    late_layers: range,
    decode_cache: DecodeCache,
    summary: dict[str, Any],
    manual_samples: list[dict[str, Any]],
    manual_rank_samples: list[dict[str, Any]],
) -> None:
    needed = {"A_prime", "B", "C"}
    if not needed.issubset(prompt_rows):
        summary["integrity_missing_pipelines"].append(
            {"prompt_id": prompt_id, "missing": sorted(needed - set(prompt_rows))}
        )
        return
    rows_a = prompt_rows["A_prime"]
    rows_b = prompt_rows["B"]
    rows_c = prompt_rows["C"]
    by_step_a = {int(row["step"]): row for row in rows_a}
    by_step_b = {int(row["step"]): row for row in rows_b}
    by_step_c = {int(row["step"]): row for row in rows_c}
    common_step_ids = sorted(set(by_step_a) & set(by_step_b) & set(by_step_c) & set(range(len(teacher_steps))))
    if not (len(rows_a) == len(rows_b) == len(rows_c) == len(teacher_steps) == len(common_step_ids)):
        summary["integrity_step_mismatches"].append(
            {
                "prompt_id": prompt_id,
                "A_prime_steps": len(rows_a),
                "B_steps": len(rows_b),
                "C_steps": len(rows_c),
                "teacher_steps": len(teacher_steps),
                "common_steps_used": len(common_step_ids),
            }
        )
    for step in common_step_ids:
        row_a = by_step_a[step]
        row_b = by_step_b[step]
        row_c = by_step_c[step]
        teacher = teacher_steps[step]
        ranks_a = row_a["metrics"]["next_token_rank"]
        ranks_b = row_b["metrics"]["next_token_rank"]
        rank_gain = sum(float(ranks_a[layer]) - float(ranks_b[layer]) for layer in late_layers) / len(late_layers)
        _add_rank_gain(summary["teacher_rank_gain"], teacher.raw_category, rank_gain)
        if len(manual_rank_samples) < 20:
            manual_rank_samples.append(
                {
                    "prompt_id": prompt_id,
                    "step": step,
                    "token_str": teacher.token_str,
                    "teacher_category": teacher.raw_category,
                    "rank_gain": rank_gain,
                    "ranks_a_first_last": [ranks_a[late_layers.start], ranks_a[late_layers.stop - 1]],
                    "ranks_b_first_last": [ranks_b[late_layers.start], ranks_b[late_layers.stop - 1]],
                }
            )
        for layer in late_layers:
            top1_a = int(row_a["metrics"]["top1_token"][layer])
            top1_b = int(row_b["metrics"]["top1_token"][layer])
            top1_c = int(row_c["metrics"]["top1_token"][layer])
            if top1_a != top1_b:
                suppressed_raw = decode_cache.category(model_key, model_id, top1_a)
                supported_raw = decode_cache.category(model_key, model_id, top1_b)
                _increment(summary["suppressed_counts"], suppressed_raw)
                _increment(summary["supported_counts"], supported_raw)
                _increment(summary["suppressed_counts_collapsed"], _collapsed_category(suppressed_raw))
                _increment(summary["supported_counts_collapsed"], _collapsed_category(supported_raw))
                summary["transition_matrix"][suppressed_raw][supported_raw] += 1
                summary["transition_matrix_collapsed"][_collapsed_category(suppressed_raw)][_collapsed_category(supported_raw)] += 1
                target_collapsed = teacher.collapsed_category
                summary["suppressed_by_target"][target_collapsed][suppressed_raw] += 1
                summary["supported_by_target"][target_collapsed][supported_raw] += 1
                if len(manual_samples) < 20:
                    manual_samples.append(
                        {
                            "prompt_id": prompt_id,
                            "step": step,
                            "layer": layer,
                            "teacher_token": teacher.token_str,
                            "teacher_category": teacher.raw_category,
                            "suppressed_token": decode_cache.decode(model_key, model_id, top1_a),
                            "suppressed_category": suppressed_raw,
                            "supported_token": decode_cache.decode(model_key, model_id, top1_b),
                            "supported_category": supported_raw,
                            "c_top1_token": decode_cache.decode(model_key, model_id, top1_c),
                        }
                    )
            set_a = set(int(x) for x in row_a["metrics"]["top20_ids"][layer])
            set_b = set(int(x) for x in row_b["metrics"]["top20_ids"][layer])
            union = set_a | set_b
            if union:
                summary["candidate_jaccard_sum"] += len(set_a & set_b) / len(union)
                summary["candidate_jaccard_count"] += 1
            for token_id in set_b - set_a:
                raw = decode_cache.category(model_key, model_id, token_id)
                _increment(summary["candidate_entries"], raw)
                _increment(summary["candidate_entries_collapsed"], _collapsed_category(raw))
            for token_id in set_a - set_b:
                raw = decode_cache.category(model_key, model_id, token_id)
                _increment(summary["candidate_exits"], raw)
                _increment(summary["candidate_exits_collapsed"], _collapsed_category(raw))
        first_layers = {}
        for label, row in [("A_prime", row_a), ("B_late_raw", row_b), ("C_it_chat", row_c)]:
            first_match = None
            for layer in range(len(row["metrics"]["top1_token"])):
                if int(row["metrics"]["top1_token"][layer]) == teacher.token_id:
                    first_match = layer
                    break
            summary["mind_change"][label]["total_steps"] += 1
            if first_match is not None:
                summary["mind_change"][label]["first_match_by_phase"][_layer_phase(first_match, len(row["metrics"]["top1_token"]), late_layers.start)] += 1
                if first_match >= late_layers.start:
                    summary["mind_change"][label]["late_win_total"] += 1
                    _increment(summary["mind_change"][label]["late_win_types"], teacher.raw_category)
                    _increment(summary["mind_change"][label]["late_win_types_collapsed"], teacher.collapsed_category)
            first_layers[label] = first_match
        if rank_gain > 0:
            summary["mind_change_improvement_examples"].append(
                {
                    "prompt_id": prompt_id,
                    "step": step,
                    "teacher_token": teacher.token_str,
                    "teacher_category": teacher.raw_category,
                    "first_match_layer": first_layers,
                    "late_rank_gain": rank_gain,
                }
            )


def _process_depth_prompt(
    prompt_id: str,
    prompt_rows: dict[str, list[dict[str, Any]]],
    teacher_steps: list[TeacherStep],
    final_layers: range,
    summary: dict[str, Any],
) -> None:
    if "A_prime_raw" not in prompt_rows:
        summary["integrity_missing_pipelines"].append(
            {"prompt_id": prompt_id, "missing": ["A_prime_raw"]}
        )
        return
    rows_a = prompt_rows["A_prime_raw"]
    by_step_a = {int(row["step"]): row for row in rows_a}
    for window in DEPTH_WINDOWS:
        rows_b = prompt_rows.get(window)
        if rows_b is None:
            summary["integrity_missing_pipelines"].append(
                {"prompt_id": prompt_id, "missing": [window]}
            )
            continue
        by_step_b = {int(row["step"]): row for row in rows_b}
        common_step_ids = sorted(set(by_step_a) & set(by_step_b) & set(range(len(teacher_steps))))
        if not (len(rows_a) == len(rows_b) == len(teacher_steps) == len(common_step_ids)):
            summary["integrity_step_mismatches"][window].append(
                {
                    "A_prime_raw_steps": len(rows_a),
                    "window_steps": len(rows_b),
                    "teacher_steps": len(teacher_steps),
                    "common_steps_used": len(common_step_ids),
                }
            )
        for step in common_step_ids:
            row_a = by_step_a[step]
            row_b = by_step_b[step]
            ranks_a = row_a["metrics"]["next_token_rank"]
            ranks_b = row_b["metrics"]["next_token_rank"]
            gain = sum(float(ranks_a[layer]) - float(ranks_b[layer]) for layer in final_layers) / len(final_layers)
            teacher_cat = teacher_steps[step].raw_category
            _add_rank_gain(summary[window], teacher_cat, gain)


def _finalize_mind_change(block: dict[str, Any]) -> dict[str, Any]:
    total = block["total_steps"]
    return {
        "total_steps": total,
        "late_win_total": block["late_win_total"],
        "late_win_fraction": _fraction(block["late_win_total"], total),
        "first_match_by_phase": {
            phase: {
                "count": count,
                "fraction": _fraction(count, total),
            }
            for phase, count in block["first_match_by_phase"].items()
        },
        "late_win_types": _normalize_counts(block["late_win_types"]),
        "late_win_types_collapsed": _normalize_counts(block["late_win_types_collapsed"]),
    }


def _empty_matched_summary() -> dict[str, Any]:
    return {
        "integrity_step_mismatches": [],
        "integrity_missing_pipelines": [],
        "suppressed_counts": _make_nested_counter(),
        "supported_counts": _make_nested_counter(),
        "suppressed_counts_collapsed": _make_collapsed_counter(),
        "supported_counts_collapsed": _make_collapsed_counter(),
        "transition_matrix": _make_transition_matrix(RAW_CATEGORIES),
        "transition_matrix_collapsed": _make_transition_matrix(COLLAPSED_CATEGORIES),
        "suppressed_by_target": {cat: _make_nested_counter() for cat in COLLAPSED_CATEGORIES},
        "supported_by_target": {cat: _make_nested_counter() for cat in COLLAPSED_CATEGORIES},
        "teacher_rank_gain": _make_rank_stats(),
        "candidate_entries": _make_nested_counter(),
        "candidate_exits": _make_nested_counter(),
        "candidate_entries_collapsed": _make_collapsed_counter(),
        "candidate_exits_collapsed": _make_collapsed_counter(),
        "candidate_jaccard_sum": 0.0,
        "candidate_jaccard_count": 0,
        "mind_change": {
            label: {
                "total_steps": 0,
                "late_win_total": 0,
                "first_match_by_phase": {"early": 0, "mid": 0, "late": 0},
                "late_win_types": _make_nested_counter(),
                "late_win_types_collapsed": _make_collapsed_counter(),
            }
            for label in ["A_prime", "B_late_raw", "C_it_chat"]
        },
        "mind_change_improvement_examples": [],
    }


def _empty_depth_summary() -> dict[str, Any]:
    summary = {window: _make_rank_stats() for window in DEPTH_WINDOWS}
    summary["integrity_step_mismatches"] = {window: [] for window in DEPTH_WINDOWS}
    summary["integrity_missing_pipelines"] = []
    return summary


def _merge_rank_stats(dest: dict[str, dict[str, float | int | None]], src: dict[str, dict[str, float | int | None]]) -> None:
    for cat, vals in src.items():
        dest[cat]["sum"] = float(dest[cat]["sum"]) + float(vals["sum"])
        dest[cat]["count"] = int(dest[cat]["count"]) + int(vals["count"])
        dest[cat]["improved"] = int(dest[cat]["improved"]) + int(vals["improved"])


def _summarize_model(model_paths: ModelPaths, decode_cache: DecodeCache) -> dict[str, Any]:
    matched_cfg = _load_json(model_paths.matched_dir / "config.json")
    depth_cfg = _load_json(model_paths.depth_dir / "config.json")
    model_id = matched_cfg.get("pt_model_id") or matched_cfg.get("it_model_id")
    n_layers = int(matched_cfg["n_layers"])
    late_start = math.floor(0.8 * n_layers)
    late_layers = range(late_start, n_layers)

    matched_integrity: dict[str, Any] = {}
    teacher_400 = _build_teacher_tokens(
        model_paths.matched_dir / "generated_texts.jsonl",
        model_paths.matched_dir / "step_metrics.jsonl",
        teacher_pipeline="C",
        integrity=matched_integrity,
    )
    matched_summary = _empty_matched_summary()
    manual_samples: list[dict[str, Any]] = []
    manual_rank_samples: list[dict[str, Any]] = []
    def _handle_matched_prompt(prompt_id: str, prompt_rows: dict[str, list[dict[str, Any]]]) -> None:
        teacher_steps = teacher_400.get(prompt_id)
        if teacher_steps is None:
            matched_summary["integrity_missing_pipelines"].append(
                {"prompt_id": prompt_id, "missing": ["teacher_tokens"]}
            )
            return
        _process_matched_prompt(
            model_key=model_paths.model,
            model_id=model_id,
            prompt_id=prompt_id,
            prompt_rows=prompt_rows,
            teacher_steps=teacher_steps,
            late_layers=late_layers,
            decode_cache=decode_cache,
            summary=matched_summary,
            manual_samples=manual_samples,
            manual_rank_samples=manual_rank_samples,
        )
    duplicates = _stream_prompt_step_rows(
        model_paths.matched_dir / "step_metrics.jsonl",
        pipelines={"A_prime", "B", "C"},
        on_prompt=_handle_matched_prompt,
    )

    depth_integrity: dict[str, Any] = {}
    depth_ablation_summary = _load_json(model_paths.depth_dir / "depth_ablation_summary.json")
    final_region = depth_ablation_summary["final_region"]
    depth_generated_path = model_paths.depth_dir / "generated_texts.jsonl"
    depth_step_metrics_path = model_paths.depth_dir / "step_metrics.jsonl"
    depth_summary = _empty_depth_summary()
    depth_duplicates: dict[str, int] = {}
    if depth_step_metrics_path.exists():
        teacher_600 = _build_teacher_tokens(
            depth_generated_path,
            depth_step_metrics_path,
            teacher_pipeline="C_it_chat",
            integrity=depth_integrity,
        )
        final_layers = range(int(final_region["start_layer"]), int(final_region["end_layer_exclusive"]))
        def _handle_depth_prompt(prompt_id: str, prompt_rows: dict[str, list[dict[str, Any]]]) -> None:
            teacher_steps = teacher_600.get(prompt_id)
            if teacher_steps is None:
                depth_summary["integrity_missing_pipelines"].append(
                    {"prompt_id": prompt_id, "missing": ["teacher_tokens"]}
                )
                return
            _process_depth_prompt(prompt_id, prompt_rows, teacher_steps, final_layers, depth_summary)
        depth_duplicates = _stream_prompt_step_rows(
            depth_step_metrics_path,
            pipelines={"A_prime_raw", "B_early_raw", "B_mid_raw", "B_late_raw", "C_it_chat"},
            on_prompt=_handle_depth_prompt,
        )
        depth_integrity["raw_teacher_rank_gain_available"] = True
    else:
        depth_integrity["raw_teacher_rank_gain_available"] = False

    out = {
        "model": model_paths.model,
        "model_id": model_id,
        "n_layers": n_layers,
        "late_region": {
            "start_layer": late_start,
            "end_layer_exclusive": n_layers,
            "display_range": f"{late_start}-{n_layers - 1}",
        },
        "depth_final_region": final_region,
        "teacherforced_400": {
            "integrity": {
                "duplicates_by_pipeline": duplicates,
                **matched_integrity,
                "step_mismatches": matched_summary["integrity_step_mismatches"][:25],
                "missing_pipelines": matched_summary["integrity_missing_pipelines"][:25],
            },
            "displacement": {
                "suppressed": _normalize_counts(matched_summary["suppressed_counts"]),
                "supported": _normalize_counts(matched_summary["supported_counts"]),
                "suppressed_collapsed": _normalize_counts(matched_summary["suppressed_counts_collapsed"]),
                "supported_collapsed": _normalize_counts(matched_summary["supported_counts_collapsed"]),
                "transition_matrix": matched_summary["transition_matrix"],
                "transition_matrix_collapsed": matched_summary["transition_matrix_collapsed"],
                "suppressed_by_target": {
                    cat: _normalize_counts(counts)
                    for cat, counts in matched_summary["suppressed_by_target"].items()
                },
                "supported_by_target": {
                    cat: _normalize_counts(counts)
                    for cat, counts in matched_summary["supported_by_target"].items()
                },
            },
            "teacher_rank_gain": _finalize_rank_stats(matched_summary["teacher_rank_gain"]),
            "candidate_reshuffling": {
                "entries": _normalize_counts(matched_summary["candidate_entries"]),
                "exits": _normalize_counts(matched_summary["candidate_exits"]),
                "entries_collapsed": _normalize_counts(matched_summary["candidate_entries_collapsed"]),
                "exits_collapsed": _normalize_counts(matched_summary["candidate_exits_collapsed"]),
                "mean_jaccard": _safe_mean(
                    matched_summary["candidate_jaccard_sum"],
                    matched_summary["candidate_jaccard_count"],
                ),
                "count": matched_summary["candidate_jaccard_count"],
            },
            "mind_change": {
                label: _finalize_mind_change(block)
                for label, block in matched_summary["mind_change"].items()
            },
            "mind_change_improvement_examples": matched_summary["mind_change_improvement_examples"][:25],
            "manual_displacement_samples": manual_samples,
            "manual_rank_gain_samples": manual_rank_samples[:20],
        },
        "depth_ablation_600": {
            "integrity": {
                "duplicates_by_pipeline": depth_duplicates,
                **depth_integrity,
                "step_mismatches": depth_summary["integrity_step_mismatches"],
                "missing_pipelines": depth_summary["integrity_missing_pipelines"][:25],
            },
            "teacher_rank_gain_by_window": {
                window: _finalize_rank_stats(stats)
                for window, stats in depth_summary.items()
                if window in DEPTH_WINDOWS
            },
            "summary_only": depth_ablation_summary,
        },
        "graft_windows": _derive_graft_windows(model_paths.model, int(depth_cfg["n_layers"]), int(depth_cfg["onset_layer"])),
    }
    return out


def _derive_graft_windows(model: str, n_layers: int, onset: int) -> dict[str, dict[str, int | str]]:
    width = n_layers - onset
    mid_start = (n_layers - width) // 2
    windows = {
        "B_early_raw": (0, width),
        "B_mid_raw": (mid_start, mid_start + width),
        "B_late_raw": (onset, n_layers),
    }
    return {
        name: {
            "start_layer": start,
            "end_layer_exclusive": end,
            "display_range": f"{start}-{end - 1}",
        }
        for name, (start, end) in windows.items()
    }


def _empty_pooled_rank_stats() -> dict[str, dict[str, float | int | None]]:
    return _make_rank_stats()


def _merge_finalized_rank_stats(dest: dict[str, dict[str, float | int | None]], src: dict[str, dict[str, Any]]) -> None:
    for cat, vals in src.items():
        count = int(vals.get("count") or 0)
        mean = vals.get("mean_rank_gain")
        improved_frac = vals.get("fraction_improved")
        if mean is not None:
            dest[cat]["sum"] = float(dest[cat]["sum"]) + float(mean) * count
        dest[cat]["count"] = int(dest[cat]["count"]) + count
        if improved_frac is not None:
            dest[cat]["improved"] = int(dest[cat]["improved"]) + int(round(float(improved_frac) * count))


def _merge_counter_stats(dest: dict[str, int], src: dict[str, dict[str, Any]]) -> None:
    for cat, vals in src.items():
        dest[cat] = dest.get(cat, 0) + int(vals.get("count") or 0)


def _load_exp12_cross_check() -> dict[str, Any] | None:
    if not EXP12_JUDGE_SUMMARY.exists():
        return None
    obj = _load_json(EXP12_JUDGE_SUMMARY)
    models = {}
    for model, payload in obj.get("models", {}).items():
        metrics = payload.get("metrics", {})
        models[model] = {
            metric: {
                condition: score_block.get("mean")
                for condition, score_block in metric_block.get("conditions", {}).items()
            }
            for metric, metric_block in metrics.items()
            if metric in {"g2", "s2"}
        }
    return {"source": str(EXP12_JUDGE_SUMMARY), "models": models}


def _build_global_summary(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pooled = {
        "teacherforced_400": {
            "suppressed_counts": _make_nested_counter(),
            "supported_counts": _make_nested_counter(),
            "suppressed_counts_collapsed": _make_collapsed_counter(),
            "supported_counts_collapsed": _make_collapsed_counter(),
            "teacher_rank_gain": _empty_pooled_rank_stats(),
        },
        "depth_ablation_600": {
            window: _empty_pooled_rank_stats() for window in DEPTH_WINDOWS
        },
    }
    for model, payload in models.items():
        if model not in DENSE5_MODELS:
            continue
        tf = payload["teacherforced_400"]
        _merge_counter_stats(pooled["teacherforced_400"]["suppressed_counts"], tf["displacement"]["suppressed"])
        _merge_counter_stats(pooled["teacherforced_400"]["supported_counts"], tf["displacement"]["supported"])
        _merge_counter_stats(pooled["teacherforced_400"]["suppressed_counts_collapsed"], tf["displacement"]["suppressed_collapsed"])
        _merge_counter_stats(pooled["teacherforced_400"]["supported_counts_collapsed"], tf["displacement"]["supported_collapsed"])
        _merge_finalized_rank_stats(pooled["teacherforced_400"]["teacher_rank_gain"], tf["teacher_rank_gain"])
        for window in DEPTH_WINDOWS:
            _merge_finalized_rank_stats(
                pooled["depth_ablation_600"][window],
                payload["depth_ablation_600"]["teacher_rank_gain_by_window"][window],
            )

    out = {
        "dense5_teacherforced_400": {
            "suppressed": _normalize_counts(pooled["teacherforced_400"]["suppressed_counts"]),
            "supported": _normalize_counts(pooled["teacherforced_400"]["supported_counts"]),
            "suppressed_collapsed": _normalize_counts(pooled["teacherforced_400"]["suppressed_counts_collapsed"]),
            "supported_collapsed": _normalize_counts(pooled["teacherforced_400"]["supported_counts_collapsed"]),
            "teacher_rank_gain": _finalize_rank_stats(pooled["teacherforced_400"]["teacher_rank_gain"]),
        },
        "dense5_depth_ablation_600": {
            window: _finalize_rank_stats(stats)
            for window, stats in pooled["depth_ablation_600"].items()
        },
    }
    deepseek = models.get("deepseek_v2_lite")
    if deepseek is not None:
        out["deepseek_separate"] = {
            "teacherforced_400": deepseek["teacherforced_400"],
            "depth_ablation_600": deepseek["depth_ablation_600"],
        }
    return out


def _assemble_summary(manifest: dict[str, Any], model_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "analysis": "exp13A-lite",
        "version": "2026-04-15",
        "description": (
            "Descriptive no-new-compute analysis of what the late corrective stage suppresses and supports "
            "using existing matched-prefix exp11 artifacts."
        ),
        "categories": {
            "raw": RAW_CATEGORIES,
            "collapsed": {
                "FORMAT": ["STRUCTURAL", "DISCOURSE", "PUNCTUATION"],
                "CONTENT": ["CONTENT"],
                "FUNCTION_OTHER": ["FUNCTION", "OTHER"],
            },
        },
        "manifest": manifest,
        "models": model_summaries,
        "pooled": _build_global_summary(model_summaries),
        "exp12_behavioral_cross_check": _load_exp12_cross_check(),
    }


def _write_summary(out_dir: Path, summary: dict[str, Any], *, filename: str = "exp13a_lite_summary.json") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / filename).write_text(json.dumps(summary, indent=2))
    (out_dir / "README.txt").write_text(
        "exp13A-lite no-new-compute summary bundle.\n"
        "Main source datasets:\n"
        f"- {MATCHED_SOURCE_ROOT}\n"
        f"- {DEPTH_SOURCE_ROOT} plus {DEPTH_FLAT_SOURCE_ROOT}\n"
        f"- {EXP12_JUDGE_SUMMARY}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze exp13A-lite from existing exp11/exp12 artifacts.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp13_late_stage_token_support_analysis/exp13A_lite_20260415"),
        help="Output directory for the summary bundle.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b", "deepseek_v2_lite"],
        help="Models to include.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    decode_cache = DecodeCache(_build_fallback_token_maps(args.models))
    model_summaries: dict[str, dict[str, Any]] = {}
    manifest = {}
    progress_log = args.out_dir / "progress.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with progress_log.open("a") as fh:
            fh.write(message + "\n")

    start_time = time.time()
    log(f"[exp13A-lite] starting full run for {len(args.models)} models")
    for idx, model in enumerate(args.models, start=1):
        model_start = time.time()
        paths = _paths_for_model(model)
        manifest[model] = {
            "matched_dir": str(paths.matched_dir),
            "depth_dir": str(paths.depth_dir),
        }
        log(f"[exp13A-lite] [{idx}/{len(args.models)}] start {model}")
        model_summaries[model] = _summarize_model(paths, decode_cache)
        elapsed = time.time() - model_start
        checkpoint = _assemble_summary(dict(manifest), dict(model_summaries))
        checkpoint["progress"] = {
            "completed_models": idx,
            "total_models": len(args.models),
            "completed_model_names": list(model_summaries.keys()),
            "last_completed_model": model,
            "elapsed_seconds": time.time() - start_time,
        }
        _write_summary(args.out_dir, checkpoint, filename="exp13a_lite_partial_summary.json")
        log(
            f"[exp13A-lite] [{idx}/{len(args.models)}] finished {model} in {elapsed / 60.0:.1f} min "
            f"(checkpoint: {args.out_dir / 'exp13a_lite_partial_summary.json'})"
        )

    summary = _assemble_summary(manifest, model_summaries)
    _write_summary(args.out_dir, summary)
    total_elapsed = time.time() - start_time
    log(
        f"[exp13A-lite] wrote summary to {args.out_dir / 'exp13a_lite_summary.json'} "
        f"after {total_elapsed / 60.0:.1f} min"
    )


if __name__ == "__main__":
    main()
