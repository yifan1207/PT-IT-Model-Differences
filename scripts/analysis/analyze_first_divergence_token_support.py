#!/usr/bin/env python3
"""Analyze what the first-divergence token pairs actually are.

This script answers a support-interpretability question for the first-divergence
factorial: are the selected token pairs mostly surface formatting/opening tokens,
or do they include content, reasoning, safety/refusal, and instruction-following
choices?

It has two layers:
1. Deterministic full-support statistics from Exp20 first-divergence records.
2. Optional LLM judging on a reproducible stratified sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


DENSE5_MODELS = ("gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b")
DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_OUT_DIR = Path("results/first_divergence_token_support")

PUNCT_RE = re.compile(r"^[\\s\\n\\r\\t\\.,;:!?\\-–—_\\*#`'\"\\[\\]\\(\\)\\{\\}<>/\\\\|=+~]+$")

LLM_CATEGORIES = (
    "surface_format_low",
    "discourse_style_opening",
    "structural_instruction_format",
    "semantic_content",
    "reasoning_answer",
    "safety_refusal_helpfulness",
    "mixed_uncertain",
)


@dataclass(frozen=True)
class EventRow:
    uid: str
    model: str
    prompt_id: str
    prompt_category: str
    source: str
    expected_behavior: str
    expected_format: str
    generated_position: int
    position_bin: str
    pt_token: str
    it_token: str
    pt_token_category: str
    it_token_category: str
    pt_token_category_collapsed: str
    it_token_category_collapsed: str
    pair_kind: str
    any_content_token: bool
    any_format_token: bool
    both_surface_format: bool
    prompt_text: str
    shared_prefix_text: str
    pt_continuation_text: str
    it_continuation_text: str


def _json_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _dataset_lookup(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id", row.get("record_id"))): row for row in _json_rows(path)}


def _find_exp20_file(root: Path, prompt_mode: str, model: str) -> Path:
    candidates = [
        root / prompt_mode / model / "exp20_validation_records.jsonl",
        root / prompt_mode / model / "exp20_records.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No Exp20 records found. Tried: " + ", ".join(str(path) for path in candidates))


def _token_str(token: dict[str, Any]) -> str:
    return str(token.get("token_str", ""))


def _concat_tokens(tokens: list[dict[str, Any]]) -> str:
    return "".join(_token_str(token) for token in tokens)


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _position_bin(step: int) -> str:
    if step == 0:
        return "pos_0"
    if step <= 2:
        return "pos_1_2"
    if step <= 4:
        return "pos_3_4"
    if step <= 9:
        return "pos_5_9"
    return "pos_10_plus"


def _is_surface_token(text: str, category: str, collapsed: str) -> bool:
    if text == "":
        return True
    if text.strip() == "":
        return True
    if collapsed == "FORMAT" and PUNCT_RE.match(text):
        return True
    if category in {"PUNCTUATION", "WHITESPACE"} and PUNCT_RE.match(text):
        return True
    return False


def _pair_kind(pt: dict[str, Any], it: dict[str, Any]) -> str:
    pt_text = _token_str(pt)
    it_text = _token_str(it)
    pt_cat = str(pt.get("token_category", "OTHER"))
    it_cat = str(it.get("token_category", "OTHER"))
    pt_collapsed = str(pt.get("token_category_collapsed", "OTHER"))
    it_collapsed = str(it.get("token_category_collapsed", "OTHER"))
    pt_surface = _is_surface_token(pt_text, pt_cat, pt_collapsed)
    it_surface = _is_surface_token(it_text, it_cat, it_collapsed)
    if pt_surface and it_surface:
        return "both_surface_format"
    if pt_cat == "CONTENT" or it_cat == "CONTENT" or pt_collapsed == "CONTENT" or it_collapsed == "CONTENT":
        return "any_content"
    if pt_collapsed == "FORMAT" or it_collapsed == "FORMAT":
        return "format_vs_nonformat"
    if pt_cat == "DISCOURSE" or it_cat == "DISCOURSE":
        return "discourse"
    if pt_cat == "FUNCTION" and it_cat == "FUNCTION":
        return "function_vs_function"
    return "other"


def _prompt_text(record: dict[str, Any]) -> str:
    formats = record.get("formats") or {}
    return str(formats.get("A") or formats.get("B") or record.get("prompt") or record.get("format_instruction") or "")


def load_events(*, exp20_root: Path, dataset: Path, prompt_mode: str, models: list[str]) -> list[EventRow]:
    dataset_by_id = _dataset_lookup(dataset)
    rows: list[EventRow] = []
    for model in models:
        path = _find_exp20_file(exp20_root, prompt_mode, model)
        for record in _json_rows(path):
            prompt_id = str(record.get("prompt_id"))
            data = dataset_by_id.get(prompt_id, {})
            readout = (record.get("readouts") or {}).get("first_diff")
            if not isinstance(readout, dict):
                continue
            event = readout.get("event") or {}
            pt_token = event.get("pt_token") or {}
            it_token = event.get("it_token") or {}
            if pt_token.get("token_id") is None or it_token.get("token_id") is None:
                continue
            step = int(event.get("step", 0))
            free_runs = record.get("free_runs") or {}
            pt_tokens = (free_runs.get("A_pt_raw") or {}).get("generated_tokens") or []
            it_tokens = (free_runs.get("C_it_chat") or {}).get("generated_tokens") or []
            shared_prefix = _concat_tokens(pt_tokens[:step])
            pt_cont = _concat_tokens(pt_tokens[step : step + 10])
            it_cont = _concat_tokens(it_tokens[step : step + 10])
            pt_cat = str(pt_token.get("token_category", "OTHER"))
            it_cat = str(it_token.get("token_category", "OTHER"))
            pt_collapsed = str(pt_token.get("token_category_collapsed", "OTHER"))
            it_collapsed = str(it_token.get("token_category_collapsed", "OTHER"))
            kind = _pair_kind(pt_token, it_token)
            uid = f"{model}::{prompt_id}"
            rows.append(
                EventRow(
                    uid=uid,
                    model=model,
                    prompt_id=prompt_id,
                    prompt_category=str(data.get("category", "UNKNOWN")),
                    source=str(data.get("source", "UNKNOWN")),
                    expected_behavior=str(data.get("expected_behavior", "")),
                    expected_format=str(data.get("expected_format", "")),
                    generated_position=step,
                    position_bin=_position_bin(step),
                    pt_token=_token_str(pt_token),
                    it_token=_token_str(it_token),
                    pt_token_category=pt_cat,
                    it_token_category=it_cat,
                    pt_token_category_collapsed=pt_collapsed,
                    it_token_category_collapsed=it_collapsed,
                    pair_kind=kind,
                    any_content_token=kind == "any_content",
                    any_format_token=(pt_collapsed == "FORMAT" or it_collapsed == "FORMAT"),
                    both_surface_format=kind == "both_surface_format",
                    prompt_text=_prompt_text(data),
                    shared_prefix_text=shared_prefix,
                    pt_continuation_text=pt_cont,
                    it_continuation_text=it_cont,
                )
            )
    return rows


def _proportion_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    # Wilson interval
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return center - half, center + half


def _fraction_table(rows: list[EventRow], attr: str) -> list[dict[str, Any]]:
    n = len(rows)
    counts = Counter(getattr(row, attr) for row in rows)
    out = []
    for key, count in counts.most_common():
        lo, hi = _proportion_ci(count, n)
        out.append({"value": key, "n": count, "fraction": count / n if n else 0, "ci95_low": lo, "ci95_high": hi})
    return out


def deterministic_summary(rows: list[EventRow]) -> dict[str, Any]:
    n = len(rows)
    bools = {
        "both_surface_format": sum(row.both_surface_format for row in rows),
        "any_content_token": sum(row.any_content_token for row in rows),
        "any_format_token": sum(row.any_format_token for row in rows),
        "position_0": sum(row.generated_position == 0 for row in rows),
        "position_ge_3": sum(row.generated_position >= 3 for row in rows),
        "position_ge_5": sum(row.generated_position >= 5 for row in rows),
        "content_or_reasoning_prompt": sum(row.prompt_category in {"CONTENT-FACT", "CONTENT-REASON"} for row in rows),
    }
    bool_summary = {}
    for key, count in bools.items():
        lo, hi = _proportion_ci(int(count), n)
        bool_summary[key] = {"n": int(count), "fraction": count / n if n else 0, "ci95_low": lo, "ci95_high": hi}
    return {
        "n": n,
        "by_model": _fraction_table(rows, "model"),
        "by_prompt_category": _fraction_table(rows, "prompt_category"),
        "by_position_bin": _fraction_table(rows, "position_bin"),
        "by_pair_kind": _fraction_table(rows, "pair_kind"),
        "by_pt_token_category_collapsed": _fraction_table(rows, "pt_token_category_collapsed"),
        "by_it_token_category_collapsed": _fraction_table(rows, "it_token_category_collapsed"),
        "boolean_support": bool_summary,
        "top_pt_tokens": Counter(row.pt_token for row in rows).most_common(40),
        "top_it_tokens": Counter(row.it_token for row in rows).most_common(40),
        "top_token_pairs": Counter((row.pt_token, row.it_token) for row in rows).most_common(60),
    }


def stratified_sample(rows: list[EventRow], *, n: int, seed: int) -> list[EventRow]:
    rng = random.Random(seed)
    strata: dict[tuple[str, str, str], list[EventRow]] = defaultdict(list)
    for row in rows:
        strata[(row.prompt_category, row.position_bin, row.pair_kind)].append(row)
    sampled: list[EventRow] = []
    keys = sorted(strata)
    rng.shuffle(keys)
    # First pass: at least one per stratum where possible.
    for key in keys:
        if len(sampled) >= n:
            break
        sampled.append(rng.choice(strata[key]))
    remaining = max(0, n - len(sampled))
    if remaining:
        all_rows = rows[:]
        rng.shuffle(all_rows)
        seen = {row.uid for row in sampled}
        for row in all_rows:
            if len(sampled) >= n:
                break
            if row.uid in seen:
                continue
            sampled.append(row)
            seen.add(row.uid)
    return sampled


def _llm_record(row: EventRow) -> dict[str, Any]:
    return {
        "uid": row.uid,
        "model": row.model,
        "prompt_category": row.prompt_category,
        "source": row.source,
        "position": row.generated_position,
        "prompt": _clip(row.prompt_text, 700),
        "shared_prefix": _clip(row.shared_prefix_text, 260),
        "pt_next_token": row.pt_token,
        "it_next_token": row.it_token,
        "pt_continuation_from_divergence": _clip(row.pt_continuation_text, 180),
        "it_continuation_from_divergence": _clip(row.it_continuation_text, 180),
    }


def _build_llm_messages(batch: list[EventRow]) -> list[dict[str, str]]:
    system = (
        "You classify PT-vs-IT first-divergence token pairs for a mechanistic "
        "interpretability paper. Decide what kind of difference the divergent "
        "next token represents. Use the prompt, shared generated prefix, the two "
        "next tokens, and short continuations. Do not judge model quality; classify "
        "the semantic role of the token-pair choice."
    )
    instructions = {
        "categories": list(LLM_CATEGORIES),
        "category_definitions": {
            "surface_format_low": "Whitespace, newline, punctuation, spacing, tokenization, or visually minor surface mark with little standalone interpretive significance.",
            "discourse_style_opening": "Generic assistant opening/style/discourse marker such as Sure/Here/I/The/Let, when not specifically required by the task.",
            "structural_instruction_format": "Token choice controls requested structure, JSON/Markdown/list/section labels, quoting, casing, language, delimiters, or other instruction-following format.",
            "semantic_content": "Token choice changes task/domain content, factual entity, answer substance, named object, or topical information.",
            "reasoning_answer": "Token choice changes reasoning trajectory, mathematical/logical operator, answer option, final answer, comparison, or stepwise inference.",
            "safety_refusal_helpfulness": "Token choice affects refusal, safety boundary, compliance/helpfulness decision, apology, or policy-style response.",
            "mixed_uncertain": "Multiple roles or insufficient context.",
        },
        "return_json_schema": {
            "items": [
                {
                    "uid": "same uid",
                    "category": "one category",
                    "substantive": "boolean; false only for pure surface_format_low or generic discourse with no task relevance",
                    "instruction_relevant": "boolean",
                    "content_or_reasoning_relevant": "boolean",
                    "confidence": "low|medium|high",
                    "brief_reason": "short phrase, <=20 words",
                }
            ]
        },
    }
    user = json.dumps({"instructions": instructions, "records": [_llm_record(row) for row in batch]}, ensure_ascii=False)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _call_openai_json(client: Any, *, model: str, messages: list[dict[str, str]], max_retries: int) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            # Prefer Responses API for newer models.
            if hasattr(client, "responses"):
                response = client.responses.create(
                    model=model,
                    input=messages,
                    text={"format": {"type": "json_object"}},
                )
                return json.loads(response.output_text)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as exc:  # pragma: no cover - network/API retry path
            last_exc = exc
            time.sleep(min(20.0, 2.0**attempt))
    raise RuntimeError(f"OpenAI call failed after {max_retries} attempts: {last_exc}") from last_exc


def run_llm_judge(
    rows: list[EventRow],
    *,
    out_dir: Path,
    sample_size: int,
    seed: int,
    model: str,
    fallback_models: list[str],
    batch_size: int,
    max_retries: int,
) -> list[dict[str, Any]]:
    from openai import OpenAI

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI()
    sample = stratified_sample(rows, n=sample_size, seed=seed)
    (out_dir / "llm_sample_input.jsonl").write_text(
        "\n".join(json.dumps(_llm_record(row), ensure_ascii=False) for row in sample) + "\n",
        encoding="utf-8",
    )
    results_path = out_dir / "llm_judgments.jsonl"
    done: set[str] = set()
    if results_path.exists():
        for row in _json_rows(results_path):
            if row.get("uid"):
                done.add(str(row["uid"]))
    models_to_try = [model, *fallback_models]
    with results_path.open("a", encoding="utf-8") as fout:
        for start in range(0, len(sample), batch_size):
            batch = [row for row in sample[start : start + batch_size] if row.uid not in done]
            if not batch:
                continue
            messages = _build_llm_messages(batch)
            parsed = None
            used_model = None
            errors = []
            for candidate in models_to_try:
                try:
                    parsed = _call_openai_json(client, model=candidate, messages=messages, max_retries=max_retries)
                    used_model = candidate
                    break
                except Exception as exc:  # pragma: no cover - network/API fallback path
                    errors.append(f"{candidate}: {exc}")
            if parsed is None:
                raise RuntimeError("; ".join(errors))
            items = parsed.get("items") if isinstance(parsed, dict) else parsed
            if not isinstance(items, list):
                raise RuntimeError(f"LLM output did not contain a list: {parsed}")
            by_uid = {row.uid: row for row in batch}
            for item in items:
                if not isinstance(item, dict) or item.get("uid") not in by_uid:
                    continue
                category = str(item.get("category", "mixed_uncertain"))
                if category not in LLM_CATEGORIES:
                    category = "mixed_uncertain"
                out = {
                    **asdict(by_uid[str(item["uid"])]),
                    "llm_model": used_model,
                    "llm_category": category,
                    "llm_substantive": bool(item.get("substantive", category not in {"surface_format_low", "discourse_style_opening"})),
                    "llm_instruction_relevant": bool(item.get("instruction_relevant", False)),
                    "llm_content_or_reasoning_relevant": bool(item.get("content_or_reasoning_relevant", False)),
                    "llm_confidence": str(item.get("confidence", "medium")),
                    "llm_reason": str(item.get("brief_reason", ""))[:160],
                }
                fout.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
                done.add(out["uid"])
            fout.flush()
            print(f"[token-support] judged {len(done)}/{len(sample)} sample records", flush=True)
    return list(_json_rows(results_path))


def _sample_stratum_key(row: dict[str, Any] | EventRow) -> tuple[str, str, str]:
    if isinstance(row, EventRow):
        return (row.prompt_category, row.position_bin, row.pair_kind)
    return (str(row.get("prompt_category")), str(row.get("position_bin")), str(row.get("pair_kind")))


def _weighted_fraction_ci(
    rows: list[dict[str, Any]],
    *,
    field: str,
    value: str,
    population_counts: Counter[tuple[str, str, str]],
    sample_counts: Counter[tuple[str, str, str]],
    n_bootstrap: int = 2000,
    seed: int = 41,
) -> dict[str, Any]:
    if not rows:
        return {"fraction": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[_sample_stratum_key(row)].append(row)

    def estimate(sampled_by_stratum: dict[tuple[str, str, str], list[dict[str, Any]]]) -> float:
        num = 0.0
        den = 0.0
        for key, vals in sampled_by_stratum.items():
            if not vals:
                continue
            weight = population_counts[key] / max(1, len(vals))
            for row in vals:
                den += weight
                if str(row.get(field)) == value:
                    num += weight
        return num / den if den else float("nan")

    point = estimate(by_stratum)
    rng = np.random.default_rng(seed)
    boots = []
    keys = sorted(by_stratum)
    for _ in range(n_bootstrap):
        sampled: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for key in keys:
            vals = by_stratum[key]
            idx = rng.integers(0, len(vals), size=len(vals))
            sampled[key] = [vals[int(i)] for i in idx]
        boots.append(estimate(sampled))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"fraction": float(point), "ci95_low": float(lo), "ci95_high": float(hi)}


def summarize_llm(rows: list[dict[str, Any]], population_rows: list[EventRow] | None = None) -> dict[str, Any]:
    n = len(rows)
    out: dict[str, Any] = {"n": n}
    for attr in ["llm_category", "llm_substantive", "llm_instruction_relevant", "llm_content_or_reasoning_relevant", "llm_confidence"]:
        counts = Counter(str(row.get(attr)) for row in rows)
        out[attr] = []
        for key, count in counts.most_common():
            lo, hi = _proportion_ci(count, n)
            out[attr].append({"value": key, "n": count, "fraction": count / n if n else 0, "ci95_low": lo, "ci95_high": hi})
    by_pos: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_pos[str(row.get("position_bin"))].append(row)
        by_prompt[str(row.get("prompt_category"))].append(row)
    out["substantive_by_position_bin"] = {
        key: _bool_rate(vals, "llm_substantive") for key, vals in sorted(by_pos.items())
    }
    out["substantive_by_prompt_category"] = {
        key: _bool_rate(vals, "llm_substantive") for key, vals in sorted(by_prompt.items())
    }
    if population_rows:
        population_counts = Counter(_sample_stratum_key(row) for row in population_rows)
        sample_counts = Counter(_sample_stratum_key(row) for row in rows)
        weighted: dict[str, list[dict[str, Any]]] = {}
        for field in [
            "llm_category",
            "llm_substantive",
            "llm_instruction_relevant",
            "llm_content_or_reasoning_relevant",
        ]:
            values = sorted({str(row.get(field)) for row in rows})
            weighted[field] = []
            for value in values:
                effect = _weighted_fraction_ci(
                    rows,
                    field=field,
                    value=value,
                    population_counts=population_counts,
                    sample_counts=sample_counts,
                )
                weighted[field].append({"value": value, **effect})
            weighted[field].sort(key=lambda row: row["fraction"], reverse=True)
        out["population_weighted"] = weighted
    return out


def _bool_rate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(bool(row.get(field)) for row in rows)
    lo, hi = _proportion_ci(k, n)
    return {"n": n, "count": k, "fraction": k / n if n else 0, "ci95_low": lo, "ci95_high": hi}


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(summary: dict[str, Any], llm_summary: dict[str, Any] | None, out_path: Path) -> None:
    bools = summary["boolean_support"]
    pair_rows = summary["by_pair_kind"][:8]
    pos_rows = summary["by_position_bin"]
    prompt_rows = summary["by_prompt_category"]

    def pct(row: dict[str, Any]) -> str:
        return f"{100 * row['fraction']:.1f}%"

    def pct_ci(row: dict[str, Any]) -> str:
        if "ci95_low" not in row:
            return pct(row)
        return f"{100 * row['fraction']:.1f}% [{100 * row['ci95_low']:.1f}%, {100 * row['ci95_high']:.1f}%]"

    lines = [
        "# First-Divergence Token Support Audit",
        "",
        f"Full deterministic support: `{summary['n']}` Dense-5 first-divergence events.",
        "",
        "## Deterministic Support",
        "",
        "| Metric | Fraction | n |",
        "|---|---:|---:|",
    ]
    for key in [
        "position_0",
        "position_ge_3",
        "position_ge_5",
        "both_surface_format",
        "any_content_token",
        "any_format_token",
        "content_or_reasoning_prompt",
    ]:
        row = bools[key]
        lines.append(f"| {key} | {pct(row)} | `{row['n']}` |")
    lines.extend(["", "Top deterministic pair kinds:", "", "| Pair kind | Fraction | n |", "|---|---:|---:|"])
    for row in pair_rows:
        lines.append(f"| {row['value']} | {pct(row)} | `{row['n']}` |")
    lines.extend(["", "Prompt categories:", "", "| Prompt category | Fraction | n |", "|---|---:|---:|"])
    for row in prompt_rows:
        lines.append(f"| {row['value']} | {pct(row)} | `{row['n']}` |")
    lines.extend(["", "Position bins:", "", "| Position bin | Fraction | n |", "|---|---:|---:|"])
    for row in pos_rows:
        lines.append(f"| {row['value']} | {pct(row)} | `{row['n']}` |")
    if llm_summary:
        lines.extend(["", "## LLM-Judged Sample", "", f"Judged sample: `{llm_summary['n']}` records.", ""])
        weighted = llm_summary.get("population_weighted") or {}
        if weighted:
            lines.extend(["Population-weighted estimates by prompt/position/pair-kind stratum:", "", "| LLM category | Fraction |", "|---|---:|"])
            for row in weighted.get("llm_category", []):
                lines.append(f"| {row['value']} | {pct_ci(row)} |")
            lines.extend(["", "| LLM flag | Fraction |", "|---|---:|"])
            for field, label in [
                ("llm_substantive", "Substantive"),
                ("llm_instruction_relevant", "Instruction-relevant"),
                ("llm_content_or_reasoning_relevant", "Content/reasoning-relevant"),
            ]:
                true_row = next((row for row in weighted.get(field, []) if row["value"] == "True"), None)
                if true_row:
                    lines.append(f"| {label} | {pct_ci(true_row)} |")
            lines.extend(["", "Unweighted judged-sample counts:", ""])
        lines.extend(["| LLM category | Fraction | n |", "|---|---:|---:|"])
        for row in llm_summary["llm_category"]:
            lines.append(f"| {row['value']} | {pct(row)} | `{row['n']}` |")
        lines.extend(["", "| LLM flag | Fraction | n |", "|---|---:|---:|"])
        for field, label in [
            ("llm_substantive", "Substantive"),
            ("llm_instruction_relevant", "Instruction-relevant"),
            ("llm_content_or_reasoning_relevant", "Content/reasoning-relevant"),
        ]:
            true_row = next((row for row in llm_summary[field] if row["value"] == "True"), None)
            if true_row:
                lines.append(f"| {label} | {pct(true_row)} | `{true_row['n']}` |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--llm-sample-size", type=int, default=750)
    parser.add_argument("--llm-batch-size", type=int, default=15)
    parser.add_argument("--llm-model", default="gpt-5.5")
    parser.add_argument("--llm-fallback-models", nargs="*", default=["gpt-5.4", "gpt-5.4-mini", "gpt-4.1"])
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    out_dir = args.out_dir or (DEFAULT_OUT_DIR / f"token_support_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_events(exp20_root=args.exp20_root, dataset=args.dataset, prompt_mode=args.prompt_mode, models=args.models)
    event_dicts = [asdict(row) for row in rows]
    write_csv(event_dicts, out_dir / "first_divergence_events.csv")
    (out_dir / "first_divergence_events.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in event_dicts) + "\n",
        encoding="utf-8",
    )
    summary = deterministic_summary(rows)
    llm_summary = None
    if args.llm:
        llm_rows = run_llm_judge(
            rows,
            out_dir=out_dir,
            sample_size=args.llm_sample_size,
            seed=args.seed,
            model=args.llm_model,
            fallback_models=args.llm_fallback_models,
            batch_size=args.llm_batch_size,
            max_retries=args.max_retries,
        )
        write_csv(llm_rows, out_dir / "llm_judgments.csv")
        llm_summary = summarize_llm(llm_rows, rows)
    output = {
        "experiment": "first_divergence_token_support",
        "exp20_root": str(args.exp20_root),
        "dataset": str(args.dataset),
        "prompt_mode": args.prompt_mode,
        "models": args.models,
        "deterministic": summary,
        "llm": llm_summary,
    }
    (out_dir / "summary.json").write_text(json.dumps(output, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_report(summary, llm_summary, out_dir / "token_support_report.md")
    print(f"[token-support] wrote {out_dir}")


if __name__ == "__main__":
    main()
