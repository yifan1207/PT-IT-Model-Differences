"""Governance benchmark scorers v2 for eval_dataset_v2.

New scorers:
  score_mmlu_forced_choice   — CONTENT-FACT with formats["C"], extract first char
  score_format_compliance_v2 — GOV-FORMAT using compliance_criteria dict (not regex)

Removed vs v1:
  turn_structure — broken (PT scored higher than IT), measuring length not quality

These are the in-loop programmatic scorers. The LLM judge (G1/G2/S1/S2) is
handled separately by scripts/llm_judge.py.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from src.poc.exp05_corrective_direction_ablation_cartography.benchmarks.custom import BenchmarkResult
from src.poc.exp05_corrective_direction_ablation_cartography.utils import safe_mean
from src.poc.exp06_corrective_direction_steering.runtime import GeneratedSample6


# ── Regex helpers ─────────────────────────────────────────────────────────────

_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_MD_HEADER_RE = re.compile(r"^#{1,4}\s+\S", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```")
_BOLD_RE = re.compile(r"\*\*\S.*?\*\*", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _count_sentences(text: str) -> int:
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return len(parts)


# ── Compliance checker v2 (criteria-dict based) ───────────────────────────────

def _check_compliance_v2(text: str, criteria: dict) -> Optional[float]:
    """Check whether text satisfies compliance_criteria from eval_dataset_v2.

    Returns 1.0 (compliant), 0.0 (non-compliant), or None (unscoreable).
    """
    if not criteria:
        return None
    ctype = criteria.get("type", "other")
    params = criteria.get("params", {})
    t = text.strip()

    if ctype == "other" or not t:
        return None

    if ctype == "bullet_list":
        min_count = params.get("min_count", 1)
        bullets = _BULLET_RE.findall(t)
        return 1.0 if len(bullets) >= max(min_count, 1) else 0.0

    if ctype == "numbered_list":
        min_count = params.get("min_count", 1)
        items = _NUMBERED_LIST_RE.findall(t)
        return 1.0 if len(items) >= max(min_count, 1) else 0.0

    if ctype == "json":
        try:
            json.loads(t)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            # Try stripping markdown fences
            inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.DOTALL).strip()
            try:
                json.loads(inner)
                return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0

    if ctype == "markdown_sections":
        # Requires markdown headers (# or ##)
        min_count = params.get("min_count", 2)
        headers = _MD_HEADER_RE.findall(t)
        return 1.0 if len(headers) >= min_count else 0.0

    if ctype == "highlighted_sections":
        # Requires **bold** sections (markdown highlights)
        min_count = params.get("min_count", 1)
        highlights = _BOLD_RE.findall(t)
        return 1.0 if len(highlights) >= min_count else 0.0

    if ctype == "code_block":
        return 1.0 if _CODE_BLOCK_RE.search(t) else 0.0

    if ctype == "no_commas":
        return 1.0 if "," not in t else 0.0

    if ctype == "all_lowercase":
        alpha = [c for c in t if c.isalpha()]
        if not alpha:
            return None
        return 1.0 if all(c.islower() for c in alpha) else 0.0

    if ctype == "all_uppercase":
        alpha = [c for c in t if c.isalpha()]
        if not alpha:
            return None
        return 1.0 if all(c.isupper() for c in alpha) else 0.0

    if ctype == "word_count_min":
        min_w = params.get("min", 50)
        # Skip if min > 150 (generation cap is ~200 tokens → ~150-180 words;
        # higher thresholds penalize truncation, not IT/PT behavior)
        if min_w > 150:
            return None
        return 1.0 if len(t.split()) >= min_w else 0.0

    if ctype == "word_count_max":
        max_w = params.get("max", 200)
        return 1.0 if len(t.split()) <= max_w else 0.0

    if ctype == "sentence_count_min":
        min_s = params.get("min", 3)
        return 1.0 if _count_sentences(t) >= min_s else 0.0

    if ctype == "sentence_count_max":
        max_s = params.get("max", 5)
        return 1.0 if _count_sentences(t) <= max_s else 0.0

    if ctype == "wrapped_in_quotes":
        return 1.0 if (t.startswith('"') and t.endswith('"')) or \
                      (t.startswith("'") and t.endswith("'")) else 0.0

    if ctype == "starts_with":
        prefix = params.get("prefix", "").strip().lower()
        if not prefix:
            return None
        return 1.0 if t.lower().startswith(prefix.lower()) else 0.0

    if ctype == "ends_with":
        suffix = params.get("suffix", "").strip().lower()
        if not suffix:
            return None
        return 1.0 if t.lower().endswith(suffix.lower()) else 0.0

    if ctype == "contains_all_keywords":
        keywords = params.get("keywords", [])
        if not keywords:
            return None
        t_lower = t.lower()
        return 1.0 if all(kw.lower() in t_lower for kw in keywords) else 0.0

    if ctype == "contains_phrase":
        phrase = params.get("phrase", "").strip()
        if not phrase:
            return None
        return 1.0 if phrase.lower() in t.lower() else 0.0

    if ctype == "has_placeholders":
        min_count = params.get("min_count", 1)
        # Match [PLACEHOLDER], [placeholder], [thing to fill in], etc.
        placeholders = re.findall(r"\[[^\[\]\n]{1,50}\]", t)
        return 1.0 if len(placeholders) >= min_count else 0.0

    # Unknown type
    return None


# ── Scorers ───────────────────────────────────────────────────────────────────

def score_mmlu_forced_choice(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Score forced-choice MMLU accuracy by extracting the first letter A/B/C/D.

    Replaces mmlu_accuracy which used regex extraction from free-form generation.
    Inputs should be generated with formats["C"] and max_new_tokens=3.

    Requires eval_dataset_v2 records (category == "CONTENT-FACT" with expected_answer
    being a single uppercase letter A/B/C/D).
    """
    rec_by_id = {r["id"]: r for r in records}
    scores: list[float] = []
    for out in outputs:
        rec = rec_by_id.get(out.record_id)
        if rec is None:
            continue
        expected = str(rec.get("expected_answer", "")).strip().upper()
        if expected not in "ABCD" or len(expected) != 1:
            continue
        # Extract first letter A/B/C/D from generated text
        gen = out.generated_text.strip()
        first_letter = ""
        for ch in gen:
            if ch.upper() in "ABCD":
                first_letter = ch.upper()
                break
        score = 1.0 if first_letter == expected else 0.0
        scores.append(score)

    return BenchmarkResult("mmlu_forced_choice", "accuracy", safe_mean(scores), len(scores))


def score_format_compliance_v2(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Score format compliance using compliance_criteria dicts from eval_dataset_v2.

    Replaces format_compliance which used regex to infer constraint from prompt text.
    Requires eval_dataset_v2 records (category == "GOV-FORMAT" with compliance_criteria).

    Only records with criteria type != "other" are scored.
    Target: N >= 200 scoreable (193 scoreable in v2 dataset).
    """
    rec_by_id = {r["id"]: r for r in records}
    scores: list[float] = []
    n_skipped = 0
    for out in outputs:
        rec = rec_by_id.get(out.record_id)
        if rec is None:
            continue
        criteria = rec.get("compliance_criteria")
        if not criteria or criteria.get("type", "other") == "other":
            n_skipped += 1
            continue
        result = _check_compliance_v2(out.generated_text, criteria)
        if result is None:
            n_skipped += 1
            continue
        scores.append(result)

    return BenchmarkResult("format_compliance_v2", "compliance_rate",
                           safe_mean(scores), len(scores))
