"""Governance-specific benchmark scorers for Exp6 (steering experiments).

These three scorers measure OUTPUT GOVERNANCE — the formatting, structure, and
register of model responses — independently of factual content correctness.

Governance scorers are evaluated alongside the exp5 scorers (factual_em,
reasoning_em, alignment_behavior) to test the specificity claim: steering
should move governance metrics without moving content metrics.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.poc.exp5.benchmarks.custom import BenchmarkResult
from src.poc.exp5.utils import safe_mean
from src.poc.exp6.runtime import GeneratedSample6


# ── Token-type classifier ─────────────────────────────────────────────────────

def _structural_token_ratio(text: str) -> float:
    """Return fraction of tokens classified as STRUCTURAL, DISCOURSE, or PUNCTUATION.

    Uses the word-level deterministic classifier from exp3 so results are
    directly comparable to §5.3 (75% of corrective mind-changes are structural).
    Text is tokenized by splitting on whitespace (word-level granularity matches
    the word_categories classifier, which reconstructs words from BPE tokens).
    """
    if not text.strip():
        return 0.0
    from src.poc.exp3.analysis.word_categories import classify_generated_tokens_by_word
    # Build synthetic token list from whitespace split (word-level proxy).
    words = text.split()
    if not words:
        return 0.0
    synthetic_tokens = [{"token_str": w + " "} for w in words]
    categories = classify_generated_tokens_by_word(synthetic_tokens)
    governance_cats = {"STRUCTURAL", "DISCOURSE", "PUNCTUATION"}
    n_gov = sum(1 for c in categories if c in governance_cats)
    return n_gov / len(categories)


# ── Turn structure patterns ───────────────────────────────────────────────────

_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s", re.MULTILINE)
_MD_HEADER_RE = re.compile(r"^#{1,4}\s+\S", re.MULTILINE)
_QA_PAIR_RE = re.compile(r"\bQuestion\s*:.*?\bAnswer\s*:", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```")
_COLON_LABEL_RE = re.compile(r"^\s*\w[\w\s]*:\s+\S", re.MULTILINE)  # "Key: value" patterns


def _count_turn_structures(text: str) -> int:
    """Count how many distinct structural formatting patterns appear in the response."""
    if not text.strip():
        return 0
    count = 0
    if _NUMBERED_LIST_RE.search(text):
        count += 1
    if _BULLET_RE.search(text):
        count += 1
    if _MD_HEADER_RE.search(text):
        count += 1
    if _QA_PAIR_RE.search(text):
        count += 1
    if _CODE_BLOCK_RE.search(text):
        count += 1
    if _COLON_LABEL_RE.search(text):
        count += 1
    return count


# ── Format compliance checker ─────────────────────────────────────────────────

def _check_format_compliance(text: str, expected_format: str | None) -> float | None:
    """Check whether the generated text complies with the expected format.

    Returns 1.0 (compliant), 0.0 (non-compliant), or None (no format expected).
    """
    if not expected_format:
        return None
    text = text.strip()
    if expected_format == "numbered_list":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if any(ln.startswith(("1.", "1)")) for ln in lines):
            return 1.0
        if any(_NUMBERED_LIST_RE.match(ln + " ") for ln in lines):
            return 1.0
        return 0.0
    if expected_format == "json":
        try:
            json.loads(text)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            # Try stripping markdown fences
            inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            try:
                json.loads(inner)
                return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0
    if expected_format == "markdown":
        return 1.0 if _MD_HEADER_RE.search(text) else 0.0
    if expected_format == "code_block":
        return 1.0 if _CODE_BLOCK_RE.search(text) else 0.0
    if expected_format == "bullet_list":
        return 1.0 if _BULLET_RE.search(text) else 0.0
    # Unknown format — skip
    return None


# ── Scorers ───────────────────────────────────────────────────────────────────

def score_structural_token_ratio(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Fraction of generated tokens classified as STRUCTURAL/DISCOURSE/PUNCTUATION.

    Core governance metric: the corrective stage preferentially modifies structural
    tokens (§5.3, 75% of corrective-stage mind-changes are structural). Steering
    governance should move this ratio; steering content features should not.
    """
    scores = [_structural_token_ratio(out.generated_text) for out in outputs]
    return BenchmarkResult("structural_token_ratio", "ratio", safe_mean(scores), len(scores))


def score_turn_structure(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Count of distinct structural formatting patterns present in the response.

    Patterns: numbered list, bullet list, markdown header, Q/A pair, code block,
    colon-label pairs. Range: 0 (unstructured) to 6 (maximally structured).

    Captures the qualitative claim that IT responses are "more structured" than PT
    in a reproducible, regex-based way without requiring an LLM judge.
    """
    scores = [float(_count_turn_structures(out.generated_text)) for out in outputs]
    return BenchmarkResult("turn_structure", "structure_count", safe_mean(scores), len(scores))


def score_format_compliance(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Binary compliance score for explicit format instructions (GOV-FORMAT records only).

    Skips records without an expected_format field (returns 0 for them but also
    tracks n so downstream plots can filter to GOV-FORMAT category).

    This is the tightest governance test: did the model actually follow "output a
    numbered list" / "output JSON" / "use markdown headers"?
    """
    scores = []
    for rec, out in zip(records, outputs):
        expected_format = rec.get("expected_format") or rec.get("metadata", {}).get("expected_format")
        result = _check_format_compliance(out.generated_text, expected_format)
        if result is not None:
            scores.append(result)
    if not scores:
        # All records lack format instructions — return NaN-like result with n=0
        return BenchmarkResult("format_compliance", "compliance", float("nan"), 0)
    return BenchmarkResult("format_compliance", "compliance", safe_mean(scores), len(scores))


def evaluate_governance_benchmark(
    name: str, records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    if name == "structural_token_ratio":
        return score_structural_token_ratio(records, outputs)
    if name == "turn_structure":
        return score_turn_structure(records, outputs)
    if name == "format_compliance":
        return score_format_compliance(records, outputs)
    raise ValueError(f"Unknown governance benchmark: {name}")
