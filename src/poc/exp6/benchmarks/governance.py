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


# ── Coherent assistant response detector ──────────────────────────────────────

_QA_CONTINUATION_RE = re.compile(
    r'\n\s*(Question|Q)\s*:',
    re.IGNORECASE,
)
_ANSWER_CONTINUATION_RE = re.compile(
    r'\n\s*Answer\s*:',
    re.IGNORECASE,
)


def _coherent_assistant_response(text: str) -> float:
    """Return 1.0 if response is an IT-style assistant response, 0.0 if PT-style.

    PT models continue the input pattern by adding new Q/A pairs and never
    "breaking out" to address the user.  IT models give a single coherent reply
    and stop.  This distinction is the core of output governance.

    Scoring:
      0  — empty / too short
      0  — character-level repetition (garbage like "Varise\\nVarise\\n..." or "\\\\\\\\")
      0  — Q/A continuation pattern (new "Question:…" or "Answer:" lines appear in output)
      1  — everything else: coherent single response
    """
    stripped = text.strip()
    if not stripped or len(stripped.split()) < 3:
        return 0.0

    # Detect character/token repetition (≥30% of words are the same word)
    words = stripped.split()
    if len(words) >= 8:
        most_common_count = max(words.count(w) for w in set(words))
        if most_common_count / len(words) > 0.30:
            return 0.0

    # Detect near-zero unique chars (e.g. all backslashes, all same letter)
    no_space = stripped.replace(" ", "").replace("\n", "")
    if len(no_space) > 15 and len(set(no_space)) < 6:
        return 0.0

    # Detect Q/A document-continuation (PT mode)
    if _QA_CONTINUATION_RE.search(stripped):
        return 0.0
    if _ANSWER_CONTINUATION_RE.search(stripped):
        return 0.0

    return 1.0


def score_coherent_assistant_rate(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Fraction of outputs that are coherent single assistant responses vs. PT-style completions.

    IT baseline expected ~0.85–0.95 (almost always responds as assistant).
    PT baseline expected ~0.05–0.15 (almost always continues the Q/A pattern).

    This is the primary IT-vs-PT governance separator: deterministic, no LLM judge,
    large effect size.
    """
    scores = [_coherent_assistant_response(out.generated_text) for out in outputs]
    return BenchmarkResult("coherent_assistant_rate", "rate", safe_mean(scores), len(scores))


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
# Detects the format constraint from the prompt text (IFEval-style), not from
# pre-assigned metadata (which was wrong for most records).

_WC_AT_LEAST_RE  = re.compile(r"at least (\d+) words?", re.IGNORECASE)
_WC_AT_MOST_RE   = re.compile(r"(?:at most|no more than|fewer than|under) (\d+) words?", re.IGNORECASE)
_WC_EXACTLY_RE   = re.compile(r"exactly (\d+) words?", re.IGNORECASE)
_STARTS_WITH_RE  = re.compile(r'start (?:your )?(?:response|reply|answer|essay|letter|email) with ["\']?([^"\'\n]{3,40})["\']?', re.IGNORECASE)
_FORBIDDEN_RE    = re.compile(r"do not (?:use|include|mention|say|write|contain)\s+(?:the words? |the phrase[s]? )?['\"]?([^'\"\n,;\.]{2,40})['\"]?", re.IGNORECASE)


def _detect_format_from_prompt(prompt: str) -> tuple[str, object] | tuple[None, None]:
    """Detect format constraint from prompt text.

    Returns (constraint_type, constraint_value):
      ("json", None), ("bullet_list", None), ("markdown", None),
      ("numbered_list", None), ("code_block", None), ("table", None),
      ("word_count_min", N), ("word_count_max", N), ("word_count_exact", N),
      ("starts_with", prefix), ("forbidden", word),
      or (None, None) if no constraint detected.

    Checks in priority order so the most specific match wins.
    """
    p = prompt.lower()
    full = prompt  # for case-sensitive checks
    # Format type checks
    if re.search(r'\bjson\b', p):
        return "json", None
    if re.search(r'bullet[- ]?point|bulleted list|in bullets|as bullets', p):
        return "bullet_list", None
    if re.search(r'\bmarkdown\b', p):
        return "markdown", None
    if re.search(r'\bnumbered list\b|number the\b', p):
        return "numbered_list", None
    if re.search(r'code block|```|in (?:python|javascript|java|c\+\+|bash) code', p):
        return "code_block", None
    if re.search(r'\btable\b|\btabular\b', p):
        return "table", None
    # Word count
    m = _WC_EXACTLY_RE.search(full)
    if m:
        return "word_count_exact", int(m.group(1))
    m = _WC_AT_LEAST_RE.search(full)
    if m:
        return "word_count_min", int(m.group(1))
    m = _WC_AT_MOST_RE.search(full)
    if m:
        return "word_count_max", int(m.group(1))
    # Starts with
    m = _STARTS_WITH_RE.search(full)
    if m:
        return "starts_with", m.group(1).strip().lower()
    # Forbidden word
    m = _FORBIDDEN_RE.search(full)
    if m:
        return "forbidden", m.group(1).strip().lower()
    return None, None


def _check_format_compliance(text: str, prompt: str) -> float | None:
    """Check whether output complies with the format constraint detected from the prompt.

    Returns 1.0 (compliant), 0.0 (non-compliant), or None (no detectable constraint).
    Covers: json, bullet_list, markdown, numbered_list, code_block, table,
            word_count_min/max/exact, starts_with, forbidden.

    This replaces the old metadata-based checker — detects constraint from prompt text,
    which is reliable (unlike the pre-assigned expected_format field which was wrong for
    most IFEval records in our dataset).
    """
    ctype, cval = _detect_format_from_prompt(prompt)
    if ctype is None:
        return None
    text = text.strip()
    if not text:
        return 0.0

    if ctype == "json":
        try:
            json.loads(text); return 1.0
        except (json.JSONDecodeError, ValueError):
            inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
            try:
                json.loads(inner); return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0

    if ctype == "bullet_list":
        # Require ≥2 content bullets (≥5 words, not template-completion lines).
        # PT text-completion produces template fillers like "* This is the Nth point."
        # or "* Text for bullet 3" by continuing the example shown in the prompt.
        _TEMPLATE_BULLET_RE = re.compile(
            r"^\s*[-*•]\s+(?:"
            r"this\s+is\s+(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|"
            r"seventh|eighth|ninth|tenth|\d+(?:st|nd|rd|th)?)\s+"
            r"|text\s+for\s+bullet"
            r"|bullet\s+point\s+\d"
            r"|(?:another\s+)?example\s+bullet"
            r"|xyz|abc|opq"
            r")",
            re.IGNORECASE,
        )
        content_bullets = [
            ln for ln in text.splitlines()
            if re.match(r"^\s*[-*•]\s+\S", ln)
            and len(ln.split()) >= 5
            and not _TEMPLATE_BULLET_RE.match(ln)
        ]
        return 1.0 if len(content_bullets) >= 2 else 0.0

    if ctype == "markdown":
        # Accept any markdown element: headers, bold, code blocks, inline code
        _MD_ANY_RE = re.compile(
            r"(?:^#{1,4}\s+\S"       # ATX headers
            r"|```"                  # fenced code block
            r"|\*\*\S"              # bold **text
            r"|__\S"                # bold __text
            r"|\[.+\]\(.+\)"        # [link](url)
            r")",
            re.MULTILINE,
        )
        return 1.0 if _MD_ANY_RE.search(text) else 0.0

    if ctype == "numbered_list":
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if any(ln.startswith(("1.", "1)")) for ln in lines):
            return 1.0
        return 1.0 if _NUMBERED_LIST_RE.search(text) else 0.0

    if ctype == "code_block":
        return 1.0 if _CODE_BLOCK_RE.search(text) else 0.0

    if ctype == "table":
        # Markdown table: has | characters and at least 2 rows
        lines_with_pipe = [ln for ln in text.splitlines() if "|" in ln]
        return 1.0 if len(lines_with_pipe) >= 2 else 0.0

    if ctype == "word_count_min":
        # Skip: generation is capped at 200 tokens so longer requirements (300+) can't be met.
        # Returning None excludes this record from the compliance score.
        return None

    if ctype == "word_count_max":
        return 1.0 if len(text.split()) <= cval else 0.0

    if ctype == "word_count_exact":
        wc = len(text.split())
        return 1.0 if abs(wc - cval) <= max(2, cval * 0.05) else 0.0

    if ctype == "starts_with":
        return 1.0 if text.lower().startswith(cval) else 0.0

    if ctype == "forbidden":
        return 1.0 if cval not in text.lower() else 0.0

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
    """Binary compliance score for explicit format instructions (IFEval-style).

    Detects the format constraint directly from the prompt text — e.g. "use JSON",
    "at least 200 words", "use bullet points", "do not use the word X".
    This is more reliable than the pre-assigned expected_format metadata, which
    was incorrectly set for most IFEval records in our dataset.

    Constraint types checked: json, bullet_list, markdown, numbered_list, code_block,
    table, word_count_min/max/exact, starts_with, forbidden.

    Expected: IT ~55-65%, PT ~5-15% (IT follows explicit instructions, PT ignores them).
    This is the primary IFEval-style format governance benchmark.
    """
    scores = []
    for rec, out in zip(records, outputs):
        prompt = rec.get("formats", {}).get("B", "") or rec.get("prompt", "")
        result = _check_format_compliance(out.generated_text, prompt)
        if result is not None:
            scores.append(result)
    if not scores:
        return BenchmarkResult("format_compliance", "compliance", float("nan"), 0)
    return BenchmarkResult("format_compliance", "compliance", safe_mean(scores), len(scores))


def score_mmlu_accuracy(
    records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    """Multiple-choice accuracy on MMLU questions.

    Checks whether the generated text contains the correct answer letter (A/B/C/D).
    Only evaluates records that have an 'expected_answer' field (MMLU records).
    """
    scores = []
    for rec, out in zip(records, outputs):
        expected = rec.get("expected_answer", "").strip().upper()
        if not expected or expected not in "ABCD":
            continue
        text = out.generated_text.strip().upper()
        # Match the letter at start, after "answer:", or as standalone
        import re
        found = re.search(r'\b([ABCD])\b', text[:50])  # check first 50 chars
        if not found:
            found = re.search(r'(?:ANSWER\s*[:\-]?\s*)([ABCD])', text[:200])
        if not found:
            found = re.search(r'\b([ABCD])\b', text[:200])
        predicted = found.group(1) if found else ""
        scores.append(float(predicted == expected))
    if not scores:
        return BenchmarkResult("mmlu_accuracy", "accuracy", float("nan"), 0)
    return BenchmarkResult("mmlu_accuracy", "accuracy", safe_mean(scores), len(scores))


def evaluate_governance_benchmark(
    name: str, records: list[dict], outputs: list[GeneratedSample6]
) -> BenchmarkResult:
    if name == "structural_token_ratio":
        return score_structural_token_ratio(records, outputs)
    if name == "turn_structure":
        return score_turn_structure(records, outputs)
    if name == "format_compliance":
        return score_format_compliance(records, outputs)
    if name == "mmlu_accuracy":
        return score_mmlu_accuracy(records, outputs)
    if name == "coherent_assistant_rate":
        return score_coherent_assistant_rate(records, outputs)
    raise ValueError(f"Unknown governance benchmark: {name}")
