"""Tests for exp6 governance benchmark scorers."""
from __future__ import annotations

import pytest
from src.poc.exp6.runtime import GeneratedSample6
from src.poc.exp6.benchmarks.governance import (
    score_structural_token_ratio,
    score_turn_structure,
    score_format_compliance,
    _structural_token_ratio,
    _count_turn_structures,
    _check_format_compliance,
)


def _sample(text: str, category: str = "GOV-FORMAT") -> GeneratedSample6:
    return GeneratedSample6(
        record_id="test_0001",
        prompt="test prompt",
        generated_text=text,
        generated_tokens=[{"token_id": 0, "token_str": t} for t in text.split()],
        category=category,
    )


def _rec(category: str = "GOV-FORMAT", expected_format: str | None = None) -> dict:
    return {
        "id": "test_0001",
        "category": category,
        "expected_format": expected_format,
    }


# ── structural_token_ratio ────────────────────────────────────────────────────

def test_structural_token_ratio_empty():
    assert _structural_token_ratio("") == 0.0


def test_structural_token_ratio_numbered_list():
    text = "1. First item 2. Second item 3. Third item"
    ratio = _structural_token_ratio(text)
    assert ratio > 0.0, "Numbered list should have structural tokens"


def test_structural_token_ratio_plain():
    text = "The elephant is a large mammal that lives in Africa and Asia."
    ratio = _structural_token_ratio(text)
    assert ratio < 0.5, "Plain sentence should have low structural ratio"


def test_structural_token_ratio_discourse_heavy():
    text = "However therefore finally in summary first second third"
    ratio = _structural_token_ratio(text)
    assert ratio > 0.0, "Discourse-heavy text should have governance tokens"


def test_score_structural_token_ratio_scorer():
    records = [_rec()]
    outputs = [_sample("1. Answer: The answer is yes. However, we should note.")]
    result = score_structural_token_ratio(records, outputs)
    assert result.benchmark == "structural_token_ratio"
    assert result.metric == "ratio"
    assert 0.0 <= result.value <= 1.0
    assert result.n == 1


# ── turn_structure ────────────────────────────────────────────────────────────

def test_turn_structure_empty():
    assert _count_turn_structures("") == 0


def test_turn_structure_numbered_list():
    text = "1. Item one\n2. Item two\n3. Item three"
    assert _count_turn_structures(text) >= 1


def test_turn_structure_markdown_header():
    text = "## Introduction\nThis is the intro.\n## Conclusion\nThis is the end."
    assert _count_turn_structures(text) >= 1


def test_turn_structure_multiple():
    text = "## Section\n1. First\n- Bullet\n```python\ncode\n```"
    count = _count_turn_structures(text)
    assert count >= 3, f"Expected ≥3 structures, got {count}"


def test_turn_structure_plain_prose():
    text = "The quick brown fox jumps over the lazy dog and runs away quickly."
    assert _count_turn_structures(text) == 0


def test_score_turn_structure_scorer():
    records = [_rec()]
    outputs = [_sample("1. First\n2. Second\n## Header")]
    result = score_turn_structure(records, outputs)
    assert result.benchmark == "turn_structure"
    assert result.value >= 1.0


# ── format_compliance ─────────────────────────────────────────────────────────

def test_format_compliance_json_valid():
    assert _check_format_compliance('{"name": "Alice", "age": 30}', "json") == 1.0


def test_format_compliance_json_invalid():
    assert _check_format_compliance("This is not JSON at all.", "json") == 0.0


def test_format_compliance_json_in_fence():
    text = '```json\n{"key": "value"}\n```'
    assert _check_format_compliance(text, "json") == 1.0


def test_format_compliance_numbered_list():
    text = "1. First item\n2. Second item\n3. Third item"
    assert _check_format_compliance(text, "numbered_list") == 1.0


def test_format_compliance_numbered_list_fail():
    text = "The answer is that these are the items."
    assert _check_format_compliance(text, "numbered_list") == 0.0


def test_format_compliance_markdown():
    text = "## Introduction\nThis is the intro section.\n## Conclusion"
    assert _check_format_compliance(text, "markdown") == 1.0


def test_format_compliance_code_block():
    text = "Here is the code:\n```python\nprint('hello')\n```"
    assert _check_format_compliance(text, "code_block") == 1.0


def test_format_compliance_no_format():
    assert _check_format_compliance("anything", None) is None


def test_score_format_compliance_non_gov_format():
    """Records without expected_format should yield nan with n=0."""
    records = [_rec(category="GOV-CONV", expected_format=None)]
    outputs = [_sample("Some response")]
    result = score_format_compliance(records, outputs)
    assert result.n == 0
    import math
    assert math.isnan(result.value)


def test_score_format_compliance_gov_format():
    records = [_rec(category="GOV-FORMAT", expected_format="numbered_list")]
    outputs = [_sample("1. First\n2. Second")]
    result = score_format_compliance(records, outputs)
    assert result.n == 1
    assert result.value == 1.0
