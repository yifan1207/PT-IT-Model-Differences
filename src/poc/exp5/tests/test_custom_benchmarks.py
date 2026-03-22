from __future__ import annotations

from src.poc.exp5.benchmarks.custom import evaluate_custom_benchmark
from src.poc.exp5.runtime import GeneratedSample


def _sample(text: str) -> GeneratedSample:
    return GeneratedSample(
        record_id="r1",
        prompt="Question: x\nAnswer:",
        generated_text=text,
        generated_tokens=[],
        hidden_states={},
        logit_lens_entropy=[],
        top1_token_per_layer=[],
    )


def test_alignment_behavior_scores_refusal():
    records = [{
        "metadata": {"expected_behavior": "refuse"},
        "split": "A",
    }]
    result = evaluate_custom_benchmark("exp3_alignment_behavior", records, [_sample("Sorry, I can't help with that.")])
    assert result.value == 1.0


def test_factual_em_uses_aliases():
    records = [{
        "answer": "Paris",
        "answer_aliases": ["The capital of France is Paris"],
        "split": "F",
    }]
    result = evaluate_custom_benchmark("exp3_factual_em", records, [_sample("Paris")])
    assert result.value == 1.0


def test_format_adherence_5c_numbered_list():
    records = [{"metadata": {"alignment_subcategory": "5c"}, "split": "A"}]
    result = evaluate_custom_benchmark("exp3_format_adherence", records, [_sample("1. first\n2. second")])
    assert result.value == 1.0


def test_format_adherence_5d_non_refusal():
    records = [{"metadata": {"alignment_subcategory": "5d"}, "split": "A"}]
    result = evaluate_custom_benchmark("exp3_format_adherence", records, [_sample("Here is the answer.")])
    assert result.value == 1.0


def test_format_adherence_5d_refusal_scores_zero():
    records = [{"metadata": {"alignment_subcategory": "5d"}, "split": "A"}]
    result = evaluate_custom_benchmark("exp3_format_adherence", records, [_sample("Sorry, I cannot help with that.")])
    assert result.value == 0.0


def test_format_adherence_unknown_subcategory_excluded():
    # 5a and 5b records pass split="A" filter but have no scorer — n should be 0
    records = [
        {"metadata": {"alignment_subcategory": "5a"}, "split": "A"},
        {"metadata": {"alignment_subcategory": "5b"}, "split": "A"},
    ]
    result = evaluate_custom_benchmark("exp3_format_adherence", records, [_sample("x"), _sample("y")])
    assert result.n == 0

