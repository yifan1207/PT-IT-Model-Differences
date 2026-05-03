from __future__ import annotations

from collections import Counter

from src.poc.cross_model.utils import load_dataset
from src.poc.exp15_symmetric_behavioral_causality.dataset import build_exp15_core_subset


def test_build_exp15_core_subset_counts():
    records = load_dataset("data/eval_dataset_v2.jsonl")
    subset = build_exp15_core_subset(records)

    assert len(subset) == 600
    assert len({record["id"] for record in subset}) == 600

    category_counts = Counter(record["category"] for record in subset)
    assert category_counts == {
        "GOV-CONV": 125,
        "GOV-REGISTER": 100,
        "SAFETY": 150,
        "GOV-FORMAT": 100,
        "CONTENT-FACT": 60,
        "CONTENT-REASON": 40,
        "BASELINE-EASY": 25,
    }

    assert sum(1 for record in subset if record["exp15_assistant_facing"]) == 225
    assert sum(1 for record in subset if record["exp15_conversation_source"]) == 100
    assert sum(1 for record in subset if record["exp15_safety_benign"]) == 75
    assert sum(1 for record in subset if record["exp15_safety_harmful"]) == 75
