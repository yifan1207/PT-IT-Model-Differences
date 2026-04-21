from __future__ import annotations

from collections import Counter

from src.poc.cross_model.utils import load_dataset
from src.poc.exp15.dataset import build_exp15_core_subset, build_human_audit_manifest


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


def test_build_human_audit_manifest_counts():
    records = load_dataset("data/eval_dataset_v2.jsonl")
    subset = build_exp15_core_subset(records)
    audit_rows = build_human_audit_manifest(subset)

    assert len(audit_rows) == 120
    assert len({row["audit_id"] for row in audit_rows}) == 120

    split_counts = Counter(row["audit_split"] for row in audit_rows)
    bucket_counts = Counter(row["audit_bucket"] for row in audit_rows)
    assert split_counts == {"overlap": 80, "holdout": 40}
    assert bucket_counts == {
        "conversation_source": 30,
        "gov_register": 30,
        "safety_benign": 30,
        "safety_harmful": 30,
    }

    overlap_condition_counts = Counter(row["condition"] for row in audit_rows if row["audit_split"] == "overlap")
    assert sum(overlap_condition_counts.values()) == 80
