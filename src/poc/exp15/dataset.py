from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Callable


CONVERSATION_SOURCES = {"MT-Bench", "WildChat-style"}
SUBSET_NAME = "exp15_eval_core_600"
SUBSET_SEED = 20260420
HUMAN_AUDIT_CONDITIONS = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
]
HUMAN_AUDIT_BUCKETS = [
    "conversation_source",
    "gov_register",
    "safety_benign",
    "safety_harmful",
]


def _hash_hex(*parts: object) -> str:
    payload = "::".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sorted_by_hash(records: list[dict], *, seed: int, namespace: str) -> list[dict]:
    return sorted(records, key=lambda record: (_hash_hex(seed, namespace, record["id"]), record["id"]))


def _record_copy(record: dict) -> dict:
    return json.loads(json.dumps(record))


def is_conversation_source(record: dict) -> bool:
    return record.get("source") in CONVERSATION_SOURCES


def is_assistant_facing(record: dict) -> bool:
    return bool(record.get("exp15_assistant_facing"))


def is_safety_harmful(record: dict) -> bool:
    behavior = (
        record.get("expected_behavior")
        or record.get("metadata", {}).get("expected_behavior")
        or ""
    )
    return record.get("category") == "SAFETY" and behavior == "refuse"


def is_safety_benign(record: dict) -> bool:
    behavior = (
        record.get("expected_behavior")
        or record.get("metadata", {}).get("expected_behavior")
        or ""
    )
    return record.get("category") == "SAFETY" and behavior in {"comply", "comply_safely"}


def _tag_records(
    records: list[dict],
    *,
    bucket: str,
    assistant_facing: bool,
    subset_name: str,
    subset_seed: int,
) -> list[dict]:
    out: list[dict] = []
    for record in records:
        cloned = _record_copy(record)
        cloned["exp15_subset_name"] = subset_name
        cloned["exp15_subset_seed"] = subset_seed
        cloned["exp15_selection_bucket"] = bucket
        cloned["exp15_assistant_facing"] = assistant_facing
        cloned["exp15_conversation_source"] = is_conversation_source(record)
        cloned["exp15_safety_harmful"] = is_safety_harmful(record)
        cloned["exp15_safety_benign"] = is_safety_benign(record)
        out.append(cloned)
    return out


def _pick_exact(
    records: list[dict],
    *,
    seed: int,
    namespace: str,
    n: int,
    predicate: Callable[[dict], bool],
) -> list[dict]:
    pool = [record for record in records if predicate(record)]
    picked = _sorted_by_hash(pool, seed=seed, namespace=namespace)[:n]
    if len(picked) != n:
        raise ValueError(f"Expected {n} records for {namespace}, found {len(picked)}")
    return picked


def build_exp15_core_subset(
    all_records: list[dict],
    *,
    seed: int = SUBSET_SEED,
    subset_name: str = SUBSET_NAME,
) -> list[dict]:
    conversation = _pick_exact(
        all_records,
        seed=seed,
        namespace="conversation_source",
        n=100,
        predicate=is_conversation_source,
    )
    gov_register = _pick_exact(
        all_records,
        seed=seed,
        namespace="gov_register",
        n=100,
        predicate=lambda record: record.get("category") == "GOV-REGISTER",
    )
    safety = _pick_exact(
        all_records,
        seed=seed,
        namespace="safety_all",
        n=150,
        predicate=lambda record: record.get("category") == "SAFETY",
    )
    gov_format = _pick_exact(
        all_records,
        seed=seed,
        namespace="gov_format",
        n=100,
        predicate=lambda record: record.get("category") == "GOV-FORMAT",
    )
    content_fact = _pick_exact(
        all_records,
        seed=seed,
        namespace="content_fact",
        n=60,
        predicate=lambda record: record.get("category") == "CONTENT-FACT",
    )
    content_reason = _pick_exact(
        all_records,
        seed=seed,
        namespace="content_reason",
        n=40,
        predicate=lambda record: record.get("category") == "CONTENT-REASON",
    )
    baseline_easy = _pick_exact(
        all_records,
        seed=seed,
        namespace="baseline_easy",
        n=25,
        predicate=lambda record: record.get("category") == "BASELINE-EASY",
    )
    extra_gov_conv = _pick_exact(
        all_records,
        seed=seed,
        namespace="gov_conv_extra",
        n=25,
        predicate=lambda record: record.get("category") == "GOV-CONV" and not is_conversation_source(record),
    )

    selected = []
    selected.extend(_tag_records(conversation, bucket="conversation_source", assistant_facing=True, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(gov_register, bucket="gov_register", assistant_facing=True, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(safety, bucket="safety", assistant_facing=False, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(gov_format, bucket="gov_format", assistant_facing=False, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(content_fact, bucket="content_fact", assistant_facing=False, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(content_reason, bucket="content_reason", assistant_facing=False, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(baseline_easy, bucket="baseline_easy", assistant_facing=False, subset_name=subset_name, subset_seed=seed))
    selected.extend(_tag_records(extra_gov_conv, bucket="gov_conv_extra", assistant_facing=True, subset_name=subset_name, subset_seed=seed))

    deduped: dict[str, dict] = {}
    for record in selected:
        if record["id"] in deduped:
            raise ValueError(f"Duplicate record selected for Exp15 subset: {record['id']}")
        deduped[record["id"]] = record

    out = sorted(deduped.values(), key=lambda record: record["id"])
    if len(out) != 600:
        raise ValueError(f"Exp15 core subset should contain 600 records, found {len(out)}")
    return out


def build_human_audit_manifest(
    selected_records: list[dict],
    *,
    seed: int = SUBSET_SEED,
) -> list[dict]:
    bucket_predicates: dict[str, Callable[[dict], bool]] = {
        "conversation_source": is_conversation_source,
        "gov_register": lambda record: record.get("category") == "GOV-REGISTER",
        "safety_benign": is_safety_benign,
        "safety_harmful": is_safety_harmful,
    }
    rows: list[dict] = []
    for bucket_idx, bucket in enumerate(HUMAN_AUDIT_BUCKETS):
        pool = [record for record in selected_records if bucket_predicates[bucket](record)]
        picked = _sorted_by_hash(pool, seed=seed, namespace=f"human_audit:{bucket}")[:30]
        if len(picked) != 30:
            raise ValueError(f"Expected 30 records for human audit bucket {bucket}, found {len(picked)}")
        for local_idx, record in enumerate(picked):
            split = "overlap" if local_idx < 20 else "holdout"
            condition = HUMAN_AUDIT_CONDITIONS[(bucket_idx + local_idx) % len(HUMAN_AUDIT_CONDITIONS)]
            rows.append(
                {
                    "audit_id": f"{split}_{bucket}_{local_idx:02d}",
                    "audit_split": split,
                    "audit_bucket": bucket,
                    "record_id": record["id"],
                    "condition": condition,
                    "category": record.get("category", ""),
                    "source": record.get("source", ""),
                    "expected_behavior": (
                        record.get("expected_behavior")
                        or record.get("metadata", {}).get("expected_behavior")
                        or ""
                    ),
                }
            )
    if len(rows) != 120:
        raise ValueError(f"Expected 120 human audit rows, found {len(rows)}")
    return sorted(rows, key=lambda row: (row["audit_split"], row["audit_bucket"], row["audit_id"]))


def subset_summary(selected_records: list[dict], audit_rows: list[dict]) -> dict:
    category_counts = Counter(record.get("category", "") for record in selected_records)
    source_counts = Counter(record.get("source", "") for record in selected_records)
    bucket_counts = Counter(record.get("exp15_selection_bucket", "") for record in selected_records)
    audit_split_counts = Counter(row["audit_split"] for row in audit_rows)
    audit_bucket_counts = Counter(row["audit_bucket"] for row in audit_rows)
    return {
        "n_records": len(selected_records),
        "category_counts": dict(sorted(category_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "selection_bucket_counts": dict(sorted(bucket_counts.items())),
        "assistant_facing_count": sum(1 for record in selected_records if record.get("exp15_assistant_facing")),
        "conversation_source_count": sum(1 for record in selected_records if record.get("exp15_conversation_source")),
        "safety_benign_count": sum(1 for record in selected_records if record.get("exp15_safety_benign")),
        "safety_harmful_count": sum(1 for record in selected_records if record.get("exp15_safety_harmful")),
        "human_audit_rows": len(audit_rows),
        "human_audit_split_counts": dict(sorted(audit_split_counts.items())),
        "human_audit_bucket_counts": dict(sorted(audit_bucket_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_human_audit_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "audit_id",
        "audit_split",
        "audit_bucket",
        "record_id",
        "condition",
        "category",
        "source",
        "expected_behavior",
        "question",
        "response",
        "human_rater1_g1",
        "human_rater1_g2",
        "human_rater1_s1",
        "human_rater1_s2",
        "human_rater2_g1",
        "human_rater2_g2",
        "human_rater2_s1",
        "human_rater2_s2",
        "adjudicated_g1",
        "adjudicated_g2",
        "adjudicated_s1",
        "adjudicated_s2",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
