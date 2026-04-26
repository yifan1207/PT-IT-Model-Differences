#!/usr/bin/env python3
"""Build blinded human-evaluation survey CSVs for Exp15.

The source Exp15 audit packs are per-model and include condition names. This
script creates rater-facing CSVs with randomized order and hidden key files for
later analysis.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("results/exp15_symmetric_behavioral_causality/data")
OUT = Path("paper_draft/human_eval_survey")
MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
PRIMARY_PAIRWISE_COMPARISONS = {"it_c_vs_dlate", "pt_late_vs_a"}
PRIMARY_PAIRWISE_TASKS = {"pairwise_g2", "pairwise_s2"}
PAIRWISE_PER_MODEL_STRATUM = 60
RATER_BATCH_SIZE = 100


def model_dir(model: str) -> Path:
    outer = ROOT / f"exp15_eval_core_600_t512_dense5_{model}"
    return outer / f"exp15_eval_core_600_t512_dense5_{model}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_batches(base_dir: Path, stem: str, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    for start in range(0, len(rows), RATER_BATCH_SIZE):
        batch = rows[start : start + RATER_BATCH_SIZE]
        batch_id = start // RATER_BATCH_SIZE + 1
        write_csv(base_dir / f"{stem}_batch_{batch_id:02d}.csv", batch, fieldnames)


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def build_pointwise() -> None:
    visible_fields = [
        "survey_id",
        "rater_id",
        "item_order",
        "category",
        "source",
        "expected_behavior",
        "question",
        "response",
        "g1",
        "g2",
        "s1",
        "s2",
        "confidence",
        "notes",
    ]
    key_fields = [
        "survey_id",
        "model",
        "audit_id",
        "audit_split",
        "audit_bucket",
        "record_id",
        "condition",
        "category",
        "source",
        "expected_behavior",
    ]

    visible_base: list[dict[str, str]] = []
    hidden_key: list[dict[str, str]] = []

    for model in MODELS:
        master = read_csv(model_dir(model) / "human_audit_master.csv")
        for row in master:
            survey_id = safe_id(f"pw_{model}_{row['audit_id']}")
            visible_base.append(
                {
                    "survey_id": survey_id,
                    "category": row.get("category", ""),
                    "source": row.get("source", ""),
                    "expected_behavior": row.get("expected_behavior", ""),
                    "question": row.get("question", ""),
                    "response": row.get("response", ""),
                    "g1": "",
                    "g2": "",
                    "s1": "",
                    "s2": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            hidden_key.append(
                {
                    "survey_id": survey_id,
                    "model": model,
                    "audit_id": row.get("audit_id", ""),
                    "audit_split": row.get("audit_split", ""),
                    "audit_bucket": row.get("audit_bucket", ""),
                    "record_id": row.get("record_id", ""),
                    "condition": row.get("condition", ""),
                    "category": row.get("category", ""),
                    "source": row.get("source", ""),
                    "expected_behavior": row.get("expected_behavior", ""),
                }
            )

    write_csv(OUT / "keys" / "pointwise_hidden_key.csv", hidden_key, key_fields)

    for rater_id, seed in [("R1", 2026042601), ("R2", 2026042602)]:
        rows = [dict(row, rater_id=rater_id) for row in visible_base]
        rng = random.Random(seed)
        rng.shuffle(rows)
        for idx, row in enumerate(rows, start=1):
            row["item_order"] = str(idx)
        write_csv(OUT / "pointwise" / f"pointwise_{rater_id}.csv", rows, visible_fields)
        write_batches(OUT / "pointwise" / "batches", f"pointwise_{rater_id}", rows, visible_fields)


def build_response_lookup() -> dict[tuple[str, str, str], dict[str, str]]:
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for model in MODELS:
        with (model_dir(model) / "sample_outputs.jsonl").open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                lookup[(model, row["record_id"], row["condition"])] = {
                    "question": row.get("prompt", ""),
                    "response": row.get("generated_text", ""),
                    "category": row.get("category", ""),
                    "source": row.get("source", ""),
                    "expected_behavior": row.get("expected_behavior", ""),
                }
    return lookup


def read_pairwise_items() -> list[dict[str, str]]:
    response_lookup = build_response_lookup()
    items: list[dict[str, str]] = []

    for model in MODELS:
        path = model_dir(model) / "judge_pairwise.jsonl"
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                comparison = item.get("comparison", "")
                task = item.get("task", "")
                if comparison not in PRIMARY_PAIRWISE_COMPARISONS:
                    continue
                if task not in PRIMARY_PAIRWISE_TASKS:
                    continue

                record_id = item["record_id"]
                cond_a = item["presented_condition_a"]
                cond_b = item["presented_condition_b"]
                row_a = response_lookup.get((model, record_id, cond_a))
                row_b = response_lookup.get((model, record_id, cond_b))
                if row_a is None or row_b is None:
                    continue
                criterion = "G2" if task == "pairwise_g2" else "S2"
                items.append(
                    {
                        "pair_id": safe_id(f"pair_{model}_{item['item_id']}"),
                        "model": model,
                        "record_id": record_id,
                        "comparison": comparison,
                        "criterion": criterion,
                        "category": item.get("category", ""),
                        "question": row_a.get("question", ""),
                        "condition_a_source": cond_a,
                        "condition_b_source": cond_b,
                        "response_a_source": row_a.get("response", ""),
                        "response_b_source": row_b.get("response", ""),
                    }
                )
    return items


def stratified_pairwise_sample(items: list[dict[str, str]]) -> list[dict[str, str]]:
    strata: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for item in items:
        strata[(item["comparison"], item["criterion"], item["model"])].append(item)

    sampled: list[dict[str, str]] = []
    rng = random.Random(2026042603)
    for key in sorted(strata):
        rows = strata[key]
        rows = list(rows)
        rng.shuffle(rows)
        sampled.extend(rows[:PAIRWISE_PER_MODEL_STRATUM])
    return sampled


def build_pairwise() -> None:
    visible_fields = [
        "pair_id",
        "rater_id",
        "item_order",
        "criterion",
        "category",
        "question",
        "response_a",
        "response_b",
        "choice",
        "confidence",
        "notes",
    ]
    key_fields = [
        "pair_id",
        "rater_id",
        "model",
        "record_id",
        "comparison",
        "criterion",
        "condition_a",
        "condition_b",
        "source_condition_a",
        "source_condition_b",
    ]

    items = stratified_pairwise_sample(read_pairwise_items())
    all_keys: list[dict[str, str]] = []

    for rater_id, seed in [("R1", 2026042604), ("R2", 2026042605)]:
        rng = random.Random(seed)
        visible_rows: list[dict[str, str]] = []
        for item in items:
            flip = rng.random() < 0.5
            if flip:
                condition_a = item["condition_b_source"]
                condition_b = item["condition_a_source"]
                response_a = item["response_b_source"]
                response_b = item["response_a_source"]
            else:
                condition_a = item["condition_a_source"]
                condition_b = item["condition_b_source"]
                response_a = item["response_a_source"]
                response_b = item["response_b_source"]

            visible_rows.append(
                {
                    "pair_id": item["pair_id"],
                    "rater_id": rater_id,
                    "criterion": item["criterion"],
                    "category": item["category"],
                    "question": item["question"],
                    "response_a": response_a,
                    "response_b": response_b,
                    "choice": "",
                    "confidence": "",
                    "notes": "",
                }
            )
            all_keys.append(
                {
                    "pair_id": item["pair_id"],
                    "rater_id": rater_id,
                    "model": item["model"],
                    "record_id": item["record_id"],
                    "comparison": item["comparison"],
                    "criterion": item["criterion"],
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "source_condition_a": item["condition_a_source"],
                    "source_condition_b": item["condition_b_source"],
                }
            )

        rng.shuffle(visible_rows)
        for idx, row in enumerate(visible_rows, start=1):
            row["item_order"] = str(idx)
        write_csv(OUT / "pairwise" / f"pairwise_primary_{rater_id}.csv", visible_rows, visible_fields)
        write_batches(
            OUT / "pairwise" / "batches",
            f"pairwise_primary_{rater_id}",
            visible_rows,
            visible_fields,
        )

    write_csv(OUT / "keys" / "pairwise_hidden_key.csv", all_keys, key_fields)


def write_readme() -> None:
    text = """# Exp15 Human Evaluation Survey Packet

This directory is ready to hand to human raters.

## Rater-facing files

- `pointwise/pointwise_R1.csv` and `pointwise/pointwise_R2.csv`: 600 blinded pointwise items each.
- `pointwise/batches/`: the same pointwise surveys split into 100-item batches.
- `pairwise/pairwise_primary_R1.csv` and `pairwise/pairwise_primary_R2.csv`: 1200 blinded pairwise items each, sampled as 60 items per model for each primary comparison and criterion.
- `pairwise/batches/`: the same pairwise surveys split into 100-item batches.

## Hidden analysis keys

Do not send `keys/` files to raters. They map survey ids back to model family and experimental condition.

## Required rater fields

Pointwise: fill `g1`, `g2`, `s1`, `s2` where applicable, plus `confidence` and optional `notes`.

Pairwise: fill `choice` with `A`, `B`, `TIE`, or `BOTH_BAD`; fill `confidence` with `1`, `2`, or `3`; optional `notes`.

Use `paper_draft/human_eval_protocol.md` for full rubric and analysis plan.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text(text)


def main() -> None:
    build_pointwise()
    build_pairwise()
    write_readme()
    print(f"Wrote survey packet to {OUT}")


if __name__ == "__main__":
    main()
