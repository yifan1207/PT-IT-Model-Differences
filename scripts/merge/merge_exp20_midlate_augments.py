#!/usr/bin/env python3
"""Merge Exp20 mid+late augmentation rows into full Exp20 records."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp20_divergence_token_counterfactual.metrics import (
    CONDITION_ORDER,
    pairwise_agreement,
    summarize_token_clusters,
)

MIDLATE_CONDITIONS = ["B_midlate_raw", "D_midlate_ptswap"]


def _iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            yield json.loads(raw.decode("utf-8", errors="ignore"))


def _load_augments(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for payload in _iter_jsonl(path):
        prompt_id = str(payload.get("prompt_id", ""))
        if prompt_id and prompt_id not in rows:
            rows[prompt_id] = payload
    return rows


def _recompute_sequence_summaries(record: dict[str, Any]) -> None:
    free_runs = record.get("free_runs") or {}
    tokens_by_condition = {
        condition: (free_runs.get(condition) or {}).get("generated_token_ids", [])
        for condition in CONDITION_ORDER
        if isinstance(free_runs.get(condition), dict)
    }
    max_new = int(record.get("max_new_tokens") or 128)
    record["pairwise_agreement"] = {
        f"{a}__{b}": pairwise_agreement(tokens_by_condition[a], tokens_by_condition[b], max_len=max_new)
        for idx, a in enumerate(CONDITION_ORDER)
        for b in CONDITION_ORDER[idx + 1 :]
        if a in tokens_by_condition and b in tokens_by_condition
    }
    record["cluster_summary"] = summarize_token_clusters(tokens_by_condition, max_len=max_new)


def _merge_one(base: dict[str, Any], augment: dict[str, Any]) -> dict[str, Any]:
    free_runs = base.setdefault("free_runs", {})
    for condition in MIDLATE_CONDITIONS:
        payload = (augment.get("free_runs") or {}).get(condition)
        if isinstance(payload, dict):
            free_runs[condition] = payload

    base_readouts = base.setdefault("readouts", {})
    for kind, aug_readout in (augment.get("readouts") or {}).items():
        if not isinstance(aug_readout, dict):
            continue
        dst = base_readouts.setdefault(kind, {})
        if not isinstance(dst, dict):
            dst = {}
            base_readouts[kind] = dst
        dst_token_class = dst.setdefault("condition_token_at_step", {})
        for condition in MIDLATE_CONDITIONS:
            payload = (aug_readout.get("condition_token_at_step") or {}).get(condition)
            if isinstance(payload, dict):
                dst_token_class[condition] = payload
        dst_conditions = dst.setdefault("conditions", {})
        for condition in MIDLATE_CONDITIONS:
            payload = (aug_readout.get("conditions") or {}).get(condition)
            if isinstance(payload, dict):
                dst_conditions[condition] = payload

    _recompute_sequence_summaries(base)
    return base


def merge_roots(
    *,
    baseline_root: Path,
    augment_root: Path,
    out_root: Path,
    models: list[str],
    prompt_modes: list[str],
    expected_per_model: int,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "baseline_root": str(baseline_root),
        "augment_root": str(augment_root),
        "out_root": str(out_root),
        "counts": {},
        "missing": [],
        "augment_missing_prompts": {},
        "condition_missing_counts": {},
    }
    for mode in prompt_modes:
        for model in models:
            base_path = baseline_root / mode / model / "exp20_records.jsonl"
            aug_path = augment_root / mode / model / "exp20_midlate_records.jsonl"
            if not base_path.exists():
                summary["missing"].append(str(base_path))
                continue
            augments = _load_augments(aug_path)
            if not augments:
                summary["missing"].append(str(aug_path))
                continue

            out_dir = out_root / mode / model
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "exp20_records.jsonl"
            count = 0
            missing_aug = 0
            missing_conditions: Counter[str] = Counter()
            with out_path.open("w") as fout:
                for base in _iter_jsonl(base_path):
                    prompt_id = str(base.get("prompt_id", ""))
                    augment = augments.get(prompt_id)
                    if augment is None:
                        missing_aug += 1
                        merged = base
                    else:
                        merged = _merge_one(base, augment)
                    for condition in MIDLATE_CONDITIONS:
                        if condition not in (merged.get("free_runs") or {}):
                            missing_conditions[condition] += 1
                    fout.write(json.dumps(merged) + "\n")
                    count += 1

            key = f"{mode}/{model}"
            summary["counts"][key] = count
            if expected_per_model and count != expected_per_model:
                summary["missing"].append(f"{key}: {count}/{expected_per_model}")
            if missing_aug:
                summary["augment_missing_prompts"][key] = missing_aug
            if missing_conditions:
                summary["condition_missing_counts"][key] = dict(missing_conditions)

    summary["ok"] = (
        not summary["missing"]
        and not summary["augment_missing_prompts"]
        and not summary["condition_missing_counts"]
    )
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "midlate_merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--augment-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"])
    parser.add_argument("--prompt-modes", nargs="+", default=["raw_shared", "native"])
    parser.add_argument("--expected-per-model", type=int, default=600)
    args = parser.parse_args()
    summary = merge_roots(
        baseline_root=args.baseline_root,
        augment_root=args.augment_root,
        out_root=args.out_root,
        models=args.models,
        prompt_modes=args.prompt_modes,
        expected_per_model=args.expected_per_model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
