#!/usr/bin/env python3
"""Merge early-swap augmentation rows into Exp20 validation records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
PROMPT_MODES = ["raw_shared", "native"]


def _iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _merge_record(base: dict, augment: dict) -> dict:
    for kind, aug_readout in (augment.get("readouts") or {}).items():
        if not isinstance(aug_readout, dict):
            continue
        base_readout = ((base.get("readouts") or {}).get(kind) or {})
        if not isinstance(base_readout, dict):
            continue
        base_readout.setdefault("condition_token_at_step", {}).update(
            aug_readout.get("condition_token_at_step") or {}
        )
        base_readout.setdefault("conditions", {}).update(aug_readout.get("conditions") or {})
    conditions = list(base.get("validation_conditions") or [])
    for name in ["D_early_ptswap", "D_earlymid_ptswap"]:
        if name not in conditions:
            conditions.append(name)
    base["validation_conditions"] = conditions
    return base


def merge_one(base_path: Path, aug_path: Path, out_path: Path) -> dict:
    aug = {str(row.get("prompt_id")): row for row in _iter_jsonl(aug_path)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    missing_aug = 0
    with out_path.open("w") as fout:
        for base in _iter_jsonl(base_path):
            prompt_id = str(base.get("prompt_id"))
            aug_row = aug.get(prompt_id)
            if aug_row is None:
                missing_aug += 1
                merged = base
            else:
                merged = _merge_record(base, aug_row)
            fout.write(json.dumps(merged) + "\n")
            count += 1
    return {"count": count, "missing_aug": missing_aug}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--augment-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--prompt-modes", nargs="+", default=PROMPT_MODES)
    args = parser.parse_args()

    summary = {"base_root": str(args.base_root), "augment_root": str(args.augment_root), "out_root": str(args.out_root), "counts": {}, "missing": [], "ok": True}
    for mode in args.prompt_modes:
        for model in args.models:
            base = args.base_root / mode / model / "exp20_validation_records.jsonl"
            aug = args.augment_root / mode / model / "exp20_validation_early_records.jsonl"
            out = args.out_root / mode / model / "exp20_validation_records.jsonl"
            if not base.exists() or not aug.exists():
                summary["missing"].append({"mode": mode, "model": model, "base_exists": base.exists(), "augment_exists": aug.exists()})
                continue
            result = merge_one(base, aug, out)
            summary["counts"][f"{mode}/{model}"] = result
            if result["count"] != 600 or result["missing_aug"]:
                summary["missing"].append({"mode": mode, "model": model, **result})
    summary["ok"] = not summary["missing"]
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "early_merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["ok"]:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
