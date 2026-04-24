#!/usr/bin/env python3
"""Create slim Exp20 baseline manifests for the mid+late augmentation run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_MODELS = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
DEFAULT_PROMPT_MODES = ["raw_shared", "native"]


def _iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            yield json.loads(raw.decode("utf-8", errors="ignore"))


def _slim_record(record: dict) -> dict:
    a_run = (record.get("free_runs") or {}).get("A_pt_raw", {})
    return {
        "prompt_id": record.get("prompt_id"),
        "model": record.get("model"),
        "prompt_mode": record.get("prompt_mode"),
        "max_new_tokens": record.get("max_new_tokens"),
        "free_runs": {
            "A_pt_raw": {
                "generated_token_ids": a_run.get("generated_token_ids", []),
            }
        },
        "divergence_events": record.get("divergence_events", {}),
    }


def create_manifests(source_root: Path, out_root: Path, models: list[str], prompt_modes: list[str]) -> dict:
    summary = {"source_root": str(source_root), "out_root": str(out_root), "counts": {}, "missing": []}
    for mode in prompt_modes:
        for model in models:
            src = source_root / mode / model / "exp20_records.jsonl"
            if not src.exists():
                summary["missing"].append(str(src))
                continue
            out_dir = out_root / mode / model
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "exp20_midlate_manifest.jsonl"
            count = 0
            with out_path.open("w") as fout:
                for record in _iter_jsonl(src):
                    fout.write(json.dumps(_slim_record(record)) + "\n")
                    count += 1
            summary["counts"][f"{mode}/{model}"] = count
    summary["ok"] = not summary["missing"]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt-modes", nargs="+", default=DEFAULT_PROMPT_MODES)
    args = parser.parse_args()
    summary = create_manifests(args.source_root, args.out_root, args.models, args.prompt_modes)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
