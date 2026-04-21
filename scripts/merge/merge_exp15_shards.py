#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _merge_jsonl_unique(
    shard_paths: list[Path],
    out_path: Path,
    *,
    key_fn,
    sort_key,
    allow_identical_duplicates: bool = False,
) -> list[dict[str, Any]]:
    merged: dict[Any, dict[str, Any]] = {}
    for shard_path in shard_paths:
        for row in _load_jsonl(shard_path):
            key = key_fn(row)
            if key in merged:
                if allow_identical_duplicates and merged[key] == row:
                    continue
                raise ValueError(f"Duplicate merged row for key={key} from {shard_path}")
            merged[key] = row
    rows = sorted(merged.values(), key=sort_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def _copy_first_existing(shard_paths: list[Path], out_path: Path) -> bool:
    for shard_path in shard_paths:
        if shard_path.exists():
            out_path.write_text(shard_path.read_text(encoding="utf-8"), encoding="utf-8")
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge exp15 shard outputs into a canonical merged dir.")
    parser.add_argument("--run-root", type=Path, default=Path("results/exp15/data"))
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    parent_dir = args.run_root / f"{args.run_prefix}_{args.model}"
    merged_dir = args.output_dir or (parent_dir / f"{args.run_prefix}_{args.model}")
    shard_dirs = [
        parent_dir / f"{args.run_prefix}_{args.model}_shard{idx}of{args.num_shards}"
        for idx in range(args.num_shards)
    ]
    missing = [path for path in shard_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard directories: {missing}")

    merged_dir.mkdir(parents=True, exist_ok=True)

    prompts_full_paths = [shard_dir / "prompts_full.jsonl" for shard_dir in shard_dirs]
    prompts_shard_paths = [shard_dir / "prompts_shard.jsonl" for shard_dir in shard_dirs]
    sample_paths = [shard_dir / "sample_outputs.jsonl" for shard_dir in shard_dirs]
    forced_choice_paths = [shard_dir / "forced_choice_outputs.jsonl" for shard_dir in shard_dirs]
    audit_paths = [shard_dir / "human_audit_manifest.jsonl" for shard_dir in shard_dirs]
    audit_template_paths = [shard_dir / "human_audit_template.csv" for shard_dir in shard_dirs]
    pipeline_manifest_paths = [shard_dir / "pipeline_manifest.json" for shard_dir in shard_dirs]
    config_paths = [shard_dir / "config.json" for shard_dir in shard_dirs]

    prompts_full_rows = _merge_jsonl_unique(
        prompts_full_paths,
        merged_dir / "prompts_full.jsonl",
        key_fn=lambda row: row["id"],
        sort_key=lambda row: row["id"],
        allow_identical_duplicates=True,
    )
    prompts_shard_rows = _merge_jsonl_unique(
        prompts_shard_paths,
        merged_dir / "prompts_shard.jsonl",
        key_fn=lambda row: row["id"],
        sort_key=lambda row: row["id"],
    )
    sample_rows = _merge_jsonl_unique(
        sample_paths,
        merged_dir / "sample_outputs.jsonl",
        key_fn=lambda row: (row["condition"], row["record_id"]),
        sort_key=lambda row: (row["condition"], row["record_id"]),
    )
    forced_choice_rows = _merge_jsonl_unique(
        forced_choice_paths,
        merged_dir / "forced_choice_outputs.jsonl",
        key_fn=lambda row: (row["condition"], row["record_id"]),
        sort_key=lambda row: (row["condition"], row["record_id"]),
    )
    _merge_jsonl_unique(
        audit_paths,
        merged_dir / "human_audit_manifest.jsonl",
        key_fn=lambda row: row["audit_id"],
        sort_key=lambda row: row["audit_id"],
        allow_identical_duplicates=True,
    )
    _copy_first_existing(audit_template_paths, merged_dir / "human_audit_template.csv")
    _copy_first_existing(pipeline_manifest_paths, merged_dir / "pipeline_manifest.json")

    config = {}
    for path in config_paths:
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8"))
            break
    config.update(
        {
            "num_shards_merged": args.num_shards,
            "n_prompts_shard_merged": len(prompts_shard_rows),
            "n_prompts_full": len(prompts_full_rows),
            "merged_from_parent_dir": str(parent_dir),
        }
    )
    (merged_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    sample_counts = Counter(row["condition"] for row in sample_rows)
    forced_choice_counts = Counter(row["condition"] for row in forced_choice_rows)
    summary = {
        "model": args.model,
        "run_prefix": args.run_prefix,
        "merged_dir": str(merged_dir),
        "n_prompts_full": len(prompts_full_rows),
        "n_prompts_shard_merged": len(prompts_shard_rows),
        "counts_by_condition": dict(sorted(sample_counts.items())),
        "forced_choice_counts_by_condition": dict(sorted(forced_choice_counts.items())),
    }
    (merged_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
