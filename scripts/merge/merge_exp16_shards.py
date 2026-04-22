#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _merge_jsonl_unique(
    shard_paths: list[Path],
    out_path: Path,
    *,
    key_fn,
    sort_key,
) -> int:
    merged: dict[Any, dict[str, Any]] = {}
    for shard_path in shard_paths:
        if not shard_path.exists():
            continue
        for row in _load_jsonl(shard_path):
            key = key_fn(row)
            if key in merged:
                raise ValueError(f"Duplicate merged row for key={key} from {shard_path}")
            merged[key] = row
    rows = sorted(merged.values(), key=sort_key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


def _merge_js_layer_stats(shard_dirs: list[Path], out_path: Path) -> None:
    payloads = [
        _read_json(shard_dir / "js_layer_stats.json")
        for shard_dir in shard_dirs
        if (shard_dir / "js_layer_stats.json").exists()
    ]
    if not payloads:
        return
    first = payloads[0]
    merged: dict[str, Any] = {
        "n_layers": first.get("n_layers"),
        "corrective_onset": first.get("corrective_onset"),
        "final_region_start": first.get("final_region_start"),
        "pairs": {},
    }
    for payload in payloads:
        for field in ("n_layers", "corrective_onset", "final_region_start"):
            if payload.get(field) != merged.get(field):
                raise ValueError(f"Shard JS payload mismatch on {field}: {payload.get(field)} != {merged.get(field)}")
        for pair_name, pair_stats in payload.get("pairs", {}).items():
            dst = merged["pairs"].setdefault(
                pair_name,
                {
                    "pair_name": pair_stats["pair_name"],
                    "current_pipeline": pair_stats["current_pipeline"],
                    "baseline_pipeline": pair_stats["baseline_pipeline"],
                    "group": pair_stats["group"],
                    "current_prompt_mode": pair_stats["current_prompt_mode"],
                    "baseline_prompt_mode": pair_stats["baseline_prompt_mode"],
                    "graft_window": pair_stats.get("graft_window"),
                    "sum": [0.0] * len(pair_stats.get("sum", [])),
                    "count": [0] * len(pair_stats.get("count", [])),
                },
            )
            for idx, value in enumerate(pair_stats.get("sum", [])):
                dst["sum"][idx] += float(value)
            for idx, value in enumerate(pair_stats.get("count", [])):
                dst["count"][idx] += int(value)
    out_path.write_text(json.dumps(merged, indent=2))


def _teacher_cap_diagnostics(teacher_manifest_rows: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    lengths = [len(row.get("token_ids", [])) for row in teacher_manifest_rows]
    if not lengths:
        return {
            "n_prompts": 0,
            "max_new_tokens_asserted": int(max_new_tokens),
            "n_at_cap": 0,
            "fraction_at_cap": None,
            "mean_teacher_length": None,
            "median_teacher_length": None,
            "max_teacher_length": None,
        }
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    median = (
        float(lengths_sorted[n // 2])
        if n % 2 == 1
        else (float(lengths_sorted[n // 2 - 1]) + float(lengths_sorted[n // 2])) / 2.0
    )
    n_at_cap = sum(length >= max_new_tokens for length in lengths_sorted)
    return {
        "n_prompts": n,
        "max_new_tokens_asserted": int(max_new_tokens),
        "n_at_cap": int(n_at_cap),
        "fraction_at_cap": float(n_at_cap / n),
        "mean_teacher_length": float(sum(lengths_sorted) / n),
        "median_teacher_length": median,
        "max_teacher_length": int(max(lengths_sorted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local exp16 shard outputs.")
    parser.add_argument("--run-root", type=Path, required=True, help="Root directory containing `shards/`.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional explicit merged output dir. Defaults to <run-root>/merged/<model>.",
    )
    args = parser.parse_args()

    shard_root = args.run_root / "shards"
    shard_dirs = [shard_root / f"{args.model}__shard{idx}of{args.num_shards}" for idx in range(args.num_shards)]
    missing = [path for path in shard_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shard dirs: {missing}")

    merged_dir = args.output_dir or (args.run_root / "merged" / args.model)
    merged_dir.mkdir(parents=True, exist_ok=True)

    shard_configs = [
        json.loads((shard_dir / "config.json").read_text())
        for shard_dir in shard_dirs
        if (shard_dir / "config.json").exists()
    ]

    prompt_count = _merge_jsonl_unique(
        [shard_dir / "prompts.jsonl" for shard_dir in shard_dirs],
        merged_dir / "prompts.jsonl",
        key_fn=lambda row: row["id"],
        sort_key=lambda row: (row.get("category", ""), row["id"]),
    )
    summary_count = _merge_jsonl_unique(
        [shard_dir / "prompt_summaries.jsonl" for shard_dir in shard_dirs],
        merged_dir / "prompt_summaries.jsonl",
        key_fn=lambda row: row["prompt_id"],
        sort_key=lambda row: row["prompt_id"],
    )
    _merge_jsonl_unique(
        [shard_dir / "generated_texts.jsonl" for shard_dir in shard_dirs],
        merged_dir / "generated_texts.jsonl",
        key_fn=lambda row: (row["prompt_id"], row["pipeline"]),
        sort_key=lambda row: (row["prompt_id"], row["pipeline"]),
    )
    teacher_manifest_count = _merge_jsonl_unique(
        [shard_dir / "teacher_token_manifest.jsonl" for shard_dir in shard_dirs],
        merged_dir / "teacher_token_manifest.jsonl",
        key_fn=lambda row: row["prompt_id"],
        sort_key=lambda row: row["prompt_id"],
    )
    _merge_jsonl_unique(
        [shard_dir / "js_prompt_region_metrics.jsonl" for shard_dir in shard_dirs],
        merged_dir / "js_prompt_region_metrics.jsonl",
        key_fn=lambda row: (row["prompt_id"], row["pair_name"]),
        sort_key=lambda row: (row["pair_name"], row["prompt_id"]),
    )
    _merge_jsonl_unique(
        [shard_dir / "js_audit_sample.jsonl" for shard_dir in shard_dirs],
        merged_dir / "js_audit_sample.jsonl",
        key_fn=lambda row: (row["prompt_id"], row["pair_name"], int(row["step"])),
        sort_key=lambda row: (row["pair_name"], row["prompt_id"], int(row["step"])),
    )
    _merge_js_layer_stats(shard_dirs, merged_dir / "js_layer_stats.json")

    merged_config = dict(shard_configs[0]) if shard_configs else {}
    merged_config.update(
        {
            "model": args.model,
            "num_shards_merged": args.num_shards,
            "n_prompts_sampled_after_sharding": prompt_count,
            "n_prompt_summaries_merged": summary_count,
            "n_teacher_manifest_rows_merged": teacher_manifest_count,
            "merged_from_run_root": str(args.run_root),
        }
    )
    (merged_dir / "config.json").write_text(json.dumps(merged_config, indent=2))

    teacher_manifest_rows = _load_jsonl(merged_dir / "teacher_token_manifest.jsonl")
    if teacher_manifest_rows:
        max_new_tokens = int(merged_config.get("max_new_tokens", 512))
        teacher_diag = _teacher_cap_diagnostics(teacher_manifest_rows, max_new_tokens=max_new_tokens)
        (merged_dir / "teacher_cap_diagnostics.json").write_text(json.dumps(teacher_diag, indent=2))

    print(
        json.dumps(
            {
                "model": args.model,
                "merged_dir": str(merged_dir),
                "prompt_count": prompt_count,
                "teacher_manifest_count": teacher_manifest_count,
            }
        )
    )


if __name__ == "__main__":
    main()
