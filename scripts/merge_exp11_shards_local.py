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
    with open(path) as f:
        for line in f:
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
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return len(rows)


def _concat_jsonl_files(shard_paths: list[Path], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fout:
        for shard_path in shard_paths:
            if not shard_path.exists():
                continue
            with open(shard_path) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)


def _merge_secondary_payloads(shard_dirs: list[Path], out_path: Path) -> None:
    secondary_payloads = [
        _read_json(shard_dir / "secondary_trajectory_stats.json")
        for shard_dir in shard_dirs
        if (shard_dir / "secondary_trajectory_stats.json").exists()
    ]
    if not secondary_payloads:
        return
    merged_secondary: dict[str, Any] = {
        "readout_name": secondary_payloads[0].get("readout_name"),
        "metrics": {},
    }
    readout_name_by_pipeline: dict[str, str] = {}
    for payload in secondary_payloads:
        readout_name_by_pipeline.update(payload.get("readout_name_by_pipeline", {}))
        for pipeline, pipeline_metrics in payload.get("metrics", {}).items():
            dst_pipeline = merged_secondary["metrics"].setdefault(pipeline, {})
            for metric_name, metric_stats in pipeline_metrics.items():
                dst_metric = dst_pipeline.setdefault(
                    metric_name,
                    {
                        "sum": [0.0] * len(metric_stats.get("sum", [])),
                        "count": [0.0] * len(metric_stats.get("count", [])),
                    },
                )
                for idx, value in enumerate(metric_stats.get("sum", [])):
                    dst_metric["sum"][idx] += value
                for idx, value in enumerate(metric_stats.get("count", [])):
                    dst_metric["count"][idx] += value
    if readout_name_by_pipeline:
        merged_secondary["readout_name_by_pipeline"] = readout_name_by_pipeline
    out_path.write_text(json.dumps(merged_secondary, indent=2))


def _merge_gradient_payloads(shard_dirs: list[Path], out_path: Path) -> None:
    gradient_payloads = [
        _read_json(shard_dir / "mechanism_gradient_directions.json")
        for shard_dir in shard_dirs
        if (shard_dir / "mechanism_gradient_directions.json").exists()
    ]
    if not gradient_payloads:
        return
    merged_gradients: dict[str, Any] = {}
    for payload in gradient_payloads:
        for pipeline_name, pipeline_payload in payload.items():
            dst = merged_gradients.setdefault(pipeline_name, {"counts": {}})
            for axis_name, layers in pipeline_payload.items():
                if axis_name == "counts":
                    for count_axis, counts in layers.items():
                        dst_counts = dst["counts"].setdefault(count_axis, [0] * len(counts))
                        for idx, value in enumerate(counts):
                            dst_counts[idx] += int(value)
                    continue
                dst_axis = dst.setdefault(axis_name, {})
                for layer_key, values in layers.items():
                    existing = dst_axis.setdefault(layer_key, [0.0] * len(values))
                    for idx, value in enumerate(values):
                        existing[idx] += float(value)
    out_path.write_text(json.dumps(merged_gradients, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge local exp11 shard outputs.")
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
    shard_dirs = [
        shard_root / f"{args.model}__shard{idx}of{args.num_shards}"
        for idx in range(args.num_shards)
    ]
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
    _concat_jsonl_files(
        [shard_dir / "step_metrics.jsonl" for shard_dir in shard_dirs],
        merged_dir / "step_metrics.jsonl",
    )
    mechanism_paths = [
        shard_dir / "mechanism_metrics.jsonl"
        for shard_dir in shard_dirs
        if (shard_dir / "mechanism_metrics.jsonl").exists()
    ]
    if mechanism_paths:
        _concat_jsonl_files(mechanism_paths, merged_dir / "mechanism_metrics.jsonl")
    _merge_secondary_payloads(shard_dirs, merged_dir / "secondary_trajectory_stats.json")
    _merge_gradient_payloads(shard_dirs, merged_dir / "mechanism_gradient_directions.json")

    merged_config = dict(shard_configs[0]) if shard_configs else {}
    merged_config.update(
        {
            "model": args.model,
            "num_shards_merged": args.num_shards,
            "n_prompts_sampled_after_sharding": prompt_count,
            "n_prompt_summaries_merged": summary_count,
            "merged_from_run_root": str(args.run_root),
        }
    )
    (merged_dir / "config.json").write_text(json.dumps(merged_config, indent=2))
    print(json.dumps({"model": args.model, "merged_dir": str(merged_dir), "prompt_count": prompt_count}))


if __name__ == "__main__":
    main()
