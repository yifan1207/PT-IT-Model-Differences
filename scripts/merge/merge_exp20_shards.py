#!/usr/bin/env python3
"""Merge Exp20 JSONL shards from multiple RunPod helper roots.

The collector can safely resume within one output directory, but this run uses
separate pods/volumes. This script is the intentionally boring join point:
dedupe by (prompt_mode, model, prompt_id), keep the first valid row, and write a
canonical tree that the normal Exp20 analyzer already understands.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _iter_candidate_files(source: Path) -> list[Path]:
    if not source.exists():
        return []
    return sorted(source.rglob("exp20_records*.jsonl"))


def _read_record(line: str) -> dict[str, Any] | None:
    if not line.strip():
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    if "model" not in payload or "prompt_id" not in payload or "prompt_mode" not in payload:
        return None
    return payload


def merge_sources(
    *,
    sources: list[Path],
    out_root: Path,
    expected_per_model: int,
) -> dict[str, Any]:
    rows: dict[tuple[str, str, str], str] = {}
    source_counts: Counter[str] = Counter()
    duplicate_counts: Counter[str] = Counter()
    malformed_counts: Counter[str] = Counter()
    files_seen: list[str] = []

    for source in sources:
        for path in _iter_candidate_files(source):
            files_seen.append(str(path))
            with path.open("rb") as handle:
                for raw in handle:
                    line = raw.decode("utf-8", errors="ignore")
                    payload = _read_record(line)
                    if payload is None:
                        malformed_counts[str(path)] += 1
                        continue
                    mode = str(payload["prompt_mode"])
                    model = str(payload["model"])
                    prompt_id = str(payload["prompt_id"])
                    key = (mode, model, prompt_id)
                    source_counts[f"{mode}/{model}"] += 1
                    if key in rows:
                        duplicate_counts[f"{mode}/{model}"] += 1
                        continue
                    rows[key] = line if line.endswith("\n") else line + "\n"

    grouped: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for (mode, model, prompt_id), line in rows.items():
        grouped[(mode, model)].append((prompt_id, line))

    out_root.mkdir(parents=True, exist_ok=True)
    final_counts: dict[str, int] = {}
    missing_conditions: list[str] = []
    for (mode, model), items in sorted(grouped.items()):
        items.sort(key=lambda item: item[0])
        out_dir = out_root / mode / model
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "exp20_records.jsonl"
        with out_path.open("w") as fout:
            for _prompt_id, line in items:
                fout.write(line)
        key = f"{mode}/{model}"
        final_counts[key] = len(items)
        if expected_per_model and len(items) < expected_per_model:
            missing_conditions.append(f"{key}: {len(items)}/{expected_per_model}")

    summary = {
        "sources": [str(path) for path in sources],
        "files_seen": files_seen,
        "n_files_seen": len(files_seen),
        "source_row_counts": dict(sorted(source_counts.items())),
        "duplicate_counts": dict(sorted(duplicate_counts.items())),
        "malformed_counts": dict(sorted(malformed_counts.items())),
        "final_counts": dict(sorted(final_counts.items())),
        "expected_per_model": expected_per_model,
        "missing_conditions": missing_conditions,
        "ok": not malformed_counts and not missing_conditions,
    }
    (out_root / "merge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge Exp20 shards from multiple local roots.")
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-per-model", type=int, default=600)
    args = parser.parse_args()
    summary = merge_sources(
        sources=args.source,
        out_root=args.out_root,
        expected_per_model=args.expected_per_model,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
