#!/usr/bin/env python3
"""Category mix for Exp23 first-divergence position thresholds.

This is a CPU-only companion to analyze_first_divergence_position_sensitivity.py.
It checks whether later first-divergence thresholds, especially generated
position >=3, collapse to a single prompt or token category.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable


DENSE5 = ("gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")
DEFAULT_RUN_ROOT = Path(
    "results/exp23_midlate_interaction_suite/"
    "exp23_dense5_full_h100x8_20260426_sh4_rw4"
)
DEFAULT_OUT_DIR = Path("results/paper_synthesis")
DEFAULT_DATASETS = (
    Path("data/eval_dataset_v2.jsonl"),
    Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"),
)

POSITION_FILTERS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("all", lambda step: True),
    ("step_0", lambda step: step == 0),
    ("step_ge1", lambda step: step >= 1),
    ("step_ge3", lambda step: step >= 3),
    ("step_ge5", lambda step: step >= 5),
    ("step_ge10", lambda step: step >= 10),
)


def _json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_dataset_metadata(paths: tuple[Path, ...]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("id") or row.get("record_id") or "")
            if prompt_id:
                out[prompt_id] = row
    return out


def _category_from_prompt_id(prompt_id: str) -> str:
    parts = prompt_id.split("_")
    if len(parts) >= 3 and parts[0].startswith("v"):
        return parts[1]
    return "UNKNOWN"


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _format_breakdown(counter: Counter[str], total: int, *, max_items: int | None = None) -> str:
    items = counter.most_common(max_items)
    return ", ".join(f"{name} {count} ({_pct(count / total)})" for name, count in items)


def _family_balanced_fraction(
    counts_by_model: dict[str, Counter[str]],
    totals_by_model: dict[str, int],
    category: str,
) -> float:
    fractions = []
    for model, total in totals_by_model.items():
        if total > 0:
            fractions.append(counts_by_model[model][category] / total)
    if not fractions:
        return 0.0
    return sum(fractions) / len(fractions)


def _load_first_diff_records(
    *,
    run_root: Path,
    models: tuple[str, ...],
    dataset_meta: dict[str, dict[str, Any]],
    prompt_mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        path = run_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp23 raw records for {model}: {path}")
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id") or "")
            payload = ((row.get("events") or {}).get("first_diff") or {})
            if not payload or payload.get("duplicate_of") or not payload.get("valid"):
                continue
            event = payload.get("event") or {}
            try:
                step = int(event.get("step"))
            except (TypeError, ValueError):
                continue
            meta = dataset_meta.get(prompt_id, {})
            pt_token = event.get("pt_token") or {}
            it_token = event.get("it_token") or {}
            rows.append(
                {
                    "model": model,
                    "prompt_id": prompt_id,
                    "step": step,
                    "prompt_category": str(meta.get("category") or _category_from_prompt_id(prompt_id)),
                    "prompt_source": str(meta.get("source") or "UNKNOWN"),
                    "pt_token_category": str(pt_token.get("token_category_collapsed") or pt_token.get("token_category") or "UNKNOWN"),
                    "it_token_category": str(it_token.get("token_category_collapsed") or it_token.get("token_category") or "UNKNOWN"),
                }
            )
    return rows


def _build_mix_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    axes = (
        ("prompt_category", "prompt_category"),
        ("prompt_source", "prompt_source"),
        ("it_token_category", "it_token_category"),
        ("pt_token_category", "pt_token_category"),
    )
    for filter_name, predicate in POSITION_FILTERS:
        kept = [record for record in records if predicate(int(record["step"]))]
        total = len(kept)
        totals_by_model = Counter(str(record["model"]) for record in kept)
        for axis_name, field in axes:
            counter = Counter(str(record[field]) for record in kept)
            counts_by_model: dict[str, Counter[str]] = defaultdict(Counter)
            for record in kept:
                counts_by_model[str(record["model"])][str(record[field])] += 1
            for category, count in counter.most_common():
                models_present = sorted(
                    model for model, model_counter in counts_by_model.items() if model_counter[category] > 0
                )
                out.append(
                    {
                        "position_filter": filter_name,
                        "axis": axis_name,
                        "category": category,
                        "count": count,
                        "total": total,
                        "record_fraction": count / total if total else 0.0,
                        "family_balanced_fraction": _family_balanced_fraction(
                            counts_by_model,
                            dict(totals_by_model),
                            category,
                        ),
                        "models_present": ";".join(models_present),
                    }
                )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_note(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Exp23 Position-Threshold Category Mix\n")
    lines.append(
        "CPU-only audit of the primary Exp23 residual-state x late-stack raw-shared records. "
        "Counts use valid `first_diff` events only; in this primary holdout, records equal prompt clusters.\n"
    )
    for filter_name, predicate in POSITION_FILTERS:
        kept = [record for record in records if predicate(int(record["step"]))]
        if not kept:
            continue
        total = len(kept)
        models = Counter(str(record["model"]) for record in kept)
        prompt = Counter(str(record["prompt_category"]) for record in kept)
        it_token = Counter(str(record["it_token_category"]) for record in kept)
        pt_token = Counter(str(record["pt_token_category"]) for record in kept)
        lines.append(f"## {filter_name}\n")
        lines.append(f"- Records: `{total}` across `{len(models)}` families ({_format_breakdown(models, total)}).")
        lines.append(f"- Prompt categories: {_format_breakdown(prompt, total)}.")
        lines.append(f"- IT divergent-token categories: {_format_breakdown(it_token, total)}.")
        lines.append(f"- PT divergent-token categories: {_format_breakdown(pt_token, total)}.\n")
    step_ge3 = [record for record in records if int(record["step"]) >= 3]
    prompt_ge3 = Counter(str(record["prompt_category"]) for record in step_ge3)
    it_ge3 = Counter(str(record["it_token_category"]) for record in step_ge3)
    lines.append("## Paper-facing summary\n")
    lines.append(
        "At generated position `>=3`, the subset is not a single-category residue: "
        f"`{len(step_ge3)}` records remain across all five dense families; prompt categories are "
        f"{_format_breakdown(prompt_ge3, len(step_ge3))}; IT divergent-token categories are "
        f"{_format_breakdown(it_ge3, len(step_ge3))}."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    dataset_meta = _load_dataset_metadata(DEFAULT_DATASETS)
    records = _load_first_diff_records(
        run_root=args.run_root,
        models=DENSE5,
        dataset_meta=dataset_meta,
        prompt_mode=args.prompt_mode,
    )
    if not records:
        raise RuntimeError("No valid Exp23 first_diff records found")

    rows = _build_mix_rows(records)
    csv_path = args.out_dir / "exp23_position_category_mix.csv"
    note_path = args.out_dir / "exp23_position_category_mix_note.md"
    _write_csv(csv_path, rows)
    _write_note(note_path, records)
    print(f"Wrote {csv_path}")
    print(f"Wrote {note_path}")


if __name__ == "__main__":
    main()
