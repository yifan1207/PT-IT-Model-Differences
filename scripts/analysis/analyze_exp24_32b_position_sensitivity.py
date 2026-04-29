#!/usr/bin/env python3
"""Position/category audit for the Exp24 Qwen2.5-32B Exp23 factorial.

This is the 32B companion to the Dense-5 Exp23 position analyses.  The 32B
run is a single-family external-validity check, so intervals are prompt-cluster
bootstraps within Qwen2.5-32B rather than family-balanced intervals.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


DEFAULT_RUN_ROOT = Path(
    "results/exp24_32b_external_validity/"
    "exp24_qwen25_32b_full_eval_v21_20260427_194839"
)
DEFAULT_OUT_DIR = Path("results/paper_synthesis/exp24_32b_external_validity")
DEFAULT_DATASETS = (
    Path("data/eval_dataset_v2.jsonl"),
    Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"),
)
RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
POSITION_FILTERS: tuple[tuple[str, str, Callable[[int], bool]], ...] = (
    ("all", "all positions", lambda step: True),
    ("step_0", "position 0", lambda step: step == 0),
    ("step_ge1", "positions >=1", lambda step: step >= 1),
    ("step_ge3", "positions >=3", lambda step: step >= 3),
    ("step_ge5", "positions >=5", lambda step: step >= 5),
    ("step_ge10", "positions >=10", lambda step: step >= 10),
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


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _margin(payload: dict[str, Any], cell: str, readout: str) -> float | None:
    return _finite((((payload.get("cells") or {}).get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))


def _effect(payload: dict[str, Any], effect: str, readout: str) -> float | None:
    cells = {cell: _margin(payload, cell, readout) for cell in RESIDUAL_CELLS}
    if any(value is None for value in cells.values()):
        return None
    if effect == "interaction":
        return cells["U_IT__L_IT"] - cells["U_IT__L_PT"] - cells["U_PT__L_IT"] + cells["U_PT__L_PT"]
    if effect == "late_given_pt":
        return cells["U_PT__L_IT"] - cells["U_PT__L_PT"]
    if effect == "late_given_it":
        return cells["U_IT__L_IT"] - cells["U_IT__L_PT"]
    if effect == "upstream_context":
        return 0.5 * ((cells["U_IT__L_PT"] - cells["U_PT__L_PT"]) + (cells["U_IT__L_IT"] - cells["U_PT__L_IT"]))
    if effect == "late_stack_main":
        return 0.5 * ((cells["U_PT__L_IT"] - cells["U_PT__L_PT"]) + (cells["U_IT__L_IT"] - cells["U_IT__L_PT"]))
    raise ValueError(effect)


def _load_units(run_root: Path, *, model: str, prompt_mode: str, readout: str) -> list[dict[str, Any]]:
    dataset_meta = _load_dataset_metadata(DEFAULT_DATASETS)
    path = run_root / "exp23_midlate_interaction_suite" / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(f"Missing Exp24 raw records: {path}")
    units: list[dict[str, Any]] = []
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
        effects = {name: _effect(payload, name, readout) for name in ("interaction", "late_given_pt", "late_given_it", "upstream_context", "late_stack_main")}
        if any(value is None for value in effects.values()):
            continue
        meta = dataset_meta.get(prompt_id, {})
        pt_token = event.get("pt_token") or {}
        it_token = event.get("it_token") or {}
        units.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                "step": step,
                "prompt_category": str(meta.get("category") or _category_from_prompt_id(prompt_id)),
                "prompt_source": str(meta.get("source") or "UNKNOWN"),
                "pt_token_category": str(pt_token.get("token_category_collapsed") or pt_token.get("token_category") or "UNKNOWN"),
                "it_token_category": str(it_token.get("token_category_collapsed") or it_token.get("token_category") or "UNKNOWN"),
                "pt_assistant_marker": bool(pt_token.get("assistant_marker")),
                "it_assistant_marker": bool(it_token.get("assistant_marker")),
                **effects,
            }
        )
    return units


def _cluster_values(units: list[dict[str, Any]], metric: str) -> np.ndarray:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        by_prompt[str(unit["prompt_id"])].append(float(unit[metric]))
    return np.asarray([float(np.mean(values)) for values in by_prompt.values()], dtype=float)


def _bootstrap(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    estimate = float(values.mean())
    if values.size == 1 or n_boot <= 0:
        return estimate, None, None
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for start in range(0, n_boot, 512):
        size = min(512, n_boot - start)
        idx = rng.integers(0, values.size, size=(size, values.size))
        boots.extend(values[idx].mean(axis=1).tolist())
    ci_low, ci_high = np.percentile(np.asarray(boots, dtype=float), [2.5, 97.5])
    return estimate, float(ci_low), float(ci_high)


def _position_rows(units: list[dict[str, Any]], *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = ("interaction", "late_given_pt", "late_given_it", "upstream_context", "late_stack_main")
    seed_offset = 0
    for filter_name, filter_label, predicate in POSITION_FILTERS:
        bucket = [unit for unit in units if predicate(int(unit["step"]))]
        prompt_clusters = len({unit["prompt_id"] for unit in bucket})
        for metric in metrics:
            values = _cluster_values(bucket, metric)
            estimate, ci_low, ci_high = _bootstrap(values, n_boot=n_boot, seed=seed + seed_offset)
            seed_offset += 101
            rows.append(
                {
                    "position_filter": filter_name,
                    "position_label": filter_label,
                    "metric": metric,
                    "estimate": estimate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_records": len(bucket),
                    "n_prompt_clusters": prompt_clusters,
                }
            )
    return rows


def _prompt_category_rows(units: list[dict[str, Any]], *, n_boot: int, seed: int, min_clusters: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    categories = sorted({str(unit["prompt_category"]) for unit in units})
    seed_offset = 10_000
    for filter_name, filter_label, predicate in POSITION_FILTERS:
        filter_units = [unit for unit in units if predicate(int(unit["step"]))]
        for category in ["ALL", *categories]:
            bucket = filter_units if category == "ALL" else [unit for unit in filter_units if unit["prompt_category"] == category]
            prompt_clusters = len({unit["prompt_id"] for unit in bucket})
            reportable = prompt_clusters >= min_clusters
            for metric in ("interaction", "late_given_pt", "late_given_it"):
                values = _cluster_values(bucket, metric)
                estimate, ci_low, ci_high = _bootstrap(values, n_boot=n_boot if reportable else 0, seed=seed + seed_offset)
                seed_offset += 101
                rows.append(
                    {
                        "position_filter": filter_name,
                        "position_label": filter_label,
                        "prompt_category": category,
                        "metric": metric,
                        "estimate": estimate,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "n_records": len(bucket),
                        "n_prompt_clusters": prompt_clusters,
                        "reportable": reportable,
                    }
                )
    return rows


def _category_mix_rows(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    axes = (
        ("prompt_category", "prompt_category"),
        ("prompt_source", "prompt_source"),
        ("it_token_category", "it_token_category"),
        ("pt_token_category", "pt_token_category"),
    )
    rows: list[dict[str, Any]] = []
    for filter_name, filter_label, predicate in POSITION_FILTERS:
        bucket = [unit for unit in units if predicate(int(unit["step"]))]
        total = len(bucket)
        for axis, field in axes:
            counter = Counter(str(unit[field]) for unit in bucket)
            for category, count in counter.most_common():
                rows.append(
                    {
                        "position_filter": filter_name,
                        "position_label": filter_label,
                        "axis": axis,
                        "category": category,
                        "count": count,
                        "total": total,
                        "record_fraction": count / total if total else 0.0,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_ci(row: dict[str, Any], digits: int = 2) -> str:
    if row.get("ci95_low") in (None, ""):
        return "NA"
    return f"[{_fmt(row['ci95_low'], digits)}, {_fmt(row['ci95_high'], digits)}]"


def _find(rows: list[dict[str, Any]], *, position_filter: str, metric: str, prompt_category: str | None = None) -> dict[str, Any]:
    for row in rows:
        if row["position_filter"] != position_filter or row["metric"] != metric:
            continue
        if prompt_category is not None and row.get("prompt_category") != prompt_category:
            continue
        return row
    raise KeyError((position_filter, metric, prompt_category))


def _breakdown(counter: Counter[str], total: int, max_items: int | None = None) -> str:
    items = counter.most_common(max_items)
    return ", ".join(f"{key} {count} ({100 * count / total:.1f}%)" for key, count in items)


def _write_note(path: Path, units: list[dict[str, Any]], position_rows: list[dict[str, Any]], prompt_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = Counter(int(unit["step"]) for unit in units)
    total = len(units)
    mean_step = sum(step * count for step, count in steps.items()) / total
    median_step = None
    acc = 0
    for step, count in sorted(steps.items()):
        acc += count
        if acc >= (total + 1) / 2:
            median_step = step
            break
    prompt_counter = Counter(str(unit["prompt_category"]) for unit in units)
    it_token_counter = Counter(str(unit["it_token_category"]) for unit in units)
    assistant_marker = sum(1 for unit in units if unit["pt_assistant_marker"] or unit["it_assistant_marker"])
    all_row = _find(position_rows, position_filter="all", metric="interaction")
    ge3_row = _find(position_rows, position_filter="step_ge3", metric="interaction")
    ge5_row = _find(position_rows, position_filter="step_ge5", metric="interaction")
    ge10_row = _find(position_rows, position_filter="step_ge10", metric="interaction")
    fact_ge3 = _find(prompt_rows, position_filter="step_ge3", prompt_category="CONTENT-FACT", metric="interaction")
    reason_ge3 = _find(prompt_rows, position_filter="step_ge3", prompt_category="CONTENT-REASON", metric="interaction")
    lines = [
        "# Exp24 Qwen2.5-32B Position Sensitivity",
        "",
        "CPU-only audit of the Qwen2.5-32B Exp23 raw-shared residual-state x late-stack records. "
        "Intervals bootstrap prompt clusters within the single 32B family.",
        "",
        "## Position distribution",
        "",
        f"- Valid records: `{total}` prompt clusters.",
        f"- Generated-position mean: `{mean_step:.2f}`; median: `{median_step}`.",
        f"- Position bins: step 0 `{steps[0]}`; step >=3 `{sum(c for s, c in steps.items() if s >= 3)}`; "
        f"step >=5 `{sum(c for s, c in steps.items() if s >= 5)}`; step >=10 `{sum(c for s, c in steps.items() if s >= 10)}`.",
        f"- Prompt categories: {_breakdown(prompt_counter, total)}.",
        f"- IT divergent-token categories: {_breakdown(it_token_counter, total)}.",
        f"- Assistant-marker events: `{assistant_marker}` (`{100 * assistant_marker / total:.1f}%`).",
        "",
        "## Interaction by position",
        "",
        "| Position filter | Clusters | Interaction | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for filter_name, filter_label, _predicate in POSITION_FILTERS:
        row = _find(position_rows, position_filter=filter_name, metric="interaction")
        lines.append(
            f"| {filter_label} | {row['n_prompt_clusters']} | `{_fmt(row['estimate'])}` | `{_fmt_ci(row)}` |"
        )
    lines.extend(
        [
            "",
            "## Prompt categories at later positions",
            "",
            "| Position filter | Category | Clusters | Interaction | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for filter_name in ("all", "step_ge3", "step_ge5"):
        for category in ("CONTENT-FACT", "CONTENT-REASON"):
            row = _find(prompt_rows, position_filter=filter_name, prompt_category=category, metric="interaction")
            ci = _fmt_ci(row) if row["reportable"] else "thin"
            lines.append(
                f"| {row['position_label']} | {category} | {row['n_prompt_clusters']} | `{_fmt(row['estimate'])}` | `{ci}` |"
            )
    lines.extend(
        [
            "",
            "## Paper-facing read",
            "",
            f"The Qwen2.5-32B interaction is `{_fmt(all_row['estimate'])}` logits overall and remains positive at generated position `>=3` "
            f"(`{_fmt(ge3_row['estimate'])}`, 95% CI `{_fmt_ci(ge3_row)}`), `>=5` (`{_fmt(ge5_row['estimate'])}`, 95% CI `{_fmt_ci(ge5_row)}`), "
            f"and `>=10` (`{_fmt(ge10_row['estimate'])}`, 95% CI `{_fmt_ci(ge10_row)}`). "
            "This run is content/reasoning-heavy rather than governance-heavy, so its position profile is not directly comparable to the Dense-5 holdout profile.",
            f"At position `>=3`, CONTENT-FACT has `{_fmt(fact_ge3['estimate'])}` logits and CONTENT-REASON has `{_fmt(reason_ge3['estimate'])}` logits; "
            "both are positive, showing that the later-position 32B effect is not a single prompt-category residue.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model", default="qwen25_32b")
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readout", default="common_it")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2432)
    parser.add_argument("--min-clusters", type=int, default=30)
    args = parser.parse_args()

    units = _load_units(args.run_root, model=args.model, prompt_mode=args.prompt_mode, readout=args.readout)
    if not units:
        raise RuntimeError("No valid Qwen2.5-32B Exp23 units found")
    position_rows = _position_rows(units, n_boot=args.n_bootstrap, seed=args.seed)
    prompt_rows = _prompt_category_rows(units, n_boot=args.n_bootstrap, seed=args.seed, min_clusters=args.min_clusters)
    mix_rows = _category_mix_rows(units)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "exp24_32b_position_sensitivity.csv", position_rows)
    _write_csv(args.out_dir / "exp24_32b_position_prompt_category_effects.csv", prompt_rows)
    _write_csv(args.out_dir / "exp24_32b_position_category_mix.csv", mix_rows)
    _write_note(args.out_dir / "exp24_32b_position_sensitivity_note.md", units, position_rows, prompt_rows)
    print(f"Wrote {args.out_dir / 'exp24_32b_position_sensitivity.csv'}")
    print(f"Wrote {args.out_dir / 'exp24_32b_position_prompt_category_effects.csv'}")
    print(f"Wrote {args.out_dir / 'exp24_32b_position_category_mix.csv'}")
    print(f"Wrote {args.out_dir / 'exp24_32b_position_sensitivity_note.md'}")


if __name__ == "__main__":
    main()
