#!/usr/bin/env python3
"""Exp23 interaction by first-divergence position and prompt category.

This CPU-only analysis answers whether the position-sensitivity pattern is
mostly a prompt-category composition effect. It computes the same Exp23
upstream-state x late-stack interaction under the common-IT readout, stratified
by prompt category and generated-position filters.
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


DENSE5 = ("gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")
RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
DEFAULT_RUN_ROOT = Path(
    "results/exp23_midlate_interaction_suite/"
    "exp23_dense5_full_h100x8_20260426_sh4_rw4"
)
DEFAULT_OUT_DIR = Path("results/paper_synthesis")
DEFAULT_DATASETS = (
    Path("data/eval_dataset_v2.jsonl"),
    Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"),
)

POSITION_FILTERS: tuple[tuple[str, str, Callable[[int], bool]], ...] = (
    ("all", "all positions", lambda step: True),
    ("step_0", "position 0", lambda step: step == 0),
    ("step_ge1", "positions >=1", lambda step: step >= 1),
    ("step_ge3", "positions >=3", lambda step: step >= 3),
    ("step_ge5", "positions >=5", lambda step: step >= 5),
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


def _load_units(
    *,
    run_root: Path,
    models: tuple[str, ...],
    prompt_mode: str,
    readout: str,
    dataset_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for model in models:
        path = run_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp23 records for {model}: {path}")
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
            cells = payload.get("cells") or {}
            margins: dict[str, float] = {}
            for cell in RESIDUAL_CELLS:
                margin = _finite(((cells.get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))
                if margin is not None:
                    margins[cell] = margin
            if len(margins) != len(RESIDUAL_CELLS):
                continue
            interaction = (
                margins["U_IT__L_IT"]
                - margins["U_IT__L_PT"]
                - margins["U_PT__L_IT"]
                + margins["U_PT__L_PT"]
            )
            late_given_pt = margins["U_PT__L_IT"] - margins["U_PT__L_PT"]
            late_given_it = margins["U_IT__L_IT"] - margins["U_IT__L_PT"]
            meta = dataset_meta.get(prompt_id, {})
            units.append(
                {
                    "model": model,
                    "prompt_id": prompt_id,
                    "prompt_category": str(meta.get("category") or _category_from_prompt_id(prompt_id)),
                    "step": step,
                    "interaction": interaction,
                    "late_given_pt": late_given_pt,
                    "late_given_it": late_given_it,
                }
            )
    return units


def _family_balanced_bootstrap(
    units: list[dict[str, Any]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_model_prompt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in units:
        by_model_prompt[str(unit["model"])][str(unit["prompt_id"])].append(float(unit[metric]))

    model_arrays: dict[str, np.ndarray] = {}
    model_counts: dict[str, int] = {}
    model_values: dict[str, float] = {}
    for model, prompt_values in by_model_prompt.items():
        prompt_means = np.array([float(np.mean(values)) for values in prompt_values.values()], dtype=float)
        if prompt_means.size:
            model_arrays[model] = prompt_means
            model_counts[model] = int(prompt_means.size)
            model_values[model] = float(prompt_means.mean())

    if not model_arrays:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "n_models": 0,
            "n_prompt_clusters": 0,
            "model_counts": {},
            "model_values": {},
        }

    estimate = float(np.mean(list(model_values.values())))
    boots: list[float] = []
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        arrays = list(model_arrays.values())
        for start in range(0, n_boot, 512):
            size = min(512, n_boot - start)
            sampled_means = []
            for arr in arrays:
                idx = rng.integers(0, len(arr), size=(size, len(arr)))
                sampled_means.append(arr[idx].mean(axis=1))
            boots.extend(np.vstack(sampled_means).mean(axis=0).tolist())
    ci_low = ci_high = None
    if boots:
        ci_low, ci_high = np.percentile(np.asarray(boots, dtype=float), [2.5, 97.5])
        ci_low = float(ci_low)
        ci_high = float(ci_high)
    return {
        "estimate": estimate,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "n_models": len(model_arrays),
        "n_prompt_clusters": int(sum(model_counts.values())),
        "model_counts": model_counts,
        "model_values": model_values,
    }


def _build_rows(units: list[dict[str, Any]], *, n_boot: int, seed: int, min_clusters: int, min_models: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_categories = sorted({str(unit["prompt_category"]) for unit in units})
    seed_offset = 0
    for filter_name, filter_label, predicate in POSITION_FILTERS:
        filter_units = [unit for unit in units if predicate(int(unit["step"]))]
        for prompt_category in ["ALL", *prompt_categories]:
            if prompt_category == "ALL":
                bucket = filter_units
            else:
                bucket = [unit for unit in filter_units if unit["prompt_category"] == prompt_category]
            n_models = len({unit["model"] for unit in bucket})
            n_clusters = len({(unit["model"], unit["prompt_id"]) for unit in bucket})
            reportable = n_models >= min_models and n_clusters >= min_clusters
            for metric in ("interaction", "late_given_pt", "late_given_it"):
                stats = _family_balanced_bootstrap(
                    bucket,
                    metric,
                    n_boot=n_boot if reportable else 0,
                    seed=seed + seed_offset,
                )
                seed_offset += 101
                rows.append(
                    {
                        "position_filter": filter_name,
                        "position_label": filter_label,
                        "prompt_category": prompt_category,
                        "metric": metric,
                        "estimate": stats["estimate"],
                        "ci95_low": stats["ci95_low"],
                        "ci95_high": stats["ci95_high"],
                        "n_models": stats["n_models"],
                        "n_prompt_clusters": stats["n_prompt_clusters"],
                        "reportable": reportable,
                        "model_counts": json.dumps(stats["model_counts"], sort_keys=True),
                        "model_values": json.dumps(stats["model_values"], sort_keys=True),
                    }
                )
    return rows


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _row_lookup(rows: list[dict[str, Any]], position_filter: str, prompt_category: str, metric: str) -> dict[str, Any]:
    for row in rows:
        if (
            row["position_filter"] == position_filter
            and row["prompt_category"] == prompt_category
            and row["metric"] == metric
        ):
            return row
    raise KeyError((position_filter, prompt_category, metric))


def _write_note(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Exp23 Position x Prompt-Category Analysis\n")
    lines.append(
        "CPU-only stratification of the primary Exp23 residual-state x late-stack factorial. "
        "The metric is the common-IT upstream-state x late-stack interaction unless noted.\n"
    )
    lines.append("## Interaction by prompt category and position\n")
    lines.append("| Position | Prompt category | Clusters | Models | Interaction | 95% CI |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for filter_name, filter_label, _predicate in POSITION_FILTERS:
        for category in ("ALL", "GOV-CONV", "GOV-FORMAT", "SAFETY"):
            row = _row_lookup(rows, filter_name, category, "interaction")
            ci = f"[{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}]" if row["ci95_low"] not in (None, "") else "NA"
            suffix = "" if row["reportable"] else " (thin)"
            lines.append(
                f"| {filter_label} | {category}{suffix} | {row['n_prompt_clusters']} | {row['n_models']} | "
                f"{_fmt(row['estimate'])} | {ci} |"
            )
    lines.append("\n## Main read\n")
    ge3_all = _row_lookup(rows, "step_ge3", "ALL", "interaction")
    ge3_conv = _row_lookup(rows, "step_ge3", "GOV-CONV", "interaction")
    ge3_format = _row_lookup(rows, "step_ge3", "GOV-FORMAT", "interaction")
    ge3_safety = _row_lookup(rows, "step_ge3", "SAFETY", "interaction")
    lines.append(
        f"At generated position `>=3`, the pooled interaction is `{_fmt(ge3_all['estimate'])}` logits. "
        f"Most records are GOV-CONV, whose within-category interaction is `{_fmt(ge3_conv['estimate'])}` logits. "
        f"GOV-FORMAT (`{_fmt(ge3_format['estimate'])}`) and SAFETY (`{_fmt(ge3_safety['estimate'])}`) remain positive but are thin, "
        "so they are useful as direction checks rather than category-level headline estimates."
    )
    lines.append(
        "The position attenuation is therefore partly a composition shift toward later conversational/governance disagreements, "
        "but not only composition: GOV-CONV itself drops from the all-position estimate to the `>=3` estimate."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readout", default="common_it")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2331)
    parser.add_argument("--min-clusters", type=int, default=30)
    parser.add_argument("--min-models", type=int, default=3)
    args = parser.parse_args()

    dataset_meta = _load_dataset_metadata(DEFAULT_DATASETS)
    units = _load_units(
        run_root=args.run_root,
        models=DENSE5,
        prompt_mode=args.prompt_mode,
        readout=args.readout,
        dataset_meta=dataset_meta,
    )
    rows = _build_rows(
        units,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        min_clusters=args.min_clusters,
        min_models=args.min_models,
    )
    _write_csv(args.out_dir / "exp23_position_prompt_category_effects.csv", rows)
    _write_note(args.out_dir / "exp23_position_prompt_category_note.md", rows)
    print(f"Wrote {args.out_dir / 'exp23_position_prompt_category_effects.csv'}")
    print(f"Wrote {args.out_dir / 'exp23_position_prompt_category_note.md'}")


if __name__ == "__main__":
    main()
