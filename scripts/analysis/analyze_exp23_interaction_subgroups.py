#!/usr/bin/env python3
"""Subgroup analysis for Exp23 residual-state x late-stack interaction.

This is an analysis-only extension of Exp23. It reuses the same factorial
estimands as analyze_exp23_midlate_interaction_suite.py, but stratifies by
prompt metadata, divergent-token category, event kind, and IT-confidence
proxies recorded in the raw residual-factorial JSONL files.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np


DENSE5_MODELS = ("gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")
RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
PRIMARY_READOUT = "common_it"

EFFECTS: dict[str, dict[str, float]] = {
    "late_weight_effect": {
        "U_PT__L_IT": 0.5,
        "U_PT__L_PT": -0.5,
        "U_IT__L_IT": 0.5,
        "U_IT__L_PT": -0.5,
    },
    "upstream_context_effect": {
        "U_IT__L_PT": 0.5,
        "U_PT__L_PT": -0.5,
        "U_IT__L_IT": 0.5,
        "U_PT__L_IT": -0.5,
    },
    "interaction": {
        "U_IT__L_IT": 1.0,
        "U_IT__L_PT": -1.0,
        "U_PT__L_IT": -1.0,
        "U_PT__L_PT": 1.0,
    },
    "late_it_given_it_upstream": {
        "U_IT__L_IT": 1.0,
        "U_IT__L_PT": -1.0,
    },
    "late_it_given_pt_upstream": {
        "U_PT__L_IT": 1.0,
        "U_PT__L_PT": -1.0,
    },
}


@dataclass(frozen=True)
class Unit:
    model: str
    prompt_id: str
    prompt_category: str
    prompt_source: str
    event_kind: str
    readout: str
    margins: dict[str, float]
    choices: dict[str, str]
    pt_token_category: str
    it_token_category: str
    pt_token_category_collapsed: str
    it_token_category_collapsed: str
    pt_assistant_marker: bool
    it_assistant_marker: bool
    step: int | None
    prefix_length: int | None
    it_baseline_margin_common_it: float | None
    it_baseline_rank_common_it: int | None


def _json_rows(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not kept:
        return None
    return float(sum(kept) / len(kept))


def _percentile_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lo, hi = np.percentile(np.array(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _unit_effect(unit: Unit, effect_name: str) -> float | None:
    total = 0.0
    for cell, coef in EFFECTS[effect_name].items():
        margin = unit.margins.get(cell)
        if margin is None:
            return None
        total += coef * margin
    return total


def _load_dataset_metadata(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("id") or row.get("record_id") or "")
            if not prompt_id:
                continue
            out[prompt_id] = row
    return out


def _load_units(run_root: Path, models: list[str], prompt_mode: str, dataset_meta: dict[str, dict[str, Any]], readouts: list[str]) -> list[Unit]:
    units: list[Unit] = []
    for model in models:
        path = run_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(f"Missing Exp23 raw records for {model}: {path}")
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id") or "")
            meta = dataset_meta.get(prompt_id, {})
            prompt_category = str(meta.get("category") or _category_from_prompt_id(prompt_id) or "UNKNOWN")
            prompt_source = str(meta.get("source") or "UNKNOWN")
            for event_kind, payload in (row.get("events") or {}).items():
                if not payload or payload.get("duplicate_of") or not payload.get("valid"):
                    continue
                event = payload.get("event") or {}
                pt_token = event.get("pt_token") or {}
                it_token = event.get("it_token") or {}
                cells = payload.get("cells") or {}
                it_baseline_common_it = ((cells.get("U_IT__L_IT") or {}).get(PRIMARY_READOUT) or {})
                it_baseline_margin = _finite(it_baseline_common_it.get("it_vs_pt_margin"))
                it_baseline_rank = _int_or_none(it_baseline_common_it.get("it_rank"))
                for readout in readouts:
                    margins: dict[str, float] = {}
                    choices: dict[str, str] = {}
                    for cell in RESIDUAL_CELLS:
                        readout_payload = (cells.get(cell) or {}).get(readout) or {}
                        margin = _finite(readout_payload.get("it_vs_pt_margin"))
                        if margin is not None:
                            margins[cell] = margin
                        choices[cell] = str(readout_payload.get("token_choice_class", "missing"))
                    if len(margins) != len(RESIDUAL_CELLS):
                        continue
                    units.append(
                        Unit(
                            model=model,
                            prompt_id=prompt_id,
                            prompt_category=prompt_category,
                            prompt_source=prompt_source,
                            event_kind=str(event_kind),
                            readout=readout,
                            margins=margins,
                            choices=choices,
                            pt_token_category=str(pt_token.get("token_category") or "UNKNOWN"),
                            it_token_category=str(it_token.get("token_category") or "UNKNOWN"),
                            pt_token_category_collapsed=str(pt_token.get("token_category_collapsed") or pt_token.get("token_category") or "UNKNOWN"),
                            it_token_category_collapsed=str(it_token.get("token_category_collapsed") or it_token.get("token_category") or "UNKNOWN"),
                            pt_assistant_marker=bool(pt_token.get("assistant_marker")),
                            it_assistant_marker=bool(it_token.get("assistant_marker")),
                            step=_int_or_none(event.get("step")),
                            prefix_length=_int_or_none(payload.get("prefix_length")),
                            it_baseline_margin_common_it=it_baseline_margin,
                            it_baseline_rank_common_it=it_baseline_rank,
                        )
                    )
    return units


def _category_from_prompt_id(prompt_id: str) -> str | None:
    parts = prompt_id.split("_")
    if len(parts) >= 3 and parts[0].startswith("v"):
        return parts[1]
    return None


def _tercile_labels_by_model(units: list[Unit], value_fn) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    by_model: dict[str, list[tuple[str, float]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for unit in units:
        key = (unit.model, unit.prompt_id + "::" + unit.event_kind)
        if key in seen:
            continue
        seen.add(key)
        val = value_fn(unit)
        if val is not None and math.isfinite(float(val)):
            by_model[unit.model].append((key[1], float(val)))
    for model, values in by_model.items():
        if len(values) < 3:
            continue
        arr = np.array([v for _, v in values], dtype=float)
        q1, q2 = np.percentile(arr, [33.333, 66.667])
        for event_key, val in values:
            if val <= q1:
                label = "low"
            elif val <= q2:
                label = "mid"
            else:
                label = "high"
            labels[(model, event_key)] = label
    return labels


def _rank_bin(rank: int | None) -> str:
    if rank is None:
        return "missing"
    if rank == 1:
        return "rank_1"
    if rank <= 5:
        return "rank_2_5"
    return "rank_gt_5"


def _step_bin(step: int | None) -> str:
    if step is None:
        return "missing"
    if step == 0:
        return "step_0"
    if step == 1:
        return "step_1"
    if step <= 3:
        return "step_2_3"
    return "step_4_plus"


def _prefix_bin(prefix_length: int | None) -> str:
    if prefix_length is None:
        return "missing"
    if prefix_length <= 4:
        return "prefix_0_4"
    if prefix_length <= 16:
        return "prefix_5_16"
    return "prefix_17_plus"


def _subgroup_value(unit: Unit, group: str, margin_terciles: dict[tuple[str, str], str]) -> str:
    event_key = unit.prompt_id + "::" + unit.event_kind
    if group == "prompt_category":
        return unit.prompt_category
    if group == "prompt_source":
        return unit.prompt_source
    if group == "event_kind":
        return unit.event_kind
    if group == "it_token_category":
        return unit.it_token_category
    if group == "it_token_category_collapsed":
        return unit.it_token_category_collapsed
    if group == "pt_token_category":
        return unit.pt_token_category
    if group == "pt_token_category_collapsed":
        return unit.pt_token_category_collapsed
    if group == "assistant_marker_event":
        return "assistant_marker" if (unit.it_assistant_marker or unit.pt_assistant_marker) else "non_assistant_marker"
    if group == "it_assistant_marker":
        return "it_assistant_marker" if unit.it_assistant_marker else "not_it_assistant_marker"
    if group == "it_margin_tercile_within_model":
        return margin_terciles.get((unit.model, event_key), "missing")
    if group == "it_rank_bin":
        return _rank_bin(unit.it_baseline_rank_common_it)
    if group == "divergence_step_bin":
        return _step_bin(unit.step)
    if group == "prefix_length_bin":
        return _prefix_bin(unit.prefix_length)
    raise ValueError(f"Unknown subgroup: {group}")


def _bootstrap_effect(units: list[Unit], effect_name: str, n_boot: int, seed: int) -> dict[str, Any]:
    by_model_prompt: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in units:
        value = _unit_effect(unit, effect_name)
        if value is not None and math.isfinite(value):
            by_model_prompt[unit.model][unit.prompt_id].append(float(value))
    model_values: dict[str, float | None] = {}
    model_counts: dict[str, int] = {}
    model_cluster_counts: dict[str, int] = {}
    model_arrays: dict[str, np.ndarray] = {}
    for model, prompt_values in by_model_prompt.items():
        prompt_means = np.array([float(np.mean(vals)) for vals in prompt_values.values() if vals], dtype=float)
        model_arrays[model] = prompt_means
        model_values[model] = float(prompt_means.mean()) if prompt_means.size else None
        model_counts[model] = sum(len(vals) for vals in prompt_values.values())
        model_cluster_counts[model] = int(prompt_means.size)
    valid_models = [model for model, value in model_values.items() if value is not None]
    estimate = _mean([model_values[model] for model in valid_models])

    boot: list[float] = []
    if valid_models and n_boot > 0:
        rng = np.random.default_rng(seed)
        arrays = [model_arrays[model] for model in valid_models if model_arrays[model].size]
        for start in range(0, n_boot, 512):
            size = min(512, n_boot - start)
            sampled_means = []
            for arr in arrays:
                idx = rng.integers(0, len(arr), size=(size, len(arr)))
                sampled_means.append(arr[idx].mean(axis=1))
            if sampled_means:
                boot.extend(np.vstack(sampled_means).mean(axis=0).tolist())
    lo, hi = _percentile_ci(boot)
    return {
        "estimate": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_models": len(valid_models),
        "n_units": len(units),
        "n_prompt_clusters": int(sum(model_cluster_counts.get(model, 0) for model in valid_models)),
        "bootstrap_unit": "prompt_cluster_within_family",
        "n_boot": len(boot),
        "model_values": model_values,
        "model_counts": model_counts,
        "model_cluster_counts": model_cluster_counts,
    }


def _choice_rate(units: list[Unit], cell: str, target: str) -> float | None:
    vals = [1.0 if unit.choices.get(cell) == target else 0.0 for unit in units if cell in unit.choices]
    return _mean(vals)


def _summarize_subgroups(units: list[Unit], groups: list[str], readouts: list[str], n_boot: int, seed: int, min_units: int, min_models: int) -> list[dict[str, Any]]:
    primary_units = [unit for unit in units if unit.readout == PRIMARY_READOUT]
    margin_terciles = _tercile_labels_by_model(primary_units, lambda u: u.it_baseline_margin_common_it)

    rows: list[dict[str, Any]] = []
    seed_offset = 0
    for group in groups:
        for readout in readouts:
            readout_units = [unit for unit in units if unit.readout == readout]
            buckets: dict[str, list[Unit]] = defaultdict(list)
            for unit in readout_units:
                buckets[_subgroup_value(unit, group, margin_terciles)].append(unit)
            for value, bucket_units in sorted(buckets.items()):
                n_models_present = len({unit.model for unit in bucket_units})
                reportable = bool(len(bucket_units) >= min_units and n_models_present >= min_models)
                for effect in EFFECTS:
                    payload = _bootstrap_effect(bucket_units, effect, n_boot if reportable else 0, seed + seed_offset)
                    seed_offset += 101
                    rows.append(
                        {
                            "group": group,
                            "value": value,
                            "readout": readout,
                            "effect": effect,
                            "estimate": payload["estimate"],
                            "ci95_low": payload["ci95_low"],
                            "ci95_high": payload["ci95_high"],
                            "n_models": payload["n_models"],
                            "n_units": payload["n_units"],
                            "n_prompt_clusters": payload["n_prompt_clusters"],
                            "bootstrap_unit": payload["bootstrap_unit"],
                            "n_boot": payload["n_boot"],
                            "reportable": reportable,
                            "model_counts": payload["model_counts"],
                            "model_cluster_counts": payload["model_cluster_counts"],
                            "model_values": payload["model_values"],
                            "p_top1_it_U_PT__L_IT": _choice_rate(bucket_units, "U_PT__L_IT", "it"),
                            "p_top1_it_U_IT__L_IT": _choice_rate(bucket_units, "U_IT__L_IT", "it"),
                        }
                    )
    return rows


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    fieldnames = [
        "group",
        "value",
        "readout",
        "effect",
        "estimate",
        "ci95_low",
        "ci95_high",
        "n_models",
        "n_units",
        "n_prompt_clusters",
        "bootstrap_unit",
        "n_boot",
        "reportable",
        "p_top1_it_U_PT__L_IT",
        "p_top1_it_U_IT__L_IT",
        "model_counts",
        "model_cluster_counts",
        "model_values",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["model_counts"] = json.dumps(out["model_counts"], sort_keys=True)
            out["model_cluster_counts"] = json.dumps(out["model_cluster_counts"], sort_keys=True)
            out["model_values"] = json.dumps(out["model_values"], sort_keys=True)
            writer.writerow(out)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{val:.{digits}f}"


def _plot_forest(rows: list[dict[str, Any]], group: str, out_path: Path, title: str) -> None:
    kept = [
        row
        for row in rows
        if row["group"] == group and row["readout"] == PRIMARY_READOUT and row["effect"] == "interaction" and row["reportable"]
    ]
    if not kept:
        return
    kept = sorted(kept, key=lambda r: float(r["estimate"] if r["estimate"] is not None else 0.0))
    labels = [f"{row['value']} (n={row['n_units']})" for row in kept]
    vals = np.array([float(row["estimate"]) for row in kept], dtype=float)
    lows = np.array([float(row["ci95_low"]) if row["ci95_low"] is not None else float(row["estimate"]) for row in kept], dtype=float)
    highs = np.array([float(row["ci95_high"]) if row["ci95_high"] is not None else float(row["estimate"]) for row in kept], dtype=float)
    y = np.arange(len(kept))
    fig_h = max(3.2, 0.42 * len(kept) + 1.2)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.errorbar(vals, y, xerr=[vals - lows, highs - vals], fmt="o", color="#2F4B7C", ecolor="#9AA6B2", capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Interaction effect on IT-vs-PT margin (logits)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_prompt_token_heatmap(rows: list[dict[str, Any]], units: list[Unit], out_path: Path) -> None:
    primary_units = [u for u in units if u.readout == PRIMARY_READOUT]
    margin_terciles = _tercile_labels_by_model(primary_units, lambda u: u.it_baseline_margin_common_it)
    buckets: dict[tuple[str, str], list[Unit]] = defaultdict(list)
    for unit in primary_units:
        # Reuse the same helper so this stays aligned with the table outputs.
        key = (
            _subgroup_value(unit, "prompt_category", margin_terciles),
            _subgroup_value(unit, "it_token_category_collapsed", margin_terciles),
        )
        buckets[key].append(unit)
    prompt_values = sorted({k[0] for k in buckets})
    token_values = sorted({k[1] for k in buckets})
    if not prompt_values or not token_values:
        return
    grid = np.full((len(prompt_values), len(token_values)), np.nan)
    counts = np.zeros((len(prompt_values), len(token_values)), dtype=int)
    for i, prompt in enumerate(prompt_values):
        for j, token in enumerate(token_values):
            bucket = buckets.get((prompt, token), [])
            counts[i, j] = len(bucket)
            models = {u.model for u in bucket}
            if len(bucket) >= 50 and len(models) >= 3:
                grid[i, j] = _bootstrap_effect(bucket, "interaction", n_boot=0, seed=0)["estimate"]
    fig, ax = plt.subplots(figsize=(1.5 * len(token_values) + 2.0, 0.55 * len(prompt_values) + 2.0))
    im = ax.imshow(grid, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(token_values)))
    ax.set_xticklabels(token_values, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(prompt_values)))
    ax.set_yticklabels(prompt_values)
    for i in range(len(prompt_values)):
        for j in range(len(token_values)):
            text = "NA" if math.isnan(grid[i, j]) else f"{grid[i, j]:.1f}"
            ax.text(j, i, f"{text}\nn={counts[i, j]}", ha="center", va="center", color="white" if not math.isnan(grid[i, j]) and grid[i, j] > 2 else "black", fontsize=8)
    ax.set_title("Exp23 interaction by prompt category and IT-token category")
    fig.colorbar(im, ax=ax, label="Interaction effect (logits)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _profile_units(units: list[Unit]) -> dict[str, Any]:
    primary = [unit for unit in units if unit.readout == PRIMARY_READOUT]
    return {
        "n_units_primary_readout": len(primary),
        "by_model": dict(Counter(unit.model for unit in primary)),
        "by_prompt_category": dict(Counter(unit.prompt_category for unit in primary)),
        "by_prompt_source": dict(Counter(unit.prompt_source for unit in primary)),
        "by_event_kind": dict(Counter(unit.event_kind for unit in primary)),
        "by_it_token_category": dict(Counter(unit.it_token_category for unit in primary)),
        "by_it_token_category_collapsed": dict(Counter(unit.it_token_category_collapsed for unit in primary)),
        "by_pt_token_category_collapsed": dict(Counter(unit.pt_token_category_collapsed for unit in primary)),
        "assistant_marker_events": dict(Counter("assistant_marker" if (unit.it_assistant_marker or unit.pt_assistant_marker) else "non_assistant_marker" for unit in primary)),
    }


def _write_report(rows: list[dict[str, Any]], profile: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Exp23 Subgroup Analysis: Context-Gated Late Readout\n")
    lines.append("This analysis stratifies the Exp23 residual-state x late-stack factorial without changing the estimand. The primary estimand is the `interaction` effect under the `common_it` readout.\n")
    lines.append("## Data Profile\n")
    lines.append(f"- Primary-readout units: `{profile['n_units_primary_readout']}`")
    for key in ["by_model", "by_prompt_category", "by_event_kind", "by_it_token_category_collapsed", "assistant_marker_events"]:
        lines.append(f"- `{key}`: `{json.dumps(profile.get(key, {}), sort_keys=True)}`")
    lines.append("\n## Reportable Interaction Strata\n")
    target_groups = [
        "prompt_category",
        "it_token_category_collapsed",
        "event_kind",
        "it_margin_tercile_within_model",
        "it_rank_bin",
        "assistant_marker_event",
        "divergence_step_bin",
    ]
    for group in target_groups:
        kept = [
            row
            for row in rows
            if row["group"] == group
            and row["readout"] == PRIMARY_READOUT
            and row["effect"] == "interaction"
            and row["reportable"]
        ]
        if not kept:
            continue
        lines.append(f"\n### {group}\n")
        lines.append("| value | records | prompt clusters | models | interaction | 95% CI |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for row in sorted(kept, key=lambda r: str(r["value"])):
            lines.append(
                f"| {row['value']} | {row['n_units']} | {row['n_prompt_clusters']} | {row['n_models']} | "
                f"{_fmt(row['estimate'])} | [{_fmt(row['ci95_low'])}, {_fmt(row['ci95_high'])}] |"
            )
    lines.append("\n## Interpretation Guardrails\n")
    lines.append("- These are descriptive subgroup checks, not new headline claims.")
    lines.append("- Bins are model-balanced where possible, but small strata are marked non-reportable in the CSV.")
    lines.append("- Confidence intervals resample prompt clusters within each model family, then average family estimates.")
    lines.append("- Confidence bins use within-model terciles of the native IT baseline IT-vs-PT margin, not calibrated probabilities.")
    categories = sorted((profile.get("by_prompt_category") or {}).keys())
    lines.append(f"- Prompt-category coverage is exactly the categories observed in this run: `{', '.join(categories)}`.")
    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, action="append", default=[Path("data/eval_dataset_v2.jsonl"), Path("data/eval_dataset_v2_holdout_0600_1199.jsonl")])
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readouts", nargs="*", default=[PRIMARY_READOUT, "common_pt"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2301)
    parser.add_argument("--min-units", type=int, default=50)
    parser.add_argument("--min-models", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis" / "subgroups")
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_meta = _load_dataset_metadata(args.dataset)
    units = _load_units(args.run_root, args.models, args.prompt_mode, dataset_meta, args.readouts)
    groups = [
        "prompt_category",
        "event_kind",
        "it_token_category_collapsed",
        "assistant_marker_event",
        "it_assistant_marker",
        "it_margin_tercile_within_model",
        "it_rank_bin",
        "divergence_step_bin",
        "prefix_length_bin",
    ]
    rows = _summarize_subgroups(
        units,
        groups,
        args.readouts,
        n_boot=args.n_bootstrap,
        seed=args.seed,
        min_units=args.min_units,
        min_models=args.min_models,
    )
    profile = _profile_units(units)
    _write_csv(rows, out_dir / "exp23_subgroup_effects.csv")
    (out_dir / "exp23_subgroup_summary.json").write_text(json.dumps({"profile": profile, "rows": rows}, indent=2, sort_keys=True))
    _write_report(rows, profile, out_dir / "exp23_subgroup_report.md")
    _plot_forest(rows, "prompt_category", out_dir / "exp23_interaction_by_prompt_category.png", "Exp23 interaction by prompt category")
    _plot_forest(rows, "it_token_category_collapsed", out_dir / "exp23_interaction_by_it_token_category.png", "Exp23 interaction by IT-token category")
    _plot_forest(rows, "it_margin_tercile_within_model", out_dir / "exp23_interaction_by_it_margin_tercile.png", "Exp23 interaction by IT-margin tercile")
    _plot_forest(rows, "event_kind", out_dir / "exp23_interaction_by_event_kind.png", "Exp23 interaction by event kind")
    _plot_prompt_token_heatmap(rows, units, out_dir / "exp23_interaction_prompt_x_token_heatmap.png")
    print(f"[exp23-subgroups] wrote {out_dir}")


if __name__ == "__main__":
    main()
