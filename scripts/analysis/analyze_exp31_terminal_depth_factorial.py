#!/usr/bin/env python3
"""Analyze Exp31 terminal-depth residual-state factorial records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
EFFECTS = {
    "late_it_given_pt_upstream": {
        "U_PT__L_IT": 1.0,
        "U_PT__L_PT": -1.0,
    },
    "late_it_given_it_upstream": {
        "U_IT__L_IT": 1.0,
        "U_IT__L_PT": -1.0,
    },
    "interaction": {
        "U_IT__L_IT": 1.0,
        "U_IT__L_PT": -1.0,
        "U_PT__L_IT": -1.0,
        "U_PT__L_PT": 1.0,
    },
}
STRATA = {
    "all_positions": None,
    "position_ge_3": 3,
    "position_ge_5": 5,
}
MODEL_ORDER = ("gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b")
BOUNDARY_ORDER = ("last3", "last1")


def _json_rows(path: Path):
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


def _records_path(
    *,
    root: Path,
    boundary_mode: str | None,
    prompt_mode: str,
    model: str,
) -> Path | None:
    candidates: list[Path] = []
    if boundary_mode:
        candidates.append(root / "residual_factorial" / boundary_mode / prompt_mode / model / "records.jsonl.gz")
    candidates.extend(
        [
            root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz",
            root / "residual_factorial" / "full_late" / prompt_mode / model / "records.jsonl.gz",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_units(path: Path, *, model: str, readouts: list[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    units: list[dict[str, Any]] = []
    quality = {
        "path": str(path),
        "valid_events": 0,
        "invalid_events": 0,
        "metadata": {},
    }
    for row in _json_rows(path):
        prompt_id = str(row.get("prompt_id"))
        row_meta = {
            "experiment": row.get("experiment"),
            "boundary_mode": row.get("boundary_mode"),
            "boundary_layer": row.get("boundary_layer"),
            "boundary_source": row.get("boundary_source"),
            "downstream_stack": row.get("downstream_stack"),
        }
        if not quality["metadata"]:
            quality["metadata"] = row_meta
        for event_kind, payload in (row.get("events") or {}).items():
            if not isinstance(payload, dict) or payload.get("duplicate_of"):
                continue
            if not payload.get("valid"):
                quality["invalid_events"] += 1
                continue
            quality["valid_events"] += 1
            event = payload.get("event") or {}
            step = int(event.get("step", -1))
            cells = payload.get("cells") or {}
            for readout in readouts:
                margins: dict[str, float] = {}
                for cell in RESIDUAL_CELLS:
                    margin = _finite(((cells.get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))
                    if margin is not None:
                        margins[cell] = margin
                if len(margins) == len(RESIDUAL_CELLS):
                    units.append(
                        {
                            "model": model,
                            "prompt_id": prompt_id,
                            "event_kind": str(event_kind),
                            "step": step,
                            "readout": readout,
                            "margins": margins,
                            **row_meta,
                        }
                    )
    return units, quality


def _unit_effect(unit: dict[str, Any], effect: str) -> float:
    return float(sum(coef * unit["margins"][cell] for cell, coef in EFFECTS[effect].items()))


def _prompt_effects(
    units: list[dict[str, Any]],
    *,
    readout: str,
    effect: str,
    min_step: int | None,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        if unit["readout"] != readout:
            continue
        if min_step is not None and int(unit["step"]) < min_step:
            continue
        grouped[str(unit["prompt_id"])].append(_unit_effect(unit, effect))
    return {prompt_id: float(np.mean(vals)) for prompt_id, vals in grouped.items() if vals}


def _ci(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_model_row(
    *,
    boundary_units: list[dict[str, Any]],
    full_units: list[dict[str, Any]],
    readout: str,
    stratum: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    min_step = STRATA[stratum]
    boundary_prompt_effects = {
        effect: _prompt_effects(boundary_units, readout=readout, effect=effect, min_step=min_step)
        for effect in EFFECTS
    }
    full_interaction = _prompt_effects(full_units, readout=readout, effect="interaction", min_step=min_step)
    common = sorted(set(boundary_prompt_effects["interaction"]) & set(full_interaction))
    out: dict[str, Any] = {
        "n_prompt_clusters": len(common),
        "full_late_overlap": len(common),
    }
    if not common:
        for effect in EFFECTS:
            out[effect] = out[f"{effect}_ci_low"] = out[f"{effect}_ci_high"] = None
        out["full_late_reference_interaction"] = None
        out["retention_fraction"] = out["retention_fraction_ci_low"] = out["retention_fraction_ci_high"] = None
        return out

    values = {
        effect: np.array([boundary_prompt_effects[effect][prompt_id] for prompt_id in common], dtype=float)
        for effect in EFFECTS
    }
    full_values = np.array([full_interaction[prompt_id] for prompt_id in common], dtype=float)
    out["full_late_reference_interaction"] = float(full_values.mean())
    for effect, arr in values.items():
        out[effect] = float(arr.mean())

    if n_boot > 0:
        idx = rng.integers(0, len(common), size=(n_boot, len(common)))
        for effect, arr in values.items():
            lo, hi = _ci(arr[idx].mean(axis=1))
            out[f"{effect}_ci_low"] = lo
            out[f"{effect}_ci_high"] = hi
        full_boot = full_values[idx].mean(axis=1)
        inter_boot = values["interaction"][idx].mean(axis=1)
        ratios = inter_boot / full_boot
        ratios = ratios[np.isfinite(ratios)]
        lo, hi = _ci(ratios)
        out["retention_fraction"] = float(out["interaction"] / out["full_late_reference_interaction"])
        out["retention_fraction_ci_low"] = lo
        out["retention_fraction_ci_high"] = hi
    else:
        for effect in EFFECTS:
            out[f"{effect}_ci_low"] = None
            out[f"{effect}_ci_high"] = None
        out["retention_fraction"] = float(out["interaction"] / out["full_late_reference_interaction"])
        out["retention_fraction_ci_low"] = None
        out["retention_fraction_ci_high"] = None
    return out


def _model_prompt_arrays(
    *,
    boundary_units: list[dict[str, Any]],
    full_units: list[dict[str, Any]],
    readout: str,
    stratum: str,
) -> dict[str, np.ndarray] | None:
    min_step = STRATA[stratum]
    boundary_effects = {
        effect: _prompt_effects(boundary_units, readout=readout, effect=effect, min_step=min_step)
        for effect in EFFECTS
    }
    full_interaction = _prompt_effects(full_units, readout=readout, effect="interaction", min_step=min_step)
    common = sorted(set(boundary_effects["interaction"]) & set(full_interaction))
    if not common:
        return None
    out = {
        effect: np.array([boundary_effects[effect][prompt_id] for prompt_id in common], dtype=float)
        for effect in EFFECTS
    }
    out["full_late_reference_interaction"] = np.array([full_interaction[prompt_id] for prompt_id in common], dtype=float)
    return out


def _aggregate_model_means(model_means: list[dict[str, float]], scope: str) -> dict[str, float]:
    if scope == "family_median":
        reducer = np.median
    elif scope == "trimmed_mean" and len(model_means) >= 3:
        def reducer(values):
            arr = np.sort(np.array(values, dtype=float))
            return float(arr[1:-1].mean())
    else:
        reducer = np.mean
    return {
        effect: float(reducer([row[effect] for row in model_means]))
        for effect in (*EFFECTS.keys(), "full_late_reference_interaction")
    }


def _aggregate_row(
    *,
    arrays_by_model: dict[str, dict[str, np.ndarray]],
    models: list[str],
    scope: str,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    kept = [model for model in models if model in arrays_by_model]
    if not kept:
        empty: dict[str, Any] = {"n_prompt_clusters": 0, "full_late_overlap": 0}
        for effect in EFFECTS:
            empty[effect] = empty[f"{effect}_ci_low"] = empty[f"{effect}_ci_high"] = None
        empty["full_late_reference_interaction"] = None
        empty["retention_fraction"] = empty["retention_fraction_ci_low"] = empty["retention_fraction_ci_high"] = None
        return empty
    model_means = []
    for model in kept:
        arrays = arrays_by_model[model]
        model_means.append(
            {
                key: float(value.mean())
                for key, value in arrays.items()
            }
        )
    point = _aggregate_model_means(model_means, scope)
    out: dict[str, Any] = {
        "n_prompt_clusters": int(sum(arrays_by_model[model]["interaction"].size for model in kept)),
        "full_late_overlap": int(sum(arrays_by_model[model]["interaction"].size for model in kept)),
        "full_late_reference_interaction": point["full_late_reference_interaction"],
    }
    for effect in EFFECTS:
        out[effect] = point[effect]
    out["retention_fraction"] = float(point["interaction"] / point["full_late_reference_interaction"])

    if n_boot <= 0:
        for effect in EFFECTS:
            out[f"{effect}_ci_low"] = None
            out[f"{effect}_ci_high"] = None
        out["retention_fraction_ci_low"] = None
        out["retention_fraction_ci_high"] = None
        return out

    boot_values = {effect: [] for effect in EFFECTS}
    boot_values["retention_fraction"] = []
    for _ in range(n_boot):
        draw_model_means = []
        for model in kept:
            arrays = arrays_by_model[model]
            n = arrays["interaction"].size
            idx = rng.integers(0, n, size=n)
            draw_model_means.append({key: float(value[idx].mean()) for key, value in arrays.items()})
        agg = _aggregate_model_means(draw_model_means, scope)
        for effect in EFFECTS:
            boot_values[effect].append(agg[effect])
        boot_values["retention_fraction"].append(
            float(agg["interaction"] / agg["full_late_reference_interaction"])
        )
    for effect in EFFECTS:
        lo, hi = _ci(np.array(boot_values[effect], dtype=float))
        out[f"{effect}_ci_low"] = lo
        out[f"{effect}_ci_high"] = hi
    lo, hi = _ci(np.array(boot_values["retention_fraction"], dtype=float))
    out["retention_fraction_ci_low"] = lo
    out["retention_fraction_ci_high"] = hi
    return out


def _flatten_row(
    *,
    model: str,
    boundary_mode: str,
    boundary_layer: int | None,
    readout: str,
    stratum: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "boundary_mode": boundary_mode,
        "boundary_layer": boundary_layer,
        "readout": readout,
        "stratum": stratum,
        "n_prompt_clusters": row.get("n_prompt_clusters"),
        "full_late_overlap": row.get("full_late_overlap"),
        "late_it_given_pt_upstream": row.get("late_it_given_pt_upstream"),
        "late_it_given_pt_upstream_ci_low": row.get("late_it_given_pt_upstream_ci_low"),
        "late_it_given_pt_upstream_ci_high": row.get("late_it_given_pt_upstream_ci_high"),
        "late_it_given_it_upstream": row.get("late_it_given_it_upstream"),
        "late_it_given_it_upstream_ci_low": row.get("late_it_given_it_upstream_ci_low"),
        "late_it_given_it_upstream_ci_high": row.get("late_it_given_it_upstream_ci_high"),
        "interaction": row.get("interaction"),
        "interaction_ci_low": row.get("interaction_ci_low"),
        "interaction_ci_high": row.get("interaction_ci_high"),
        "full_late_reference_interaction": row.get("full_late_reference_interaction"),
        "retention_fraction": row.get("retention_fraction"),
        "retention_fraction_ci_low": row.get("retention_fraction_ci_low"),
        "retention_fraction_ci_high": row.get("retention_fraction_ci_high"),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "model",
        "boundary_mode",
        "boundary_layer",
        "readout",
        "stratum",
        "n_prompt_clusters",
        "full_late_overlap",
        "late_it_given_pt_upstream",
        "late_it_given_pt_upstream_ci_low",
        "late_it_given_pt_upstream_ci_high",
        "late_it_given_it_upstream",
        "late_it_given_it_upstream_ci_low",
        "late_it_given_it_upstream_ci_high",
        "interaction",
        "interaction_ci_low",
        "interaction_ci_high",
        "full_late_reference_interaction",
        "retention_fraction",
        "retention_fraction_ci_low",
        "retention_fraction_ci_high",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_retention(rows: list[dict[str, Any]], out_path: Path) -> None:
    plot_rows = [
        row for row in rows
        if row["model"] in {"Dense-5 family mean", "Gemma-removed Dense-4"}
        and row["readout"] == "common_it"
        and row["stratum"] == "all_positions"
        and row["boundary_mode"] in BOUNDARY_ORDER
    ]
    if not plot_rows:
        return
    labels = []
    values = []
    lows = []
    highs = []
    colors = []
    for row in plot_rows:
        labels.append(f"{row['model'].replace(' family mean', '')}\n{row['boundary_mode']}")
        values.append(float(row["retention_fraction"]))
        lows.append(float(row["retention_fraction"] - row["retention_fraction_ci_low"]))
        highs.append(float(row["retention_fraction_ci_high"] - row["retention_fraction"]))
        colors.append("#4C78A8" if row["model"].startswith("Dense-5") else "#F58518")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(values))
    ax.bar(x, values, color=colors)
    ax.errorbar(x, values, yerr=[lows, highs], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Retention fraction vs full-late")
    ax.set_title("Exp31 terminal-depth retention, common-IT readout")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--full-late-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(MODEL_ORDER))
    parser.add_argument("--boundary-modes", nargs="*", default=["last3", "last1"])
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readouts", nargs="*", default=["common_it", "common_pt"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_units: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    quality: dict[str, Any] = {"boundary_runs": {}, "full_late": {}}
    boundary_layers: dict[str, dict[str, int | None]] = defaultdict(dict)

    for boundary_mode in args.boundary_modes:
        for model in args.models:
            path = _records_path(root=args.run_root, boundary_mode=boundary_mode, prompt_mode=args.prompt_mode, model=model)
            if path is None:
                raise FileNotFoundError(f"Missing boundary records for {boundary_mode}/{model} under {args.run_root}")
            units, q = _load_units(path, model=model, readouts=args.readouts)
            all_units[boundary_mode][model] = units
            quality["boundary_runs"][f"{boundary_mode}/{model}"] = q
            boundary_layers[boundary_mode][model] = q.get("metadata", {}).get("boundary_layer")

    full_units: dict[str, list[dict[str, Any]]] = {}
    for model in args.models:
        path = _records_path(root=args.run_root, boundary_mode="full_late", prompt_mode=args.prompt_mode, model=model)
        if path is None:
            path = _records_path(root=args.full_late_root, boundary_mode=None, prompt_mode=args.prompt_mode, model=model)
        if path is None:
            raise FileNotFoundError(f"Missing full-late records for {model} in {args.run_root} or {args.full_late_root}")
        units, q = _load_units(path, model=model, readouts=args.readouts)
        full_units[model] = units
        quality["full_late"][model] = q

    rows: list[dict[str, Any]] = []
    nested: dict[str, Any] = defaultdict(lambda: defaultdict(dict))

    for boundary_mode in args.boundary_modes:
        for readout in args.readouts:
            for stratum in STRATA:
                arrays_by_model: dict[str, dict[str, np.ndarray]] = {}
                for model in args.models:
                    row = _bootstrap_model_row(
                        boundary_units=all_units[boundary_mode][model],
                        full_units=full_units[model],
                        readout=readout,
                        stratum=stratum,
                        n_boot=args.n_bootstrap,
                        rng=rng,
                    )
                    nested[model].setdefault(boundary_mode, {}).setdefault(readout, {})[stratum] = row
                    rows.append(
                        _flatten_row(
                            model=model,
                            boundary_mode=boundary_mode,
                            boundary_layer=boundary_layers[boundary_mode][model],
                            readout=readout,
                            stratum=stratum,
                            row=row,
                        )
                    )
                    arrs = _model_prompt_arrays(
                        boundary_units=all_units[boundary_mode][model],
                        full_units=full_units[model],
                        readout=readout,
                        stratum=stratum,
                    )
                    if arrs is not None:
                        arrays_by_model[model] = arrs

                scopes = {
                    "Dense-5 family mean": list(args.models),
                    "Gemma-removed Dense-4": [model for model in args.models if model != "gemma3_4b"],
                    "family median": list(args.models),
                    "trimmed mean": list(args.models),
                }
                for scope, scope_models in scopes.items():
                    agg_scope = "family_median" if scope == "family median" else ("trimmed_mean" if scope == "trimmed mean" else "mean")
                    row = _aggregate_row(
                        arrays_by_model=arrays_by_model,
                        models=scope_models,
                        scope=agg_scope,
                        n_boot=args.n_bootstrap,
                        rng=rng,
                    )
                    nested[scope].setdefault(boundary_mode, {}).setdefault(readout, {})[stratum] = row
                    rows.append(
                        _flatten_row(
                            model=scope,
                            boundary_mode=boundary_mode,
                            boundary_layer=None,
                            readout=readout,
                            stratum=stratum,
                            row=row,
                        )
                    )

    summary = {
        "experiment": "exp31_terminal_depth_factorial",
        "run_root": str(args.run_root),
        "full_late_root": str(args.full_late_root),
        "models": args.models,
        "boundary_modes": args.boundary_modes,
        "readouts": args.readouts,
        "strata": list(STRATA),
        "bootstrap_unit": "prompt_cluster_within_model_equal_family_weight",
        "n_bootstrap": args.n_bootstrap,
        "quality": quality,
        "summary_rows": nested,
    }
    (out_dir / "terminal_depth_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(rows, out_dir / "terminal_depth_effects.csv")
    _write_csv([row for row in rows if row["model"] in args.models], out_dir / "terminal_depth_family_table.csv")
    _write_csv([row for row in rows if row["stratum"] != "all_positions"], out_dir / "terminal_depth_position_table.csv")
    _plot_retention(rows, out_dir / "terminal_depth_retention.png")
    print(f"[exp31] wrote {out_dir / 'terminal_depth_summary.json'}")


if __name__ == "__main__":
    main()
