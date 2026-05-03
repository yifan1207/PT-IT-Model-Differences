#!/usr/bin/env python3
"""Analyze Exp40 pre-late commitment controls.

The audit joins Exp40 boundary readouts to the existing Exp23 2x2 records.
It tests whether the upstream x late-stack interaction is reducible to
pre-late token commitment.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DENSE5_MODELS = ("gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b")
RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_EXP20_FALLBACK_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final"
)


@dataclass(frozen=True)
class Exp23Unit:
    model: str
    prompt_id: str
    readout: str
    generated_position: int | None
    interaction: float
    late_it_given_pt_upstream: float
    late_it_given_it_upstream: float


@dataclass(frozen=True)
class BoundaryUnit:
    model: str
    prompt_id: str
    readout: str
    generated_position: int | None
    boundary_pt_margin: float
    boundary_it_margin: float
    boundary_delta_margin: float
    final_pt_native_margin: float | None
    final_it_native_margin: float | None


@dataclass(frozen=True)
class JoinedUnit:
    model: str
    prompt_id: str
    readout: str
    generated_position: int | None
    interaction: float
    late_it_given_pt_upstream: float
    late_it_given_it_upstream: float
    boundary_pt_margin: float
    boundary_it_margin: float
    boundary_delta_margin: float
    final_pt_native_margin: float | None
    final_it_native_margin: float | None


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


def _ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lo, hi = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):+.{digits}f}"


def _margin(payload: dict[str, Any], cell: str, readout: str) -> float | None:
    return _finite((((payload.get("cells") or {}).get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))


def _load_exp23_units(
    *,
    exp23_root: Path,
    models: list[str],
    prompt_mode: str,
    readouts: list[str],
) -> tuple[dict[tuple[str, str, str], Exp23Unit], dict[str, Any]]:
    units: dict[tuple[str, str, str], Exp23Unit] = {}
    quality: dict[str, Any] = {"missing_files": [], "valid_events": 0, "invalid_events": 0}
    for model in models:
        path = exp23_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            quality["missing_files"].append(str(path))
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id"))
            event_payload = (row.get("events") or {}).get("first_diff")
            if not isinstance(event_payload, dict) or event_payload.get("duplicate_of"):
                continue
            if not event_payload.get("valid"):
                quality["invalid_events"] += 1
                continue
            quality["valid_events"] += 1
            event = event_payload.get("event") or {}
            generated_position = int(event["step"]) if event.get("step") is not None else None
            for readout in readouts:
                margins = {cell: _margin(event_payload, cell, readout) for cell in RESIDUAL_CELLS}
                if any(value is None for value in margins.values()):
                    continue
                interaction = (
                    margins["U_IT__L_IT"]
                    - margins["U_IT__L_PT"]
                    - margins["U_PT__L_IT"]
                    + margins["U_PT__L_PT"]
                )
                late_it_given_pt = margins["U_PT__L_IT"] - margins["U_PT__L_PT"]
                late_it_given_it = margins["U_IT__L_IT"] - margins["U_IT__L_PT"]
                units[(model, prompt_id, readout)] = Exp23Unit(
                    model=model,
                    prompt_id=prompt_id,
                    readout=readout,
                    generated_position=generated_position,
                    interaction=float(interaction),
                    late_it_given_pt_upstream=float(late_it_given_pt),
                    late_it_given_it_upstream=float(late_it_given_it),
                )
    return units, quality


def _boundary_margin(row: dict[str, Any], upstream: str, readout: str) -> float | None:
    return _finite(
        (((row.get("boundary_readouts") or {}).get(upstream) or {}).get(readout) or {}).get("it_vs_pt_margin")
    )


def _final_native_margin(row: dict[str, Any], variant: str) -> float | None:
    return _finite(((row.get("final_native") or {}).get(variant) or {}).get("it_vs_pt_margin"))


def _load_boundary_units(
    *,
    run_root: Path,
    models: list[str],
    prompt_mode: str,
    readouts: list[str],
) -> tuple[dict[tuple[str, str, str], BoundaryUnit], dict[str, Any]]:
    units: dict[tuple[str, str, str], BoundaryUnit] = {}
    quality: dict[str, Any] = {"missing_files": [], "valid_events": 0, "invalid_events": 0}
    for model in models:
        path = run_root / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            quality["missing_files"].append(str(path))
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id"))
            if not row.get("valid"):
                quality["invalid_events"] += 1
                continue
            quality["valid_events"] += 1
            generated_position = row.get("generated_position")
            generated_position = int(generated_position) if generated_position is not None else None
            for readout in readouts:
                b_pt = _boundary_margin(row, "pt", readout)
                b_it = _boundary_margin(row, "it", readout)
                if b_pt is None or b_it is None:
                    continue
                units[(model, prompt_id, readout)] = BoundaryUnit(
                    model=model,
                    prompt_id=prompt_id,
                    readout=readout,
                    generated_position=generated_position,
                    boundary_pt_margin=float(b_pt),
                    boundary_it_margin=float(b_it),
                    boundary_delta_margin=float(b_it - b_pt),
                    final_pt_native_margin=_final_native_margin(row, "pt"),
                    final_it_native_margin=_final_native_margin(row, "it"),
                )
    return units, quality


def _find_exp20_file(root: Path, fallback_root: Path | None, prompt_mode: str, model: str) -> Path:
    candidates = [
        root / prompt_mode / model / "exp20_validation_records.jsonl",
        root / prompt_mode / model / "exp20_records.jsonl",
    ]
    if fallback_root is not None:
        candidates.extend(
            [
                fallback_root / prompt_mode / model / "exp20_validation_records.jsonl",
                fallback_root / prompt_mode / model / "exp20_records.jsonl",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No Exp20 file for {prompt_mode}/{model}. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _window_margin(condition_payload: dict[str, Any], window: str, endpoint: str) -> float | None:
    return _finite(
        (((condition_payload.get("windows") or {}).get(window) or {}).get("it_minus_pt_margin") or {}).get(endpoint)
    )


def _load_exp20_layerwise_boundary_units(
    *,
    exp20_root: Path,
    exp20_fallback_root: Path | None,
    models: list[str],
    prompt_mode: str,
    readouts: list[str],
) -> tuple[dict[tuple[str, str, str], BoundaryUnit], dict[str, Any]]:
    """Load a CPU-only pre-late commitment proxy from Exp20 layerwise traces.

    Exp20 records store native layerwise token margins for the same
    first-divergence token pair.  The start of the late_reconciliation window is
    the closest stored proxy to the Exp23 boundary state entering the late
    stack.  It is not a common-readout boundary score; the summary labels this
    source explicitly.
    """

    units: dict[tuple[str, str, str], BoundaryUnit] = {}
    quality: dict[str, Any] = {
        "missing_files": [],
        "valid_events": 0,
        "invalid_events": 0,
        "boundary_source_note": "exp20_native_layerwise_late_window_start_proxy",
    }
    for model in models:
        try:
            path = _find_exp20_file(exp20_root, exp20_fallback_root, prompt_mode, model)
        except FileNotFoundError as exc:
            quality["missing_files"].append(str(exc))
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id"))
            payload = (row.get("readouts") or {}).get("first_diff")
            if not isinstance(payload, dict):
                quality["invalid_events"] += 1
                continue
            event = payload.get("event") or {}
            generated_position = int(event["step"]) if event.get("step") is not None else None
            conditions = payload.get("conditions") or {}
            pt_payload = conditions.get("A_pt_raw") or {}
            it_payload = conditions.get("C_it_chat") or {}
            b_pt = _window_margin(pt_payload, "late_reconciliation", "start_value")
            b_it = _window_margin(it_payload, "late_reconciliation", "start_value")
            f_pt = _window_margin(pt_payload, "late_reconciliation", "end_value")
            f_it = _window_margin(it_payload, "late_reconciliation", "end_value")
            if b_pt is None or b_it is None:
                quality["invalid_events"] += 1
                continue
            quality["valid_events"] += 1
            for readout in readouts:
                units[(model, prompt_id, readout)] = BoundaryUnit(
                    model=model,
                    prompt_id=prompt_id,
                    readout=readout,
                    generated_position=generated_position,
                    boundary_pt_margin=float(b_pt),
                    boundary_it_margin=float(b_it),
                    boundary_delta_margin=float(b_it - b_pt),
                    final_pt_native_margin=f_pt,
                    final_it_native_margin=f_it,
                )
    return units, quality


def _join_units(exp23: dict[tuple[str, str, str], Exp23Unit], boundary: dict[tuple[str, str, str], BoundaryUnit]) -> list[JoinedUnit]:
    out: list[JoinedUnit] = []
    for key in sorted(set(exp23) & set(boundary)):
        e = exp23[key]
        b = boundary[key]
        out.append(
            JoinedUnit(
                model=e.model,
                prompt_id=e.prompt_id,
                readout=e.readout,
                generated_position=e.generated_position if e.generated_position is not None else b.generated_position,
                interaction=e.interaction,
                late_it_given_pt_upstream=e.late_it_given_pt_upstream,
                late_it_given_it_upstream=e.late_it_given_it_upstream,
                boundary_pt_margin=b.boundary_pt_margin,
                boundary_it_margin=b.boundary_it_margin,
                boundary_delta_margin=b.boundary_delta_margin,
                final_pt_native_margin=b.final_pt_native_margin,
                final_it_native_margin=b.final_it_native_margin,
            )
        )
    return out


def _family_bootstrap(
    units: list[JoinedUnit],
    metric: Callable[[JoinedUnit], float | None],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {}
    for model in sorted({unit.model for unit in units}):
        values = [metric(unit) for unit in units if unit.model == model]
        arr = np.asarray([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=float)
        if arr.size:
            arrays[model] = arr
    model_estimates = {model: float(arr.mean()) for model, arr in arrays.items()}
    estimate = float(np.mean(list(model_estimates.values()))) if model_estimates else None
    boot_by_model: dict[str, np.ndarray] = {}
    model_cis: dict[str, Any] = {}
    for model, arr in arrays.items():
        if n_boot > 0:
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boot = arr[idx].mean(axis=1)
        else:
            boot = np.array([], dtype=float)
        boot_by_model[model] = boot
        lo, hi = _ci(boot.tolist())
        model_cis[model] = {
            "estimate": model_estimates[model],
            "ci95_low": lo,
            "ci95_high": hi,
            "n": int(arr.size),
            "n_boot": int(boot.size),
        }
    if boot_by_model and len(boot_by_model) == len(arrays):
        boot_dense = np.stack([boot_by_model[model] for model in arrays], axis=0).mean(axis=0)
    else:
        boot_dense = np.array([], dtype=float)
    lo, hi = _ci(boot_dense.tolist())
    return {
        "estimate": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_models": len(arrays),
        "n": int(sum(arr.size for arr in arrays.values())),
        "model_estimates": model_estimates,
        "model_cis": model_cis,
        "n_boot": int(boot_dense.size),
    }


def _assign_terciles(units: list[JoinedUnit], attr: str) -> dict[tuple[str, str, str], str]:
    labels: dict[tuple[str, str, str], str] = {}
    for model in sorted({unit.model for unit in units}):
        model_units = [unit for unit in units if unit.model == model]
        values = np.asarray([float(getattr(unit, attr)) for unit in model_units], dtype=float)
        if values.size < 3:
            continue
        q1, q2 = np.percentile(values, [33.333, 66.667])
        for unit in model_units:
            value = float(getattr(unit, attr))
            if value <= q1:
                label = "low"
            elif value <= q2:
                label = "mid"
            else:
                label = "high"
            labels[(unit.model, unit.prompt_id, unit.readout)] = label
    return labels


METRICS: dict[str, Callable[[JoinedUnit], float | None]] = {
    "interaction": lambda unit: unit.interaction,
    "late_it_given_pt_upstream": lambda unit: unit.late_it_given_pt_upstream,
    "late_it_given_it_upstream": lambda unit: unit.late_it_given_it_upstream,
    "boundary_pt_margin": lambda unit: unit.boundary_pt_margin,
    "boundary_it_margin": lambda unit: unit.boundary_it_margin,
    "boundary_delta_margin": lambda unit: unit.boundary_delta_margin,
    "boundary_it_positive_rate": lambda unit: 1.0 if unit.boundary_it_margin > 0 else 0.0,
    "boundary_pt_positive_rate": lambda unit: 1.0 if unit.boundary_pt_margin > 0 else 0.0,
}


def _summarize_scopes(
    units: list[JoinedUnit],
    *,
    readout: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    readout_units = [unit for unit in units if unit.readout == readout]
    b_it_terciles = _assign_terciles(readout_units, "boundary_it_margin")
    delta_terciles = _assign_terciles(readout_units, "boundary_delta_margin")
    scope_defs: list[tuple[str, Callable[[JoinedUnit], bool], str]] = [
        ("all", lambda unit: True, "All joined first-divergence events"),
        ("boundary_it_le_zero", lambda unit: unit.boundary_it_margin <= 0, "IT boundary does not yet favor t_IT"),
        ("boundary_it_low_tercile", lambda unit: b_it_terciles.get((unit.model, unit.prompt_id, unit.readout)) == "low", "Lowest within-family IT boundary margin tercile"),
        ("boundary_it_mid_tercile", lambda unit: b_it_terciles.get((unit.model, unit.prompt_id, unit.readout)) == "mid", "Middle within-family IT boundary margin tercile"),
        ("boundary_it_high_tercile", lambda unit: b_it_terciles.get((unit.model, unit.prompt_id, unit.readout)) == "high", "Highest within-family IT boundary margin tercile"),
        ("boundary_delta_low_tercile", lambda unit: delta_terciles.get((unit.model, unit.prompt_id, unit.readout)) == "low", "Lowest within-family IT-minus-PT boundary margin tercile"),
        ("boundary_delta_mid_tercile", lambda unit: delta_terciles.get((unit.model, unit.prompt_id, unit.readout)) == "mid", "Middle within-family IT-minus-PT boundary margin tercile"),
        ("boundary_delta_high_tercile", lambda unit: delta_terciles.get((unit.model, unit.prompt_id, unit.readout)) == "high", "Highest within-family IT-minus-PT boundary margin tercile"),
    ]
    out: dict[str, Any] = {}
    for scope_idx, (scope, predicate, description) in enumerate(scope_defs):
        scope_units = [unit for unit in readout_units if predicate(unit)]
        out[scope] = {
            "description": description,
            "n_units": len(scope_units),
            "n_by_model": {model: sum(1 for unit in scope_units if unit.model == model) for model in DENSE5_MODELS},
            "metrics": {
                metric_name: _family_bootstrap(
                    scope_units,
                    metric,
                    n_boot=n_boot,
                    seed=seed + 1000 * scope_idx + metric_idx * 37,
                )
                for metric_idx, (metric_name, metric) in enumerate(METRICS.items())
            },
        }
    return out


def _design_state_rows(units: list[JoinedUnit], *, overlap_only: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if overlap_only:
        ranges: dict[str, tuple[float, float]] = {}
        for model in sorted({unit.model for unit in units}):
            pts = np.asarray([unit.boundary_pt_margin for unit in units if unit.model == model], dtype=float)
            its = np.asarray([unit.boundary_it_margin for unit in units if unit.model == model], dtype=float)
            if pts.size and its.size:
                lo = max(float(np.min(pts)), float(np.min(its)))
                hi = min(float(np.max(pts)), float(np.max(its)))
                if lo <= hi:
                    ranges[model] = (lo, hi)
    else:
        ranges = {}
    models = sorted({unit.model for unit in units})
    rows: list[list[float]] = []
    ys: list[float] = []
    for unit in units:
        for is_it, boundary_margin, effect in (
            (0.0, unit.boundary_pt_margin, unit.late_it_given_pt_upstream),
            (1.0, unit.boundary_it_margin, unit.late_it_given_it_upstream),
        ):
            if overlap_only:
                if unit.model not in ranges:
                    continue
                lo, hi = ranges[unit.model]
                if boundary_margin < lo or boundary_margin > hi:
                    continue
            row = [1.0 if unit.model == model else 0.0 for model in models]
            row.extend([float(boundary_margin), float(is_it)])
            rows.append(row)
            ys.append(float(effect))
    return np.asarray(rows, dtype=float), np.asarray(ys, dtype=float), models


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float] | None:
    if X.shape[0] <= X.shape[1] or X.size == 0:
        return None
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    pred = X @ beta
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return beta, r2


def _state_regression_once(units: list[JoinedUnit], *, overlap_only: bool) -> dict[str, float] | None:
    X, y, models = _design_state_rows(units, overlap_only=overlap_only)
    fit = _ols(X, y)
    if fit is None:
        return None
    beta, r2 = fit
    return {
        "boundary_margin_slope": float(beta[len(models)]),
        "it_upstream_provenance_coef": float(beta[len(models) + 1]),
        "r2": float(r2),
        "n_rows": int(X.shape[0]),
        "n_models": int(len(models)),
    }


def _pair_regression_once(units: list[JoinedUnit]) -> dict[str, float] | None:
    models = sorted({unit.model for unit in units})
    rows: list[list[float]] = []
    ys: list[float] = []
    for unit in units:
        row = [1.0 if unit.model == model else 0.0 for model in models]
        row.append(float(unit.boundary_delta_margin))
        rows.append(row)
        ys.append(float(unit.interaction))
    X = np.asarray(rows, dtype=float)
    y = np.asarray(ys, dtype=float)
    fit = _ols(X, y)
    if fit is None:
        return None
    beta, r2 = fit
    intercepts = beta[: len(models)]
    return {
        "interaction_at_zero_boundary_delta": float(np.mean(intercepts)),
        "boundary_delta_slope": float(beta[len(models)]),
        "r2": float(r2),
        "n_rows": int(X.shape[0]),
        "n_models": int(len(models)),
    }


def _bootstrap_regression(
    units: list[JoinedUnit],
    fit_fn: Callable[[list[JoinedUnit]], dict[str, float] | None],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    point = fit_fn(units)
    if point is None:
        return {"ok": False, "reason": "point_fit_failed"}
    rng = np.random.default_rng(seed)
    by_model: dict[str, list[JoinedUnit]] = defaultdict(list)
    for unit in units:
        by_model[unit.model].append(unit)
    boot_values: dict[str, list[float]] = {key: [] for key in point if key not in {"n_rows", "n_models"}}
    ok = 0
    for _ in range(n_boot):
        sampled: list[JoinedUnit] = []
        for model_units in by_model.values():
            idx = rng.integers(0, len(model_units), size=len(model_units))
            sampled.extend(model_units[int(i)] for i in idx)
        fit = fit_fn(sampled)
        if fit is None:
            continue
        ok += 1
        for key in boot_values:
            boot_values[key].append(float(fit[key]))
    out = {"ok": True, "point": point, "n_boot": ok}
    for key, values in boot_values.items():
        lo, hi = _ci(values)
        out[key] = {
            "estimate": float(point[key]),
            "ci95_low": lo,
            "ci95_high": hi,
        }
    return out


def _summarize_regressions(
    units: list[JoinedUnit],
    *,
    readout: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    readout_units = [unit for unit in units if unit.readout == readout]
    return {
        "state_level_all_support": _bootstrap_regression(
            readout_units,
            lambda sampled: _state_regression_once(sampled, overlap_only=False),
            n_boot=n_boot,
            seed=seed,
        ),
        "state_level_overlap_support": _bootstrap_regression(
            readout_units,
            lambda sampled: _state_regression_once(sampled, overlap_only=True),
            n_boot=n_boot,
            seed=seed + 101,
        ),
        "pair_level_delta_adjusted": _bootstrap_regression(
            readout_units,
            _pair_regression_once,
            n_boot=n_boot,
            seed=seed + 202,
        ),
    }


def _write_effects_csv(summary: dict[str, Any], out_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for readout, payload in (summary.get("readouts") or {}).items():
        for scope, scope_payload in (payload.get("scopes") or {}).items():
            for metric, metric_payload in (scope_payload.get("metrics") or {}).items():
                rows.append(
                    {
                        "readout": readout,
                        "analysis": "scope",
                        "scope": scope,
                        "metric": metric,
                        "estimate": metric_payload.get("estimate"),
                        "ci95_low": metric_payload.get("ci95_low"),
                        "ci95_high": metric_payload.get("ci95_high"),
                        "n_models": metric_payload.get("n_models"),
                        "n": metric_payload.get("n"),
                        "n_boot": metric_payload.get("n_boot"),
                    }
                )
        for regression_name, regression_payload in (payload.get("regressions") or {}).items():
            for metric, metric_payload in regression_payload.items():
                if not isinstance(metric_payload, dict) or "estimate" not in metric_payload:
                    continue
                point = regression_payload.get("point") or {}
                rows.append(
                    {
                        "readout": readout,
                        "analysis": "regression",
                        "scope": regression_name,
                        "metric": metric,
                        "estimate": metric_payload.get("estimate"),
                        "ci95_low": metric_payload.get("ci95_low"),
                        "ci95_high": metric_payload.get("ci95_high"),
                        "n_models": point.get("n_models"),
                        "n": point.get("n_rows"),
                        "n_boot": regression_payload.get("n_boot"),
                    }
                )
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "readout",
                "analysis",
                "scope",
                "metric",
                "estimate",
                "ci95_low",
                "ci95_high",
                "n_models",
                "n",
                "n_boot",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: dict[str, Any], out_path: Path) -> None:
    readout_payload = (summary.get("readouts") or {}).get("common_it") or {}
    scopes = readout_payload.get("scopes") or {}
    names = ["boundary_it_low_tercile", "boundary_it_mid_tercile", "boundary_it_high_tercile"]
    labels = ["Low", "Mid", "High"]
    vals = [((scopes.get(name) or {}).get("metrics") or {}).get("interaction", {}) for name in names]
    y = np.asarray([v.get("estimate", np.nan) for v in vals], dtype=float)
    lo = np.asarray([v.get("ci95_low", np.nan) for v in vals], dtype=float)
    hi = np.asarray([v.get("ci95_high", np.nan) for v in vals], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    x = np.arange(len(names))
    ax.bar(x, y, color=["#4C78A8", "#72B7B2", "#F58518"])
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Upstream x late interaction (logits)")
    ax.set_xlabel("Within-family pre-late IT-boundary margin")
    ax.set_title("Exp40: interaction by pre-late commitment")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_report(summary: dict[str, Any], out_path: Path) -> None:
    common = (summary.get("readouts") or {}).get("common_it") or {}
    scopes = common.get("scopes") or {}
    all_interaction = ((scopes.get("all") or {}).get("metrics") or {}).get("interaction") or {}
    no_commit = ((scopes.get("boundary_it_le_zero") or {}).get("metrics") or {}).get("interaction") or {}
    low = ((scopes.get("boundary_it_low_tercile") or {}).get("metrics") or {}).get("interaction") or {}
    reg = ((common.get("regressions") or {}).get("state_level_all_support") or {}).get("it_upstream_provenance_coef") or {}
    pair = ((common.get("regressions") or {}).get("pair_level_delta_adjusted") or {}).get("interaction_at_zero_boundary_delta") or {}
    text = [
        "# Exp40 Pre-Late Commitment Control",
        "",
        "This audit tests whether the Exp23 upstream x late-stack interaction is reducible to pre-late token commitment.",
        "",
        "## Common-IT Readout",
        "",
        (
            f"- All joined events: interaction `{_fmt(all_interaction.get('estimate'))}` "
            f"`[{_fmt(all_interaction.get('ci95_low'))}, {_fmt(all_interaction.get('ci95_high'))}]`."
        ),
        (
            f"- IT-boundary margin <= 0 subset: interaction `{_fmt(no_commit.get('estimate'))}` "
            f"`[{_fmt(no_commit.get('ci95_low'))}, {_fmt(no_commit.get('ci95_high'))}]`, "
            f"n={no_commit.get('n')}."
        ),
        (
            f"- Lowest within-family IT-boundary-margin tercile: interaction `{_fmt(low.get('estimate'))}` "
            f"`[{_fmt(low.get('ci95_low'))}, {_fmt(low.get('ci95_high'))}]`, n={low.get('n')}."
        ),
        (
            f"- State-level regression, late-stack replacement effect ~ boundary margin + IT-upstream indicator "
            f"+ family fixed effects: IT-upstream coefficient `{_fmt(reg.get('estimate'))}` "
            f"`[{_fmt(reg.get('ci95_low'))}, {_fmt(reg.get('ci95_high'))}]`."
        ),
        (
            f"- Pair-level regression, interaction ~ boundary-margin delta + family fixed effects: "
            f"interaction at zero boundary-margin delta `{_fmt(pair.get('estimate'))}` "
            f"`[{_fmt(pair.get('ci95_low'))}, {_fmt(pair.get('ci95_high'))}]`."
        ),
        "",
        "Interpretation: if the no/low-commitment subsets and commitment-adjusted coefficients remain positive, the Exp23 interaction is not explained by pre-late token commitment alone.",
    ]
    out_path.write_text("\n".join(text) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp40 pre-late commitment control.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--boundary-source",
        choices=["exp40_exact", "exp20_layerwise_proxy"],
        default="exp40_exact",
        help=(
            "exp40_exact loads boundary records collected by "
            "src.poc.exp40_prelate_commitment_control.collect; "
            "exp20_layerwise_proxy uses stored Exp20 native layerwise late-window "
            "start margins as a CPU-only pre-late commitment proxy."
        ),
    )
    parser.add_argument(
        "--exp23-root",
        type=Path,
        default=Path("results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4"),
    )
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readouts", nargs="*", default=["common_it", "common_pt"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp23_units, exp23_quality = _load_exp23_units(
        exp23_root=args.exp23_root,
        models=args.models,
        prompt_mode=args.prompt_mode,
        readouts=args.readouts,
    )
    if args.boundary_source == "exp40_exact":
        boundary_units, boundary_quality = _load_boundary_units(
            run_root=args.run_root,
            models=args.models,
            prompt_mode=args.prompt_mode,
            readouts=args.readouts,
        )
    else:
        boundary_units, boundary_quality = _load_exp20_layerwise_boundary_units(
            exp20_root=args.exp20_root,
            exp20_fallback_root=args.exp20_fallback_root,
            models=args.models,
            prompt_mode=args.prompt_mode,
            readouts=args.readouts,
        )
    joined = _join_units(exp23_units, boundary_units)
    summary = {
        "experiment": "exp40_prelate_commitment_control",
        "boundary_source": args.boundary_source,
        "run_root": str(args.run_root),
        "exp23_root": str(args.exp23_root),
        "exp20_root": str(args.exp20_root),
        "exp20_fallback_root": str(args.exp20_fallback_root),
        "models": args.models,
        "prompt_mode": args.prompt_mode,
        "readouts_requested": args.readouts,
        "quality": {
            "exp23": exp23_quality,
            "exp40": boundary_quality,
            "joined_units": len(joined),
            "joined_primary_prompts": sum(1 for unit in joined if unit.readout == args.readouts[0]),
        },
        "readouts": {
            readout: {
                "scopes": _summarize_scopes(joined, readout=readout, n_boot=args.n_bootstrap, seed=args.seed),
                "regressions": _summarize_regressions(joined, readout=readout, n_boot=args.n_bootstrap, seed=args.seed + 100),
            }
            for readout in args.readouts
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_effects_csv(summary, out_dir / "effects.csv")
    _plot(summary, out_dir / "prelate_commitment_bins.png")
    _write_report(summary, out_dir / "exp40_prelate_commitment_report.md")
    print(f"[exp40] wrote {summary_path}")


if __name__ == "__main__":
    main()
