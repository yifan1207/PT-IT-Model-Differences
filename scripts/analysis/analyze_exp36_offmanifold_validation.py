#!/usr/bin/env python3
"""Analyze Exp36 off-manifold validation records."""

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
READOUTS = ("common_it", "common_pt")
ANOMALY_FEATURES = ("seq_rms", "last_rms", "seq_std", "last_std")
TRAJECTORY_METRICS = ("late_kl_mean", "remaining_adj_js", "future_top1_flips", "top5_churn")


@dataclass(frozen=True)
class Unit:
    model: str
    prompt_id: str
    readout: str
    position: int
    alphas: tuple[float, ...]
    late_advantage: dict[float, float]
    endpoint_interaction: float
    slope: float
    spearman_alpha: float | None
    monotone: bool
    random_displacements: tuple[float, ...]
    cells: dict[str, dict[str, Any]]
    a_pair: float | None
    a_path: float | None


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


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def _ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lo, hi = np.percentile(np.asarray(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _cell_name(alpha: float, host: str) -> str:
    return f"alpha_{float(alpha):.2f}_L_{host.upper()}"


def _random_cell_name(seed_index: int, host: str) -> str:
    return f"random_{seed_index}_L_{host.upper()}"


def _readout_margin(cells: dict[str, Any], cell_name: str, readout: str) -> float | None:
    return _finite((((cells.get(cell_name) or {}).get("readouts") or {}).get(readout) or {}).get("it_vs_pt_margin"))


def _readout_payload(cells: dict[str, Any], cell_name: str, readout: str) -> dict[str, Any]:
    return (((cells.get(cell_name) or {}).get("readouts") or {}).get(readout) or {})


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    rx = _rankdata(x)
    ry = _rankdata(y)
    sx = float(rx.std())
    sy = float(ry.std())
    if sx == 0.0 or sy == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _linear_slope(xs: list[float], ys: list[float]) -> float:
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered * x_centered))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def _bootstrap_family_mean(
    values_by_model: dict[str, list[float]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    arrays = {
        model: np.asarray([v for v in values if math.isfinite(float(v))], dtype=float)
        for model, values in values_by_model.items()
    }
    arrays = {model: arr for model, arr in arrays.items() if arr.size}
    model_estimates = {model: float(arr.mean()) for model, arr in arrays.items()}
    model_cis: dict[str, Any] = {}
    boot_by_model: dict[str, np.ndarray] = {}
    for model, arr in arrays.items():
        boot = np.array([], dtype=float)
        if n_boot > 0:
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boot = arr[idx].mean(axis=1)
            boot_by_model[model] = boot
        lo, hi = _ci(boot.tolist())
        model_cis[model] = {
            "estimate": model_estimates[model],
            "ci95_low": lo,
            "ci95_high": hi,
            "n": int(arr.size),
            "n_boot": int(boot.size),
        }
    if not arrays:
        return {
            "estimate": None,
            "ci95_low": None,
            "ci95_high": None,
            "model_cis": model_cis,
            "n_models": 0,
            "n": 0,
            "n_boot": 0,
        }
    models = list(arrays)
    estimate = float(np.mean([model_estimates[model] for model in models]))
    boot_dense = (
        np.stack([boot_by_model[model] for model in models], axis=0).mean(axis=0)
        if len(boot_by_model) == len(models)
        else np.array([], dtype=float)
    )
    lo, hi = _ci(boot_dense.tolist())
    return {
        "estimate": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "model_estimates": model_estimates,
        "model_cis": model_cis,
        "n_models": len(models),
        "n": int(sum(arr.size for arr in arrays.values())),
        "n_boot": int(boot_dense.size),
    }


def _bootstrap_units(
    units: list[Unit],
    metric: Callable[[Unit], float | None],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    values_by_model: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        value = metric(unit)
        if value is not None and math.isfinite(float(value)):
            values_by_model[unit.model].append(float(value))
    return _bootstrap_family_mean(values_by_model, n_boot=n_boot, seed=seed)


def _reference_stats(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], tuple[float, float]]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for record in records:
        if not record.get("valid"):
            continue
        model = str(record.get("model"))
        cells = record.get("cells") or {}
        for host, cell_name in (("pt", "alpha_0.00_L_PT"), ("it", "alpha_1.00_L_IT")):
            raw = (cells.get(cell_name) or {}).get("anomaly_raw") or {}
            for feature in ANOMALY_FEATURES:
                value = _finite(raw.get(feature))
                if value is not None:
                    values[(model, host, feature)].append(value)
    out: dict[tuple[str, str, str], tuple[float, float]] = {}
    for key, vals in values.items():
        arr = np.asarray(vals, dtype=float)
        std = float(arr.std(ddof=0))
        out[key] = (float(arr.mean()), std if std > 1e-12 else 1.0)
    return out


def _anomaly_score(record: dict[str, Any], cell_name: str, refs: dict[tuple[str, str, str], tuple[float, float]]) -> float | None:
    model = str(record.get("model"))
    cell = (record.get("cells") or {}).get(cell_name) or {}
    host = str(cell.get("host_variant"))
    raw = cell.get("anomaly_raw") or {}
    scores = []
    for feature in ANOMALY_FEATURES:
        value = _finite(raw.get(feature))
        ref = refs.get((model, host, feature))
        if value is None or ref is None:
            continue
        mean, std = ref
        scores.append(abs((value - mean) / std))
    return float(max(scores)) if scores else None


def _record_anomaly_pair(record: dict[str, Any], refs: dict[tuple[str, str, str], tuple[float, float]]) -> float | None:
    values = [
        _anomaly_score(record, "alpha_0.00_L_IT", refs),
        _anomaly_score(record, "alpha_1.00_L_PT", refs),
    ]
    kept = [v for v in values if v is not None]
    return float(max(kept)) if kept else None


def _record_anomaly_path(record: dict[str, Any], refs: dict[tuple[str, str, str], tuple[float, float]]) -> float | None:
    values = []
    for alpha in record.get("alphas") or []:
        for host in ("pt", "it"):
            value = _anomaly_score(record, _cell_name(float(alpha), host), refs)
            if value is not None:
                values.append(value)
    return float(max(values)) if values else None


def _build_units(records: list[dict[str, Any]], refs: dict[tuple[str, str, str], tuple[float, float]]) -> list[Unit]:
    units: list[Unit] = []
    for record in records:
        if not record.get("valid"):
            continue
        model = str(record.get("model"))
        prompt_id = str(record.get("prompt_id"))
        position = int(record.get("generated_position", -1))
        cells = record.get("cells") or {}
        alphas = tuple(float(alpha) for alpha in (record.get("alphas") or []))
        if not alphas:
            continue
        a_pair = _record_anomaly_pair(record, refs)
        a_path = _record_anomaly_path(record, refs)
        for readout in READOUTS:
            late_advantage: dict[float, float] = {}
            for alpha in alphas:
                y_it_late = _readout_margin(cells, _cell_name(alpha, "it"), readout)
                y_pt_late = _readout_margin(cells, _cell_name(alpha, "pt"), readout)
                if y_it_late is not None and y_pt_late is not None:
                    late_advantage[alpha] = y_it_late - y_pt_late
            if 0.0 not in late_advantage or 1.0 not in late_advantage:
                continue
            ordered_alphas = [alpha for alpha in alphas if alpha in late_advantage]
            ordered_values = [late_advantage[alpha] for alpha in ordered_alphas]
            random_displacements = []
            for seed_idx in range(int(record.get("n_random", 0))):
                y_it_late = _readout_margin(cells, _random_cell_name(seed_idx, "it"), readout)
                y_pt_late = _readout_margin(cells, _random_cell_name(seed_idx, "pt"), readout)
                if y_it_late is not None and y_pt_late is not None:
                    random_displacements.append((y_it_late - y_pt_late) - late_advantage[0.0])
            units.append(
                Unit(
                    model=model,
                    prompt_id=prompt_id,
                    readout=readout,
                    position=position,
                    alphas=alphas,
                    late_advantage=late_advantage,
                    endpoint_interaction=late_advantage[1.0] - late_advantage[0.0],
                    slope=_linear_slope(ordered_alphas, ordered_values),
                    spearman_alpha=_spearman(ordered_alphas, ordered_values),
                    monotone=all(b >= a for a, b in zip(ordered_values, ordered_values[1:], strict=False)),
                    random_displacements=tuple(random_displacements),
                    cells=cells,
                    a_pair=a_pair,
                    a_path=a_path,
                )
            )
    return units


def _load_records(run_root: Path, models: list[str], prompt_mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    quality = {"missing_files": [], "valid_records": 0, "invalid_records": 0}
    for model in models:
        candidates = [
            run_root / prompt_mode / model / "records.jsonl.gz",
            run_root / "raw_shared" / model / "records.jsonl.gz",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            quality["missing_files"].append(str(candidates[0]))
            continue
        for row in _json_rows(path):
            records.append(row)
            if row.get("valid"):
                quality["valid_records"] += 1
            else:
                quality["invalid_records"] += 1
    return records, quality


def _threshold_by_model(units: list[Unit], attr: str, q: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for model in sorted({unit.model for unit in units}):
        vals = [getattr(unit, attr) for unit in units if unit.model == model and getattr(unit, attr) is not None]
        if vals:
            out[model] = float(np.quantile(np.asarray(vals, dtype=float), q))
    return out


def _interaction_summary(units: list[Unit], readout: str, n_boot: int, seed: int) -> dict[str, Any]:
    subset = [unit for unit in units if unit.readout == readout]
    return _bootstrap_units(subset, lambda unit: unit.endpoint_interaction, n_boot=n_boot, seed=seed)


def _dose_response_rows(units: list[Unit], readout: str, n_boot: int, seed: int) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout]
    alphas = sorted({alpha for unit in subset for alpha in unit.late_advantage})
    rows = []
    for idx, alpha in enumerate(alphas):
        effect = _bootstrap_units(subset, lambda unit, alpha=alpha: unit.late_advantage.get(alpha), n_boot=n_boot, seed=seed + idx)
        entropy_values = []
        it_choice_values = []
        for unit in subset:
            payload = _readout_payload(unit.cells, _cell_name(alpha, "it"), readout)
            entropy = _finite(payload.get("entropy"))
            if entropy is not None:
                entropy_values.append(entropy)
            choice = payload.get("token_choice_class")
            if choice is not None:
                it_choice_values.append(1.0 if choice == "it" else 0.0)
        rows.append(
            {
                "readout": readout,
                "alpha": alpha,
                "late_advantage": effect["estimate"],
                "ci95_low": effect["ci95_low"],
                "ci95_high": effect["ci95_high"],
                "top1_it_rate_it_late": _mean(it_choice_values),
                "entropy_it_late": _mean(entropy_values),
                "n": effect["n"],
                "n_models": effect["n_models"],
            }
        )
    return rows


def _slope_rows(units: list[Unit], readout: str, n_boot: int, seed: int) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout]
    rows: list[dict[str, Any]] = []
    dense = _bootstrap_units(subset, lambda unit: unit.slope, n_boot=n_boot, seed=seed)
    positive = _bootstrap_units(subset, lambda unit: 1.0 if unit.slope > 0 else 0.0, n_boot=n_boot, seed=seed + 1)
    monotone = _bootstrap_units(subset, lambda unit: 1.0 if unit.monotone else 0.0, n_boot=n_boot, seed=seed + 2)
    spearman = _bootstrap_units(subset, lambda unit: unit.spearman_alpha, n_boot=n_boot, seed=seed + 3)
    rows.append(
        {
            "scope": "Dense-5 family mean",
            "readout": readout,
            "slope": dense["estimate"],
            "ci95_low": dense["ci95_low"],
            "ci95_high": dense["ci95_high"],
            "prompt_positive_slope_rate": positive["estimate"],
            "monotone_rate": monotone["estimate"],
            "spearman_alpha": spearman["estimate"],
            "n": dense["n"],
        }
    )
    for model, payload in dense.get("model_cis", {}).items():
        model_units = [unit for unit in subset if unit.model == model]
        rows.append(
            {
                "scope": f"family:{model}",
                "readout": readout,
                "slope": payload["estimate"],
                "ci95_low": payload["ci95_low"],
                "ci95_high": payload["ci95_high"],
                "prompt_positive_slope_rate": _mean([1.0 if unit.slope > 0 else 0.0 for unit in model_units]),
                "monotone_rate": _mean([1.0 if unit.monotone else 0.0 for unit in model_units]),
                "spearman_alpha": _mean([unit.spearman_alpha for unit in model_units if unit.spearman_alpha is not None]),
                "n": payload["n"],
            }
        )
    return rows


def _random_rows(units: list[Unit], readout: str, n_boot: int, seed: int) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout]
    observed = _bootstrap_units(subset, lambda unit: unit.endpoint_interaction, n_boot=n_boot, seed=seed)
    random_signed = _bootstrap_units(
        subset,
        lambda unit: _mean(list(unit.random_displacements)) if unit.random_displacements else None,
        n_boot=n_boot,
        seed=seed + 1,
    )
    random_abs = _bootstrap_units(
        subset,
        lambda unit: _mean([abs(value) for value in unit.random_displacements]) if unit.random_displacements else None,
        n_boot=n_boot,
        seed=seed + 2,
    )
    ratio = None
    if observed["estimate"] not in (None, 0) and random_abs["estimate"] is not None:
        ratio = float(random_abs["estimate"] / abs(observed["estimate"]))
    rows = [
        {
            "scope": "Dense-5 family mean",
            "readout": readout,
            "control": "signed_permutation",
            "observed_endpoint_interaction": observed["estimate"],
            "observed_ci95_low": observed["ci95_low"],
            "observed_ci95_high": observed["ci95_high"],
            "random_endpoint_displacement_signed": random_signed["estimate"],
            "random_endpoint_displacement_abs": random_abs["estimate"],
            "random_abs_over_observed_abs": ratio,
            "n": observed["n"],
        }
    ]
    for model, model_obs in observed.get("model_cis", {}).items():
        model_units = [unit for unit in subset if unit.model == model]
        signed_vals = [_mean(list(unit.random_displacements)) for unit in model_units if unit.random_displacements]
        abs_vals = [_mean([abs(value) for value in unit.random_displacements]) for unit in model_units if unit.random_displacements]
        signed = _mean([value for value in signed_vals if value is not None])
        abs_mean = _mean([value for value in abs_vals if value is not None])
        model_ratio = None
        if model_obs["estimate"] not in (None, 0) and abs_mean is not None:
            model_ratio = float(abs_mean / abs(model_obs["estimate"]))
        rows.append(
            {
                "scope": f"family:{model}",
                "readout": readout,
                "control": "signed_permutation",
                "observed_endpoint_interaction": model_obs["estimate"],
                "observed_ci95_low": model_obs["ci95_low"],
                "observed_ci95_high": model_obs["ci95_high"],
                "random_endpoint_displacement_signed": signed,
                "random_endpoint_displacement_abs": abs_mean,
                "random_abs_over_observed_abs": model_ratio,
                "n": model_obs["n"],
            }
        )
    return rows


def _low_anomaly_rows(units: list[Unit], readout: str, n_boot: int, seed: int) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout]
    all_effect = _bootstrap_units(subset, lambda unit: unit.endpoint_interaction, n_boot=n_boot, seed=seed)
    rows = [
        {
            "subset": "all",
            "selector": "none",
            "readout": readout,
            "retained_events": all_effect["n"],
            "interaction": all_effect["estimate"],
            "ci95_low": all_effect["ci95_low"],
            "ci95_high": all_effect["ci95_high"],
            "retention_vs_all": 1.0,
        }
    ]
    selectors = [
        ("low_anomaly_half", 0.50, "A_pair <= model median"),
        ("low_anomaly_tertile", 1.0 / 3.0, "A_pair <= model 33rd percentile"),
        ("drop_top_10pct_anomaly", 0.90, "A_pair <= model 90th percentile"),
    ]
    for idx, (name, quantile, selector) in enumerate(selectors, start=1):
        thresholds = _threshold_by_model(subset, "a_pair", quantile)
        kept = [
            unit
            for unit in subset
            if unit.a_pair is not None and unit.model in thresholds and unit.a_pair <= thresholds[unit.model]
        ]
        effect = _bootstrap_units(kept, lambda unit: unit.endpoint_interaction, n_boot=n_boot, seed=seed + idx)
        retention = None
        if all_effect["estimate"] not in (None, 0) and effect["estimate"] is not None:
            retention = float(effect["estimate"] / all_effect["estimate"])
        rows.append(
            {
                "subset": name,
                "selector": selector,
                "readout": readout,
                "retained_events": effect["n"],
                "interaction": effect["estimate"],
                "ci95_low": effect["ci95_low"],
                "ci95_high": effect["ci95_high"],
                "retention_vs_all": retention,
            }
        )
    return rows


def _anomaly_rows(units: list[Unit], readout: str, n_boot: int, seed: int) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout and unit.a_pair is not None]
    rows: list[dict[str, Any]] = []
    rho_values_by_model: dict[str, list[float]] = defaultdict(list)
    for model in sorted({unit.model for unit in subset}):
        model_units = [unit for unit in subset if unit.model == model]
        rho = _spearman([float(unit.a_pair) for unit in model_units], [unit.endpoint_interaction for unit in model_units])
        if rho is not None:
            rho_values_by_model[model].append(rho)
    dense_rho = _mean([vals[0] for vals in rho_values_by_model.values() if vals])
    for q_idx in range(4):
        kept: list[Unit] = []
        for model in sorted({unit.model for unit in subset}):
            model_units = [unit for unit in subset if unit.model == model]
            values = np.asarray([float(unit.a_pair) for unit in model_units], dtype=float)
            lo = float(np.quantile(values, q_idx / 4.0))
            hi = float(np.quantile(values, (q_idx + 1) / 4.0))
            if q_idx == 3:
                kept.extend([unit for unit in model_units if float(unit.a_pair) >= lo and float(unit.a_pair) <= hi])
            else:
                kept.extend([unit for unit in model_units if float(unit.a_pair) >= lo and float(unit.a_pair) < hi])
        effect = _bootstrap_units(kept, lambda unit: unit.endpoint_interaction, n_boot=n_boot, seed=seed + q_idx)
        rows.append(
            {
                "a_pair_slice": f"quartile_{q_idx + 1}",
                "readout": readout,
                "retained_events": effect["n"],
                "interaction": effect["estimate"],
                "ci95_low": effect["ci95_low"],
                "ci95_high": effect["ci95_high"],
                "spearman_rho_vs_prompt_interaction": dense_rho,
            }
        )
    return rows


def _position_rows(units: list[Unit], readout: str, min_pos: int, n_boot: int, seed: int) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout]
    scopes = [("all_positions", subset), (f"position_ge_{min_pos}", [unit for unit in subset if unit.position >= min_pos])]
    rows = []
    for idx, (scope, kept) in enumerate(scopes):
        interaction = _bootstrap_units(kept, lambda unit: unit.endpoint_interaction, n_boot=n_boot, seed=seed + idx)
        slope = _bootstrap_units(kept, lambda unit: unit.slope, n_boot=n_boot, seed=seed + idx + 10)
        rows.append(
            {
                "scope": scope,
                "readout": readout,
                "retained_events": interaction["n"],
                "interaction": interaction["estimate"],
                "interaction_ci95_low": interaction["ci95_low"],
                "interaction_ci95_high": interaction["ci95_high"],
                "interpolation_slope": slope["estimate"],
                "slope_ci95_low": slope["ci95_low"],
                "slope_ci95_high": slope["ci95_high"],
            }
        )
    return rows


def _degen_rows(units: list[Unit], readout: str) -> list[dict[str, Any]]:
    subset = [unit for unit in units if unit.readout == readout]
    groups = {
        "diagonal_endpoints": ["alpha_0.00_L_PT", "alpha_1.00_L_IT"],
        "offdiag_endpoints": ["alpha_0.00_L_IT", "alpha_1.00_L_PT"],
        "interpolation_mid": [
            "alpha_0.25_L_PT",
            "alpha_0.25_L_IT",
            "alpha_0.50_L_PT",
            "alpha_0.50_L_IT",
            "alpha_0.75_L_PT",
            "alpha_0.75_L_IT",
        ],
    }
    random_names = sorted(
        {
            name
            for unit in subset
            for name in unit.cells
            if name.startswith("random_")
        }
    )
    groups["random_controls"] = random_names

    diag_metric_means: dict[str, float] = {}
    for metric in TRAJECTORY_METRICS:
        vals = []
        for unit in subset:
            for name in groups["diagonal_endpoints"]:
                value = _finite((_readout_payload(unit.cells, name, readout).get("trajectory") or {}).get(metric))
                if value is not None:
                    vals.append(value)
        mean_value = _mean(vals)
        if mean_value is not None:
            diag_metric_means[metric] = mean_value

    rows = []
    for group, names in groups.items():
        entropy_vals = []
        confidence_vals = []
        other_vals = []
        trajectory_ratios = []
        for unit in subset:
            for name in names:
                payload = _readout_payload(unit.cells, name, readout)
                entropy = _finite(payload.get("entropy"))
                confidence = _finite(payload.get("top1_confidence"))
                if entropy is not None:
                    entropy_vals.append(entropy)
                if confidence is not None:
                    confidence_vals.append(confidence)
                if payload.get("token_choice_class") is not None:
                    other_vals.append(1.0 if payload.get("token_choice_class") == "other" else 0.0)
        for metric, diag_mean in diag_metric_means.items():
            vals = []
            for unit in subset:
                for name in names:
                    value = _finite((_readout_payload(unit.cells, name, readout).get("trajectory") or {}).get(metric))
                    if value is not None:
                        vals.append(value)
            group_mean = _mean(vals)
            if group_mean is not None and abs(diag_mean) > 1e-12:
                trajectory_ratios.append(group_mean / diag_mean)
        rows.append(
            {
                "cell_group": group,
                "readout": readout,
                "entropy_mean": _mean(entropy_vals),
                "top1_confidence_mean": _mean(confidence_vals),
                "other_token_rate": _mean(other_vals),
                "worst_trajectory_ratio": max(trajectory_ratios) if trajectory_ratios else None,
                "n_cell_values": len(entropy_vals),
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_exp23_reference(path: Path | None, readout: str) -> float | None:
    if path is None or not path.exists():
        return None
    try:
        summary = json.loads(path.read_text())
        return _finite(
            summary["residual_factorial"]["effects"][readout]["interaction"]["estimate"]
        )
    except Exception:
        return None


def _plot_dose(rows: list[dict[str, Any]], out_path: Path, readout: str) -> None:
    kept = [row for row in rows if row["readout"] == readout]
    x = np.asarray([float(row["alpha"]) for row in kept], dtype=float)
    y = np.asarray([float(row["late_advantage"]) for row in kept], dtype=float)
    lo = np.asarray([float(row["ci95_low"]) for row in kept], dtype=float)
    hi = np.asarray([float(row["ci95_high"]) for row in kept], dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(x, y, marker="o", color="#2878B5")
    ax.fill_between(x, lo, hi, color="#2878B5", alpha=0.18)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("PT-to-IT boundary interpolation alpha")
    ax.set_ylabel("LateAdvantage(alpha), logits")
    ax.set_title(f"Exp36 interpolation dose-response ({readout})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _plot_low_anomaly(rows: list[dict[str, Any]], out_path: Path, readout: str) -> None:
    kept = [row for row in rows if row["readout"] == readout]
    labels = [str(row["subset"]).replace("_", "\n") for row in kept]
    y = np.asarray([float(row["interaction"]) for row in kept], dtype=float)
    lo = np.asarray([float(row["ci95_low"]) for row in kept], dtype=float)
    hi = np.asarray([float(row["ci95_high"]) for row in kept], dtype=float)
    x = np.arange(len(kept))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(x, y, color="#54A24B")
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Endpoint interaction, logits")
    ax.set_title(f"Exp36 low-anomaly robustness ({readout})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _write_report(
    *,
    out_path: Path,
    summary: dict[str, Any],
    dose_rows: list[dict[str, Any]],
    low_rows: list[dict[str, Any]],
    random_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
) -> None:
    primary = summary["readouts"]["common_it"]
    endpoint = primary["endpoint_interaction"]
    slope = primary["slope"]
    exp23_ref = primary.get("exp23_reference_interaction")
    endpoint_diff = primary.get("exp23_endpoint_delta")
    low_half = next(row for row in low_rows if row["readout"] == "common_it" and row["subset"] == "low_anomaly_half")
    random_dense = next(row for row in random_rows if row["readout"] == "common_it" and row["scope"] == "Dense-5 family mean")
    pos_ge = next(row for row in position_rows if row["readout"] == "common_it" and row["scope"].startswith("position_ge_"))
    lines = [
        "# Exp36 Off-Manifold Validation Report",
        "",
        "## Primary Common-IT Readout",
        "",
        f"- Endpoint interaction: `{endpoint['estimate']:.6f}` logits "
        f"(`{endpoint['ci95_low']:.6f}`, `{endpoint['ci95_high']:.6f}`).",
        f"- Interpolation slope: `{slope['estimate']:.6f}` "
        f"(`{slope['ci95_low']:.6f}`, `{slope['ci95_high']:.6f}`).",
        f"- Low-anomaly half interaction: `{low_half['interaction']:.6f}` "
        f"(`{low_half['ci95_low']:.6f}`, `{low_half['ci95_high']:.6f}`), "
        f"retention `{low_half['retention_vs_all']:.3f}`.",
        f"- Signed-permutation random/observed abs ratio: "
        f"`{random_dense['random_abs_over_observed_abs']:.3f}`.",
        f"- Position robustness `{pos_ge['scope']}` interaction: `{pos_ge['interaction']:.6f}`.",
    ]
    if exp23_ref is not None and endpoint_diff is not None:
        lines.append(f"- Exp23 stored interaction reference: `{exp23_ref:.6f}`; Exp36 delta: `{endpoint_diff:.6f}`.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Exp36 does not prove that hybrid states are natural trajectories. It tests whether the Exp23 interaction is graded along the PT-to-IT boundary path, survives pre-output low-anomaly filtering, and is not reproduced by same-norm label-destroyed controls.",
            "",
            "## Dose Response",
            "",
            "| alpha | LateAdvantage | CI low | CI high |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in dose_rows:
        if row["readout"] == "common_it":
            lines.append(
                f"| {row['alpha']:.2f} | {row['late_advantage']:.6f} | {row['ci95_low']:.6f} | {row['ci95_high']:.6f} |"
            )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp36 off-manifold validation.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--position-min", type=int, default=3)
    parser.add_argument(
        "--exp23-summary",
        type=Path,
        default=Path("results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    records, quality = _load_records(args.run_root, args.models, args.prompt_mode)
    refs = _reference_stats(records)
    units = _build_units(records, refs)
    readout_summaries: dict[str, Any] = {}
    all_dose_rows: list[dict[str, Any]] = []
    all_slope_rows: list[dict[str, Any]] = []
    all_random_rows: list[dict[str, Any]] = []
    all_low_rows: list[dict[str, Any]] = []
    all_anomaly_rows: list[dict[str, Any]] = []
    all_degen_rows: list[dict[str, Any]] = []
    all_position_rows: list[dict[str, Any]] = []
    for idx, readout in enumerate(READOUTS):
        endpoint = _interaction_summary(units, readout, args.n_bootstrap, args.seed + idx * 101)
        slope_summary = _bootstrap_units(
            [unit for unit in units if unit.readout == readout],
            lambda unit: unit.slope,
            n_boot=args.n_bootstrap,
            seed=args.seed + idx * 101 + 1,
        )
        exp23_ref = _load_exp23_reference(args.exp23_summary, readout)
        readout_summaries[readout] = {
            "endpoint_interaction": endpoint,
            "slope": slope_summary,
            "exp23_reference_interaction": exp23_ref,
            "exp23_endpoint_delta": endpoint["estimate"] - exp23_ref if endpoint["estimate"] is not None and exp23_ref is not None else None,
        }
        all_dose_rows.extend(_dose_response_rows(units, readout, args.n_bootstrap, args.seed + idx * 101 + 10))
        all_slope_rows.extend(_slope_rows(units, readout, args.n_bootstrap, args.seed + idx * 101 + 20))
        all_random_rows.extend(_random_rows(units, readout, args.n_bootstrap, args.seed + idx * 101 + 30))
        all_low_rows.extend(_low_anomaly_rows(units, readout, args.n_bootstrap, args.seed + idx * 101 + 40))
        all_anomaly_rows.extend(_anomaly_rows(units, readout, args.n_bootstrap, args.seed + idx * 101 + 50))
        all_degen_rows.extend(_degen_rows(units, readout))
        all_position_rows.extend(_position_rows(units, readout, args.position_min, args.n_bootstrap, args.seed + idx * 101 + 60))

    summary = {
        "experiment": "exp36_offmanifold_validation",
        "run_root": str(args.run_root),
        "models": args.models,
        "quality": quality,
        "n_units": len(units),
        "readouts": readout_summaries,
        "bootstrap_unit": "prompt_cluster_within_family",
        "n_bootstrap": args.n_bootstrap,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(out_dir / "interpolation_effects.csv", all_dose_rows + all_slope_rows)
    _write_csv(out_dir / "random_control_effects.csv", all_random_rows)
    _write_csv(out_dir / "anomaly_effects.csv", all_low_rows)
    _write_csv(out_dir / "anomaly_quartiles.csv", all_anomaly_rows)
    _write_csv(out_dir / "degen_metrics.csv", all_degen_rows)
    _write_csv(out_dir / "family_effects.csv", all_slope_rows)
    _write_csv(out_dir / "position_robustness.csv", all_position_rows)
    _plot_dose(all_dose_rows, out_dir / "interpolation_dose_response.png", "common_it")
    _plot_low_anomaly(all_low_rows, out_dir / "low_anomaly_robustness.png", "common_it")
    _write_report(
        out_path=out_dir / "exp36_offmanifold_validation_report.md",
        summary=summary,
        dose_rows=all_dose_rows,
        low_rows=all_low_rows,
        random_rows=all_random_rows,
        position_rows=all_position_rows,
    )
    print(f"[exp36] wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
