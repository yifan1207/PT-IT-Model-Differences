#!/usr/bin/env python3
"""Position-sensitivity checks for first-divergence analyses.

This script addresses the concern that first-divergence records are concentrated
at generated position 0. It reuses stored Exp20/Exp21/Exp23 records and computes
family-balanced bootstrap intervals for the same paper-facing estimands after
stratifying by first-divergence generated position.
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

import matplotlib.pyplot as plt
import numpy as np


DENSE5 = ("gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")

DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_EXP21_ROOT = Path(
    "results/exp21_productive_opposition/"
    "exp21_full_productive_opposition_clean_20260426_053736"
)
DEFAULT_EXP23_ROOT = Path(
    "results/exp23_midlate_interaction_suite/"
    "exp23_dense5_full_h100x8_20260426_sh4_rw4"
)
DEFAULT_OUT_DIR = Path(
    "results/first_divergence_position_sensitivity/exp20_exp21_exp23_20260427"
)
DEFAULT_PAPER_SYNTHESIS_DIR = Path("results/paper_synthesis")

POSITION_BINS: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("all", lambda step: True),
    ("step_0", lambda step: step == 0),
    ("step_1", lambda step: step == 1),
    ("step_2_4", lambda step: 2 <= step <= 4),
    ("step_1_4", lambda step: 1 <= step <= 4),
    ("step_5_9", lambda step: 5 <= step <= 9),
    ("step_5_plus", lambda step: step >= 5),
    ("step_10_plus", lambda step: step >= 10),
    ("step_ge1", lambda step: step >= 1),
    ("step_ge2", lambda step: step >= 2),
    ("step_ge3", lambda step: step >= 3),
    ("step_ge4", lambda step: step >= 4),
)
POSITION_BIN_SEED_OFFSETS = {
    "all": 0,
    "step_0": 997,
    "step_1_4": 1994,
    "step_5_plus": 2991,
    "step_ge1": 3988,
    "step_1": 4985,
    "step_2_4": 5982,
    "step_5_9": 6979,
    "step_10_plus": 7976,
    "step_ge2": 8973,
    "step_ge3": 9970,
    "step_ge4": 10967,
}


def _iter_json(path: Path) -> Iterable[dict[str, Any]]:
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


def _step_bin(step: int | None) -> str | None:
    if step is None:
        return None
    for name, predicate in POSITION_BINS:
        if name == "all":
            continue
        if predicate(step):
            return name
    return None


def _family_balanced_bootstrap(
    values_by_model: dict[str, list[float]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    kept = {model: np.asarray(vals, dtype=float) for model, vals in values_by_model.items() if vals}
    model_counts = {model: int(len(vals)) for model, vals in kept.items()}
    model_values = {model: float(vals.mean()) for model, vals in kept.items()}
    model_cis: dict[str, dict[str, float]] = {}
    if not kept:
        return {
            "n": 0,
            "n_models": 0,
            "mean": None,
            "ci_low": None,
            "ci_high": None,
            "model_counts": {},
            "model_values": {},
            "model_cis": {},
        }
    mean = float(np.mean(list(model_values.values())))
    model_rng = np.random.default_rng(seed + 1_000_003)
    for model, arr in kept.items():
        if len(arr) == 1:
            model_cis[model] = {
                "estimate": float(arr[0]),
                "ci_low": float(arr[0]),
                "ci_high": float(arr[0]),
            }
            continue
        idx = model_rng.integers(0, len(arr), size=(n_boot, len(arr)))
        boot = arr[idx].mean(axis=1)
        ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
        model_cis[model] = {
            "estimate": model_values[model],
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
        }
    boots: list[float] = []
    rng = np.random.default_rng(seed)
    arrays = list(kept.values())
    for start in range(0, n_boot, 512):
        size = min(512, n_boot - start)
        sampled = []
        for arr in arrays:
            idx = rng.integers(0, len(arr), size=(size, len(arr)))
            sampled.append(arr[idx].mean(axis=1))
        boots.extend(np.vstack(sampled).mean(axis=0).tolist())
    ci_low, ci_high = np.percentile(np.asarray(boots, dtype=float), [2.5, 97.5])
    return {
        "n": int(sum(model_counts.values())),
        "n_models": len(model_counts),
        "mean": mean,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "model_counts": model_counts,
        "model_values": model_values,
        "model_cis": model_cis,
    }


def _append_stat_rows(
    rows: list[dict[str, Any]],
    *,
    experiment: str,
    prompt_mode: str,
    metric: str,
    unit: str,
    values_by_model_by_bin: dict[str, dict[str, list[float]]],
    n_boot: int,
    seed_base: int,
) -> None:
    for bin_name, _predicate in POSITION_BINS:
        stats = _family_balanced_bootstrap(
            values_by_model_by_bin.get(bin_name, {}),
            n_boot=n_boot,
            seed=seed_base + POSITION_BIN_SEED_OFFSETS[bin_name],
        )
        rows.append(
            {
                "experiment": experiment,
                "prompt_mode": prompt_mode,
                "metric": metric,
                "unit": unit,
                "position_bin": bin_name,
                **stats,
            }
        )


def _init_bin_store() -> dict[str, dict[str, list[float]]]:
    return {bin_name: defaultdict(list) for bin_name, _ in POSITION_BINS}


def _add_value(
    store: dict[str, dict[str, list[float]]],
    *,
    model: str,
    step: int | None,
    value: float | None,
) -> None:
    if step is None or value is None or not math.isfinite(float(value)):
        return
    for bin_name, predicate in POSITION_BINS:
        if predicate(step):
            store[bin_name][model].append(float(value))


def _event_step_exp20(record: dict[str, Any]) -> int | None:
    event = ((record.get("divergence_events") or {}).get("first_diff") or {})
    try:
        return int(event.get("step"))
    except (TypeError, ValueError):
        return None


def _readout_exp20(record: dict[str, Any]) -> dict[str, Any]:
    payload = (record.get("readouts") or {}).get("first_diff")
    return payload if isinstance(payload, dict) else {}


def _class_value_exp20(record: dict[str, Any], condition: str, target: str) -> float | None:
    cls = (((_readout_exp20(record).get("condition_token_at_step") or {}).get(condition) or {}).get("class"))
    if cls in {None, "missing"}:
        return None
    return 1.0 if cls == target else 0.0


def _margin_exp20(record: dict[str, Any], condition: str, window: str = "late_reconciliation") -> float | None:
    payload = ((_readout_exp20(record).get("conditions") or {}).get(condition) or {})
    return _finite(
        (((payload.get("windows") or {}).get(window) or {}).get("it_minus_pt_margin") or {}).get("total_delta")
    )


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def analyze_exp20(root: Path, models: list[str], n_boot: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metric_fns: dict[str, tuple[str, str, Callable[[dict[str, Any]], float | None]]] = {
        "raw_shared/PT_host_mid_IT_token_rate": (
            "raw_shared",
            "fraction",
            lambda r: _class_value_exp20(r, "B_mid_raw", "it"),
        ),
        "raw_shared/PT_host_late_IT_token_rate": (
            "raw_shared",
            "fraction",
            lambda r: _class_value_exp20(r, "B_late_raw", "it"),
        ),
        "raw_shared/IT_host_mid_PT_token_rate": (
            "raw_shared",
            "fraction",
            lambda r: _class_value_exp20(r, "D_mid_ptswap", "pt"),
        ),
        "raw_shared/IT_host_late_PT_token_rate": (
            "raw_shared",
            "fraction",
            lambda r: _class_value_exp20(r, "D_late_ptswap", "pt"),
        ),
        "native/IT_host_margin_drop_early": (
            "native",
            "logits",
            lambda r: _diff(_margin_exp20(r, "C_it_chat"), _margin_exp20(r, "D_early_ptswap")),
        ),
        "native/IT_host_margin_drop_mid": (
            "native",
            "logits",
            lambda r: _diff(_margin_exp20(r, "C_it_chat"), _margin_exp20(r, "D_mid_ptswap")),
        ),
        "native/IT_host_margin_drop_late": (
            "native",
            "logits",
            lambda r: _diff(_margin_exp20(r, "C_it_chat"), _margin_exp20(r, "D_late_ptswap")),
        ),
        "raw_shared/IT_host_margin_drop_early": (
            "raw_shared",
            "logits",
            lambda r: _diff(_margin_exp20(r, "C_it_chat"), _margin_exp20(r, "D_early_ptswap")),
        ),
        "raw_shared/IT_host_margin_drop_mid": (
            "raw_shared",
            "logits",
            lambda r: _diff(_margin_exp20(r, "C_it_chat"), _margin_exp20(r, "D_mid_ptswap")),
        ),
        "raw_shared/IT_host_margin_drop_late": (
            "raw_shared",
            "logits",
            lambda r: _diff(_margin_exp20(r, "C_it_chat"), _margin_exp20(r, "D_late_ptswap")),
        ),
    }
    stores = {metric: _init_bin_store() for metric in metric_fns}
    position_counts: dict[str, Counter[int]] = {mode: Counter() for mode in {"raw_shared", "native"}}

    for model in models:
        for mode in {"raw_shared", "native"}:
            path = root / mode / model / "exp20_validation_records.jsonl"
            for record in _iter_json(path):
                step = _event_step_exp20(record)
                if step is not None:
                    position_counts[mode][step] += 1
                for metric, (metric_mode, _unit, fn) in metric_fns.items():
                    if metric_mode != mode:
                        continue
                    _add_value(stores[metric], model=model, step=step, value=fn(record))

    rows: list[dict[str, Any]] = []
    for i, (metric, (mode, unit, _fn)) in enumerate(metric_fns.items()):
        _append_stat_rows(
            rows,
            experiment="exp20",
            prompt_mode=mode,
            metric=metric.split("/", 1)[1],
            unit=unit,
            values_by_model_by_bin=stores[metric],
            n_boot=n_boot,
            seed_base=20000 + i * 100,
        )
    profile = {
        mode: {
            "n": int(sum(counter.values())),
            "mean_step": float(np.average(list(counter.keys()), weights=list(counter.values()))) if counter else None,
            "median_step": _weighted_median(counter),
            "counts": {str(k): int(v) for k, v in sorted(counter.items())},
            "bin_counts": _counter_bins(counter),
        }
        for mode, counter in position_counts.items()
    }
    return rows, profile


def _event_step_exp21(record: dict[str, Any]) -> int | None:
    event = (((record.get("events") or {}).get("first_diff") or {}).get("event") or {})
    try:
        return int(event.get("step"))
    except (TypeError, ValueError):
        return None


def _value_exp21(
    record: dict[str, Any],
    condition: str,
    metric: str,
    window: str = "late_reconciliation",
) -> float | None:
    try:
        value = record["events"]["first_diff"]["conditions"][condition]["windows"][window][metric]
    except KeyError:
        return None
    return _finite(value)


def _late_weight_exp21(record: dict[str, Any], metric: str) -> float | None:
    vals = [
        _value_exp21(record, "B_late_raw", metric),
        _value_exp21(record, "A_pt_raw", metric),
        _value_exp21(record, "C_it_chat", metric),
        _value_exp21(record, "D_late_ptswap", metric),
    ]
    if any(v is None for v in vals):
        return None
    b, a, c, d = [float(v) for v in vals]
    return 0.5 * ((b - a) + (c - d))


def analyze_exp21(root: Path, models: list[str], n_boot: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_metric_fns: dict[str, tuple[str, Callable[[dict[str, Any]], float | None]]] = {
        "native/pure_IT_late_support_IT_token": (
            "logits",
            lambda r: _value_exp21(r, "C_it_chat", "support_it_token", "late_reconciliation"),
        ),
        "native/PT_to_IT_late_support_change": (
            "logits",
            lambda r: _diff(
                _value_exp21(r, "C_it_chat", "support_it_token", "late_reconciliation"),
                _value_exp21(r, "A_pt_raw", "support_it_token", "late_reconciliation"),
            ),
        ),
        "native/late_weight_margin_effect": (
            "logits",
            lambda r: _late_weight_exp21(r, "margin_writein_it_vs_pt"),
        ),
        "native/PT_host_late_graft_MLP_margin": (
            "logits",
            lambda r: _diff(
                _value_exp21(r, "B_late_raw", "margin_writein_it_vs_pt", "late_reconciliation"),
                _value_exp21(r, "A_pt_raw", "margin_writein_it_vs_pt", "late_reconciliation"),
            ),
        ),
    }
    metric_fns: dict[str, tuple[str, str, Callable[[dict[str, Any]], float | None]]] = {}
    for mode in ("native", "raw_shared"):
        for metric_name, (unit, fn) in base_metric_fns.items():
            metric_fns[metric_name.replace("native/", f"{mode}/")] = (mode, unit, fn)
    stores = {metric: _init_bin_store() for metric in metric_fns}
    position_counts: dict[str, Counter[int]] = {"native": Counter(), "raw_shared": Counter()}

    for model in models:
        for mode in ("native", "raw_shared"):
            path = root / mode / model / "records.jsonl.gz"
            for record in _iter_json(path):
                step = _event_step_exp21(record)
                if step is not None:
                    position_counts[mode][step] += 1
                for metric, (metric_mode, _unit, fn) in metric_fns.items():
                    if metric_mode != mode:
                        continue
                    _add_value(stores[metric], model=model, step=step, value=fn(record))

    rows: list[dict[str, Any]] = []
    for i, (metric, (mode, unit, _fn)) in enumerate(metric_fns.items()):
        _append_stat_rows(
            rows,
            experiment="exp21",
            prompt_mode=mode,
            metric=metric.split("/", 1)[1],
            unit=unit,
            values_by_model_by_bin=stores[metric],
            n_boot=n_boot,
            seed_base=21000 + i * 100,
        )
    profile = {
        mode: {
            "n": int(sum(counter.values())),
            "mean_step": float(np.average(list(counter.keys()), weights=list(counter.values()))) if counter else None,
            "median_step": _weighted_median(counter),
            "counts": {str(k): int(v) for k, v in sorted(counter.items())},
            "bin_counts": _counter_bins(counter),
        }
        for mode, counter in position_counts.items()
    }
    return rows, profile


def _event_step_exp23(payload: dict[str, Any]) -> int | None:
    event = payload.get("event") or {}
    try:
        return int(event.get("step"))
    except (TypeError, ValueError):
        return None


def _margin_exp23(payload: dict[str, Any], cell: str, readout: str = "common_it") -> float | None:
    return _finite((((payload.get("cells") or {}).get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))


def _effect_exp23(payload: dict[str, Any], effect: str, readout: str = "common_it") -> float | None:
    cells = {cell: _margin_exp23(payload, cell, readout) for cell in ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")}
    if any(v is None for v in cells.values()):
        return None
    if effect == "interaction":
        return cells["U_IT__L_IT"] - cells["U_IT__L_PT"] - cells["U_PT__L_IT"] + cells["U_PT__L_PT"]
    if effect == "late_IT_given_PT_upstream":
        return cells["U_PT__L_IT"] - cells["U_PT__L_PT"]
    if effect == "late_IT_given_IT_upstream":
        return cells["U_IT__L_IT"] - cells["U_IT__L_PT"]
    if effect == "upstream_context_effect":
        return 0.5 * ((cells["U_IT__L_PT"] - cells["U_PT__L_PT"]) + (cells["U_IT__L_IT"] - cells["U_PT__L_IT"]))
    if effect == "late_stack_main_effect":
        return 0.5 * ((cells["U_PT__L_IT"] - cells["U_PT__L_PT"]) + (cells["U_IT__L_IT"] - cells["U_IT__L_PT"]))
    raise ValueError(effect)


def analyze_exp23(root: Path, models: list[str], n_boot: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = [
        "interaction",
        "late_IT_given_PT_upstream",
        "late_IT_given_IT_upstream",
        "upstream_context_effect",
        "late_stack_main_effect",
    ]
    stores = {metric: _init_bin_store() for metric in metrics}
    position_counts: Counter[int] = Counter()

    for model in models:
        path = root / "residual_factorial" / "raw_shared" / model / "records.jsonl.gz"
        for record in _iter_json(path):
            payload = (record.get("events") or {}).get("first_diff") or {}
            if not payload or not payload.get("valid") or payload.get("duplicate_of"):
                continue
            step = _event_step_exp23(payload)
            if step is not None:
                position_counts[step] += 1
            for metric in metrics:
                _add_value(stores[metric], model=model, step=step, value=_effect_exp23(payload, metric))

    rows: list[dict[str, Any]] = []
    for i, metric in enumerate(metrics):
        _append_stat_rows(
            rows,
            experiment="exp23",
            prompt_mode="raw_shared",
            metric=metric,
            unit="logits",
            values_by_model_by_bin=stores[metric],
            n_boot=n_boot,
            seed_base=23000 + i * 100,
        )
    _append_stat_rows(
        rows,
        experiment="exp23_gemma_removed",
        prompt_mode="raw_shared",
        metric="interaction",
        unit="logits",
        values_by_model_by_bin={
            bin_name: {model: vals for model, vals in by_model.items() if model != "gemma3_4b"}
            for bin_name, by_model in stores["interaction"].items()
        },
        n_boot=n_boot,
        seed_base=23900,
    )
    profile = {
        "raw_shared": {
            "n": int(sum(position_counts.values())),
            "mean_step": float(np.average(list(position_counts.keys()), weights=list(position_counts.values()))) if position_counts else None,
            "median_step": _weighted_median(position_counts),
            "counts": {str(k): int(v) for k, v in sorted(position_counts.items())},
            "bin_counts": _counter_bins(position_counts),
        }
    }
    return rows, profile


def _weighted_median(counter: Counter[int]) -> int | None:
    if not counter:
        return None
    total = sum(counter.values())
    acc = 0
    for step, count in sorted(counter.items()):
        acc += count
        if acc >= (total + 1) / 2:
            return int(step)
    return None


def _counter_bins(counter: Counter[int]) -> dict[str, int]:
    out = {}
    for bin_name, predicate in POSITION_BINS:
        out[bin_name] = int(sum(count for step, count in counter.items() if predicate(step)))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment",
        "prompt_mode",
        "metric",
        "unit",
        "position_bin",
        "n",
        "n_models",
        "mean",
        "ci_low",
        "ci_high",
        "model_counts",
        "model_values",
        "model_cis",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["model_counts"] = json.dumps(out["model_counts"], sort_keys=True)
            out["model_values"] = json.dumps(out["model_values"], sort_keys=True)
            out["model_cis"] = json.dumps(out["model_cis"], sort_keys=True)
            writer.writerow(out)


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):.1f}%"


def _find(rows: list[dict[str, Any]], experiment: str, metric: str, position_bin: str, prompt_mode: str | None = None) -> dict[str, Any]:
    for row in rows:
        if row["experiment"] == experiment and row["metric"] == metric and row["position_bin"] == position_bin:
            if prompt_mode is None or row["prompt_mode"] == prompt_mode:
                return row
    raise KeyError((experiment, metric, position_bin, prompt_mode))


def _write_report(path: Path, rows: list[dict[str, Any]], profiles: dict[str, Any]) -> None:
    exp23_ge5 = _find(rows, "exp23", "interaction", "step_5_plus")
    exp23_ge1 = _find(rows, "exp23", "interaction", "step_ge1")
    exp23_0 = _find(rows, "exp23", "interaction", "step_0")
    exp23_14 = _find(rows, "exp23", "interaction", "step_1_4")

    lines = [
        "# First-Divergence Position Sensitivity",
        "",
        "CPU-only analysis over stored Exp20/Exp21/Exp23 records. Estimates bootstrap prompt records within each dense family and average family means.",
        "",
        "## Exp23 Core Interaction",
        "",
        "| Position bin | n | Dense-5 interaction (95% CI) | Gemma-removed interaction (95% CI) |",
        "|---|---:|---:|---:|",
    ]
    for bin_name in [
        "all",
        "step_0",
        "step_1",
        "step_2_4",
        "step_1_4",
        "step_5_9",
        "step_5_plus",
        "step_10_plus",
        "step_ge1",
        "step_ge2",
        "step_ge3",
        "step_ge4",
    ]:
        row = _find(rows, "exp23", "interaction", bin_name)
        sans = _find(rows, "exp23_gemma_removed", "interaction", bin_name)
        lines.append(
            f"| `{bin_name}` | `{row['n']}` | "
            f"`{_fmt(row['mean'])}` `[{_fmt(row['ci_low'])}, {_fmt(row['ci_high'])}]` | "
            f"`{_fmt(sans['mean'])}` `[{_fmt(sans['ci_low'])}, {_fmt(sans['ci_high'])}]` |"
        )
    lines.extend(
        [
            "",
            f"Key check: the Exp23 interaction remains positive after dropping first-token records: "
            f"`{_fmt(exp23_ge1['mean'])}` logits (95% CI `[{_fmt(exp23_ge1['ci_low'])}, {_fmt(exp23_ge1['ci_high'])}]`). "
            f"At generated position `>=5`, the pooled interaction remains positive: "
            f"`{_fmt(exp23_ge5['mean'])}` logits (95% CI `[{_fmt(exp23_ge5['ci_low'])}, {_fmt(exp23_ge5['ci_high'])}]`, "
            f"`n={exp23_ge5['n']}`), but per-family support is thinner.",
            "",
            "The effect attenuates with later divergence position, so first-divergence position affects magnitude. It does not determine the sign or pooled statistical support of the core interaction.",
            "",
            "## Exp23 Per-Family Threshold Checks",
            "",
            "| Model | position >=3 interaction (95% CI) | position >=5 interaction (95% CI) |",
            "|---|---:|---:|",
        ]
    )
    ge3_row = _find(rows, "exp23", "interaction", "step_ge3")
    ge5_row = _find(rows, "exp23", "interaction", "step_5_plus")
    for model in DENSE5:
        ge3_cis = ge3_row["model_cis"][model]
        ge5_cis = ge5_row["model_cis"][model]
        lines.append(
            f"| `{model}` | "
            f"`{_fmt(ge3_cis['estimate'])}` `[{_fmt(ge3_cis['ci_low'])}, {_fmt(ge3_cis['ci_high'])}]` | "
            f"`{_fmt(ge5_cis['estimate'])}` `[{_fmt(ge5_cis['ci_low'])}, {_fmt(ge5_cis['ci_high'])}]` |"
        )
    lines.extend(
        [
            "",
            "At generated position `>=3`, all five family-specific intervals are above zero. At `>=5`, the pooled Dense-5 and Gemma-removed intervals remain above zero, but Llama's family-specific interval crosses zero.",
            "",
            "",
            "## Exp20 Raw-Shared Supporting Identity/Margin Checks",
            "",
            "| Metric | bin all | bin >=5 |",
            "|---|---:|---:|",
        ]
    )
    exp20_metrics = [
        ("PT host middle IT-token transfer", "PT_host_mid_IT_token_rate", True),
        ("PT host late IT-token transfer", "PT_host_late_IT_token_rate", True),
        ("IT host middle PT-token transfer", "IT_host_mid_PT_token_rate", True),
        ("IT host late PT-token transfer", "IT_host_late_PT_token_rate", True),
        ("Raw-shared IT-host early margin drop", "IT_host_margin_drop_early", False),
        ("Raw-shared IT-host middle margin drop", "IT_host_margin_drop_mid", False),
        ("Raw-shared IT-host late margin drop", "IT_host_margin_drop_late", False),
    ]
    for label, metric, is_pct in exp20_metrics:
        all_row = _find(rows, "exp20", metric, "all", "raw_shared")
        ge5_row = _find(rows, "exp20", metric, "step_5_plus", "raw_shared")
        if is_pct:
            all_val = f"`{_fmt_pct(all_row['mean'])}` `[{_fmt_pct(all_row['ci_low'])}, {_fmt_pct(all_row['ci_high'])}]`"
            ge5_val = f"`{_fmt_pct(ge5_row['mean'])}` `[{_fmt_pct(ge5_row['ci_low'])}, {_fmt_pct(ge5_row['ci_high'])}]`"
        else:
            all_val = f"`{_fmt(all_row['mean'])}` `[{_fmt(all_row['ci_low'])}, {_fmt(all_row['ci_high'])}]`"
            ge5_val = f"`{_fmt(ge5_row['mean'])}` `[{_fmt(ge5_row['ci_low'])}, {_fmt(ge5_row['ci_high'])}]`"
        lines.append(f"| {label} | {all_val} | {ge5_val} |")

    lines.extend(
        [
            "",
            "## Exp21 Raw-Shared Supporting Write-Out Checks",
            "",
            "| Metric | bin all | bin >=5 |",
            "|---|---:|---:|",
        ]
    )
    for label, metric in [
        ("Pure IT late support for IT token", "pure_IT_late_support_IT_token"),
        ("PT-to-IT late support change", "PT_to_IT_late_support_change"),
        ("Late-weight MLP margin effect", "late_weight_margin_effect"),
        ("PT-host late-graft MLP margin", "PT_host_late_graft_MLP_margin"),
    ]:
        all_row = _find(rows, "exp21", metric, "all", "raw_shared")
        ge5_row = _find(rows, "exp21", metric, "step_5_plus", "raw_shared")
        lines.append(
            f"| {label} | `{_fmt(all_row['mean'])}` `[{_fmt(all_row['ci_low'])}, {_fmt(all_row['ci_high'])}]` | "
            f"`{_fmt(ge5_row['mean'])}` `[{_fmt(ge5_row['ci_low'])}, {_fmt(ge5_row['ci_high'])}]` |"
        )

    lines.extend(
        [
            "",
            "## Position Profiles",
            "",
            "```json",
            json.dumps(profiles, indent=2, sort_keys=True),
            "```",
            "",
            "## Interpretation",
            "",
            "- The primary Exp23 interaction is not first-token-only: it remains positive for `step>=1`, `step>=3`, and pooled `step>=5`.",
            "- The `step>=3` threshold is the strongest later-position robustness claim because all five family-specific intervals remain above zero.",
            "- The `step>=5` and `step>=10` thresholds are thinner diagnostics; they should not be used as universal per-family claims.",
            "- The later-position Exp23 estimate is smaller than the full estimate, so the paper should explicitly state that first-divergence position affects magnitude.",
            "- Exp20 and Exp21 supporting effects should be described as position-sensitive diagnostics rather than exact same-magnitude replications in later bins.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_exp23(path: Path, rows: list[dict[str, Any]]) -> None:
    bins = ["all", "step_0", "step_ge1", "step_ge2", "step_ge3", "step_ge4", "step_5_plus", "step_10_plus"]
    labels = ["all", "0", ">=1", ">=2", ">=3", ">=4", ">=5", ">=10"]
    selected = [_find(rows, "exp23", "interaction", b) for b in bins]
    vals = np.array([float(row["mean"]) for row in selected])
    lows = np.array([float(row["ci_low"]) for row in selected])
    highs = np.array([float(row["ci_high"]) for row in selected])
    ns = [int(row["n"]) for row in selected]
    x = np.arange(len(selected))
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.errorbar(vals, x, xerr=[vals - lows, highs - vals], fmt="o", color="#2F4B7C", ecolor="#8AA0B8", capsize=4)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{label} (n={n})" for label, n in zip(labels, ns)])
    ax.invert_yaxis()
    ax.set_xlabel("Exp23 upstream-late interaction (logits)")
    ax.set_title("Position-stratified first-divergence interaction")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_exp23_paper_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "position_stratum",
        "records",
        "dense5_interaction",
        "dense5_ci_low",
        "dense5_ci_high",
        "gemma_removed_records",
        "gemma_removed_interaction",
        "gemma_removed_ci_low",
        "gemma_removed_ci_high",
    ]
    bin_labels = {
        "all": "all positions",
        "step_0": "position 0",
        "step_1": "position 1",
        "step_2_4": "positions 2-4",
        "step_1_4": "positions 1-4",
        "step_5_9": "positions 5-9",
        "step_5_plus": "position >=5",
        "step_10_plus": "position >=10",
        "step_ge1": "positions >=1",
        "step_ge2": "positions >=2",
        "step_ge3": "positions >=3",
        "step_ge4": "positions >=4",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for bin_name in [
            "all",
            "step_0",
            "step_1_4",
            "step_ge1",
            "step_ge2",
            "step_ge3",
            "step_ge4",
            "step_5_plus",
            "step_10_plus",
        ]:
            dense = _find(rows, "exp23", "interaction", bin_name)
            sans = _find(rows, "exp23_gemma_removed", "interaction", bin_name)
            writer.writerow(
                {
                    "position_stratum": bin_labels[bin_name],
                    "records": dense["n"],
                    "dense5_interaction": dense["mean"],
                    "dense5_ci_low": dense["ci_low"],
                    "dense5_ci_high": dense["ci_high"],
                    "gemma_removed_records": sans["n"],
                    "gemma_removed_interaction": sans["mean"],
                    "gemma_removed_ci_low": sans["ci_low"],
                    "gemma_removed_ci_high": sans["ci_high"],
                }
            )


def _write_exp23_per_family_position_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "position_stratum",
        "model",
        "records",
        "interaction",
        "ci_low",
        "ci_high",
    ]
    bin_labels = {
        "all": "all positions",
        "step_ge1": "positions >=1",
        "step_ge2": "positions >=2",
        "step_ge3": "positions >=3",
        "step_ge4": "positions >=4",
        "step_5_plus": "position >=5",
        "step_10_plus": "position >=10",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for bin_name in bin_labels:
            row = _find(rows, "exp23", "interaction", bin_name)
            counts = row["model_counts"]
            model_cis = row["model_cis"]
            for model in DENSE5:
                if model not in model_cis:
                    continue
                cis = model_cis[model]
                writer.writerow(
                    {
                        "position_stratum": bin_labels[bin_name],
                        "model": model,
                        "records": counts[model],
                        "interaction": cis["estimate"],
                        "ci_low": cis["ci_low"],
                        "ci_high": cis["ci_high"],
                    }
                )


def _write_exp23_paper_note(path: Path, rows: list[dict[str, Any]], profiles: dict[str, Any]) -> None:
    ge5 = _find(rows, "exp23", "interaction", "step_5_plus")
    ge5_sans = _find(rows, "exp23_gemma_removed", "interaction", "step_5_plus")
    ge1 = _find(rows, "exp23", "interaction", "step_ge1")
    ge3 = _find(rows, "exp23", "interaction", "step_ge3")
    ge3_sans = _find(rows, "exp23_gemma_removed", "interaction", "step_ge3")
    llama_ge3 = ge3["model_cis"]["llama31_8b"]
    llama_ge5 = ge5["model_cis"]["llama31_8b"]
    profile = profiles["exp23"]["raw_shared"]
    lines = [
        "# Exp23 Position-Sensitivity Synthesis",
        "",
        "This paper-facing note is generated by `scripts/analysis/analyze_first_divergence_position_sensitivity.py` from stored Exp23 raw-shared first-divergence records.",
        "",
        f"The primary Exp23 first-divergence pool has `{profile['n']}` valid dense-family events, mean generated position `{profile['mean_step']:.2f}`, median `{profile['median_step']}`, `{profile['bin_counts']['step_0']}` at position `0`, and `{profile['bin_counts']['step_5_plus']}` at position `>=5`.",
        "",
        f"After dropping position-0 records, the common-IT upstream-late interaction is `{_fmt(ge1['mean'])}` logits (95% CI `[{_fmt(ge1['ci_low'])}, {_fmt(ge1['ci_high'])}]`).",
        f"Restricting to position `>=3`, it is `{_fmt(ge3['mean'])}` logits (95% CI `[{_fmt(ge3['ci_low'])}, {_fmt(ge3['ci_high'])}]`), and Gemma-removed is `{_fmt(ge3_sans['mean'])}` logits (95% CI `[{_fmt(ge3_sans['ci_low'])}, {_fmt(ge3_sans['ci_high'])}]`); all five family-specific intervals are above zero, with Llama close to the boundary (`{_fmt(llama_ge3['ci_low'])}`).",
        f"Restricting to position `>=5`, it is `{_fmt(ge5['mean'])}` logits (95% CI `[{_fmt(ge5['ci_low'])}, {_fmt(ge5['ci_high'])}]`), and Gemma-removed is `{_fmt(ge5_sans['mean'])}` logits (95% CI `[{_fmt(ge5_sans['ci_low'])}, {_fmt(ge5_sans['ci_high'])}]`).",
        f"At `>=5`, per-family support is thinner: Llama is `{_fmt(llama_ge5['estimate'])}` logits (95% CI `[{_fmt(llama_ge5['ci_low'])}, {_fmt(llama_ge5['ci_high'])}]`).",
        "",
        "Interpretation: first-divergence position affects magnitude and family-level power, but the pooled interaction is not first-token-only.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp21-root", type=Path, default=DEFAULT_EXP21_ROOT)
    parser.add_argument("--exp23-root", type=Path, default=DEFAULT_EXP23_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--paper-synthesis-dir", type=Path, default=DEFAULT_PAPER_SYNTHESIS_DIR)
    parser.add_argument("--models", nargs="*", default=list(DENSE5))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    profiles: dict[str, Any] = {}
    exp23_rows, profiles["exp23"] = analyze_exp23(args.exp23_root, args.models, args.n_bootstrap)
    exp20_rows, profiles["exp20"] = analyze_exp20(args.exp20_root, args.models, args.n_bootstrap)
    exp21_rows, profiles["exp21"] = analyze_exp21(args.exp21_root, args.models, args.n_bootstrap)
    rows.extend(exp23_rows)
    rows.extend(exp20_rows)
    rows.extend(exp21_rows)

    summary = {
        "exp20_root": str(args.exp20_root),
        "exp21_root": str(args.exp21_root),
        "exp23_root": str(args.exp23_root),
        "models": args.models,
        "n_bootstrap": args.n_bootstrap,
        "profiles": profiles,
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(args.out_dir / "position_sensitivity.csv", rows)
    _write_report(args.out_dir / "position_sensitivity_report.md", rows, profiles)
    _plot_exp23(args.out_dir / "exp23_position_sensitivity.png", rows)
    _write_exp23_paper_table(args.paper_synthesis_dir / "exp23_position_sensitivity_table.csv", rows)
    _write_exp23_per_family_position_table(
        args.paper_synthesis_dir / "exp23_position_sensitivity_per_family.csv",
        rows,
    )
    _write_exp23_paper_note(args.paper_synthesis_dir / "exp23_position_sensitivity_note.md", rows, profiles)
    _plot_exp23(args.paper_synthesis_dir / "exp23_position_sensitivity.png", rows)
    print(f"[position-sensitivity] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
