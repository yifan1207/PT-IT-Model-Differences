#!/usr/bin/env python3
"""Compare Exp29 boundary-29 factorial against a same-prompt full-late reference."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
EFFECTS = {
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


def _event_units(run_root: Path, *, model: str, prompt_mode: str, readouts: list[str]) -> list[dict[str, Any]]:
    path = run_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
    if not path.exists():
        raise FileNotFoundError(path)
    units: list[dict[str, Any]] = []
    for row in _json_rows(path):
        prompt_id = str(row.get("prompt_id"))
        row_meta = {
            "experiment": row.get("experiment"),
            "boundary_layer": row.get("boundary_layer"),
            "boundary_source": row.get("boundary_source"),
            "downstream_stack": row.get("downstream_stack"),
        }
        for event_kind, payload in (row.get("events") or {}).items():
            if not isinstance(payload, dict) or payload.get("duplicate_of") or not payload.get("valid"):
                continue
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
                            "prompt_id": prompt_id,
                            "event_kind": str(event_kind),
                            "step": step,
                            "readout": readout,
                            "margins": margins,
                            **row_meta,
                        }
                    )
    return units


def _unit_effect(unit: dict[str, Any], effect: str) -> float:
    return float(sum(coef * unit["margins"][cell] for cell, coef in EFFECTS[effect].items()))


def _prompt_means(units: list[dict[str, Any]], *, readout: str, effect: str, min_step: int | None) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        if unit["readout"] != readout:
            continue
        if min_step is not None and int(unit["step"]) < min_step:
            continue
        grouped[str(unit["prompt_id"])].append(_unit_effect(unit, effect))
    return {prompt_id: float(np.mean(vals)) for prompt_id, vals in grouped.items() if vals}


def _bootstrap_mean(values: np.ndarray, *, n_boot: int, seed: int) -> tuple[float | None, float | None, float | None]:
    if values.size == 0:
        return None, None, None
    point = float(values.mean())
    if n_boot <= 0:
        return point, None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def _bootstrap_fraction(
    numerator: dict[str, float],
    denominator: dict[str, float],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    common = sorted(set(numerator) & set(denominator))
    if not common:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n_prompt_clusters": 0}
    num = np.array([numerator[prompt_id] for prompt_id in common], dtype=float)
    den = np.array([denominator[prompt_id] for prompt_id in common], dtype=float)
    point = float(num.mean() / den.mean()) if den.mean() else None
    lo = hi = None
    if n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(common), size=(n_boot, len(common)))
        den_boot = den[idx].mean(axis=1)
        num_boot = num[idx].mean(axis=1)
        ratios = num_boot / den_boot
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            lo, hi = np.percentile(ratios, [2.5, 97.5])
            lo = float(lo)
            hi = float(hi)
    return {
        "estimate": point,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_prompt_clusters": len(common),
    }


def _summarize_run(
    units: list[dict[str, Any]],
    *,
    readouts: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for readout in readouts:
        out[readout] = {}
        for stratum, min_step in (("all_positions", None), ("position_ge_3", 3)):
            out[readout][stratum] = {}
            for effect in EFFECTS:
                means = _prompt_means(units, readout=readout, effect=effect, min_step=min_step)
                values = np.array(list(means.values()), dtype=float)
                estimate, lo, hi = _bootstrap_mean(values, n_boot=n_boot, seed=seed + hash((readout, stratum, effect)) % 10000)
                out[readout][stratum][effect] = {
                    "estimate": estimate,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_prompt_clusters": int(values.size),
                }
    return out


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run_label in ("last3", "full_late_reference"):
        for readout, strata in summary["runs"][run_label]["effects"].items():
            for stratum, effects in strata.items():
                for effect, payload in effects.items():
                    rows.append(
                        {
                            "run_label": run_label,
                            "readout": readout,
                            "stratum": stratum,
                            "effect": effect,
                            **payload,
                        }
                    )
    for readout, strata in summary["last3_fraction_of_full_late"].items():
        for stratum, payload in strata.items():
            rows.append(
                {
                    "run_label": "last3_fraction_of_full_late",
                    "readout": readout,
                    "stratum": stratum,
                    "effect": "interaction_fraction",
                    **payload,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_label",
                "readout",
                "stratum",
                "effect",
                "estimate",
                "ci95_low",
                "ci95_high",
                "n_prompt_clusters",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last3-run-root", type=Path, required=True)
    parser.add_argument("--full-late-run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model", default="llama31_8b")
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readouts", nargs="*", default=["common_it", "common_pt"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.last3_run_root / "analysis" / "exp29_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    last3_units = _event_units(args.last3_run_root, model=args.model, prompt_mode=args.prompt_mode, readouts=args.readouts)
    full_units = _event_units(args.full_late_run_root, model=args.model, prompt_mode=args.prompt_mode, readouts=args.readouts)
    last3_effects = _summarize_run(last3_units, readouts=args.readouts, n_boot=args.n_bootstrap, seed=args.seed)
    full_effects = _summarize_run(full_units, readouts=args.readouts, n_boot=args.n_bootstrap, seed=args.seed + 1000)

    fractions: dict[str, Any] = {}
    for readout in args.readouts:
        fractions[readout] = {}
        for stratum, min_step in (("all_positions", None), ("position_ge_3", 3)):
            last3_interaction = _prompt_means(last3_units, readout=readout, effect="interaction", min_step=min_step)
            full_interaction = _prompt_means(full_units, readout=readout, effect="interaction", min_step=min_step)
            fractions[readout][stratum] = _bootstrap_fraction(
                last3_interaction,
                full_interaction,
                n_boot=args.n_bootstrap,
                seed=args.seed + hash((readout, stratum, "fraction")) % 10000,
            )

    summary = {
        "experiment": "exp29_last3_stack_factorial",
        "model": args.model,
        "prompt_mode": args.prompt_mode,
        "runs": {
            "last3": {
                "run_root": str(args.last3_run_root),
                "n_units": len(last3_units),
                "metadata_sample": {
                    key: last3_units[0].get(key) if last3_units else None
                    for key in ("experiment", "boundary_layer", "boundary_source", "downstream_stack")
                },
                "effects": last3_effects,
            },
            "full_late_reference": {
                "run_root": str(args.full_late_run_root),
                "n_units": len(full_units),
                "metadata_sample": {
                    key: full_units[0].get(key) if full_units else None
                    for key in ("experiment", "boundary_layer", "boundary_source", "downstream_stack")
                },
                "effects": full_effects,
            },
        },
        "last3_fraction_of_full_late": fractions,
        "bootstrap_unit": "prompt_cluster",
        "n_bootstrap": args.n_bootstrap,
    }
    (out_dir / "exp29_last3_comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(summary, out_dir / "exp29_last3_comparison_effects.csv")
    print(json.dumps(summary["last3_fraction_of_full_late"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
