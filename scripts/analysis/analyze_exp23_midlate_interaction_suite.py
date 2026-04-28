#!/usr/bin/env python3
"""Analyze Exp23 combined mid-late interaction suite outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DENSE5_MODELS = ("gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b")
RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
EFFECTS = {
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
    event_kind: str
    readout: str
    margins: dict[str, float]
    choices: dict[str, str]
    late_kl: dict[str, float | None]
    noop_margin_abs_delta_max: float | None
    valid: bool


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


def _unit_effect(unit: Unit, coefficients: dict[str, float]) -> float | None:
    total = 0.0
    for cell, coef in coefficients.items():
        if cell not in unit.margins:
            return None
        total += coef * unit.margins[cell]
    return total


def _mean(values: list[float]) -> float | None:
    kept = [float(v) for v in values if math.isfinite(float(v))]
    if not kept:
        return None
    return float(sum(kept) / len(kept))


def _percentile_ci(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lo, hi = np.percentile(np.array(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_effect_by_model(
    units_by_model: dict[str, list[Unit]],
    *,
    effect_name: str,
    readout: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    coeffs = EFFECTS[effect_name]
    rng = np.random.default_rng(seed)
    model_arrays: dict[str, np.ndarray] = {}
    model_unit_counts: dict[str, int] = {}
    model_values: dict[str, float | None] = {}
    model_cis: dict[str, dict[str, Any]] = {}
    for model, units in units_by_model.items():
        model_units = [unit for unit in units if unit.readout == readout]
        prompt_values: dict[str, list[float]] = defaultdict(list)
        for unit in model_units:
            value = _unit_effect(unit, coeffs)
            if value is not None and math.isfinite(float(value)):
                prompt_values[unit.prompt_id].append(float(value))
        prompt_means = np.array([float(np.mean(vals)) for vals in prompt_values.values() if vals], dtype=float)
        model_unit_counts[model] = sum(len(vals) for vals in prompt_values.values())
        model_arrays[model] = prompt_means
        model_values[model] = float(prompt_means.mean()) if prompt_means.size else None
        boot_model = np.array([], dtype=float)
        if prompt_means.size and n_boot > 0:
            idx = rng.integers(0, prompt_means.size, size=(n_boot, prompt_means.size))
            boot_model = prompt_means[idx].mean(axis=1)
        lo_model, hi_model = _percentile_ci(boot_model.tolist())
        model_cis[model] = {
            "estimate": model_values[model],
            "ci95_low": lo_model,
            "ci95_high": hi_model,
            "n_units": int(model_unit_counts[model]),
            "n_prompt_clusters": int(prompt_means.size),
            "n_boot": int(boot_model.size),
        }
    valid_models = [model for model, value in model_values.items() if value is not None]
    dense_mean = _mean([model_values[model] for model in valid_models if model_values[model] is not None])

    boot_by_model: dict[str, np.ndarray] = {}
    if valid_models and n_boot > 0:
        for model in valid_models:
            arr = model_arrays[model]
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boot_by_model[model] = arr[idx].mean(axis=1)
    boot = np.stack([boot_by_model[model] for model in valid_models], axis=0).mean(axis=0) if boot_by_model else np.array([], dtype=float)
    lo, hi = _percentile_ci(boot.tolist())
    leave_one_out: dict[str, dict[str, Any]] = {}
    for held_out in valid_models:
        kept_models = [model for model in valid_models if model != held_out]
        point = _mean([model_values[model] for model in kept_models if model_values[model] is not None])
        boot_loo = (
            np.stack([boot_by_model[model] for model in kept_models], axis=0).mean(axis=0)
            if kept_models and boot_by_model
            else np.array([], dtype=float)
        )
        lo_loo, hi_loo = _percentile_ci(boot_loo.tolist())
        leave_one_out[held_out] = {
            "estimate": point,
            "ci95_low": lo_loo,
            "ci95_high": hi_loo,
            "n_models": len(kept_models),
            "n_boot": int(boot_loo.size),
        }
    return {
        "effect": effect_name,
        "readout": readout,
        "estimate": dense_mean,
        "ci95_low": lo,
        "ci95_high": hi,
        "models": model_values,
        "model_cis": model_cis,
        "leave_one_out": leave_one_out,
        "n_models": len(valid_models),
        "n_units": int(sum(model_unit_counts.get(model, 0) for model in valid_models)),
        "n_prompt_clusters": int(sum(model_arrays[model].size for model in valid_models)),
        "bootstrap_unit": "prompt_cluster_within_family",
        "n_boot": int(boot.size),
    }


def _choice_rate(units: list[Unit], *, cell: str, target: str, readout: str) -> float | None:
    vals = [1.0 if unit.choices.get(cell) == target else 0.0 for unit in units if unit.readout == readout and cell in unit.choices]
    return _mean(vals)


def _load_residual_units(run_root: Path, models: list[str], prompt_mode: str, readouts: list[str]) -> tuple[list[Unit], dict[str, Any]]:
    units: list[Unit] = []
    quality = {
        "missing_model_files": [],
        "invalid_events": 0,
        "valid_events": 0,
        "noop_margin_abs_delta_max": None,
    }
    noop_deltas: list[float] = []
    for model in models:
        path = run_root / "residual_factorial" / prompt_mode / model / "records.jsonl.gz"
        if not path.exists():
            quality["missing_model_files"].append(str(path))
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id"))
            for event_kind, payload in (row.get("events") or {}).items():
                if not payload or payload.get("duplicate_of"):
                    continue
                if not payload.get("valid"):
                    quality["invalid_events"] += 1
                    continue
                quality["valid_events"] += 1
                cells = payload.get("cells") or {}
                noop_checks = payload.get("noop_patch_checks") or {}
                for check in noop_checks.values():
                    val = _finite(check.get("common_it_margin_abs_delta"))
                    if val is not None:
                        noop_deltas.append(val)
                for readout in readouts:
                    margins: dict[str, float] = {}
                    choices: dict[str, str] = {}
                    late_kl: dict[str, float | None] = {}
                    for cell in RESIDUAL_CELLS:
                        readout_payload = (cells.get(cell) or {}).get(readout) or {}
                        margin = _finite(readout_payload.get("it_vs_pt_margin"))
                        if margin is not None:
                            margins[cell] = margin
                        choices[cell] = str(readout_payload.get("token_choice_class", "missing"))
                        late_kl[cell] = _finite((readout_payload.get("trajectory") or {}).get("late_kl_mean"))
                    if len(margins) == len(RESIDUAL_CELLS):
                        units.append(
                            Unit(
                                model=model,
                                prompt_id=prompt_id,
                                event_kind=str(event_kind),
                                readout=readout,
                                margins=margins,
                                choices=choices,
                                late_kl=late_kl,
                                noop_margin_abs_delta_max=max(noop_deltas) if noop_deltas else None,
                                valid=True,
                            )
                        )
    quality["noop_margin_abs_delta_max"] = max(noop_deltas) if noop_deltas else None
    return units, quality


def _summarize_residual(
    *,
    run_root: Path,
    models: list[str],
    prompt_mode: str,
    readouts: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    units, quality = _load_residual_units(run_root, models, prompt_mode, readouts)
    units_by_model: dict[str, list[Unit]] = defaultdict(list)
    for unit in units:
        units_by_model[unit.model].append(unit)

    effects = {
        readout: {
            effect: _bootstrap_effect_by_model(
                units_by_model,
                effect_name=effect,
                readout=readout,
                n_boot=n_boot,
                seed=seed + idx * 17,
            )
            for idx, effect in enumerate(EFFECTS)
        }
        for readout in readouts
    }

    choice_rates: dict[str, Any] = {}
    for readout in readouts:
        choice_rates[readout] = {
            cell: {
                "it": _choice_rate(units, cell=cell, target="it", readout=readout),
                "pt": _choice_rate(units, cell=cell, target="pt", readout=readout),
                "other": _choice_rate(units, cell=cell, target="other", readout=readout),
            }
            for cell in RESIDUAL_CELLS
        }

    by_model_counts = {model: len([u for u in units if u.model == model and u.readout == readouts[0]]) for model in models}
    return {
        "part": "residual_factorial",
        "prompt_mode": prompt_mode,
        "readouts": readouts,
        "n_units_by_model": by_model_counts,
        "n_units_total_primary_readout": sum(by_model_counts.values()),
        "quality": quality,
        "effects": effects,
        "choice_rates": choice_rates,
    }


def _maybe_load_json(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def _write_effects_csv(summary: dict[str, Any], out_path: Path) -> None:
    rows = []
    residual = summary.get("residual_factorial") or {}
    for readout, effects in (residual.get("effects") or {}).items():
        for effect, payload in effects.items():
            rows.append(
                {
                    "part": "residual_factorial",
                    "readout": readout,
                    "effect": effect,
                    "scope": "dense_pooled",
                    "held_out_model": "",
                    "estimate": payload.get("estimate"),
                    "ci95_low": payload.get("ci95_low"),
                    "ci95_high": payload.get("ci95_high"),
                    "n_models": payload.get("n_models"),
                    "n_units": payload.get("n_units"),
                    "n_prompt_clusters": payload.get("n_prompt_clusters"),
                    "bootstrap_unit": payload.get("bootstrap_unit"),
                    "n_boot": payload.get("n_boot"),
                }
            )
            for model, model_payload in (payload.get("model_cis") or {}).items():
                rows.append(
                    {
                        "part": "residual_factorial",
                        "readout": readout,
                        "effect": effect,
                        "scope": f"family:{model}",
                        "held_out_model": "",
                        "estimate": model_payload.get("estimate"),
                        "ci95_low": model_payload.get("ci95_low"),
                        "ci95_high": model_payload.get("ci95_high"),
                        "n_models": 1,
                        "n_units": model_payload.get("n_units"),
                        "n_prompt_clusters": model_payload.get("n_prompt_clusters"),
                        "bootstrap_unit": payload.get("bootstrap_unit"),
                        "n_boot": model_payload.get("n_boot"),
                    }
                )
            for held_out, loo_payload in (payload.get("leave_one_out") or {}).items():
                rows.append(
                    {
                        "part": "residual_factorial",
                        "readout": readout,
                        "effect": effect,
                        "scope": "leave_one_family_out",
                        "held_out_model": held_out,
                        "estimate": loo_payload.get("estimate"),
                        "ci95_low": loo_payload.get("ci95_low"),
                        "ci95_high": loo_payload.get("ci95_high"),
                        "n_models": loo_payload.get("n_models"),
                        "n_units": "",
                        "n_prompt_clusters": "",
                        "bootstrap_unit": payload.get("bootstrap_unit"),
                        "n_boot": loo_payload.get("n_boot"),
                    }
                )
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "part",
                "readout",
                "effect",
                "scope",
                "held_out_model",
                "estimate",
                "ci95_low",
                "ci95_high",
                "n_models",
                "n_units",
                "n_prompt_clusters",
                "bootstrap_unit",
                "n_boot",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary(summary: dict[str, Any], out_path: Path) -> None:
    residual = summary.get("residual_factorial") or {}
    effects = (residual.get("effects") or {}).get("common_it") or {}
    names = ["late_weight_effect", "upstream_context_effect", "interaction"]
    vals = [effects.get(name, {}).get("estimate") for name in names]
    lows = [effects.get(name, {}).get("ci95_low") for name in names]
    highs = [effects.get(name, {}).get("ci95_high") for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.0))
    x = np.arange(len(names))
    y = np.array([float(v) if v is not None else np.nan for v in vals])
    yerr_low = np.array([float(v - lo) if v is not None and lo is not None else 0.0 for v, lo in zip(vals, lows, strict=False)])
    yerr_high = np.array([float(hi - v) if v is not None and hi is not None else 0.0 for v, hi in zip(vals, highs, strict=False)])
    axes[0].bar(x, y, color=["#4C78A8", "#F58518", "#54A24B"])
    axes[0].errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", color="black", capsize=3)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["late-stack\nmain", "upstream\nstate", "interaction"])
    axes[0].set_ylabel("IT-vs-PT margin, common IT readout")
    axes[0].set_title("Residual-state x late-stack factorial")

    rates = (residual.get("choice_rates") or {}).get("common_it") or {}
    cell_names = list(RESIDUAL_CELLS)
    it_rates = [rates.get(cell, {}).get("it") for cell in cell_names]
    axes[1].bar(np.arange(len(cell_names)), [float(v) if v is not None else np.nan for v in it_rates], color="#72B7B2")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("P(top-1 is IT divergent token)")
    axes[1].set_xticks(np.arange(len(cell_names)))
    axes[1].set_xticklabels(["PT/PT", "PT/IT", "IT/PT", "IT/IT"], rotation=20)
    axes[1].set_title("Token choice under common IT readout")

    interaction = effects.get("interaction") or {}
    model_cis = interaction.get("model_cis") or {}
    model_names = [model for model in DENSE5_MODELS if model in model_cis]
    if model_names:
        model_est = np.array([float(model_cis[model]["estimate"]) for model in model_names])
        model_lo = np.array([float(model_cis[model]["ci95_low"]) for model in model_names])
        model_hi = np.array([float(model_cis[model]["ci95_high"]) for model in model_names])
        x2 = np.arange(len(model_names))
        axes[2].bar(x2, model_est, color="#B279A2")
        axes[2].errorbar(
            x2,
            model_est,
            yerr=[model_est - model_lo, model_hi - model_est],
            fmt="none",
            color="black",
            capsize=3,
        )
        axes[2].axhline(0, color="black", linewidth=0.8)
        axes[2].set_xticks(x2)
        axes[2].set_xticklabels([model.replace("_", "\n") for model in model_names], fontsize=8)
    axes[2].set_ylabel("Interaction, logits")
    axes[2].set_title("Family-level interaction CIs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp23 mid-late interaction suite.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readouts", nargs="*", default=["common_it", "common_pt", "native_pt", "native_it"])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--part-a-summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    residual = _summarize_residual(
        run_root=args.run_root,
        models=args.models,
        prompt_mode=args.prompt_mode,
        readouts=args.readouts,
        n_boot=args.n_bootstrap,
        seed=args.seed,
    )
    summary = {
        "experiment": "exp23_midlate_interaction_suite",
        "run_root": str(args.run_root),
        "models": args.models,
        "part_a_mlp_kl_factorial": _maybe_load_json(args.part_a_summary),
        "residual_factorial": residual,
    }
    summary_path = out_dir / "exp23_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_effects_csv(summary, out_dir / "exp23_effects.csv")
    _plot_summary(summary, out_dir / "exp23_midlate_interaction.png")
    print(f"[exp23] wrote {summary_path}")


if __name__ == "__main__":
    main()
