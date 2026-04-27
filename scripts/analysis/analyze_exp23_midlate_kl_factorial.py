#!/usr/bin/env python3
"""Analyze Exp23 matched-prefix mid x late KL factorial outputs.

The runner output is the Exp14/Exp11 ``step_metrics.jsonl`` format. This
script recomputes final-20% ``KL(layer || own final)`` means and bootstraps the
mid+late interaction from prompt-level records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


MODEL_ORDER = [
    "gemma3_4b",
    "qwen3_4b",
    "qwen25_32b",
    "llama31_8b",
    "mistral_7b",
    "olmo2_7b",
    "olmo2_32b",
    "deepseek_v2_lite",
]
DENSE5_MODELS = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]

PT_CONDITIONS = ["A_prime_raw", "B_mid_raw", "B_late_raw", "B_midlate_raw"]
IT_CONDITIONS = ["C_it_chat", "D_mid_ptswap", "D_late_ptswap", "D_midlate_ptswap"]
ALL_CONDITIONS = PT_CONDITIONS + IT_CONDITIONS


@dataclass(frozen=True)
class EffectDef:
    side: str
    name: str
    kind: str
    coefficients: dict[str, float]


@dataclass
class PromptStats:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += float(value)
        self.count += 1


EFFECT_DEFS = [
    EffectDef("pt", "E_mid_pt", "delta", {"B_mid_raw": 1.0, "A_prime_raw": -1.0}),
    EffectDef("pt", "E_late_pt", "delta", {"B_late_raw": 1.0, "A_prime_raw": -1.0}),
    EffectDef("pt", "E_midlate_pt", "delta", {"B_midlate_raw": 1.0, "A_prime_raw": -1.0}),
    EffectDef(
        "pt",
        "I_pt",
        "interaction",
        {"B_midlate_raw": 1.0, "B_mid_raw": -1.0, "B_late_raw": -1.0, "A_prime_raw": 1.0},
    ),
    EffectDef("pt", "L_given_M_pt", "simple_effect", {"B_midlate_raw": 1.0, "B_mid_raw": -1.0}),
    EffectDef("pt", "L_alone_pt", "simple_effect", {"B_late_raw": 1.0, "A_prime_raw": -1.0}),
    EffectDef("it", "E_mid_it", "signed_delta", {"D_mid_ptswap": 1.0, "C_it_chat": -1.0}),
    EffectDef("it", "E_late_it", "signed_delta", {"D_late_ptswap": 1.0, "C_it_chat": -1.0}),
    EffectDef("it", "E_midlate_it", "signed_delta", {"D_midlate_ptswap": 1.0, "C_it_chat": -1.0}),
    EffectDef(
        "it",
        "I_it",
        "signed_interaction",
        {"D_midlate_ptswap": 1.0, "D_mid_ptswap": -1.0, "D_late_ptswap": -1.0, "C_it_chat": 1.0},
    ),
    EffectDef("it", "C_mid_it", "collapse", {"C_it_chat": 1.0, "D_mid_ptswap": -1.0}),
    EffectDef("it", "C_late_it", "collapse", {"C_it_chat": 1.0, "D_late_ptswap": -1.0}),
    EffectDef("it", "C_midlate_it", "collapse", {"C_it_chat": 1.0, "D_midlate_ptswap": -1.0}),
    EffectDef(
        "it",
        "I_collapse_it",
        "collapse_interaction",
        {"D_mid_ptswap": 1.0, "D_late_ptswap": 1.0, "D_midlate_ptswap": -1.0, "C_it_chat": -1.0},
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_jsonl(path: Path):
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


def _collect_model_dirs(run_root: Path, models: list[str]) -> list[tuple[str, Path]]:
    base = run_root / "merged" if (run_root / "merged").exists() else run_root
    subdirs = {path.name: path for path in base.iterdir() if path.is_dir()}
    return [(model, subdirs[model]) for model in models if model in subdirs]


def _final_region(config: dict[str, Any]) -> tuple[int, int]:
    n_layers = int(config["n_layers"])
    late_layers = config.get("late_mechanism_layers")
    if late_layers:
        return int(late_layers[0]), int(late_layers[-1]) + 1
    width = max(1, math.ceil(0.2 * n_layers))
    return n_layers - width, n_layers


def _slice_mean(values: list[Any] | None, start: int, end: int) -> float | None:
    if values is None:
        return None
    subset = [float(value) for value in values[start:end] if value is not None]
    if not subset:
        return None
    return float(sum(subset) / len(subset))


def _prompt_sets(condition_stats: dict[str, dict[str, PromptStats]]) -> dict[str, set[str]]:
    return {condition: set(by_prompt) for condition, by_prompt in condition_stats.items()}


def _common_prompt_ids(model_payload: dict[str, Any], conditions: list[str]) -> list[str]:
    prompt_sets = model_payload["_prompt_sets"]
    missing = [condition for condition in conditions if not prompt_sets.get(condition)]
    if missing:
        return []
    common = set(prompt_sets[conditions[0]])
    for condition in conditions[1:]:
        common &= prompt_sets[condition]
    return sorted(common)


def _condition_mean_from_prompts(
    model_payload: dict[str, Any],
    condition: str,
    prompt_ids: list[str],
) -> float | None:
    stats_by_prompt: dict[str, PromptStats] = model_payload["_prompt_stats"].get(condition, {})
    total = 0.0
    count = 0
    for prompt_id in prompt_ids:
        stats = stats_by_prompt.get(prompt_id)
        if stats is None:
            continue
        total += stats.total
        count += stats.count
    if count == 0:
        return None
    return total / count


def _effect_value(
    model_payload: dict[str, Any],
    effect: EffectDef,
    prompt_ids: list[str],
) -> float | None:
    value = 0.0
    for condition, coefficient in effect.coefficients.items():
        condition_mean = _condition_mean_from_prompts(model_payload, condition, prompt_ids)
        if condition_mean is None:
            return None
        value += coefficient * condition_mean
    return value


def _bootstrap_percentile(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    lo, hi = np.percentile(np.array(values, dtype=float), [2.5, 97.5])
    return float(lo), float(hi)


def _summarize_model(model: str, model_dir: Path) -> dict[str, Any]:
    config = _read_json(model_dir / "config.json")
    step_metrics_path = model_dir / "step_metrics.jsonl"
    if not step_metrics_path.exists():
        raise FileNotFoundError(step_metrics_path)

    final_start, final_end = _final_region(config)
    condition_stats: dict[str, dict[str, PromptStats]] = {condition: {} for condition in ALL_CONDITIONS}
    prompt_modes_seen: dict[str, str | None] = {
        condition: (config.get("pipeline_prompt_modes") or {}).get(condition)
        for condition in ALL_CONDITIONS
    }

    for row in _iter_jsonl(step_metrics_path):
        condition = row.get("pipeline")
        if condition not in condition_stats:
            continue
        value = _slice_mean((row.get("metrics") or {}).get("kl_to_own_final"), final_start, final_end)
        if value is None:
            continue
        prompt_id = str(row["prompt_id"])
        condition_stats[condition].setdefault(prompt_id, PromptStats()).add(value)

    condition_summaries: dict[str, Any] = {}
    for condition, by_prompt in condition_stats.items():
        total = sum(stats.total for stats in by_prompt.values())
        count = sum(stats.count for stats in by_prompt.values())
        condition_summaries[condition] = {
            "mean_final20_kl": total / count if count else None,
            "n_prompt_ids": len(by_prompt),
            "n_token_steps": count,
            "prompt_mode": prompt_modes_seen.get(condition),
        }

    prompt_sets = _prompt_sets(condition_stats)
    expected_modes = {
        "A_prime_raw": "raw_format_b",
        "B_mid_raw": "raw_format_b",
        "B_late_raw": "raw_format_b",
        "B_midlate_raw": "raw_format_b",
        "C_it_chat": "it_chat_template",
        "D_mid_ptswap": "it_chat_template",
        "D_late_ptswap": "it_chat_template",
        "D_midlate_ptswap": "it_chat_template",
    }
    validation = {
        "missing_conditions": [
            condition for condition in ALL_CONDITIONS if not condition_stats[condition]
        ],
        "prompt_counts_by_condition": {
            condition: len(prompt_sets[condition]) for condition in ALL_CONDITIONS
        },
        "pt_common_prompt_count": len(_common_prompt_ids({"_prompt_sets": prompt_sets}, PT_CONDITIONS)),
        "it_common_prompt_count": len(_common_prompt_ids({"_prompt_sets": prompt_sets}, IT_CONDITIONS)),
        "prompt_mode_mismatches": {
            condition: {
                "expected": expected,
                "observed": prompt_modes_seen.get(condition),
            }
            for condition, expected in expected_modes.items()
            if prompt_modes_seen.get(condition) not in {None, expected}
        },
    }

    return {
        "model": model,
        "model_dir": str(model_dir),
        "n_layers": int(config["n_layers"]),
        "final_region": {
            "start_layer": final_start,
            "end_layer_exclusive": final_end,
            "display_range": f"{final_start}-{final_end - 1}",
        },
        "conditions": condition_summaries,
        "validation": validation,
        "_prompt_stats": condition_stats,
        "_prompt_sets": prompt_sets,
    }


def _effect_row(
    *,
    model_name: str,
    model_payload: dict[str, Any],
    effect: EffectDef,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    prompt_ids = _common_prompt_ids(model_payload, list(effect.coefficients))
    observed = _effect_value(model_payload, effect, prompt_ids) if prompt_ids else None
    boots: list[float] = []
    if prompt_ids and n_boot > 0:
        rng = random.Random(seed)
        n = len(prompt_ids)
        for _ in range(n_boot):
            sample = [prompt_ids[rng.randrange(n)] for _ in range(n)]
            value = _effect_value(model_payload, effect, sample)
            if value is not None and math.isfinite(value):
                boots.append(float(value))
    ci_low, ci_high = _bootstrap_percentile(boots)
    return {
        "side": effect.side,
        "model": model_name,
        "effect": effect.name,
        "kind": effect.kind,
        "n_models": 1 if observed is not None else 0,
        "n_prompt_ids": len(prompt_ids),
        "mean": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _pooled_effect_row(
    *,
    model_payloads: dict[str, Any],
    model_names: list[str],
    effect: EffectDef,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    usable_models = [
        model
        for model in model_names
        if model in model_payloads and _common_prompt_ids(model_payloads[model], list(effect.coefficients))
    ]
    per_model_values = [
        _effect_value(
            model_payloads[model],
            effect,
            _common_prompt_ids(model_payloads[model], list(effect.coefficients)),
        )
        for model in usable_models
    ]
    per_model_values = [float(value) for value in per_model_values if value is not None]
    observed = float(np.mean(per_model_values)) if per_model_values else None

    boots: list[float] = []
    if usable_models and n_boot > 0:
        rng = random.Random(seed)
        common_by_model = {
            model: _common_prompt_ids(model_payloads[model], list(effect.coefficients))
            for model in usable_models
        }
        for _ in range(n_boot):
            sampled_model_values: list[float] = []
            for model in usable_models:
                prompt_ids = common_by_model[model]
                n = len(prompt_ids)
                sample = [prompt_ids[rng.randrange(n)] for _ in range(n)]
                value = _effect_value(model_payloads[model], effect, sample)
                if value is not None and math.isfinite(value):
                    sampled_model_values.append(float(value))
            if sampled_model_values:
                boots.append(float(np.mean(sampled_model_values)))
    ci_low, ci_high = _bootstrap_percentile(boots)
    return {
        "side": effect.side,
        "model": "dense5",
        "effect": effect.name,
        "kind": effect.kind,
        "n_models": len(usable_models),
        "n_prompt_ids": sum(
            len(_common_prompt_ids(model_payloads[model], list(effect.coefficients)))
            for model in usable_models
        ),
        "mean": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _dense_condition_means(model_payloads: dict[str, Any], model_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for condition in ALL_CONDITIONS:
        vals = [
            model_payloads[model]["conditions"][condition]["mean_final20_kl"]
            for model in model_names
            if model in model_payloads
            and model_payloads[model]["conditions"][condition]["mean_final20_kl"] is not None
        ]
        out[condition] = float(np.mean(vals)) if vals else None
    return out


def _public_model_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def analyze(run_root: Path, models: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    model_payloads: dict[str, Any] = {}
    for model, model_dir in _collect_model_dirs(run_root, models):
        model_payloads[model] = _summarize_model(model, model_dir)

    dense_models = [model for model in DENSE5_MODELS if model in model_payloads]
    effects: list[dict[str, Any]] = []
    for idx, effect in enumerate(EFFECT_DEFS):
        effects.append(
            _pooled_effect_row(
                model_payloads=model_payloads,
                model_names=dense_models,
                effect=effect,
                n_boot=n_boot,
                seed=seed + 10_000 * idx,
            )
        )
        for model in models:
            if model not in model_payloads:
                continue
            effects.append(
                _effect_row(
                    model_name=model,
                    model_payload=model_payloads[model],
                    effect=effect,
                    n_boot=n_boot,
                    seed=seed + 10_000 * idx + MODEL_ORDER.index(model) + 1,
                )
            )

    return {
        "analysis": "exp23_midlate_kl_factorial",
        "run_root": str(run_root),
        "models_requested": models,
        "models_present": list(model_payloads),
        "dense_models_present": dense_models,
        "dense5_condition_means": _dense_condition_means(model_payloads, dense_models),
        "models": {
            model: _public_model_payload(payload)
            for model, payload in model_payloads.items()
        },
        "effects": effects,
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _write_report(summary: dict[str, Any], out_path: Path) -> None:
    dense_rows = {
        row["effect"]: row
        for row in summary["effects"]
        if row["model"] == "dense5"
    }
    lines = [
        "# Exp23 Mid x Late KL Factorial",
        "",
        f"Run root: `{summary['run_root']}`",
        f"Dense models present: {', '.join(summary['dense_models_present']) or 'none'}",
        "",
        "## Dense-5 Effects",
        "",
        "| Effect | Mean | 95% CI |",
        "|---|---:|---:|",
    ]
    for name in [
        "E_mid_pt",
        "E_late_pt",
        "E_midlate_pt",
        "I_pt",
        "L_given_M_pt",
        "L_alone_pt",
        "E_mid_it",
        "E_late_it",
        "E_midlate_it",
        "I_it",
        "C_mid_it",
        "C_late_it",
        "C_midlate_it",
        "I_collapse_it",
    ]:
        row = dense_rows.get(name, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_float(row.get("mean")),
                    f"[{_format_float(row.get('ci_low'))}, {_format_float(row.get('ci_high'))}]",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            "| Model | Missing conditions | PT common prompts | IT common prompts | Prompt-mode mismatches |",
            "|---|---|---:|---:|---|",
        ]
    )
    for model, payload in summary["models"].items():
        validation = payload["validation"]
        lines.append(
            "| "
            + " | ".join(
                [
                    model,
                    ", ".join(validation["missing_conditions"]) or "none",
                    str(validation["pt_common_prompt_count"]),
                    str(validation["it_common_prompt_count"]),
                    json.dumps(validation["prompt_mode_mismatches"], sort_keys=True)
                    if validation["prompt_mode_mismatches"]
                    else "none",
                ]
            )
            + " |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp23_midlate_kl_factorial_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    fieldnames = ["side", "model", "effect", "kind", "n_models", "n_prompt_ids", "mean", "ci_low", "ci_high"]
    with (out_dir / "exp23_midlate_kl_factorial_effects.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["effects"])
    _write_report(summary, out_dir / "exp23_midlate_kl_factorial_report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    args = parser.parse_args()

    summary = analyze(args.run_root, list(args.models), n_boot=args.n_bootstrap, seed=args.seed)
    out_dir = args.out_dir or (args.run_root / "analysis")
    write_outputs(summary, out_dir)
    print(f"[exp23] wrote {out_dir}")


if __name__ == "__main__":
    main()
