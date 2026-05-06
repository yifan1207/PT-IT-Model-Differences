"""Analyze Exp51 native-history cross-patching outputs."""

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

from src.poc.exp51_native_history_crosspatch import DEFAULT_MODELS, DEFAULT_READOUTS, PRIMARY_HORIZONS

RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
EFFECTS = {
    "late_it_given_pt_upstream": {"U_PT__L_IT": 1.0, "U_PT__L_PT": -1.0},
    "late_it_given_it_upstream": {"U_IT__L_IT": 1.0, "U_IT__L_PT": -1.0},
    "interaction": {
        "U_IT__L_IT": 1.0,
        "U_IT__L_PT": -1.0,
        "U_PT__L_IT": -1.0,
        "U_PT__L_PT": 1.0,
    },
}


@dataclass(frozen=True)
class EffectUnit:
    model: str
    prompt_id: str
    history_source: str
    horizon: int
    readout: str
    category: str
    margins: dict[str, float]
    noop_delta: float | None


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
    if not vals:
        return None
    return float(np.mean(np.array(vals, dtype=float)))


def _ci(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def _effect_value(unit: EffectUnit, effect_name: str) -> float | None:
    total = 0.0
    for cell, coef in EFFECTS[effect_name].items():
        if cell not in unit.margins:
            return None
        total += coef * unit.margins[cell]
    return float(total)


def _load_units_and_support(
    *,
    run_root: Path,
    models: list[str],
    history_sources: list[str],
    prompt_mode: str,
    readouts: list[str],
) -> tuple[list[EffectUnit], dict[str, Any]]:
    units: list[EffectUnit] = []
    support: dict[str, Any] = {
        "missing_files": [],
        "rows": 0,
        "valid_prompt_rows": 0,
        "invalid_prompt_rows": 0,
        "by_model_history_horizon": {},
        "noop_margin_abs_delta_max": None,
    }
    noop_deltas: list[float] = []

    def key(model: str, history_source: str, horizon: str) -> str:
        return f"{model}::{history_source}::h{horizon}"

    for model in models:
        for history_source in history_sources:
            path = run_root / history_source / prompt_mode / model / "records.jsonl.gz"
            if not path.exists():
                support["missing_files"].append(str(path))
                continue
            for row in _json_rows(path):
                support["rows"] += 1
                if not row.get("valid", True):
                    support["invalid_prompt_rows"] += 1
                else:
                    support["valid_prompt_rows"] += 1
                category = str(row.get("category") or "UNKNOWN")
                prompt_id = str(row.get("prompt_id"))
                hist = row.get("native_history") or {}
                for horizon, status in (hist.get("horizon_status") or {}).items():
                    bucket = support["by_model_history_horizon"].setdefault(
                        key(model, history_source, horizon),
                        {
                            "model": model,
                            "history_source": history_source,
                            "horizon": int(horizon),
                            "n_prompts_total": 0,
                            "n_prefixes_not_terminated": 0,
                            "n_local_disagreements": 0,
                            "n_valid_after_real_token_mask": 0,
                            "reasons": defaultdict(int),
                        },
                    )
                    bucket["n_prompts_total"] += 1
                    if status.get("prefix_not_terminated"):
                        bucket["n_prefixes_not_terminated"] += 1
                    if status.get("local_disagreement"):
                        bucket["n_local_disagreements"] += 1
                    if status.get("valid_after_real_token_mask"):
                        bucket["n_valid_after_real_token_mask"] += 1
                    reason = str(status.get("reason") or "kept")
                    bucket["reasons"][reason] += 1

                for event_kind, payload in (row.get("events") or {}).items():
                    if not payload or payload.get("duplicate_of") or not payload.get("valid"):
                        continue
                    event = payload.get("event") or {}
                    horizon = int(event.get("horizon", str(event_kind).split("h")[-1]))
                    cells = payload.get("cells") or {}
                    for check in (payload.get("noop_patch_checks") or {}).values():
                        val = _finite(check.get("common_it_margin_abs_delta"))
                        if val is not None:
                            noop_deltas.append(val)
                    for readout in readouts:
                        margins: dict[str, float] = {}
                        for cell in RESIDUAL_CELLS:
                            margin = _finite(((cells.get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))
                            if margin is not None:
                                margins[cell] = margin
                        if len(margins) == len(RESIDUAL_CELLS):
                            units.append(
                                EffectUnit(
                                    model=model,
                                    prompt_id=prompt_id,
                                    history_source=history_source,
                                    horizon=horizon,
                                    readout=readout,
                                    category=category,
                                    margins=margins,
                                    noop_delta=max(noop_deltas) if noop_deltas else None,
                                )
                            )
    for bucket in support["by_model_history_horizon"].values():
        bucket["reasons"] = dict(bucket["reasons"])
        total = max(1, int(bucket["n_prompts_total"]))
        bucket["valid_event_rate"] = float(bucket["n_valid_after_real_token_mask"] / total)
        bucket["local_disagreement_rate"] = float(bucket["n_local_disagreements"] / total)
    support["noop_margin_abs_delta_max"] = max(noop_deltas) if noop_deltas else None
    return units, support


def _prompt_cluster_values(units: list[EffectUnit], effect_name: str) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for unit in units:
        value = _effect_value(unit, effect_name)
        if value is not None and math.isfinite(value):
            out[(unit.model, unit.prompt_id)].append(value)
    return out


def _cluster_bootstrap(
    units: list[EffectUnit],
    *,
    effect_name: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for unit in units:
        value = _effect_value(unit, effect_name)
        if value is not None and math.isfinite(value):
            by_model[unit.model][unit.prompt_id].append(value)
    model_estimates: dict[str, float | None] = {}
    model_counts: dict[str, dict[str, int]] = {}
    boot_by_model: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)
    for model, by_prompt in by_model.items():
        prompt_means = np.array([float(np.mean(vals)) for vals in by_prompt.values() if vals], dtype=float)
        model_counts[model] = {"n_prompt_clusters": int(prompt_means.size), "n_units": int(sum(len(v) for v in by_prompt.values()))}
        model_estimates[model] = float(prompt_means.mean()) if prompt_means.size else None
        if prompt_means.size and n_boot > 0:
            idx = rng.integers(0, prompt_means.size, size=(n_boot, prompt_means.size))
            boot_by_model[model] = prompt_means[idx].mean(axis=1)
    valid_models = [m for m, v in model_estimates.items() if v is not None]
    estimate = _mean([model_estimates[m] for m in valid_models if model_estimates[m] is not None])
    boot = (
        np.stack([boot_by_model[m] for m in valid_models], axis=0).mean(axis=0)
        if valid_models and boot_by_model and all(m in boot_by_model for m in valid_models)
        else np.array([], dtype=float)
    )
    lo, hi = _ci(boot)
    return {
        "estimate": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "models": model_estimates,
        "model_counts": model_counts,
        "n_models": len(valid_models),
        "n_units": int(sum(model_counts.get(m, {}).get("n_units", 0) for m in valid_models)),
        "n_prompt_clusters": int(sum(model_counts.get(m, {}).get("n_prompt_clusters", 0) for m in valid_models)),
        "bootstrap_unit": "prompt_cluster_within_family",
        "n_boot": int(boot.size),
    }


def _sign_flip_null(
    units: list[EffectUnit],
    *,
    effect_name: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    clusters = _prompt_cluster_values(units, effect_name)
    values = np.array([float(np.mean(vals)) for vals in clusters.values() if vals], dtype=float)
    if values.size == 0 or n_boot <= 0:
        return {"p_upper": None, "null_p999": None, "n_clusters": int(values.size)}
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_boot, values.size))
    null = (signs * values[None, :]).mean(axis=1)
    return {
        "observed": observed,
        "p_upper": float((np.sum(null >= observed) + 1) / (null.size + 1)),
        "null_p999": float(np.percentile(null, 99.9)),
        "n_clusters": int(values.size),
        "n_boot": int(null.size),
    }


def analyze_run(
    *,
    run_root: Path,
    models: list[str],
    history_sources: list[str],
    prompt_mode: str,
    readouts: list[str],
    primary_horizons: list[int],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    units, support = _load_units_and_support(
        run_root=run_root,
        models=models,
        history_sources=history_sources,
        prompt_mode=prompt_mode,
        readouts=readouts,
    )
    effects: dict[str, Any] = {}
    for history_source in history_sources:
        effects[history_source] = {}
        for readout in readouts:
            effects[history_source][readout] = {"by_horizon": {}, "primary_horizons": {}}
            for horizon in sorted({unit.horizon for unit in units if unit.history_source == history_source}):
                subset = [
                    unit
                    for unit in units
                    if unit.history_source == history_source and unit.readout == readout and unit.horizon == horizon
                ]
                effects[history_source][readout]["by_horizon"][str(horizon)] = {
                    effect_name: _cluster_bootstrap(
                        subset,
                        effect_name=effect_name,
                        n_boot=n_boot,
                        seed=seed + horizon * 19 + idx * 101,
                    )
                    for idx, effect_name in enumerate(EFFECTS)
                }
            primary_subset = [
                unit
                for unit in units
                if unit.history_source == history_source
                and unit.readout == readout
                and unit.horizon in set(primary_horizons)
            ]
            effects[history_source][readout]["primary_horizons"] = {
                effect_name: _cluster_bootstrap(
                    primary_subset,
                    effect_name=effect_name,
                    n_boot=n_boot,
                    seed=seed + idx * 997,
                )
                for idx, effect_name in enumerate(EFFECTS)
            }
            effects[history_source][readout]["primary_label_swap_null"] = {
                effect_name: _sign_flip_null(
                    primary_subset,
                    effect_name=effect_name,
                    n_boot=n_boot,
                    seed=seed + idx * 613,
                )
                for idx, effect_name in enumerate(EFFECTS)
            }
    return {
        "experiment": "exp51_native_history_crosspatch",
        "run_root": str(run_root),
        "models": models,
        "history_sources": history_sources,
        "prompt_mode": prompt_mode,
        "readouts": readouts,
        "primary_horizons": primary_horizons,
        "support": support,
        "effects": effects,
    }


def write_effects_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for history_source, readout_payload in summary["effects"].items():
        for readout, payload in readout_payload.items():
            for scope, scope_payload in payload.items():
                if scope == "primary_label_swap_null":
                    continue
                if scope == "by_horizon":
                    for horizon, effects in scope_payload.items():
                        for effect_name, stats in effects.items():
                            rows.append(
                                {
                                    "history_source": history_source,
                                    "readout": readout,
                                    "scope": "horizon",
                                    "horizon": horizon,
                                    "effect": effect_name,
                                    **{k: stats.get(k) for k in ("estimate", "ci95_low", "ci95_high", "n_units", "n_prompt_clusters", "n_models")},
                                }
                            )
                elif scope == "primary_horizons":
                    for effect_name, stats in scope_payload.items():
                        rows.append(
                            {
                                "history_source": history_source,
                                "readout": readout,
                                "scope": "primary_horizons",
                                "horizon": ",".join(str(x) for x in summary["primary_horizons"]),
                                "effect": effect_name,
                                **{k: stats.get(k) for k in ("estimate", "ci95_low", "ci95_high", "n_units", "n_prompt_clusters", "n_models")},
                            }
                        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "history_source",
                "readout",
                "scope",
                "horizon",
                "effect",
                "estimate",
                "ci95_low",
                "ci95_high",
                "n_units",
                "n_prompt_clusters",
                "n_models",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_support_csv(summary: dict[str, Any], path: Path) -> None:
    buckets = list((summary.get("support") or {}).get("by_model_history_horizon", {}).values())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "model",
            "history_source",
            "horizon",
            "n_prompts_total",
            "n_prefixes_not_terminated",
            "n_local_disagreements",
            "n_valid_after_real_token_mask",
            "valid_event_rate",
            "local_disagreement_rate",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for bucket in sorted(buckets, key=lambda x: (x["history_source"], x["model"], x["horizon"])):
            writer.writerow({name: bucket.get(name) for name in fieldnames})


def plot_interaction(summary: dict[str, Any], path: Path) -> None:
    readout = "common_it" if "common_it" in summary["readouts"] else summary["readouts"][0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for history_source, color in (("it", "#1f77b4"), ("pt", "#ff7f0e")):
        if history_source not in summary["effects"]:
            continue
        by_h = summary["effects"][history_source][readout]["by_horizon"]
        horizons = sorted(int(h) for h in by_h)
        estimates = [by_h[str(h)]["interaction"]["estimate"] for h in horizons]
        lows = [by_h[str(h)]["interaction"]["ci95_low"] for h in horizons]
        highs = [by_h[str(h)]["interaction"]["ci95_high"] for h in horizons]
        ax.plot(horizons, estimates, marker="o", label=f"{history_source}-history", color=color)
        if all(v is not None for v in lows + highs + estimates):
            ax.fill_between(horizons, lows, highs, color=color, alpha=0.15)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Native-history horizon h")
    ax.set_ylabel("Interaction C (logits)")
    ax.set_title(f"Exp51 native-history interaction ({readout})")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_claims(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Exp51 Paper-Facing Claims",
        "",
        "Use only scoped language: native-history local disagreements, not broad natural behavior.",
        "",
    ]
    for history_source in summary["history_sources"]:
        payload = summary["effects"].get(history_source, {}).get("common_it", {}).get("primary_horizons", {})
        inter = payload.get("interaction", {})
        lines.append(
            f"- `{history_source}` history primary-horizon interaction: "
            f"{inter.get('estimate')} [{inter.get('ci95_low')}, {inter.get('ci95_high')}], "
            f"n_units={inter.get('n_units')}, n_prompt_clusters={inter.get('n_prompt_clusters')}."
        )
    lines.append("")
    lines.append("If the `it`-history interaction is positive and the PT-history mirror is weaker, say the split persists on IT-shaped histories. If both are positive, call it local-readout generalization rather than IT-history specificity.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Exp51 native-history cross-patching records.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--history-sources", nargs="*", default=["it"])
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--readouts", nargs="*", default=list(DEFAULT_READOUTS))
    parser.add_argument("--primary-horizons", nargs="*", type=int, default=list(PRIMARY_HORIZONS))
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = analyze_run(
        run_root=args.run_root,
        models=list(args.models),
        history_sources=list(args.history_sources),
        prompt_mode=args.prompt_mode,
        readouts=list(args.readouts),
        primary_horizons=[int(x) for x in args.primary_horizons],
        n_boot=int(args.n_bootstrap),
        seed=int(args.seed),
    )
    analysis_dir = args.run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_effects_csv(summary, analysis_dir / "effects.csv")
    write_support_csv(summary, analysis_dir / "support.csv")
    plot_interaction(summary, analysis_dir / "native_history_interaction.png")
    write_claims(summary, analysis_dir / "paper_claims_exp51.md")


def main() -> None:
    main_with_args(None)


if __name__ == "__main__":
    main()
