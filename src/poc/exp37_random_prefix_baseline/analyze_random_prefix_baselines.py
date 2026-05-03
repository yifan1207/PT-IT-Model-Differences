"""Analyze Exp37 matched-prefix selection baselines."""

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

from src.poc.exp37_random_prefix_baseline import (
    ARM_SOURCE_KEYS,
    DENSE5_MODELS,
    PREDIV_KEY,
    RANDOM_IT_KEY,
    RANDOM_PT_KEY,
    REFERENCE_KEY,
)

RESIDUAL_CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
SOURCE_BALANCED_KEY = "random_local_disagreement__source_balanced"


@dataclass(frozen=True)
class Unit:
    key: str
    model: str
    prompt_id: str
    readout: str
    margins: dict[str, float]
    metadata: dict[str, Any]


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


def interaction_from_margins(margins: dict[str, float]) -> dict[str, float] | None:
    if any(cell not in margins for cell in RESIDUAL_CELLS):
        return None
    late_pt = float(margins["U_PT__L_IT"] - margins["U_PT__L_PT"])
    late_it = float(margins["U_IT__L_IT"] - margins["U_IT__L_PT"])
    return {
        "late_effect_from_pt_upstream": late_pt,
        "late_effect_from_it_upstream": late_it,
        "interaction": float(late_it - late_pt),
    }


def _load_units(run_root: Path, key: str, models: list[str], readouts: list[str]) -> list[Unit]:
    units: list[Unit] = []
    for model in models:
        path = run_root / "residual_factorial" / key / "raw_shared" / model / "records.jsonl.gz"
        if not path.exists():
            continue
        for row in _json_rows(path):
            prompt_id = str(row.get("prompt_id"))
            payload = (row.get("events") or {}).get("first_diff")
            if not isinstance(payload, dict) or not payload.get("valid"):
                continue
            event = payload.get("event") or {}
            cells = payload.get("cells") or {}
            for readout in readouts:
                margins: dict[str, float] = {}
                for cell in RESIDUAL_CELLS:
                    margin = _finite(((cells.get(cell) or {}).get(readout) or {}).get("it_vs_pt_margin"))
                    if margin is not None:
                        margins[cell] = margin
                if len(margins) == len(RESIDUAL_CELLS):
                    units.append(
                        Unit(
                            key=key,
                            model=model,
                            prompt_id=prompt_id,
                            readout=readout,
                            margins=margins,
                            metadata=event,
                        )
                    )
    return units


def _effect_value(unit: Unit, effect: str) -> float | None:
    effects = interaction_from_margins(unit.margins)
    if effects is None:
        return None
    return effects.get(effect)


def _cluster_values(units: list[Unit], effect: str) -> list[tuple[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        value = _effect_value(unit, effect)
        if value is not None and math.isfinite(value):
            grouped[unit.prompt_id].append(float(value))
    return [(prompt_id, float(np.mean(vals))) for prompt_id, vals in grouped.items() if vals]


def _bootstrap_model_mean(
    values_by_model: dict[str, list[tuple[str, float]]],
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    model_arrays: dict[str, np.ndarray] = {}
    model_summary: dict[str, Any] = {}
    for model, pairs in values_by_model.items():
        arr = np.array([v for _, v in pairs], dtype=float)
        model_arrays[model] = arr
        if arr.size == 0:
            model_summary[model] = {"estimate": None, "ci95_low": None, "ci95_high": None, "n_prompt_clusters": 0}
            continue
        boots = np.array([], dtype=float)
        if n_boot > 0:
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boots = arr[idx].mean(axis=1)
        lo, hi = (np.percentile(boots, [2.5, 97.5]).tolist() if boots.size else (None, None))
        model_summary[model] = {
            "estimate": float(arr.mean()),
            "ci95_low": float(lo) if lo is not None else None,
            "ci95_high": float(hi) if hi is not None else None,
            "n_prompt_clusters": int(arr.size),
            "n_units": int(arr.size),
        }
    valid_models = [model for model, arr in model_arrays.items() if arr.size > 0]
    point = float(np.mean([model_arrays[model].mean() for model in valid_models])) if valid_models else None
    pooled_boot = np.array([], dtype=float)
    if valid_models and n_boot > 0:
        boot_by_model = []
        for model in valid_models:
            arr = model_arrays[model]
            idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
            boot_by_model.append(arr[idx].mean(axis=1))
        pooled_boot = np.stack(boot_by_model, axis=0).mean(axis=0)
    lo, hi = (np.percentile(pooled_boot, [2.5, 97.5]).tolist() if pooled_boot.size else (None, None))
    return {
        "estimate": point,
        "ci95_low": float(lo) if lo is not None else None,
        "ci95_high": float(hi) if hi is not None else None,
        "n_models": len(valid_models),
        "n_units": int(sum(len(values_by_model.get(model, [])) for model in valid_models)),
        "n_prompt_clusters": int(sum(len(values_by_model.get(model, [])) for model in valid_models)),
        "models": model_summary,
        "bootstrap_unit": "prompt_cluster_within_family",
        "n_boot": int(pooled_boot.size),
    }


def _summary_for_units(
    units: list[Unit],
    *,
    models: list[str],
    readout: str,
    effect: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    values_by_model = {
        model: _cluster_values([u for u in units if u.model == model and u.readout == readout], effect)
        for model in models
    }
    return _bootstrap_model_mean(values_by_model, n_boot=n_boot, seed=seed)


def _source_balanced_units(units_by_key: dict[str, list[Unit]], readouts: list[str]) -> list[Unit]:
    out: list[Unit] = []
    for readout in readouts:
        grouped: dict[tuple[str, str], list[Unit]] = defaultdict(list)
        for key in (RANDOM_PT_KEY, RANDOM_IT_KEY):
            for unit in units_by_key.get(key, []):
                if unit.readout == readout:
                    grouped[(unit.model, unit.prompt_id)].append(unit)
        for (model, prompt_id), group in grouped.items():
            if not group:
                continue
            mean_margins = {
                cell: float(np.mean([unit.margins[cell] for unit in group if cell in unit.margins]))
                for cell in RESIDUAL_CELLS
            }
            metadata = dict(group[0].metadata)
            metadata["source_balanced_n_sources"] = len(group)
            out.append(
                Unit(
                    key=SOURCE_BALANCED_KEY,
                    model=model,
                    prompt_id=prompt_id,
                    readout=readout,
                    margins=mean_margins,
                    metadata=metadata,
                )
            )
    return out


def _covariates(unit: Unit) -> dict[str, float | str | None]:
    meta = unit.metadata
    pt_ent = _finite(meta.get("pt_entropy"))
    it_ent = _finite(meta.get("it_entropy"))
    pt_conf = _finite(meta.get("pt_confidence"))
    it_conf = _finite(meta.get("it_confidence"))
    pt_margin = _finite(meta.get("pt_top1_top2_margin"))
    it_margin = _finite(meta.get("it_top1_top2_margin"))
    return {
        "position": _finite(meta.get("sampled_prefix_step", meta.get("source_position", meta.get("step")))),
        "entropy": None if pt_ent is None or it_ent is None else 0.5 * (pt_ent + it_ent),
        "confidence": None if pt_conf is None or it_conf is None else 0.5 * (pt_conf + it_conf),
        "margin": None if pt_margin is None or it_margin is None else 0.5 * (pt_margin + it_margin),
        "token_category_pair": meta.get("token_category_pair"),
    }


def _smd(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2 or len(b) < 2:
        return None
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    pooled = math.sqrt((float(arr_a.var(ddof=1)) + float(arr_b.var(ddof=1))) / 2.0)
    if pooled == 0:
        return 0.0
    return float(abs(arr_a.mean() - arr_b.mean()) / pooled)


def _balance(ref_units: list[Unit], base_units: list[Unit]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in ("position", "entropy", "confidence", "margin"):
        ref_vals = [_covariates(unit)[name] for unit in ref_units]
        base_vals = [_covariates(unit)[name] for unit in base_units]
        ref_nums = [float(v) for v in ref_vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        base_nums = [float(v) for v in base_vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
        out[name] = {
            "smd": _smd(ref_nums, base_nums),
            "reference_mean": float(np.mean(ref_nums)) if ref_nums else None,
            "baseline_mean": float(np.mean(base_nums)) if base_nums else None,
            "n_reference": len(ref_nums),
            "n_baseline": len(base_nums),
        }
    max_smd = max(
        [payload["smd"] for payload in out.values() if payload.get("smd") is not None],
        default=None,
    )
    status = "clean" if max_smd is not None and max_smd <= 0.1 else ("usable_with_caution" if max_smd is not None and max_smd <= 0.2 else "imbalanced")
    return {"covariates": out, "max_smd": max_smd, "balance_status": status}


def _bin_edges(values: list[float], n_bins: int) -> list[float]:
    if len(values) < 2:
        return []
    qs = np.linspace(0, 100, n_bins + 1)[1:-1]
    edges = sorted(set(float(x) for x in np.percentile(np.array(values, dtype=float), qs)))
    return edges


def _bucket(value: float | None, edges: list[float]) -> int | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return int(np.searchsorted(np.array(edges, dtype=float), float(value), side="right"))


def _position_bucket(value: float | None) -> str | None:
    if value is None:
        return None
    pos = int(value)
    if pos == 0:
        return "0"
    if pos <= 2:
        return "1_2"
    if pos <= 4:
        return "3_4"
    return "5_plus"


def _overlap_matched(ref_units: list[Unit], base_units: list[Unit]) -> tuple[list[Unit], list[Unit], dict[str, Any]]:
    all_units = ref_units + base_units
    covs = {id(unit): _covariates(unit) for unit in all_units}
    edges = {
        name: _bin_edges(
            [float(covs[id(unit)][name]) for unit in all_units if isinstance(covs[id(unit)][name], (int, float))],
            5,
        )
        for name in ("entropy", "confidence", "margin")
    }

    def key(unit: Unit) -> tuple[Any, ...]:
        cov = covs[id(unit)]
        return (
            unit.model,
            _position_bucket(cov["position"] if isinstance(cov["position"], (int, float)) else None),
            _bucket(cov["entropy"] if isinstance(cov["entropy"], (int, float)) else None, edges["entropy"]),
            _bucket(cov["confidence"] if isinstance(cov["confidence"], (int, float)) else None, edges["confidence"]),
            _bucket(cov["margin"] if isinstance(cov["margin"], (int, float)) else None, edges["margin"]),
        )

    ref_keys = {key(unit) for unit in ref_units}
    base_keys = {key(unit) for unit in base_units}
    overlap = ref_keys & base_keys
    ref_kept = [unit for unit in ref_units if key(unit) in overlap]
    base_kept = [unit for unit in base_units if key(unit) in overlap]
    return ref_kept, base_kept, {
        "n_reference_before": len(ref_units),
        "n_baseline_before": len(base_units),
        "n_reference_after": len(ref_kept),
        "n_baseline_after": len(base_kept),
        "n_overlap_bins": len(overlap),
    }


def _load_qc(run_root: Path, models: list[str]) -> dict[str, Any]:
    by_model: dict[str, Any] = {}
    for model in models:
        path = run_root / "manifests" / "_qc" / model / "manifest_qc.jsonl"
        counts: dict[str, int] = defaultdict(int)
        if path.exists():
            for row in _json_rows(path):
                key = str(row.get("key", "unknown"))
                status = str(row.get("status", "unknown"))
                counts[f"{key}:{status}"] += 1
        by_model[model] = dict(counts)
    return by_model


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    readouts = args.readouts
    units_by_key = {key: _load_units(args.run_root, key, args.models, readouts) for key in ARM_SOURCE_KEYS}
    units_by_key[SOURCE_BALANCED_KEY] = _source_balanced_units(units_by_key, readouts)
    keys = [REFERENCE_KEY, SOURCE_BALANCED_KEY, RANDOM_PT_KEY, RANDOM_IT_KEY, PREDIV_KEY]

    summaries: dict[str, Any] = {}
    for key in keys:
        summaries[key] = {}
        for readout in readouts:
            summaries[key][readout] = {
                effect: _summary_for_units(
                    [unit for unit in units_by_key.get(key, []) if unit.readout == readout],
                    models=args.models,
                    readout=readout,
                    effect=effect,
                    n_boot=args.n_bootstrap,
                    seed=args.seed + idx * 101 + len(key),
                )
                for idx, effect in enumerate(
                    ("late_effect_from_pt_upstream", "late_effect_from_it_upstream", "interaction")
                )
            }

    ref_est = ((summaries.get(REFERENCE_KEY) or {}).get(args.primary_readout) or {}).get("interaction", {}).get("estimate")
    ratios: dict[str, Any] = {}
    for key in keys:
        est = ((summaries.get(key) or {}).get(args.primary_readout) or {}).get("interaction", {}).get("estimate")
        ratios[key] = (float(est) / float(ref_est)) if est is not None and ref_est not in (None, 0) else None

    balance: dict[str, Any] = {}
    ref_units = [unit for unit in units_by_key.get(REFERENCE_KEY, []) if unit.readout == args.primary_readout]
    for key in keys:
        if key == REFERENCE_KEY:
            continue
        base_units = [unit for unit in units_by_key.get(key, []) if unit.readout == args.primary_readout]
        before = _balance(ref_units, base_units)
        ref_matched, base_matched, overlap = _overlap_matched(ref_units, base_units)
        after = _balance(ref_matched, base_matched)
        matched_ref_summary = _summary_for_units(
            ref_matched,
            models=args.models,
            readout=args.primary_readout,
            effect="interaction",
            n_boot=args.n_bootstrap,
            seed=args.seed + 909 + len(key),
        )
        matched_base_summary = _summary_for_units(
            base_matched,
            models=args.models,
            readout=args.primary_readout,
            effect="interaction",
            n_boot=args.n_bootstrap,
            seed=args.seed + 1009 + len(key),
        )
        balance[key] = {
            "before": before,
            "after": after,
            "overlap": overlap,
            "matched_reference_interaction": matched_ref_summary,
            "matched_baseline_interaction": matched_base_summary,
            "matched_ratio_to_first_diff": (
                float(matched_base_summary["estimate"]) / float(matched_ref_summary["estimate"])
                if matched_base_summary.get("estimate") is not None
                and matched_ref_summary.get("estimate") not in (None, 0)
                else None
            ),
        }

    return {
        "experiment": "exp37_random_prefix_baseline",
        "run_root": str(args.run_root),
        "models": args.models,
        "readouts": readouts,
        "primary_readout": args.primary_readout,
        "keys": keys,
        "summaries": summaries,
        "ratio_to_first_diff": ratios,
        "matching_balance": balance,
        "manifest_qc": _load_qc(args.run_root, args.models),
    }


def _write_csv(summary: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    ratios = summary.get("ratio_to_first_diff") or {}
    for key, by_readout in (summary.get("summaries") or {}).items():
        for readout, effects in by_readout.items():
            for effect, payload in effects.items():
                rows.append(
                    {
                        "key": key,
                        "readout": readout,
                        "effect": effect,
                        "scope": "dense5",
                        "estimate": payload.get("estimate"),
                        "ci95_low": payload.get("ci95_low"),
                        "ci95_high": payload.get("ci95_high"),
                        "n_models": payload.get("n_models"),
                        "n_prompt_clusters": payload.get("n_prompt_clusters"),
                        "ratio_to_first_diff": ratios.get(key) if effect == "interaction" else "",
                    }
                )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "key",
                "readout",
                "effect",
                "scope",
                "estimate",
                "ci95_low",
                "ci95_high",
                "n_models",
                "n_prompt_clusters",
                "ratio_to_first_diff",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot(summary: dict[str, Any], path: Path) -> None:
    readout = summary["primary_readout"]
    keys = [REFERENCE_KEY, SOURCE_BALANCED_KEY, RANDOM_PT_KEY, RANDOM_IT_KEY, PREDIV_KEY]
    labels = ["first\ndiff", "random\nbalanced", "random\nPT path", "random\nIT path", "prediv\nfuture"]
    vals = []
    lows = []
    highs = []
    for key in keys:
        payload = summary["summaries"].get(key, {}).get(readout, {}).get("interaction", {})
        vals.append(payload.get("estimate"))
        lows.append(payload.get("ci95_low"))
        highs.append(payload.get("ci95_high"))
    y = np.array([float(v) if v is not None else np.nan for v in vals])
    yerr_low = np.array([float(v - lo) if v is not None and lo is not None else 0.0 for v, lo in zip(vals, lows, strict=False)])
    yerr_high = np.array([float(hi - v) if v is not None and hi is not None else 0.0 for v, hi in zip(vals, highs, strict=False)])
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    colors = ["#2F5D7C", "#6DAA75", "#9CCB86", "#C9A54A", "#B05D5D"]
    x = np.arange(len(keys))
    ax.bar(x, y, color=colors)
    ax.errorbar(x, y, yerr=[yerr_low, yerr_high], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Upstream x late interaction (logits)")
    ax.set_title("Exp37 matched-prefix selection baselines")
    ref = vals[0]
    if ref is not None:
        ax.axhline(float(ref), color="#2F5D7C", linestyle="--", linewidth=1.0, alpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--readouts", nargs="*", default=["common_it", "common_pt"])
    parser.add_argument("--primary-readout", default="common_it")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=37)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.run_root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = analyze(args)
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    _write_csv(summary, out_dir / "effects.csv")
    _plot(summary, out_dir / "exp37_matched_prefix_baselines.png")
    print(f"[exp37] wrote {summary_path}")


if __name__ == "__main__":
    main()

