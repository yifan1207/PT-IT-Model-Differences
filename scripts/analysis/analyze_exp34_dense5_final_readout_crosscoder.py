#!/usr/bin/env python3
"""Preflight and synthesize Exp34 Dense-5 final-readout crosscoder runs."""

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

from src.poc.cross_model.config import get_spec


DENSE5_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
DEFAULT_EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DEFAULT_EXP20_FALLBACK_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final"
)
DEFAULT_LLAMA_ROOT = Path(
    "results/exp30_final_readout_crosscoder_mediation/"
    "exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/selected_d131072_k64"
)
SUMMARY_COLUMNS = [
    "model",
    "result_root",
    "layers",
    "dict_config",
    "n_rank_events",
    "n_mediation_records",
    "ve_pt_min",
    "ve_it_min",
    "effective_l0_mean",
    "alive_fraction_min",
    "alive_fraction_max",
    "interaction_full",
    "causal_top200_interaction_drop",
    "causal_top200_ci_low",
    "causal_top200_ci_high",
    "causal_top200_mediation_fraction",
    "causal_top200_fraction_ci_low",
    "causal_top200_fraction_ci_high",
    "causal_matched_random200_drop_mean",
    "causal_matched_random200_drop_ci_low",
    "causal_matched_random200_drop_ci_high",
    "causal_matched_random200_mediation_fraction_mean",
    "position_ge_3_mediation_fraction",
    "layer_first_score_mass",
    "layer_second_score_mass",
    "final_layer_score_mass",
    "quality_gate",
    "paper_gate",
    "missing_reason",
]


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _median(values: list[Any]) -> float | None:
    vals = [_finite(v) for v in values]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.median(vals))


def _trimmed_mean(values: list[Any]) -> float | None:
    vals = sorted(v for v in (_finite(v) for v in values) if v is not None)
    if not vals:
        return None
    if len(vals) >= 5:
        vals = vals[1:-1]
    return float(np.mean(vals))


def _bootstrap_ci(values: list[Any], *, seed: int = 34, n_boot: int = 10000) -> tuple[float | None, float | None]:
    vals = np.asarray([v for v in (_finite(v) for v in values) if v is not None], dtype=np.float64)
    if vals.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        means[idx] = float(rng.choice(vals, size=vals.size, replace=True).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def final_three_layers(model: str) -> list[int]:
    spec = get_spec(model)
    return list(range(spec.n_layers - 3, spec.n_layers))


def selected_config(model: str) -> dict[str, Any]:
    spec = get_spec(model)
    if spec.d_model == 4096:
        return {"selected_name": "d131072_k64", "dict_size": 131072, "k": 64}
    if spec.d_model == 2560:
        return {"selected_name": "d81920_k64", "dict_size": 81920, "k": 64}
    return {"selected_name": f"d{32 * spec.d_model}_k64", "dict_size": 32 * spec.d_model, "k": 64}


def _manifest_path(root: Path, prompt_mode: str, model: str) -> Path | None:
    candidates = [
        root / prompt_mode / model / "exp20_validation_records.jsonl",
        root / prompt_mode / model / "exp20_records.jsonl",
    ]
    return next((path for path in candidates if path.exists()), None)


def _count_manifest(path: Path, event_kinds: list[str]) -> dict[str, Any]:
    total = 0
    counts = {kind: 0 for kind in event_kinds}
    pos_ge_3 = {kind: 0 for kind in event_kinds}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            events = row.get("divergence_events") or {}
            for kind in event_kinds:
                event = events.get(kind)
                if event and "duplicate_of" not in event:
                    counts[kind] += 1
                    if int(event.get("step", -1)) >= 3:
                        pos_ge_3[kind] += 1
    return {"records": total, "event_counts": counts, "position_ge_3_counts": pos_ge_3}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    warnings = []
    blocking_warnings = []
    if not args.dataset.exists():
        blocking_warnings.append(f"missing dataset: {args.dataset}")
    for model in args.models:
        spec = get_spec(model)
        cfg = selected_config(model)
        manifest = _manifest_path(args.exp20_root, args.prompt_mode, model)
        manifest_source = "primary"
        if manifest is None:
            manifest = _manifest_path(args.exp20_fallback_root, args.prompt_mode, model)
            manifest_source = "fallback"
        manifest_counts = {}
        ok = manifest is not None
        if manifest is None:
            blocking_warnings.append(f"{model}: missing Exp20 manifest for {args.prompt_mode}")
        else:
            manifest_counts = _count_manifest(manifest, args.event_kinds)
            n_records = int(manifest_counts.get("records", 0))
            first_diff = int((manifest_counts.get("event_counts") or {}).get("first_diff", 0))
            requested = args.rank_prompts + args.mediate_prompts
            if n_records < requested:
                blocking_warnings.append(
                    f"{model}: only {n_records} prompt records; requested {requested}"
                )
                ok = False
            if first_diff < args.rank_prompts:
                blocking_warnings.append(
                    f"{model}: only {first_diff} first_diff events; rank slice needs {args.rank_prompts}"
                )
                ok = False
            elif first_diff < requested:
                warnings.append(
                    f"{model}: only {first_diff} first_diff events; "
                    f"requested prompt window is {args.rank_prompts}+{args.mediate_prompts}; "
                    "analysis will report the actual valid-event count"
                )
        rows.append(
            {
                "model": model,
                "hf_pt": spec.pt_id,
                "hf_it": spec.it_id,
                "n_layers": spec.n_layers,
                "d_model": spec.d_model,
                "final_three_layers": final_three_layers(model),
                "selected_config": cfg,
                "manifest_path": str(manifest) if manifest else None,
                "manifest_source": manifest_source if manifest else None,
                **manifest_counts,
                "ok": ok,
            }
        )
    payload = {
        "experiment": "exp34_dense5_final_readout_crosscoder",
        "dataset": str(args.dataset),
        "prompt_mode": args.prompt_mode,
        "event_kinds": args.event_kinds,
        "rank_prompts": args.rank_prompts,
        "mediate_prompts": args.mediate_prompts,
        "models": rows,
        "warnings": warnings,
        "blocking_warnings": blocking_warnings,
        "ok": not blocking_warnings,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "exp34_preflight.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def _parse_model_roots(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--model-root expects model=path, got {item!r}")
        model, path = item.split("=", 1)
        out[model] = Path(path)
    return out


def _resolve_selected_root(path: Path, preferred_name: str | None = None) -> Path | None:
    if (path / "analysis" / "summary.json").exists():
        return path
    candidates = []
    if preferred_name:
        candidates.append(path / f"selected_{preferred_name}")
    candidates.extend(sorted(path.glob("selected_*")))
    existing = [candidate for candidate in candidates if (candidate / "analysis" / "summary.json").exists()]
    if not existing:
        return None
    return sorted(set(existing), key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _effect_row(effect_rows: list[dict[str, Any]], feature_set: str, k: int, seed: str = "none") -> dict[str, Any] | None:
    for row in effect_rows:
        if row.get("feature_set") == feature_set and int(row.get("k", -1)) == k and str(row.get("control_seed")) == seed:
            return row
    return None


def _quality_from_configs(root: Path, expected_layers: list[int]) -> dict[str, Any]:
    rows = []
    for layer in expected_layers:
        path = root / "dictionaries" / f"layer_{layer}" / "config.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("crosscoder") or {}
        metrics = payload.get("metrics") or {}
        rows.append(
            {
                "layer": int(layer),
                "dict_size": cfg.get("dict_size"),
                "k": cfg.get("k"),
                "ve_pt": _finite(metrics.get("heldout_variance_explained_pt")),
                "ve_it": _finite(metrics.get("heldout_variance_explained_it")),
                "l0": _finite(metrics.get("effective_l0")),
                "alive": _finite(metrics.get("alive_fraction")),
            }
        )
    if not rows:
        return {"rows": [], "quality_gate": "missing"}
    k_values = [row["k"] for row in rows if row.get("k") is not None]
    expected_k = int(k_values[0]) if k_values else None
    ve_pt_min = min((row["ve_pt"] for row in rows if row["ve_pt"] is not None), default=None)
    ve_it_min = min((row["ve_it"] for row in rows if row["ve_it"] is not None), default=None)
    l0_mean = _mean([row["l0"] for row in rows])
    alive_min = min((row["alive"] for row in rows if row["alive"] is not None), default=None)
    alive_max = max((row["alive"] for row in rows if row["alive"] is not None), default=None)
    pass_gate = (
        len(rows) == len(expected_layers)
        and ve_pt_min is not None
        and ve_pt_min >= 0.75
        and ve_it_min is not None
        and ve_it_min >= 0.75
        and expected_k is not None
        and l0_mean is not None
        and abs(l0_mean - expected_k) <= 0.10 * expected_k
        and alive_min is not None
        and alive_min >= 0.01
        and alive_max is not None
        and alive_max <= 0.20
    )
    dict_sizes = sorted({str(row["dict_size"]) for row in rows if row.get("dict_size") is not None})
    ks = sorted({str(row["k"]) for row in rows if row.get("k") is not None})
    return {
        "rows": rows,
        "dict_config": f"d{'+'.join(dict_sizes)}_k{'+'.join(ks)}" if dict_sizes and ks else "",
        "ve_pt_min": ve_pt_min,
        "ve_it_min": ve_it_min,
        "effective_l0_mean": l0_mean,
        "alive_fraction_min": alive_min,
        "alive_fraction_max": alive_max,
        "quality_gate": "pass" if pass_gate else "fail",
    }


def _top200_layer_mass(root: Path, expected_layers: list[int]) -> dict[int, float]:
    path = root / "feature_stats" / "causal_feature_scores.csv"
    if not path.exists():
        return {}
    rows = _csv_rows(path)
    positives = [row for row in rows if (_finite(row.get("score_mean")) or 0.0) > 0.0][:200]
    denom = sum(float(row["score_mean"]) for row in positives)
    if denom <= 0:
        return {layer: 0.0 for layer in expected_layers}
    out = {layer: 0.0 for layer in expected_layers}
    for row in positives:
        layer = int(row["layer"])
        out[layer] = out.get(layer, 0.0) + float(row["score_mean"]) / denom
    return out


def _model_result_root(args: argparse.Namespace, model: str, explicit_roots: dict[str, Path]) -> Path:
    if model in explicit_roots:
        return explicit_roots[model]
    if model == "llama31_8b" and args.llama_root:
        local_candidate = _resolve_selected_root(args.run_root / model, preferred_name=selected_config(model)["selected_name"])
        if local_candidate is not None:
            return args.run_root / model
        return args.llama_root
    return args.run_root / model


def _row_for_model(args: argparse.Namespace, model: str, explicit_roots: dict[str, Path]) -> tuple[dict[str, Any], list[str]]:
    warnings = []
    expected_layers = final_three_layers(model)
    cfg = selected_config(model)
    base_root = _model_result_root(args, model, explicit_roots)
    root = _resolve_selected_root(base_root, preferred_name=cfg["selected_name"])
    if root is None:
        return (
            {
                "model": model,
                "result_root": str(base_root),
                "layers": " ".join(map(str, expected_layers)),
                "dict_config": cfg["selected_name"],
                "quality_gate": "missing",
                "paper_gate": "missing",
                "missing_reason": "analysis/summary.json not found",
            },
            [f"{model}: missing selected result under {base_root}"],
        )

    quality = _quality_from_configs(root, expected_layers)
    effects = _csv_rows(root / "analysis" / "effects.csv")
    top = _effect_row(effects, "causal_top", 200, "none")
    rand_rows = [row for row in effects if row.get("feature_set") == "causal_matched_random" and int(row.get("k", -1)) == 200]
    if top is None:
        warnings.append(f"{model}: missing causal_top k=200 row")
    if not rand_rows:
        warnings.append(f"{model}: missing causal_matched_random k=200 rows")
    rank_summary_path = root / "feature_stats" / "causal_feature_scores_summary.json"
    rank_summary = json.loads(rank_summary_path.read_text(encoding="utf-8")) if rank_summary_path.exists() else {}
    mass = _top200_layer_mass(root, expected_layers)
    records_path = root / "mediation" / "records.jsonl.gz"
    n_records = sum(1 for _ in _json_rows(records_path)) if records_path.exists() else None
    causal_drop = _finite(top.get("interaction_drop_mean")) if top else None
    causal_frac = _finite(top.get("mediation_fraction_mean")) if top else None
    rand_drop = _mean([row.get("interaction_drop_mean") for row in rand_rows])
    quality_gate = quality.get("quality_gate", "missing")
    if quality_gate == "pass" and causal_drop is not None and rand_drop is not None and causal_drop > 0.0 and rand_drop < causal_drop:
        paper_gate = "pass" if causal_drop >= 0.20 and rand_drop <= 0.05 else "weak"
    elif top is None or not rand_rows:
        paper_gate = "missing"
    else:
        paper_gate = "fail"
    row = {
        "model": model,
        "result_root": str(root),
        "layers": " ".join(map(str, expected_layers)),
        "dict_config": quality.get("dict_config") or cfg["selected_name"],
        "n_rank_events": rank_summary.get("n_events"),
        "n_mediation_records": n_records,
        "ve_pt_min": quality.get("ve_pt_min"),
        "ve_it_min": quality.get("ve_it_min"),
        "effective_l0_mean": quality.get("effective_l0_mean"),
        "alive_fraction_min": quality.get("alive_fraction_min"),
        "alive_fraction_max": quality.get("alive_fraction_max"),
        "interaction_full": _finite(top.get("interaction_full_mean")) if top else None,
        "causal_top200_interaction_drop": causal_drop,
        "causal_top200_ci_low": _finite(top.get("interaction_drop_ci_lo")) if top else None,
        "causal_top200_ci_high": _finite(top.get("interaction_drop_ci_hi")) if top else None,
        "causal_top200_mediation_fraction": causal_frac,
        "causal_top200_fraction_ci_low": _finite(top.get("mediation_fraction_ci_lo")) if top else None,
        "causal_top200_fraction_ci_high": _finite(top.get("mediation_fraction_ci_hi")) if top else None,
        "causal_matched_random200_drop_mean": rand_drop,
        "causal_matched_random200_drop_ci_low": _mean([row.get("interaction_drop_ci_lo") for row in rand_rows]),
        "causal_matched_random200_drop_ci_high": _mean([row.get("interaction_drop_ci_hi") for row in rand_rows]),
        "causal_matched_random200_mediation_fraction_mean": _mean([row.get("mediation_fraction_mean") for row in rand_rows]),
        "position_ge_3_mediation_fraction": _finite(top.get("position_ge_3_mediation_fraction_mean")) if top else None,
        "layer_first_score_mass": mass.get(expected_layers[0]),
        "layer_second_score_mass": mass.get(expected_layers[1]),
        "final_layer_score_mass": mass.get(expected_layers[2]),
        "quality_gate": quality_gate,
        "paper_gate": paper_gate,
        "missing_reason": "",
    }
    return row, warnings


def _summary_row(label: str, rows: list[dict[str, Any]], reducer: str) -> dict[str, Any]:
    if reducer == "median":
        fn = _median
    elif reducer == "trimmed":
        fn = _trimmed_mean
    else:
        fn = _mean
    out = {key: "" for key in SUMMARY_COLUMNS}
    out["model"] = label
    out["result_root"] = "summary"
    out["n_rank_events"] = sum(int(row.get("n_rank_events") or 0) for row in rows) or None
    out["n_mediation_records"] = sum(int(row.get("n_mediation_records") or 0) for row in rows) or None
    numeric = [
        "ve_pt_min",
        "ve_it_min",
        "effective_l0_mean",
        "alive_fraction_min",
        "alive_fraction_max",
        "interaction_full",
        "causal_top200_interaction_drop",
        "causal_top200_mediation_fraction",
        "causal_matched_random200_drop_mean",
        "causal_matched_random200_mediation_fraction_mean",
        "position_ge_3_mediation_fraction",
        "layer_first_score_mass",
        "layer_second_score_mass",
        "final_layer_score_mass",
    ]
    for key in numeric:
        out[key] = fn([row.get(key) for row in rows])
    drop_lo, drop_hi = _bootstrap_ci([row.get("causal_top200_interaction_drop") for row in rows])
    frac_lo, frac_hi = _bootstrap_ci([row.get("causal_top200_mediation_fraction") for row in rows])
    rand_lo, rand_hi = _bootstrap_ci([row.get("causal_matched_random200_drop_mean") for row in rows])
    out["causal_top200_ci_low"] = drop_lo
    out["causal_top200_ci_high"] = drop_hi
    out["causal_top200_fraction_ci_low"] = frac_lo
    out["causal_top200_fraction_ci_high"] = frac_hi
    out["causal_matched_random200_drop_ci_low"] = rand_lo
    out["causal_matched_random200_drop_ci_high"] = rand_hi
    pass_count = sum(1 for row in rows if row.get("paper_gate") == "pass")
    quality_count = sum(1 for row in rows if row.get("quality_gate") == "pass")
    out["quality_gate"] = f"{quality_count}/{len(rows)} pass" if rows else "0/0 pass"
    out["paper_gate"] = f"{pass_count}/{len(rows)} pass" if rows else "0/0 pass"
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_COLUMNS})


def _plot_curve(path: Path, model_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    available = [row for row in model_rows if _finite(row.get("causal_top200_mediation_fraction")) is not None]
    if not available:
        ax.text(0.5, 0.5, "No Exp34 mediation rows available yet", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [row["model"] for row in available]
        x = np.arange(len(labels))
        top = [float(row["causal_top200_mediation_fraction"]) for row in available]
        rand = [float(row["causal_matched_random200_mediation_fraction_mean"] or 0.0) for row in available]
        width = 0.36
        ax.bar(x - width / 2, top, width, label="causal top-200", color="#4C78A8")
        ax.bar(x + width / 2, rand, width, label="matched random-200", color="#BAB0AC")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("Mediation fraction")
        ax.set_title("Exp34 Dense-5 terminal crosscoder mediation")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_layer_mass(path: Path, model_rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    rows = [row for row in model_rows if _finite(row.get("final_layer_score_mass")) is not None]
    if not rows:
        ax.text(0.5, 0.5, "No layer-mass rows available yet", ha="center", va="center")
        ax.set_axis_off()
    else:
        labels = [row["model"] for row in rows]
        first = np.asarray([float(row.get("layer_first_score_mass") or 0.0) for row in rows])
        second = np.asarray([float(row.get("layer_second_score_mass") or 0.0) for row in rows])
        final = np.asarray([float(row.get("final_layer_score_mass") or 0.0) for row in rows])
        x = np.arange(len(labels))
        ax.bar(x, first, label="first of final-three", color="#72B7B2")
        ax.bar(x, second, bottom=first, label="second", color="#F58518")
        ax.bar(x, final, bottom=first + second, label="final", color="#4C78A8")
        ax.set_ylim(0, 1)
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylabel("Top-200 positive score mass")
        ax.set_title("Causal feature score mass by terminal layer")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _write_note(path: Path, rows: list[dict[str, Any]], warnings: list[str]) -> None:
    model_rows = [row for row in rows if row.get("result_root") != "summary"]
    pass_count = sum(1 for row in model_rows if row.get("paper_gate") == "pass")
    lines = [
        "# Exp34 Dense-5 Final-Readout Crosscoder Synthesis",
        "",
        f"Models with paper-gate pass: `{pass_count}/{len(model_rows)}`.",
        "",
        "| Model | Quality | Paper | Top-200 drop | Top-200 frac | Matched-random drop | Final-layer mass |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            "| {model} | {quality_gate} | {paper_gate} | {drop} | {frac} | {rand} | {mass} |".format(
                model=row.get("model"),
                quality_gate=row.get("quality_gate"),
                paper_gate=row.get("paper_gate"),
                drop=_fmt(row.get("causal_top200_interaction_drop")),
                frac=_fmt(row.get("causal_top200_mediation_fraction")),
                rand=_fmt(row.get("causal_matched_random200_drop_mean")),
                mass=_fmt(row.get("final_layer_score_mass")),
            )
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Treat Exp34 as feature-level mediation evidence only when causal features beat "
            "same-layer matched random controls and dictionary health gates pass. The primary "
            "estimand is interaction drop in logits; mediation fraction is descriptive.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any, digits: int = 3) -> str:
    val = _finite(value)
    if val is None:
        return ""
    return f"{val:.{digits}f}"


def run_synthesis(args: argparse.Namespace) -> dict[str, Any]:
    explicit_roots = _parse_model_roots(args.model_root)
    model_rows = []
    warnings = []
    for model in args.models:
        row, row_warnings = _row_for_model(args, model, explicit_roots)
        model_rows.append(row)
        warnings.extend(row_warnings)
    available = [row for row in model_rows if row.get("paper_gate") != "missing"]
    summary_rows = []
    if available:
        family_label = (
            "Dense-5 family mean"
            if {row.get("model") for row in available} == set(DENSE5_MODELS)
            else f"Dense-{len(available)} available-family mean"
        )
        summary_rows.append(_summary_row(family_label, available, "mean"))
        if any(row.get("model") == "gemma3_4b" for row in available):
            no_gemma = [row for row in available if row.get("model") != "gemma3_4b"]
            summary_rows.append(_summary_row("Gemma-removed Dense-4", no_gemma, "mean"))
        summary_rows.append(_summary_row("family median", available, "median"))
        summary_rows.append(_summary_row("trimmed mean", available, "trimmed"))
    rows = model_rows + summary_rows
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / "exp34_dense5_crosscoder_family_table.csv", rows)
    _write_csv(args.out_dir / "exp34_dense5_crosscoder_quality_table.csv", model_rows)
    _plot_curve(args.out_dir / "exp34_dense5_crosscoder_mediation_curve.png", model_rows)
    _plot_layer_mass(args.out_dir / "exp34_dense5_crosscoder_layer_mass.png", model_rows)
    _write_note(args.out_dir / "exp34_dense5_crosscoder_note.md", rows, warnings)
    payload = {
        "experiment": "exp34_dense5_final_readout_crosscoder",
        "run_root": str(args.run_root),
        "out_dir": str(args.out_dir),
        "models": model_rows,
        "summary_rows": summary_rows,
        "warnings": warnings,
        "paper_ready": sum(1 for row in model_rows if row.get("paper_gate") == "pass") >= 4,
    }
    (args.out_dir / "exp34_dense5_crosscoder_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("results/exp34_dense5_final_readout_crosscoder/latest"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/paper_synthesis/exp34_dense5_final_readout_crosscoder"))
    parser.add_argument("--models", nargs="+", default=DENSE5_MODELS)
    parser.add_argument("--model-root", nargs="*", default=[], help="Optional explicit model=path result roots.")
    parser.add_argument("--llama-root", type=Path, default=DEFAULT_LLAMA_ROOT)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"))
    parser.add_argument("--exp20-root", type=Path, default=DEFAULT_EXP20_ROOT)
    parser.add_argument("--exp20-fallback-root", type=Path, default=DEFAULT_EXP20_FALLBACK_ROOT)
    parser.add_argument("--prompt-mode", default="raw_shared")
    parser.add_argument("--event-kinds", nargs="+", default=["first_diff"])
    parser.add_argument("--rank-prompts", type=int, default=160)
    parser.add_argument("--mediate-prompts", type=int, default=440)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.preflight_only:
        run_preflight(args)
    else:
        run_synthesis(args)


if __name__ == "__main__":
    main()
