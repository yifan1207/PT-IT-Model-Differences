#!/usr/bin/env python3
"""Analyze Exp47 same-base recipe/domain specificity outputs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.poc.exp06_corrective_direction_steering.benchmarks.governance_v2 import _check_compliance_v2


DEFAULT_MODELS = (
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_openmath2",
)
INSTRUCTION_LIKE = {
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_hermes3",
}
OPENMATH = "llama31_openmath2"
READOUTS = ("common_it", "common_pt")
CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else None


def _percentile_ci(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    lo, hi = np.percentile(values.astype(float), [2.5, 97.5])
    return float(lo), float(hi)


def _bootstrap_prompt_mean(rows: list[dict[str, Any]], key: str, *, n_boot: int, seed: int) -> dict[str, Any]:
    by_prompt: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        val = _finite(row.get(key))
        if val is not None:
            by_prompt[str(row.get("prompt_id"))].append(val)
    prompt_vals = np.array([float(np.mean(vals)) for vals in by_prompt.values() if vals], dtype=float)
    estimate = float(prompt_vals.mean()) if prompt_vals.size else None
    boot = np.array([], dtype=float)
    if prompt_vals.size and n_boot > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, prompt_vals.size, size=(n_boot, prompt_vals.size))
        boot = prompt_vals[idx].mean(axis=1)
    lo, hi = _percentile_ci(boot)
    return {
        "estimate": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_units": int(sum(len(vals) for vals in by_prompt.values())),
        "n_prompt_clusters": int(prompt_vals.size),
        "n_boot": int(boot.size),
    }


def _bootstrap_alias_mean(rows: list[dict[str, Any]], key: str, aliases: list[str], *, n_boot: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    alias_vals: dict[str, np.ndarray] = {}
    point_vals = []
    for alias in aliases:
        stat = _bootstrap_prompt_mean([row for row in rows if row.get("model") == alias], key, n_boot=0, seed=seed)
        if stat["estimate"] is not None:
            point_vals.append(float(stat["estimate"]))
        by_prompt: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row.get("model") != alias:
                continue
            val = _finite(row.get(key))
            if val is not None:
                by_prompt[str(row.get("prompt_id"))].append(val)
        vals = np.array([float(np.mean(v)) for v in by_prompt.values() if v], dtype=float)
        if vals.size:
            alias_vals[alias] = vals
    estimate = float(np.mean(point_vals)) if point_vals else None
    boot = np.array([], dtype=float)
    if alias_vals and n_boot > 0:
        samples = []
        for _ in range(n_boot):
            per_alias = []
            for vals in alias_vals.values():
                idx = rng.integers(0, vals.size, size=vals.size)
                per_alias.append(float(vals[idx].mean()))
            samples.append(float(np.mean(per_alias)))
        boot = np.array(samples, dtype=float)
    lo, hi = _percentile_ci(boot)
    return {
        "estimate": estimate,
        "ci95_low": lo,
        "ci95_high": hi,
        "n_aliases": len(alias_vals),
        "n_units": sum(1 for row in rows if row.get("model") in alias_vals and _finite(row.get(key)) is not None),
        "n_boot": int(boot.size),
    }


def _slice_for_record(category: str, source: str, metadata: dict[str, Any] | None) -> list[str]:
    out = ["full_1400"]
    if category in {"GOV-FORMAT", "GOV-CONV", "GOV-REGISTER"}:
        out.append("instruction_format")
    if category == "SAFETY" and (metadata or {}).get("expected_behavior") == "comply":
        out.append("instruction_format")
    if category == "CONTENT-REASON" and "GSM8K" in str(source):
        out.append("math_domain")
    if category == "CONTENT-FACT":
        out.append("content_fact")
    return out


def _safe_ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or not math.isfinite(num) or not math.isfinite(den):
        return None
    if abs(den) < 1e-8:
        return None
    return float(num / den)


def _ratio_stable(m: float | None, lo: float | None, hi: float | None) -> bool:
    return m is not None and lo is not None and hi is not None and m > 0 and lo > 0


def _dataset_lookup(dataset: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with dataset.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("id", row.get("record_id")))] = row
    return out


def load_token_effect_rows(run_root: Path, models: list[str], dataset: Path) -> list[dict[str, Any]]:
    records = _dataset_lookup(dataset)
    rows: list[dict[str, Any]] = []
    for model in models:
        path = run_root / "residual_factorial" / "raw_shared" / model / "records.jsonl.gz"
        if not path.exists():
            continue
        for outer in _json_rows(path):
            prompt_id = str(outer.get("prompt_id"))
            dataset_record = records.get(prompt_id, {})
            category = str(dataset_record.get("category", outer.get("category", "")))
            source = str(dataset_record.get("source", ""))
            metadata = dataset_record.get("metadata") or {}
            slices = _slice_for_record(category, source, metadata)
            for event_kind, payload in (outer.get("events") or {}).items():
                if not isinstance(payload, dict) or payload.get("duplicate_of") or not payload.get("valid"):
                    continue
                event = payload.get("event") or {}
                position = int(event.get("step", payload.get("position", 0)) or 0)
                pt_token = event.get("pt_token") or {}
                it_token = event.get("it_token") or {}
                token_category = str(it_token.get("token_category_collapsed") or it_token.get("token_category") or "")
                cells = payload.get("cells") or {}
                for readout in READOUTS:
                    margins = {}
                    choices = {}
                    for cell in CELLS:
                        rp = (cells.get(cell) or {}).get(readout) or {}
                        margin = _finite(rp.get("it_vs_pt_margin"))
                        if margin is not None:
                            margins[cell] = margin
                        choices[cell] = rp.get("token_choice_class")
                    if len(margins) != len(CELLS):
                        continue
                    p = margins["U_PT__L_IT"] - margins["U_PT__L_PT"]
                    m = margins["U_IT__L_IT"] - margins["U_IT__L_PT"]
                    c = m - p
                    native_shift = margins["U_IT__L_IT"] - margins["U_PT__L_PT"]
                    base_row = {
                        "model": model,
                        "recipe_group": "instruction_like" if model in INSTRUCTION_LIKE else ("math_domain" if model == OPENMATH else "other"),
                        "prompt_id": prompt_id,
                        "event_kind": event_kind,
                        "readout": readout,
                        "category": category,
                        "source": source,
                        "position": position,
                        "position_ge_3": int(position >= 3),
                        "position_ge_5": int(position >= 5),
                        "position_bin": "p0" if position == 0 else ("p1_2" if position < 3 else ("p3_5" if position < 6 else "p6_plus")),
                        "token_category": token_category,
                        "pt_token_id": pt_token.get("token_id"),
                        "it_token_id": it_token.get("token_id"),
                        "pt_token_text": pt_token.get("token_str") or payload.get("pt_token_text"),
                        "it_token_text": it_token.get("token_str") or payload.get("it_token_text"),
                        "P": p,
                        "M": m,
                        "C": c,
                        "interaction": c,
                        "amplification_M_over_P": _safe_ratio(m, p),
                        "portable_share_P_over_M": _safe_ratio(p, m),
                        "coadapted_share_C_over_M": _safe_ratio(c, m),
                        "native_diagonal_shift": native_shift,
                        "interaction_over_native_shift": _safe_ratio(c, native_shift),
                        "U_PT__L_PT": margins["U_PT__L_PT"],
                        "U_PT__L_IT": margins["U_PT__L_IT"],
                        "U_IT__L_PT": margins["U_IT__L_PT"],
                        "U_IT__L_IT": margins["U_IT__L_IT"],
                        "choice_U_PT__L_IT": choices.get("U_PT__L_IT"),
                        "choice_U_IT__L_IT": choices.get("U_IT__L_IT"),
                    }
                    for slice_name in slices:
                        rows.append({**base_row, "slice": slice_name})
    return rows


def summarize_token_effects(rows: list[dict[str, Any]], *, n_boot: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"per_alias": {}, "group_contrasts": {}}
    metrics = ("P", "M", "C", "native_diagonal_shift", "interaction_over_native_shift")
    for readout in READOUTS:
        for slice_name in sorted({row["slice"] for row in rows}):
            for model in sorted({row["model"] for row in rows}):
                subset = [row for row in rows if row["readout"] == readout and row["slice"] == slice_name and row["model"] == model]
                if not subset:
                    continue
                stats = {metric: _bootstrap_prompt_mean(subset, metric, n_boot=n_boot, seed=abs(hash((readout, slice_name, model, metric))) % (2**32)) for metric in metrics}
                p_est = stats["P"]["estimate"]
                m_est = stats["M"]["estimate"]
                c_est = stats["C"]["estimate"]
                m_lo = stats["M"]["ci95_low"]
                m_hi = stats["M"]["ci95_high"]
                ratio_ok = _ratio_stable(m_est, m_lo, m_hi)
                row = {
                    "scope": "alias",
                    "model": model,
                    "recipe_group": "instruction_like" if model in INSTRUCTION_LIKE else ("math_domain" if model == OPENMATH else "other"),
                    "slice": slice_name,
                    "readout": readout,
                    "n_units": stats["C"]["n_units"],
                    "n_prompt_clusters": stats["C"]["n_prompt_clusters"],
                    "P": p_est,
                    "P_ci95_low": stats["P"]["ci95_low"],
                    "P_ci95_high": stats["P"]["ci95_high"],
                    "M": m_est,
                    "M_ci95_low": m_lo,
                    "M_ci95_high": m_hi,
                    "C": c_est,
                    "C_ci95_low": stats["C"]["ci95_low"],
                    "C_ci95_high": stats["C"]["ci95_high"],
                    "native_diagonal_shift": stats["native_diagonal_shift"]["estimate"],
                    "interaction_over_native_shift": stats["interaction_over_native_shift"]["estimate"],
                    "ratio_stable": int(ratio_ok),
                    "portable_share_P_over_M": _safe_ratio(p_est, m_est) if ratio_ok else None,
                    "coadapted_share_C_over_M": _safe_ratio(c_est, m_est) if ratio_ok else None,
                    "amplification_M_over_P": _safe_ratio(m_est, p_est) if ratio_ok else None,
                }
                out.append(row)
                summary["per_alias"][f"{readout}/{slice_name}/{model}"] = row
            instr_models = sorted({row["model"] for row in rows if row["model"] in INSTRUCTION_LIKE})
            subset = [row for row in rows if row["readout"] == readout and row["slice"] == slice_name and row["model"] in instr_models]
            if subset:
                stat = _bootstrap_alias_mean(subset, "C", instr_models, n_boot=n_boot, seed=abs(hash(("group", readout, slice_name))) % (2**32))
                out.append(
                    {
                        "scope": "instruction_like_mean",
                        "model": "instruction_like_mean",
                        "recipe_group": "instruction_like",
                        "slice": slice_name,
                        "readout": readout,
                        "C": stat["estimate"],
                        "C_ci95_low": stat["ci95_low"],
                        "C_ci95_high": stat["ci95_high"],
                        "n_units": stat["n_units"],
                        "n_aliases": stat["n_aliases"],
                    }
                )
                summary["group_contrasts"][f"{readout}/{slice_name}/instruction_like_mean_C"] = stat
    return out, summary


def _regression_adjusted_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    subset = [
        row for row in rows
        if row["slice"] == "instruction_format"
        and row["readout"] == "common_it"
        and row["model"] in (INSTRUCTION_LIKE | {OPENMATH})
        and _finite(row.get("C")) is not None
    ]
    if not subset:
        return {"ok": False, "reason": "no_rows"}
    y = np.array([float(row["C"]) for row in subset], dtype=float)
    levels: list[tuple[str, str]] = []
    for field in ("category", "position_bin", "token_category"):
        vals = sorted({str(row.get(field, "")) for row in subset})
        levels.extend((field, val) for val in vals[1:])
    x = np.ones((len(subset), 2 + len(levels)), dtype=float)
    x[:, 1] = np.array([1.0 if row["model"] in INSTRUCTION_LIKE else 0.0 for row in subset], dtype=float)
    for j, (field, val) in enumerate(levels, start=2):
        x[:, j] = np.array([1.0 if str(row.get(field, "")) == val else 0.0 for row in subset], dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {
        "ok": True,
        "contrast": "instruction_like_minus_openmath_on_instruction_format",
        "readout": "common_it",
        "estimate": float(beta[1]),
        "n_units": len(subset),
        "controls": ["category", "position_bin", "token_category"],
    }


def _sign_flip_null(rows: list[dict[str, Any]], *, n_perm: int, seed: int) -> dict[str, Any]:
    subset = [
        row for row in rows
        if row["readout"] == "common_it"
        and row["slice"] == "instruction_format"
        and row["model"] in INSTRUCTION_LIKE
        and _finite(row.get("C")) is not None
    ]
    by_prompt: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in subset:
        by_prompt[(str(row["model"]), str(row["prompt_id"]))].append(float(row["C"]))
    vals = np.array([float(np.mean(v)) for v in by_prompt.values() if v], dtype=float)
    if vals.size == 0:
        return {"ok": False, "reason": "no_values"}
    observed = float(vals.mean())
    rng = np.random.default_rng(seed)
    samples = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=vals.size)
        samples[i] = float((signs * vals).mean())
    return {
        "ok": True,
        "observed": observed,
        "null_mean": float(samples.mean()),
        "null_std": float(samples.std(ddof=1)),
        "p_upper": float((1 + np.sum(samples >= observed)) / (n_perm + 1)),
        "q95": float(np.percentile(samples, 95)),
        "q99": float(np.percentile(samples, 99)),
        "q99_9": float(np.percentile(samples, 99.9)),
        "n_prompt_clusters": int(vals.size),
        "n_permutations": int(n_perm),
    }


def _score_gsm8k(record: dict[str, Any], text: str) -> float | None:
    if record.get("category") != "CONTENT-REASON":
        return None
    expected = str(record.get("expected_answer", "")).replace(",", "").strip()
    if not expected:
        return None
    nums = re.findall(r"-?\d+(?:,\d+)*", text.replace(",", ""))
    return float(bool(nums) and nums[-1].replace(",", "") == expected)


def load_behavior_rows(run_root: Path, models: list[str], dataset: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_by_id = _dataset_lookup(dataset)
    one_rows: list[dict[str, Any]] = []
    rollout_scores: list[dict[str, Any]] = []
    for model in models:
        one_path = run_root / "behavior_bridge" / "raw" / model / "one_step_records.jsonl.gz"
        if one_path.exists():
            for row in _json_rows(one_path):
                for cell in CELLS:
                    payload = ((row.get("cells") or {}).get(cell) or {}).get("common_it") or {}
                    one_rows.append(
                        {
                            "model": model,
                            "prompt_id": row.get("prompt_id"),
                            "event_kind": row.get("event_kind"),
                            "category": row.get("category"),
                            "position": row.get("position"),
                            "cell": cell,
                            "top1_t_ft": float(bool(payload.get("top1_is_t_it"))),
                            "top5_t_ft": float(bool(payload.get("t_it_top5"))),
                            "pairwise_ft_win": float(bool(payload.get("pairwise_it_win"))),
                            "rank_t_ft": payload.get("t_it_rank"),
                            "margin_ft_minus_base": payload.get("margin_it_minus_pt"),
                        }
                    )
        rollout_path = run_root / "behavior_bridge" / "raw" / model / "rollout_records.jsonl.gz"
        if rollout_path.exists():
            for row in _json_rows(rollout_path):
                prompt_id = str(row.get("prompt_id"))
                record = dataset_by_id.get(prompt_id, {})
                text = str(row.get("continuation_text", ""))
                if record.get("category") == "GOV-FORMAT":
                    score = _check_compliance_v2(text, record.get("compliance_criteria") or {})
                    task = "ifeval_format"
                elif record.get("category") == "CONTENT-REASON":
                    score = _score_gsm8k(record, text)
                    task = "gsm8k_exact"
                else:
                    score = None
                    task = "other"
                if score is None:
                    continue
                rollout_scores.append(
                    {
                        "model": model,
                        "prompt_id": prompt_id,
                        "category": record.get("category"),
                        "source": record.get("source"),
                        "cell": row.get("cell"),
                        "task": task,
                        "score": float(score),
                        "generated_tokens_count": row.get("generated_tokens_count"),
                    }
                )
    return one_rows, rollout_scores


def summarize_behavior(one_rows: list[dict[str, Any]], rollout_scores: list[dict[str, Any]], *, n_boot: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    one_summary: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in one_rows}):
        by_event: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in one_rows:
            if row["model"] == model:
                by_event[(str(row["prompt_id"]), str(row["event_kind"]))][str(row["cell"])] = row
        contrasts: list[dict[str, Any]] = []
        for (prompt_id, event_kind), cells in by_event.items():
            if not set(CELLS).issubset(cells):
                continue
            for metric in ("top1_t_ft", "top5_t_ft", "pairwise_ft_win", "margin_ft_minus_base"):
                vals = {cell: _finite(cells[cell].get(metric)) for cell in CELLS}
                if any(vals[cell] is None for cell in CELLS):
                    continue
                contrasts.append(
                    {
                        "model": model,
                        "prompt_id": prompt_id,
                        "event_kind": event_kind,
                        "metric": metric,
                        "U_D_L_D_minus_U_B_L_D": vals["U_IT__L_IT"] - vals["U_PT__L_IT"],
                        "factorial_interaction": (vals["U_IT__L_IT"] - vals["U_IT__L_PT"]) - (vals["U_PT__L_IT"] - vals["U_PT__L_PT"]),
                    }
                )
        for metric in ("top1_t_ft", "top5_t_ft", "pairwise_ft_win", "margin_ft_minus_base"):
            for key in ("U_D_L_D_minus_U_B_L_D", "factorial_interaction"):
                subset = [row for row in contrasts if row["metric"] == metric]
                stat = _bootstrap_prompt_mean(subset, key, n_boot=n_boot, seed=abs(hash(("one", model, metric, key))) % (2**32))
                one_summary.append({"model": model, "metric": metric, "effect": key, **stat})

    matrix: list[dict[str, Any]] = []
    recovery: list[dict[str, Any]] = []
    for model in sorted({row["model"] for row in rollout_scores}):
        for task in sorted({row["task"] for row in rollout_scores if row["model"] == model}):
            cells = {}
            for cell in CELLS:
                subset = [row for row in rollout_scores if row["model"] == model and row["task"] == task and row["cell"] == cell]
                stat = _bootstrap_prompt_mean(subset, "score", n_boot=n_boot, seed=abs(hash(("roll", model, task, cell))) % (2**32))
                cells[cell] = stat["estimate"]
                matrix.append({"model": model, "task": task, "cell": cell, **stat})
            rec = _safe_ratio(
                None if cells["U_PT__L_IT"] is None or cells["U_PT__L_PT"] is None else cells["U_PT__L_IT"] - cells["U_PT__L_PT"],
                None if cells["U_IT__L_IT"] is None or cells["U_PT__L_PT"] is None else cells["U_IT__L_IT"] - cells["U_PT__L_PT"],
            )
            recovery.append(
                {
                    "model": model,
                    "task": task,
                    "late_only_behavioral_recovery_share": rec,
                    "U_PT__L_PT": cells["U_PT__L_PT"],
                    "U_PT__L_IT": cells["U_PT__L_IT"],
                    "U_IT__L_PT": cells["U_IT__L_PT"],
                    "U_IT__L_IT": cells["U_IT__L_IT"],
                }
            )
    return one_summary, matrix, recovery


def _plot(token_summary: list[dict[str, Any]], behavior_recovery: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in token_summary
        if row.get("scope") == "alias" and row.get("readout") == "common_it" and row.get("slice") in {"instruction_format", "math_domain", "full_1400"}
    ]
    models = [m for m in DEFAULT_MODELS if any(row.get("model") == m for row in rows)]
    slices = ["instruction_format", "math_domain", "full_1400"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.24
    x = np.arange(len(models), dtype=float)
    colors = {"instruction_format": "#4c78a8", "math_domain": "#f58518", "full_1400": "#54a24b"}
    for i, slice_name in enumerate(slices):
        vals = []
        los = []
        his = []
        for model in models:
            row = next((r for r in rows if r["model"] == model and r["slice"] == slice_name), None)
            val = _finite(row.get("interaction_over_native_shift")) if row else None
            if val is None:
                val = _finite(row.get("C")) if row else None
            vals.append(float(val) if val is not None else 0.0)
            lo = _finite(row.get("C_ci95_low")) if row else None
            hi = _finite(row.get("C_ci95_high")) if row else None
            los.append(lo)
            his.append(hi)
        axes[0].bar(x + (i - 1) * width, vals, width=width, label=slice_name, color=colors[slice_name])
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(x, [m.replace("llama31_", "").replace("_", "\n") for m in models], fontsize=8)
    axes[0].set_ylabel("Interaction / native shift (fallback: C logits)")
    axes[0].set_title("Token-Level Recipe/Domain Specificity")
    axes[0].legend(frameon=False, fontsize=8)

    token_by_model = {
        row["model"]: _finite(row.get("interaction_over_native_shift"))
        for row in rows
        if row.get("slice") == "instruction_format"
    }
    xs = []
    ys = []
    labels = []
    for row in behavior_recovery:
        model = str(row.get("model"))
        xval = token_by_model.get(model)
        yval = _finite(row.get("late_only_behavioral_recovery_share"))
        if xval is None or yval is None:
            continue
        xs.append(xval)
        ys.append(yval)
        labels.append(f"{model.replace('llama31_', '')}/{row.get('task')}")
    axes[1].scatter(xs, ys, s=40, color="#b279a2")
    for xval, yval, label in zip(xs, ys, labels, strict=False):
        axes[1].annotate(label, (xval, yval), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].axvline(0, color="#333333", linewidth=0.8)
    axes[1].set_xlabel("Token interaction / native shift")
    axes[1].set_ylabel("Late-only behavioral recovery share")
    axes[1].set_title("Automatic Behavioral Bridge")
    fig.tight_layout()
    fig.savefig(out_dir / "recipe_domain_two_panel.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = list(args.models or DEFAULT_MODELS)
    analysis_dir = args.run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    token_rows = load_token_effect_rows(args.run_root, models, args.dataset)
    token_summary, token_summary_payload = summarize_token_effects(token_rows, n_boot=args.n_boot)
    adjusted = _regression_adjusted_contrast(token_rows)
    null = _sign_flip_null(token_rows, n_perm=args.n_permutations, seed=args.seed)
    one_rows, rollout_scores = load_behavior_rows(args.run_root, models, args.dataset)
    one_summary, behavior_matrix, behavior_recovery = summarize_behavior(one_rows, rollout_scores, n_boot=args.n_boot)

    position_rows: list[dict[str, Any]] = []
    for readout in READOUTS:
        for model in models:
            for subset_name, predicate in (
                ("all", lambda r: True),
                ("position_ge_3", lambda r: bool(r.get("position_ge_3"))),
                ("position_ge_5", lambda r: bool(r.get("position_ge_5"))),
            ):
                subset = [r for r in token_rows if r["readout"] == readout and r["slice"] == "full_1400" and r["model"] == model and predicate(r)]
                stat = _bootstrap_prompt_mean(subset, "C", n_boot=args.n_boot, seed=abs(hash(("pos", readout, model, subset_name))) % (2**32))
                position_rows.append({"model": model, "readout": readout, "subset": subset_name, "metric": "C", **stat})
    category_rows: list[dict[str, Any]] = []
    for readout in READOUTS:
        for model in models:
            for category in sorted({r["category"] for r in token_rows if r["model"] == model}):
                subset = [r for r in token_rows if r["readout"] == readout and r["model"] == model and r["category"] == category and r["slice"] == "full_1400"]
                stat = _bootstrap_prompt_mean(subset, "C", n_boot=args.n_boot, seed=abs(hash(("cat", readout, model, category))) % (2**32))
                category_rows.append({"model": model, "readout": readout, "category": category, "metric": "C", **stat})

    _write_csv(analysis_dir / "effects.csv", token_rows)
    _write_csv(analysis_dir / "portable_coadapted_table.csv", token_summary)
    _write_csv(analysis_dir / "position_sensitivity.csv", position_rows)
    _write_csv(analysis_dir / "prompt_category_effects.csv", category_rows)
    _write_csv(analysis_dir / "matched_support_effects.csv", [adjusted])
    _write_csv(analysis_dir / "one_step_effects.csv", one_summary)
    _write_csv(analysis_dir / "behavior_four_cell_matrix.csv", behavior_matrix)
    _write_csv(analysis_dir / "behavior_recovery_summary.csv", behavior_recovery)
    _write_csv(analysis_dir / "ifeval_bridge_summary.csv", [r for r in behavior_recovery if r.get("task") == "ifeval_format"])
    _write_csv(analysis_dir / "gsm8k_bridge_summary.csv", [r for r in behavior_recovery if r.get("task") == "gsm8k_exact"])
    _write_json(analysis_dir / "label_swap_null.json", null)
    summary = {
        "experiment": "exp47_same_base_recipe_specificity",
        "run_root": str(args.run_root),
        "models": models,
        "n_token_effect_rows": len(token_rows),
        "n_one_step_rows": len(one_rows),
        "n_rollout_score_rows": len(rollout_scores),
        "token_summary": token_summary_payload,
        "matched_support": adjusted,
        "label_swap_null": null,
        "behavior_recovery": behavior_recovery,
    }
    _write_json(analysis_dir / "summary.json", summary)
    _plot(token_summary, behavior_recovery, analysis_dir)
    report = [
        "# Exp47 Same-Base Recipe Specificity Report",
        "",
        f"- Run root: `{args.run_root}`",
        f"- Models: `{', '.join(models)}`",
        f"- Token rows: `{len(token_rows)}`",
        f"- One-step rows: `{len(one_rows)}`",
        f"- Automatic rollout score rows: `{len(rollout_scores)}`",
        f"- Matched-support contrast: `{adjusted.get('estimate')}`",
        f"- Label-swap p_upper: `{null.get('p_upper')}`",
    ]
    (analysis_dir / "exp47_report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"ok": True, "analysis_dir": str(analysis_dir), "n_token_rows": len(token_rows)}, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=47)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
