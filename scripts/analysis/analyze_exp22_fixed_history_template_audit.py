#!/usr/bin/env python
"""Analyze Exp22 fixed-history template audit records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.analyze_exp22_endpoint_deconfounded_gap import (  # noqa: E402
    COVARIATES,
    OUTCOMES,
    PRIMARY_MODELS,
    add_endpoint_bins,
    balance_table,
    compute_cem_weights,
    dense_equal_model_effects,
)
from src.poc.exp22_endpoint_deconfounded_gap.fixed_history import CELLS, TEACHER_SOURCES  # noqa: E402
from src.poc.exp22_endpoint_deconfounded_gap.metrics import as_float, derived_step_metrics  # noqa: E402

COMPARISONS = {
    "native_fixed_effect": ("pt_raw", "it_native"),
    "raw_fixed_effect": ("pt_raw", "it_raw"),
    "template_delta": ("it_raw", "it_native"),
}


def _open_jsonl(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def _record_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    paths = []
    for cell in CELLS:
        paths.extend(sorted(root.glob(f"**/{cell}/records.jsonl.gz")))
        paths.extend(sorted(root.glob(f"**/{cell}/records.jsonl")))
    return sorted(set(paths))


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def flatten_records(root: Path, skip_first_n: int) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    branch_counts: dict[str, dict[str, int]] = {}
    record_alignment: dict[tuple[str, str, str], dict[str, Any]] = {}
    malformed_examples: list[dict[str, Any]] = []
    paths = _record_paths(root)
    for path in paths:
        with _open_jsonl(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    malformed_examples.append({"path": str(path), "error": repr(exc)})
                    continue
                model = str(rec.get("model", "unknown"))
                cell = str(rec.get("cell", path.parent.name))
                teacher_source = str(rec.get("teacher_source", "it_native"))
                branch = f"{model}/{teacher_source}/{cell}"
                branch_counts.setdefault(branch, {"records": 0, "malformed": 0, "valid": 0})
                branch_counts[branch]["records"] += 1
                if cell not in CELLS or rec.get("malformed") or "probes" not in rec:
                    branch_counts[branch]["malformed"] += 1
                    if len(malformed_examples) < 20:
                        malformed_examples.append(
                            {
                                "path": str(path),
                                "prompt_id": rec.get("prompt_id"),
                                "model": model,
                                "teacher_source": teacher_source,
                                "cell": cell,
                                "error": rec.get("error"),
                            }
                        )
                    continue
                branch_counts[branch]["valid"] += 1
                prompt_id = str(rec.get("prompt_id"))
                n_steps = int(rec.get("n_steps", 0))
                forced_ids = [int(x) for x in rec.get("forced_ids", [])]
                forced_hash = hash(tuple(forced_ids))
                align = record_alignment.setdefault(
                    (model, teacher_source, prompt_id),
                    {"cells": {}, "n_steps": {}, "forced_hashes": {}},
                )
                align["cells"][cell] = True
                align["n_steps"][cell] = n_steps
                align["forced_hashes"][cell] = forced_hash

                target_logprob = rec.get("target_logprob", [])
                target_rank = rec.get("target_rank", [])
                top1_matches_forced = rec.get("top1_matches_forced", [])
                for step_idx in range(skip_first_n, n_steps):
                    support_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "model": model,
                            "teacher_source": teacher_source,
                            "cell": cell,
                            "step_idx": step_idx,
                            "forced_id": int(forced_ids[step_idx]) if step_idx < len(forced_ids) else None,
                            "target_logprob": as_float(target_logprob[step_idx]) if step_idx < len(target_logprob) else float("nan"),
                            "target_rank": as_float(target_rank[step_idx]) if step_idx < len(target_rank) else float("nan"),
                            "top1_matches_forced": float(bool(top1_matches_forced[step_idx])) if step_idx < len(top1_matches_forced) else float("nan"),
                        }
                    )
                for probe_family, payload in rec.get("probes", {}).items():
                    step_count = min(n_steps, len(payload.get("kl_to_final", [])))
                    for step_idx in range(skip_first_n, step_count):
                        try:
                            metrics = derived_step_metrics(payload, step_idx)
                        except Exception:
                            continue
                        if not all(_finite(metrics[cov]) for cov in COVARIATES):
                            continue
                        if not _finite(metrics["late_kl_mean"]):
                            continue
                        row = {
                            "prompt_id": prompt_id,
                            "model": model,
                            "teacher_source": teacher_source,
                            "cell": cell,
                            "variant": str(rec.get("variant", "unknown")),
                            "prompt_mode": rec.get("prompt_mode"),
                            "probe_family": str(probe_family),
                            "step_idx": step_idx,
                            "forced_id": int(forced_ids[step_idx]) if step_idx < len(forced_ids) else None,
                            "target_logprob": as_float(target_logprob[step_idx]) if step_idx < len(target_logprob) else float("nan"),
                            "target_rank": as_float(target_rank[step_idx]) if step_idx < len(target_rank) else float("nan"),
                            "top1_matches_forced": float(bool(top1_matches_forced[step_idx])) if step_idx < len(top1_matches_forced) else float("nan"),
                        }
                        row.update(metrics)
                        rows.append(row)
    record_alignment_rows = []
    for (model, teacher_source, prompt_id), payload in record_alignment.items():
        cells = set(payload["cells"])
        n_steps_values = set(payload["n_steps"].values())
        forced_hash_values = set(payload["forced_hashes"].values())
        record_alignment_rows.append(
            {
                "model": model,
                "teacher_source": teacher_source,
                "prompt_id": prompt_id,
                "n_cells": len(cells),
                "missing_cells": ",".join(sorted(set(CELLS) - cells)),
                "same_n_steps": len(n_steps_values) <= 1,
                "same_forced_ids": len(forced_hash_values) <= 1,
            }
        )
    summary = {
        "paths": [str(path) for path in paths],
        "branch_counts": branch_counts,
        "malformed_examples": malformed_examples,
        "n_rows": len(rows),
        "record_alignment": record_alignment_rows,
    }
    return pd.DataFrame(rows), pd.DataFrame(support_rows), summary


def _comparison_frame(df: pd.DataFrame, control_cell: str, treatment_cell: str) -> pd.DataFrame:
    sub = df[df["cell"].isin([control_cell, treatment_cell])].copy()
    sub["variant"] = np.where(sub["cell"] == treatment_cell, "it", "pt")
    sub["variant_it"] = (sub["variant"] == "it").astype(int)
    return sub


def _paired_effect(
    df: pd.DataFrame,
    *,
    control_cell: str,
    treatment_cell: str,
    outcome: str,
    probe_family: str,
) -> tuple[float, dict[str, pd.DataFrame]]:
    sub = df[df["cell"].isin([control_cell, treatment_cell])].copy()
    if probe_family != "pooled_probe_fe":
        sub = sub[sub["probe_family"] == probe_family].copy()
    base_keys = ["model", "prompt_id", "step_idx"]
    if "teacher_source" in sub.columns:
        base_keys.insert(1, "teacher_source")
    keys = base_keys + ["probe_family"] if probe_family == "pooled_probe_fe" else base_keys
    control = sub[sub["cell"] == control_cell][keys + [outcome]].rename(columns={outcome: "control"})
    treatment = sub[sub["cell"] == treatment_cell][keys + [outcome]].rename(columns={outcome: "treatment"})
    merged = control.merge(treatment, on=keys, how="inner")
    if merged.empty:
        return float("nan"), {}
    merged["diff"] = merged["treatment"].astype(float) - merged["control"].astype(float)
    model_means = []
    by_model: dict[str, pd.DataFrame] = {}
    for model, group in merged.groupby("model", sort=True):
        # The paired CI is a prompt-cluster bootstrap, so use the same
        # prompt-cluster estimand for the printed point estimate.
        cluster_cols = [col for col in ["teacher_source", "prompt_id"] if col in group.columns]
        prompt_means = group.groupby(cluster_cols, sort=False)["diff"].mean().reset_index()
        if prompt_means.empty:
            continue
        by_model[str(model)] = prompt_means
        model_means.append(float(prompt_means["diff"].mean()))
    if not model_means:
        return float("nan"), {}
    return float(np.mean(model_means)), by_model


def _stable_seed_offset(*parts: Any) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % 100_000


def _paired_bootstrap_ci(
    by_model: dict[str, pd.DataFrame],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if n_boot <= 0 or not by_model:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    clustered: dict[str, pd.Series] = {}
    for model, model_df in by_model.items():
        cluster_cols = [col for col in ["teacher_source", "prompt_id"] if col in model_df.columns]
        prompt_means = model_df.groupby(cluster_cols, sort=False)["diff"].mean()
        if not prompt_means.empty:
            clustered[model] = prompt_means
    boot: list[float] = []
    for _ in range(n_boot):
        diffs: list[float] = []
        for prompt_means in clustered.values():
            vals = prompt_means.to_numpy(dtype=float)
            idx = rng.integers(0, len(vals), size=len(vals))
            diff = float(np.mean(vals[idx]))
            if math.isfinite(diff):
                diffs.append(diff)
        if diffs:
            boot.append(float(np.mean(diffs)))
    if len(boot) < 10:
        return float("nan"), float("nan")
    lo, hi = np.quantile(np.asarray(boot), [0.025, 0.975])
    return float(lo), float(hi)


def paired_effects(
    df: pd.DataFrame,
    *,
    comparison_name: str,
    control_cell: str,
    treatment_cell: str,
    n_boot: int,
    seed: int,
    teacher_source: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if teacher_source is not None and "teacher_source" in df.columns:
        df = df[df["teacher_source"] == teacher_source].copy()
    probes = sorted(df["probe_family"].unique())
    probes.append("pooled_probe_fe")
    for probe_family in probes:
        for outcome in OUTCOMES:
            estimate, by_model = _paired_effect(
                df,
                control_cell=control_cell,
                treatment_cell=treatment_cell,
                outcome=outcome,
                probe_family=probe_family,
            )
            lo, hi = _paired_bootstrap_ci(
                by_model,
                n_boot=n_boot,
                seed=seed + _stable_seed_offset(comparison_name, probe_family, outcome),
            )
            rows.append(
                {
                    "comparison": comparison_name,
                    "teacher_source": teacher_source,
                    "method": "paired_same_prompt_step",
                    "control_cell": control_cell,
                    "treatment_cell": treatment_cell,
                    "probe_family": probe_family,
                    "outcome": outcome,
                    "estimate": estimate,
                    "ci95_low": lo,
                    "ci95_high": hi,
                    "n_models": len(by_model),
                    "models": ",".join(sorted(by_model)),
                }
            )
    return rows


def cem_effects(
    df: pd.DataFrame,
    *,
    comparison_name: str,
    control_cell: str,
    treatment_cell: str,
    n_bins: int,
    n_boot: int,
    seed: int,
    teacher_source: str | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    if teacher_source is not None and "teacher_source" in df.columns:
        df = df[df["teacher_source"] == teacher_source].copy()
    comp = _comparison_frame(df, control_cell, treatment_cell)
    cem_df, matching = compute_cem_weights(comp, n_bins=n_bins)
    bal = balance_table(cem_df)
    rows = dense_equal_model_effects(
        cem_df,
        method="cem",
        weight_col="cem_weight",
        n_boot=n_boot,
        seed=seed,
    )
    for row in rows:
        row["comparison"] = comparison_name
        row["teacher_source"] = teacher_source
        row["control_cell"] = control_cell
        row["treatment_cell"] = treatment_cell
        row["estimate"] = row.pop("estimate_it_minus_pt")
    return rows, bal, matching


def support_summary(support: pd.DataFrame) -> pd.DataFrame:
    if support.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [col for col in ["model", "teacher_source", "cell"] if col in support.columns]
    for key, group in support.groupby(group_cols, sort=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_values, strict=True))
        row.update(
            {
                "n_steps": int(len(group)),
                "mean_target_logprob": float(group["target_logprob"].mean()),
                "median_target_rank": float(group["target_rank"].median()),
                "mean_target_rank": float(group["target_rank"].mean()),
                "top1_match_rate": float(group["top1_matches_forced"].mean()),
            }
        )
        rows.append(
            row
        )
    return pd.DataFrame(rows)


def alignment_summary(df: pd.DataFrame, load_summary: dict[str, Any]) -> dict[str, Any]:
    record_alignment = pd.DataFrame(load_summary.get("record_alignment", []))
    record_missing = 0
    record_bad_ids = 0
    if not record_alignment.empty:
        record_missing = int((record_alignment["missing_cells"].astype(str) != "").sum())
        record_bad_ids = int((~record_alignment["same_forced_ids"].astype(bool)).sum())
    missing_rows = 0
    if not df.empty:
        align_cols = [col for col in ["model", "teacher_source", "prompt_id", "probe_family", "step_idx"] if col in df.columns]
        cell_counts = df.groupby(align_cols)["cell"].nunique()
        missing_rows = int((len(CELLS) - cell_counts).clip(lower=0).sum())
    return {
        "record_missing_cell_count": record_missing,
        "record_forced_id_mismatch_count": record_bad_ids,
        "missing_aligned_step_rows": missing_rows,
    }


def plot_effects(effects: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = effects[
        (effects["method"] == "paired_same_prompt_step")
        & (effects["probe_family"].isin(["raw", "tuned"]))
        & (effects["outcome"] == "late_kl_mean")
    ].copy()
    if primary.empty:
        return
    source = primary.get("teacher_source")
    if source is not None:
        primary["label"] = primary["teacher_source"].fillna("unknown").astype(str) + "/" + primary["comparison"] + "/" + primary["probe_family"]
    else:
        primary["label"] = primary["comparison"] + "/" + primary["probe_family"]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(primary))
    ax.bar(x, primary["estimate"], color="#4C78A8")
    lower = primary["estimate"].to_numpy(dtype=float) - primary["ci95_low"].to_numpy(dtype=float)
    upper = primary["ci95_high"].to_numpy(dtype=float) - primary["estimate"].to_numpy(dtype=float)
    yerr = np.vstack([np.maximum(lower, 0.0), np.maximum(upper, 0.0)])
    ax.errorbar(x, primary["estimate"], yerr=yerr, fmt="none", color="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, primary["label"], rotation=25, ha="right")
    ax.set_ylabel("late KL effect (nats)")
    ax.set_title("Exp22 fixed-history template audit")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_note(summary: dict[str, Any], out_path: Path) -> None:
    rows = summary.get("controlled_estimates", [])

    def lookup(teacher_source: str, comparison: str, method: str, probe: str, outcome: str) -> dict[str, Any]:
        for row in rows:
            if (
                row.get("teacher_source") == teacher_source
                and
                row.get("comparison") == comparison
                and row.get("method") == method
                and row.get("probe_family") == probe
                and row.get("outcome") == outcome
            ):
                return row
        return {}

    teacher_sources = summary.get("teacher_sources") or ["it_native"]
    blocks = []
    for teacher_source in teacher_sources:
        native = lookup(teacher_source, "native_fixed_effect", "paired_same_prompt_step", "raw", "late_kl_mean")
        raw = lookup(teacher_source, "raw_fixed_effect", "paired_same_prompt_step", "raw", "late_kl_mean")
        delta = lookup(teacher_source, "template_delta", "paired_same_prompt_step", "raw", "late_kl_mean")
        blocks.append(
            f"""Teacher source: `{teacher_source}`
- Native fixed effect (`it_native - pt_raw`): {native.get('estimate', float('nan')):.6g}, 95% CI [{native.get('ci95_low', float('nan')):.6g}, {native.get('ci95_high', float('nan')):.6g}]
- Raw/no-template fixed effect (`it_raw - pt_raw`): {raw.get('estimate', float('nan')):.6g}, 95% CI [{raw.get('ci95_low', float('nan')):.6g}, {raw.get('ci95_high', float('nan')):.6g}]
- Template delta (`it_native - it_raw`): {delta.get('estimate', float('nan')):.6g}, 95% CI [{delta.get('ci95_low', float('nan')):.6g}, {delta.get('ci95_high', float('nan')):.6g}]"""
        )
    primary_block = "\n\n".join(blocks)
    text = f"""# Exp22 Fixed-History Template Audit

Run: `{Path(summary["root"]).name}`.

Primary paired raw-lens late-KL estimates:
{primary_block}

Quality gates: malformed max {summary['quality']['max_malformed_rate']:.4g}, missing aligned step rows {summary['quality']['missing_aligned_step_rows']}, minimum CEM retention {summary['quality']['min_retained_fraction']:.4g}, maximum post-match SMD {summary['quality']['max_smd_after']:.4g}.

Interpretation rule:
- If the native fixed-history effect remains positive, the native-template convergence-gap result is not a free-generation length artifact.
- If the raw/no-template effect weakens or reverses, describe raw prompting as an off-distribution serialization that modulates expression of the gap.
- If the native fixed-history effect fails, report this as a serious limitation and do not strengthen the main claim.
"""
    out_path.write_text(text)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, support, load_summary = flatten_records(root, skip_first_n=args.skip_first_n)
    if df.empty:
        raise RuntimeError(f"No valid fixed-history Exp22 rows found under {root}")
    if args.models:
        df = df[df["model"].isin(args.models)].copy()
        support = support[support["model"].isin(args.models)].copy() if not support.empty else support

    effect_rows: list[dict[str, Any]] = []
    balance_frames: list[pd.DataFrame] = []
    matching_by_comparison: dict[str, Any] = {}
    teacher_sources = sorted(str(x) for x in df.get("teacher_source", pd.Series(["it_native"])).dropna().unique())
    for source_offset, teacher_source in enumerate(teacher_sources):
        source_df = df[df["teacher_source"] == teacher_source].copy() if "teacher_source" in df.columns else df.copy()
        for offset, (name, (control, treatment)) in enumerate(COMPARISONS.items()):
            seed_offset = args.seed + source_offset * 1009 + offset * 101
            effect_rows.extend(
                paired_effects(
                    source_df,
                    comparison_name=name,
                    control_cell=control,
                    treatment_cell=treatment,
                    n_boot=args.n_boot,
                    seed=seed_offset,
                    teacher_source=teacher_source,
                )
            )
            cem_rows, bal, matching = cem_effects(
                source_df,
                comparison_name=name,
                control_cell=control,
                treatment_cell=treatment,
                n_bins=args.n_bins,
                n_boot=args.n_boot,
                seed=seed_offset + 31,
                teacher_source=teacher_source,
            )
            effect_rows.extend(cem_rows)
            bal = bal.copy()
            bal["teacher_source"] = teacher_source
            bal["comparison"] = name
            bal["control_cell"] = control
            bal["treatment_cell"] = treatment
            balance_frames.append(bal)
            matching_by_comparison[f"{teacher_source}/{name}"] = matching

    effects = pd.DataFrame(effect_rows)
    balance = pd.concat(balance_frames, ignore_index=True) if balance_frames else pd.DataFrame()
    support_out = support_summary(support)
    align = alignment_summary(df, load_summary)
    malformed_rates: dict[str, float] = {}
    for branch, counts in load_summary["branch_counts"].items():
        total = max(int(counts.get("records", 0)), 1)
        malformed_rates[branch] = float(counts.get("malformed", 0) / total)
    max_smd_after = float(np.nanmax(np.abs(balance["smd_after"].to_numpy()))) if not balance.empty else float("inf")
    min_retained = min(
        (float(matching.get("min_retained_fraction", 0.0)) for matching in matching_by_comparison.values()),
        default=0.0,
    )
    quality = {
        "max_malformed_rate": max(malformed_rates.values()) if malformed_rates else 0.0,
        "min_retained_fraction": min_retained,
        "max_smd_after": max_smd_after,
        **align,
    }
    quality["ok"] = bool(
        quality["max_malformed_rate"] <= args.max_malformed_rate
        and quality["missing_aligned_step_rows"] == 0
        and quality["record_forced_id_mismatch_count"] == 0
        and quality["min_retained_fraction"] >= args.min_retained_fraction
        and quality["max_smd_after"] <= args.max_smd
    )

    effects_path = out_dir / "controlled_effects.csv"
    balance_path = out_dir / "balance.csv"
    support_path = out_dir / "support.csv"
    effects.to_csv(effects_path, index=False)
    balance.to_csv(balance_path, index=False)
    support_out.to_csv(support_path, index=False)
    plot_path = out_dir / "fixed_history_template_audit.png"
    plot_effects(effects, plot_path)
    summary = {
        "root": str(root),
        "n_rows": int(len(df)),
        "models": args.models,
        "teacher_sources": teacher_sources,
        "cells": list(CELLS),
        "comparisons": COMPARISONS,
        "covariates": COVARIATES,
        "outcomes": OUTCOMES,
        "load_summary": load_summary,
        "matching": matching_by_comparison,
        "quality": quality,
        "controlled_estimates": effects.to_dict(orient="records"),
        "support": support_out.to_dict(orient="records"),
        "malformed_rates": malformed_rates,
        "artifacts": {
            "controlled_effects_csv": str(effects_path),
            "balance_csv": str(balance_path),
            "support_csv": str(support_path),
            "plot": str(plot_path),
            "paper_note": str(out_dir / "paper_claims_exp22_fixed_history.md"),
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_note(summary, out_dir / "paper_claims_exp22_fixed_history.md")
    if args.fail_on_quality and not quality["ok"]:
        raise SystemExit(f"Exp22 fixed-history quality gate failed: {quality}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=PRIMARY_MODELS)
    parser.add_argument("--skip-first-n", type=int, default=5)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-malformed-rate", type=float, default=0.01)
    parser.add_argument("--min-retained-fraction", type=float, default=0.75)
    parser.add_argument("--max-smd", type=float, default=0.1)
    parser.add_argument("--fail-on-quality", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(json.dumps({"quality": summary["quality"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
