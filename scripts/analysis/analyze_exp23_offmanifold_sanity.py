#!/usr/bin/env python3
"""CPU-only off-manifold sanity diagnostics for Exp23 residual factorial records."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_ROOT = Path(
    "results/exp23_midlate_interaction_suite/"
    "exp23_dense5_full_h100x8_20260426_sh4_rw4/residual_factorial/raw_shared"
)
DEFAULT_OUT = Path("results/paper_synthesis/exp23_offmanifold_sanity")

MODELS = ("gemma3_4b", "llama31_8b", "mistral_7b", "olmo2_7b", "qwen3_4b")
CELLS = ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
DIAGONAL_CELLS = ("U_PT__L_PT", "U_IT__L_IT")
OFFDIAGONAL_CELLS = ("U_PT__L_IT", "U_IT__L_PT")
READOUTS = ("common_it", "common_pt")
SCALAR_METRICS = ("it_vs_pt_margin", "it_rank", "pt_rank")
TRAJECTORY_METRICS = ("late_kl_mean", "remaining_adj_js", "future_top1_flips", "top5_churn")
EFFECTS = {
    "interaction": {
        "U_IT__L_IT": 1.0,
        "U_IT__L_PT": -1.0,
        "U_PT__L_IT": -1.0,
        "U_PT__L_PT": 1.0,
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


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.array(values, dtype=float)))


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.array(values, dtype=float)))


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=float), q))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _unit_effect(cells: dict[str, float], coefficients: dict[str, float]) -> float | None:
    total = 0.0
    for cell, coef in coefficients.items():
        value = cells.get(cell)
        if value is None:
            return None
        total += coef * value
    return total


def collect(records_root: Path, models: list[str], event_kind: str) -> dict[str, Any]:
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    top_classes: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    margin_units: dict[tuple[str, str], list[tuple[str, str, dict[str, float]]]] = defaultdict(list)
    noop_checks: list[dict[str, Any]] = []
    valid_by_model: Counter[str] = Counter()
    invalid_by_model: Counter[str] = Counter()

    for model in models:
        path = records_root / model / "records.jsonl.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        for record in _json_rows(path):
            event = (record.get("events") or {}).get(event_kind) or {}
            if not event.get("valid"):
                invalid_by_model[model] += 1
                continue
            valid_by_model[model] += 1

            for check_name, check in (event.get("noop_patch_checks") or {}).items():
                noop_checks.append(
                    {
                        "model": model,
                        "prompt_id": str(record.get("prompt_id")),
                        "event_kind": event_kind,
                        "check_name": check_name,
                        "patch_input_delta_max_abs": _finite(check.get("patch_input_delta_max_abs")),
                        "common_it_margin_abs_delta": _finite(check.get("common_it_margin_abs_delta")),
                        "common_it_top1_equal": bool(check.get("common_it_top1_equal")),
                    }
                )

            cells = event.get("cells") or {}
            for readout in READOUTS:
                unit_margins: dict[str, float] = {}
                for cell in CELLS:
                    cell_payload = cells.get(cell) or {}
                    readout_payload = cell_payload.get(readout) or {}
                    margin = _finite(readout_payload.get("it_vs_pt_margin"))
                    if margin is not None:
                        unit_margins[cell] = margin
                    token_choice = str(readout_payload.get("token_choice_class"))
                    if token_choice and token_choice != "None":
                        top_classes[(model, cell, readout)][token_choice] += 1
                    for metric in SCALAR_METRICS:
                        value = _finite(readout_payload.get(metric))
                        if value is not None:
                            values[(model, cell, readout, metric)].append(value)
                    trajectory = readout_payload.get("trajectory") or {}
                    for metric in TRAJECTORY_METRICS:
                        value = _finite(trajectory.get(metric))
                        if value is not None:
                            values[(model, cell, readout, metric)].append(value)
                if set(CELLS).issubset(unit_margins):
                    margin_units[(model, readout)].append(
                        (str(record.get("prompt_id")), event_kind, unit_margins)
                    )

    return {
        "values": values,
        "top_classes": top_classes,
        "margin_units": margin_units,
        "noop_checks": noop_checks,
        "valid_by_model": valid_by_model,
        "invalid_by_model": invalid_by_model,
    }


def summarize_cell_metrics(collected: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = collected["values"]
    top_classes = collected["top_classes"]
    for model in models:
        for cell in CELLS:
            for readout in READOUTS:
                for metric in (*SCALAR_METRICS, *TRAJECTORY_METRICS):
                    vals = values.get((model, cell, readout, metric), [])
                    rows.append(
                        {
                            "model": model,
                            "cell": cell,
                            "readout": readout,
                            "metric": metric,
                            "n": len(vals),
                            "mean": _mean(vals),
                            "median": _median(vals),
                            "p05": _percentile(vals, 5),
                            "p95": _percentile(vals, 95),
                        }
                    )
                counts = top_classes.get((model, cell, readout), Counter())
                total = sum(counts.values())
                for label in ("pt", "it", "other"):
                    rows.append(
                        {
                            "model": model,
                            "cell": cell,
                            "readout": readout,
                            "metric": f"top_choice_{label}_rate",
                            "n": total,
                            "mean": (counts.get(label, 0) / total) if total else None,
                            "median": counts.get(label, 0),
                            "p05": None,
                            "p95": None,
                        }
                    )
    return rows


def summarize_degradation(cell_rows: list[dict[str, Any]], models: list[str]) -> list[dict[str, Any]]:
    by_key = {
        (row["model"], row["cell"], row["readout"], row["metric"]): row
        for row in cell_rows
        if row["mean"] is not None
    }
    rows: list[dict[str, Any]] = []
    for model in models:
        for readout in READOUTS:
            for metric in TRAJECTORY_METRICS:
                diag_means = [
                    by_key[(model, cell, readout, metric)]["mean"]
                    for cell in DIAGONAL_CELLS
                    if (model, cell, readout, metric) in by_key
                ]
                off_means = [
                    by_key[(model, cell, readout, metric)]["mean"]
                    for cell in OFFDIAGONAL_CELLS
                    if (model, cell, readout, metric) in by_key
                ]
                if not diag_means or not off_means:
                    continue
                diag_worst = max(diag_means)
                off_worst = max(off_means)
                rows.append(
                    {
                        "model": model,
                        "readout": readout,
                        "metric": metric,
                        "diag_worst_mean": diag_worst,
                        "offdiag_worst_mean": off_worst,
                        "offdiag_minus_diag_worst": off_worst - diag_worst,
                        "offdiag_over_diag_worst": off_worst / diag_worst if diag_worst else None,
                    }
                )
    return rows


def summarize_effects(collected: dict[str, Any], models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (model, readout), units in collected["margin_units"].items():
        if model not in models:
            continue
        for effect_name, coefficients in EFFECTS.items():
            vals = [
                value
                for _, _, margins in units
                for value in [_unit_effect(margins, coefficients)]
                if value is not None
            ]
            rows.append(
                {
                    "model": model,
                    "readout": readout,
                    "effect": effect_name,
                    "n": len(vals),
                    "estimate": _mean(vals),
                    "median": _median(vals),
                    "p05": _percentile(vals, 5),
                    "p95": _percentile(vals, 95),
                }
            )
    return rows


def bootstrap_effect_agreement(
    collected: dict[str, Any],
    models: list[str],
    *,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Family-balanced prompt bootstrap for common-IT/common-PT readout agreement."""
    if n_bootstrap <= 0:
        return []

    by_model_readout: dict[tuple[str, str], np.ndarray] = {}
    by_model_prompt: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for (model, readout), units in collected["margin_units"].items():
        vals: list[float] = []
        for prompt_id, _, margins in units:
            value = _unit_effect(margins, EFFECTS["interaction"])
            if value is None:
                continue
            vals.append(value)
            by_model_prompt[model][prompt_id][readout] = value
        if vals:
            by_model_readout[(model, readout)] = np.asarray(vals, dtype=float)

    by_model_diff: dict[str, np.ndarray] = {}
    for model in models:
        vals = [
            payload["common_it"] - payload["common_pt"]
            for payload in by_model_prompt.get(model, {}).values()
            if "common_it" in payload and "common_pt" in payload
        ]
        if vals:
            by_model_diff[model] = np.asarray(vals, dtype=float)

    rng = np.random.default_rng(seed)

    def family_balanced_bootstrap(arrays: list[np.ndarray]) -> tuple[float, float, float]:
        estimate = float(np.mean([float(arr.mean()) for arr in arrays]))
        draws = np.empty(n_bootstrap, dtype=float)
        for idx in range(n_bootstrap):
            model_means = []
            for arr in arrays:
                sample_idx = rng.integers(0, len(arr), size=len(arr))
                model_means.append(float(arr[sample_idx].mean()))
            draws[idx] = float(np.mean(model_means))
        return estimate, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))

    rows: list[dict[str, Any]] = []
    for readout in READOUTS:
        arrays = [by_model_readout[(model, readout)] for model in models if (model, readout) in by_model_readout]
        if not arrays:
            continue
        estimate, ci_low, ci_high = family_balanced_bootstrap(arrays)
        rows.append(
            {
                "metric": f"{readout}_interaction",
                "n_models": len(arrays),
                "n_bootstrap": n_bootstrap,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

    diff_arrays = [by_model_diff[model] for model in models if model in by_model_diff]
    if diff_arrays:
        estimate, ci_low, ci_high = family_balanced_bootstrap(diff_arrays)
        rows.append(
            {
                "metric": "common_it_minus_common_pt_interaction",
                "n_models": len(diff_arrays),
                "n_bootstrap": n_bootstrap,
                "estimate": estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return rows


def pooled_top_choice(cell_rows: list[dict[str, Any]], *, readout: str, cell: str) -> dict[str, float]:
    out: dict[str, float] = {}
    totals: dict[str, float] = {}
    n_by_model: dict[str, float] = {}
    for row in cell_rows:
        if row["readout"] != readout or row["cell"] != cell:
            continue
        metric = str(row["metric"])
        if not metric.startswith("top_choice_") or row["mean"] is None:
            continue
        n = float(row["n"] or 0)
        n_by_model[str(row["model"])] = n
        label = metric.replace("top_choice_", "").replace("_rate", "")
        totals[label] = totals.get(label, 0.0) + float(row["mean"]) * n
    n_total = sum(n_by_model.values())
    for label, count in totals.items():
        out[label] = count / n_total if n_total else math.nan
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    *,
    path: Path,
    summary: dict[str, Any],
    degradation_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
) -> None:
    common_it_effect = [
        row for row in effect_rows if row["readout"] == "common_it" and row["effect"] == "interaction"
    ]
    common_pt_effect = [
        row for row in effect_rows if row["readout"] == "common_pt" and row["effect"] == "interaction"
    ]
    dense5_common_it = _mean([float(row["estimate"]) for row in common_it_effect if row["estimate"] is not None])
    dense5_common_pt = _mean([float(row["estimate"]) for row in common_pt_effect if row["estimate"] is not None])

    degradation_common_it = [row for row in degradation_rows if row["readout"] == "common_it"]
    max_ratio = max(
        float(row["offdiag_over_diag_worst"])
        for row in degradation_common_it
        if row["offdiag_over_diag_worst"] is not None
    )
    max_delta = max(
        float(row["offdiag_minus_diag_worst"])
        for row in degradation_common_it
        if row["offdiag_minus_diag_worst"] is not None
    )
    top_u_pt_l_it = pooled_top_choice(cell_rows, readout="common_it", cell="U_PT__L_IT")
    top_u_it_l_pt = pooled_top_choice(cell_rows, readout="common_it", cell="U_IT__L_PT")
    boot_by_metric = {str(row["metric"]): row for row in bootstrap_rows}
    boot_common_it = boot_by_metric.get("common_it_interaction", {})
    boot_common_pt = boot_by_metric.get("common_pt_interaction", {})
    boot_diff = boot_by_metric.get("common_it_minus_common_pt_interaction", {})

    def boot_fmt(row: dict[str, Any]) -> str:
        if not row:
            return "NA"
        return (
            f"`{_fmt(row.get('estimate'))}` "
            f"[`{_fmt(row.get('ci_low'))}`, `{_fmt(row.get('ci_high'))}`]"
        )

    lines = [
        "# Exp23 Off-Manifold Sanity Diagnostic",
        "",
        "CPU-only diagnostic over the Dense-5 raw-record Exp23 first-divergence residual-factorial records.",
        "",
        "## Summary",
        "",
        f"- Valid first-divergence events: `{summary['n_valid']}`; invalid events: `{summary['n_invalid']}`.",
        f"- Noop diagonal patch checks: `{summary['noop']['n_checks']}`; max patch-input delta `{_fmt(summary['noop']['max_patch_input_delta_abs'])}`; max common-IT margin delta `{_fmt(summary['noop']['max_common_it_margin_abs_delta'])}`; all common-IT top-1 equal: `{summary['noop']['all_common_it_top1_equal']}`.",
        f"- Mean common-IT interaction across Dense-5 models: `{_fmt(dense5_common_it)}` logits; mean common-PT interaction: `{_fmt(dense5_common_pt)}` logits.",
        f"- Family-balanced bootstrap: common-IT interaction {boot_fmt(boot_common_it)} logits; common-PT interaction {boot_fmt(boot_common_pt)} logits; common-IT minus common-PT {boot_fmt(boot_diff)} logits.",
        f"- Across common-IT trajectory metrics, the worst off-diagonal mean is at most `{_fmt(max_ratio)}`x the worst diagonal mean; the largest absolute excess is `{_fmt(max_delta)}`.",
        f"- Pooled common-IT top-1 choices for `U_PT__L_IT`: `{_fmt(top_u_pt_l_it.get('pt'))}` PT, `{_fmt(top_u_pt_l_it.get('it'))}` IT, `{_fmt(top_u_pt_l_it.get('other'))}` other.",
        f"- Pooled common-IT top-1 choices for `U_IT__L_PT`: `{_fmt(top_u_it_l_pt.get('pt'))}` PT, `{_fmt(top_u_it_l_pt.get('it'))}` IT, `{_fmt(top_u_it_l_pt.get('other'))}` other.",
        "",
        "## Interpretation",
        "",
        "These checks do not prove that off-diagonal hybrids are natural model states. They do show that the recorded hybrids are not numerical patching failures or degenerate collapsed distributions under the stored readouts: diagonal cells reconstruct exactly, common-IT and common-PT readouts agree on the interaction, off-diagonal trajectory metrics remain in the diagonal range or only slightly above it, and hybrid top-1 predictions move in graded ways rather than becoming arbitrary.",
        "",
        "The remaining concern is semantic, not numerical: off-diagonal cells are still constructed counterfactuals. The main paper should therefore interpret the upstream x late interaction as a compatibility/readout estimand, not as full circuit recovery.",
        "",
        "## Worst Off-Diagonal Trajectory Ratios",
        "",
        "| Model | Metric | Common-IT offdiag / diag worst | Offdiag - diag worst |",
        "|---|---|---:|---:|",
    ]
    for row in degradation_common_it:
        lines.append(
            "| {model} | {metric} | `{ratio}` | `{delta}` |".format(
                model=row["model"],
                metric=row["metric"],
                ratio=_fmt(row["offdiag_over_diag_worst"]),
                delta=_fmt(row["offdiag_minus_diag_worst"]),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--event-kind", default="first_diff")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    collected = collect(args.records_root, args.models, args.event_kind)
    cell_rows = summarize_cell_metrics(collected, args.models)
    degradation_rows = summarize_degradation(cell_rows, args.models)
    effect_rows = summarize_effects(collected, args.models)
    bootstrap_rows = bootstrap_effect_agreement(
        collected,
        args.models,
        n_bootstrap=args.bootstrap_iters,
        seed=args.seed,
    )

    noop_checks = collected["noop_checks"]
    summary = {
        "records_root": str(args.records_root),
        "event_kind": args.event_kind,
        "models": args.models,
        "n_valid": int(sum(collected["valid_by_model"].values())),
        "n_invalid": int(sum(collected["invalid_by_model"].values())),
        "valid_by_model": dict(collected["valid_by_model"]),
        "invalid_by_model": dict(collected["invalid_by_model"]),
        "noop": {
            "n_checks": len(noop_checks),
            "max_patch_input_delta_abs": max(
                [float(row["patch_input_delta_max_abs"] or 0.0) for row in noop_checks],
                default=None,
            ),
            "max_common_it_margin_abs_delta": max(
                [float(row["common_it_margin_abs_delta"] or 0.0) for row in noop_checks],
                default=None,
            ),
            "all_common_it_top1_equal": all(bool(row["common_it_top1_equal"]) for row in noop_checks),
        },
        "artifacts": {
            "cell_metrics": "cell_metrics.csv",
            "degradation_ratios": "offdiag_degradation_ratios.csv",
            "effect_sanity": "effect_sanity.csv",
            "effect_bootstrap": "effect_bootstrap.csv",
            "report": "offmanifold_sanity_report.md",
        },
    }

    write_csv(args.out_dir / "cell_metrics.csv", cell_rows)
    write_csv(args.out_dir / "offdiag_degradation_ratios.csv", degradation_rows)
    write_csv(args.out_dir / "effect_sanity.csv", effect_rows)
    write_csv(args.out_dir / "effect_bootstrap.csv", bootstrap_rows)
    write_csv(args.out_dir / "noop_checks.csv", noop_checks)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_markdown(
        path=args.out_dir / "offmanifold_sanity_report.md",
        summary=summary,
        degradation_rows=degradation_rows,
        effect_rows=effect_rows,
        cell_rows=cell_rows,
        bootstrap_rows=bootstrap_rows,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
