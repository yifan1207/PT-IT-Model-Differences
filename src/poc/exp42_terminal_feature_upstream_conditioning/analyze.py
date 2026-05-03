"""Analyze Exp42 terminal feature upstream-conditioning records."""

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


PRIMARY_MODELS = ("gemma3_4b", "llama31_8b", "mistral_7b", "qwen3_4b")
PRIMARY_KS = (1, 2, 5, 10, 20, 50, 100, 200)


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


def _mean(values: list[Any]) -> float | None:
    vals = [v for v in (_finite(x) for x in values) if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _event_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, str]:
    seed = "none" if row.get("control_seed") is None else str(row.get("control_seed"))
    return (
        str(row.get("model")),
        str(row.get("prompt_id")),
        str(row.get("event_kind")),
        str(row.get("feature_set")),
        int(row.get("k", 0)),
        seed,
    )


def _load_records(run_root: Path, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        candidates = [
            run_root / "raw" / model / "records.jsonl.gz",
            run_root / model / "records.jsonl.gz",
            run_root / "raw" / model / "records_w0.jsonl.gz",
        ]
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            continue
        rows.extend(_json_rows(path))
    return rows


def _activation_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # First aggregate terminal-layer rows within an event/cell/feature-set.
    by_cell: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("record_type") != "activation":
            continue
        key = (*_event_key(row), str(row.get("cell")))
        by_cell[key].append(row)

    cell_aggs: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in by_cell.items():
        model, prompt_id, event_kind, feature_set, k, seed, cell = key
        cell_aggs[key] = {
            "model": model,
            "prompt_id": prompt_id,
            "event_kind": event_kind,
            "feature_set": feature_set,
            "k": int(k),
            "control_seed": None if seed == "none" else int(seed),
            "cell": cell,
            "position_ge_3": bool(group[0].get("position_ge_3")),
            "n_layers": len(group),
            "decoder_weighted_sum": _mean([sum(_finite(g.get("decoder_weighted_sum")) or 0.0 for g in group)]),
            "decoder_margin_weighted_sum": _mean([sum(_finite(g.get("decoder_margin_weighted_sum")) or 0.0 for g in group)]),
            "sum_activation": _mean([sum(_finite(g.get("sum_activation")) or 0.0 for g in group)]),
            "selected_feature_l0": _mean([sum(_finite(g.get("selected_feature_l0")) or 0.0 for g in group)]),
            "reconstruction_error_rel": _mean([g.get("reconstruction_error_rel") for g in group]),
            "feature_l0": _mean([g.get("feature_l0") for g in group]),
            "selected_union_mass_fraction": _mean([g.get("selected_union_mass_fraction") for g in group]),
        }

    out: list[dict[str, Any]] = []
    base_keys = {
        (model, prompt_id, event_kind, feature_set, k, seed)
        for (model, prompt_id, event_kind, feature_set, k, seed, _cell) in cell_aggs
    }
    for base in sorted(base_keys):
        model, prompt_id, event_kind, feature_set, k, seed = base
        lit_native = cell_aggs.get((*base, "U_IT__L_IT"))
        lit_ptup = cell_aggs.get((*base, "U_PT__L_IT"))
        lpt_native = cell_aggs.get((*base, "U_PT__L_PT"))
        lpt_itup = cell_aggs.get((*base, "U_IT__L_PT"))
        if not lit_native or not lit_ptup:
            continue
        row = {
            "model": model,
            "prompt_id": prompt_id,
            "event_kind": event_kind,
            "feature_set": feature_set,
            "k": int(k),
            "control_seed": None if seed == "none" else int(seed),
            "position_ge_3": bool(lit_native.get("position_ge_3")),
            "activation_gate_decoder_weighted": float(lit_native["decoder_weighted_sum"] - lit_ptup["decoder_weighted_sum"]),
            "activation_gate_decoder_margin_weighted": float(
                lit_native["decoder_margin_weighted_sum"] - lit_ptup["decoder_margin_weighted_sum"]
            ),
            "activation_gate_sum": float(lit_native["sum_activation"] - lit_ptup["sum_activation"]),
            "activation_gate_l0": float(lit_native["selected_feature_l0"] - lit_ptup["selected_feature_l0"]),
            "uit_lit_decoder_weighted": lit_native["decoder_weighted_sum"],
            "upt_lit_decoder_weighted": lit_ptup["decoder_weighted_sum"],
            "uit_lit_reconstruction_error_rel": lit_native["reconstruction_error_rel"],
            "upt_lit_reconstruction_error_rel": lit_ptup["reconstruction_error_rel"],
            "uit_lit_feature_l0": lit_native["feature_l0"],
            "upt_lit_feature_l0": lit_ptup["feature_l0"],
            "uit_lit_selected_union_mass_fraction": lit_native["selected_union_mass_fraction"],
            "upt_lit_selected_union_mass_fraction": lit_ptup["selected_union_mass_fraction"],
        }
        if lpt_native and lpt_itup:
            row["pt_branch_activation_gate_decoder_weighted"] = float(
                lpt_native["decoder_weighted_sum"] - lpt_itup["decoder_weighted_sum"]
            )
        else:
            row["pt_branch_activation_gate_decoder_weighted"] = None
        out.append(row)
    return out


def _ablation_event_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("record_type") != "ablation":
            continue
        out.append(
            {
                "model": row.get("model"),
                "prompt_id": row.get("prompt_id"),
                "event_kind": row.get("event_kind"),
                "feature_set": row.get("feature_set"),
                "k": int(row.get("k", 0)),
                "control_seed": row.get("control_seed"),
                "position_ge_3": bool(row.get("position_ge_3")),
                "interaction_full": _finite(row.get("interaction_full")),
                "interaction_ablate": _finite(row.get("interaction_ablate")),
                "interaction_drop": _finite(row.get("interaction_drop")),
                "mediation_fraction": _finite(row.get("mediation_fraction")),
                "drop_native_upstream": _finite(row.get("drop_native_upstream")),
                "drop_pt_upstream": _finite(row.get("drop_pt_upstream")),
                "feature_causal_gate": _finite(row.get("feature_causal_gate")),
            }
        )
    return out


def _summarize_events(events: list[dict[str, Any]], metric: str) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        seed = "none" if row.get("control_seed") is None else str(row.get("control_seed"))
        grouped[(str(row["model"]), str(row["feature_set"]), int(row["k"]), seed)].append(row)
    out = {}
    for key, group in grouped.items():
        vals = [row.get(metric) for row in group]
        out[key] = {
            "model": key[0],
            "feature_set": key[1],
            "k": key[2],
            "control_seed": None if key[3] == "none" else int(key[3]),
            "n": len(group),
            f"{metric}_mean": _mean(vals),
            f"{metric}_pos3_mean": _mean([row.get(metric) for row in group if row.get("position_ge_3")]),
        }
    return out


def _join_event_rows(activation_events: list[dict[str, Any]], ablation_events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    abl_by_key = {_event_key(row): row for row in ablation_events}
    out = []
    for row in activation_events:
        joined = {**row}
        abl = abl_by_key.get(_event_key(row))
        if abl:
            for key in (
                "interaction_full",
                "interaction_drop",
                "mediation_fraction",
                "drop_native_upstream",
                "drop_pt_upstream",
                "feature_causal_gate",
            ):
                joined[key] = abl.get(key)
        out.append(joined)
    return out


def _causal_minus_control(events: list[dict[str, Any]], metric: str, control: str) -> list[dict[str, Any]]:
    by_base: dict[tuple[str, str, str, int], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in events:
        key = (str(row["model"]), str(row["prompt_id"]), str(row["event_kind"]), int(row["k"]))
        by_base[key][str(row["feature_set"])].append(row)
    out = []
    for (model, prompt_id, event_kind, k), groups in by_base.items():
        causal = groups.get("causal_top") or []
        controls = groups.get(control) or []
        if not causal or not controls:
            continue
        cval = _finite(causal[0].get(metric))
        rval = _mean([row.get(metric) for row in controls])
        if cval is None or rval is None:
            continue
        out.append(
            {
                "model": model,
                "prompt_id": prompt_id,
                "event_kind": event_kind,
                "k": int(k),
                "control": control,
                "metric": metric,
                "causal_value": cval,
                "control_value": rval,
                "causal_minus_control": cval - rval,
                "position_ge_3": bool(causal[0].get("position_ge_3")),
            }
        )
    return out


def _bootstrap_family_mean(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    models: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    by_model: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("model") in models:
            val = _finite(row.get(metric))
            if val is not None:
                by_model[str(row["model"])].append(val)
    family_means = {model: _mean(vals) for model, vals in by_model.items()}
    present = [m for m in models if by_model.get(m)]
    estimate = _mean([family_means[m] for m in present])
    if len(present) < 2 or n_boot <= 0:
        return {"estimate": estimate, "ci_low": None, "ci_high": None, "n_families": len(present), "family_means": family_means}
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        vals = []
        for model in present:
            arr = np.asarray(by_model[model], dtype=np.float64)
            vals.append(float(rng.choice(arr, size=arr.size, replace=True).mean()))
        boot[idx] = float(np.mean(vals))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_families": len(present),
        "family_means": family_means,
    }


def _saturation(events: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    by_model: dict[str, dict[int, float | None]] = defaultdict(dict)
    for row in events:
        if row.get("feature_set") == "causal_top" and row.get("control_seed") is None:
            by_model[str(row["model"])].setdefault(int(row["k"]), [])
    tmp: dict[str, dict[int, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for row in events:
        if row.get("feature_set") == "causal_top" and row.get("control_seed") is None:
            tmp[str(row["model"])][int(row["k"])].append(row.get(metric))
    out = []
    for model, k_values in tmp.items():
        means = {k: _mean(vals) for k, vals in k_values.items()}
        target = means.get(200)
        result = {"model": model, "metric": metric, "top200": target}
        for pct in (50, 80, 90):
            kk = None
            if target is not None and target > 0:
                threshold = target * (pct / 100.0)
                for k in sorted(means):
                    val = means[k]
                    if val is not None and val >= threshold:
                        kk = k
                        break
            result[f"k{pct}"] = kk
        out.append(result)
    return out


def _plot(run_root: Path, summary_rows: list[dict[str, Any]], diff_rows: list[dict[str, Any]]) -> None:
    out_dir = run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = sorted({row["model"] for row in summary_rows if row.get("k") == 200})
    causal = []
    random = []
    top_active = []
    for model in models:
        causal.append(
            _mean(
                [
                    row.get("activation_gate_decoder_weighted_mean")
                    for row in summary_rows
                    if row["model"] == model and row["feature_set"] == "causal_top" and int(row["k"]) == 200
                ]
            )
            or 0.0
        )
        random.append(
            _mean(
                [
                    row.get("activation_gate_decoder_weighted_mean")
                    for row in summary_rows
                    if row["model"] == model and row["feature_set"] == "causal_matched_random" and int(row["k"]) == 200
                ]
            )
            or 0.0
        )
        top_active.append(
            _mean(
                [
                    row.get("activation_gate_decoder_weighted_mean")
                    for row in summary_rows
                    if row["model"] == model and row["feature_set"] == "top_active_noncausal" and int(row["k"]) == 200
                ]
            )
            or 0.0
        )
    if models:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(models))
        width = 0.25
        ax.bar(x - width, causal, width, label="causal top-200")
        ax.bar(x, random, width, label="matched random")
        ax.bar(x + width, top_active, width, label="top-active noncausal")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("Activation gate, decoder-weighted")
        ax.set_title("Exp42 feature activation gate")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "feature_gating_by_family.png", dpi=180)
        plt.close(fig)

    curve_rows = [r for r in diff_rows if r["metric"] == "feature_causal_gate" and r["control"] == "causal_matched_random"]
    if curve_rows:
        ks = sorted({int(r["k"]) for r in curve_rows})
        vals = [_mean([r["causal_minus_control"] for r in curve_rows if int(r["k"]) == k]) or 0.0 for k in ks]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(ks, vals, marker="o")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("Ablated causal features k")
        ax.set_ylabel("Causal gate minus matched-random")
        ax.set_title("Exp42 ablation-gate saturation")
        fig.tight_layout()
        fig.savefig(out_dir / "feature_ablation_saturation.png", dpi=180)
        plt.close(fig)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    rows = _load_records(args.run_root, args.models)
    activation_events = _activation_event_rows(rows)
    ablation_events = _ablation_event_rows(rows)
    joined = _join_event_rows(activation_events, ablation_events)

    activation_summary = _summarize_events(joined, "activation_gate_decoder_weighted")
    activation_margin_summary = _summarize_events(joined, "activation_gate_decoder_margin_weighted")
    causal_summary = _summarize_events(joined, "feature_causal_gate")
    interaction_drop_summary = _summarize_events(joined, "interaction_drop")

    summary_rows = []
    keys = sorted(set(activation_summary) | set(activation_margin_summary) | set(causal_summary) | set(interaction_drop_summary))
    for key in keys:
        row = {
            **activation_summary.get(key, {}),
            **{
                k: v
                for k, v in activation_margin_summary.get(key, {}).items()
                if k.endswith("_mean") or k.endswith("_pos3_mean")
            },
            **{
                k: v
                for k, v in causal_summary.get(key, {}).items()
                if k.endswith("_mean") or k.endswith("_pos3_mean")
            },
            **{
                k: v
                for k, v in interaction_drop_summary.get(key, {}).items()
                if k.endswith("_mean") or k.endswith("_pos3_mean")
            },
        }
        if "model" not in row:
            model, feature_set, k, seed = key
            row.update({"model": model, "feature_set": feature_set, "k": k, "control_seed": None if seed == "none" else int(seed)})
        summary_rows.append(row)

    diff_rows: list[dict[str, Any]] = []
    for metric in ("activation_gate_decoder_weighted", "activation_gate_decoder_margin_weighted", "feature_causal_gate", "interaction_drop"):
        for control in ("causal_matched_random", "top_active_noncausal"):
            diff_rows.extend(_causal_minus_control(joined, metric, control))

    family_estimates = {}
    for metric in ("activation_gate_decoder_weighted", "activation_gate_decoder_margin_weighted", "feature_causal_gate", "interaction_drop"):
        for control in ("causal_matched_random", "top_active_noncausal"):
            for k in args.k_list:
                subset = [
                    row
                    for row in diff_rows
                    if row["metric"] == metric and row["control"] == control and int(row["k"]) == int(k)
                ]
                family_estimates[f"{metric}__causal_minus_{control}__k{k}"] = _bootstrap_family_mean(
                    subset,
                    metric="causal_minus_control",
                    models=args.models,
                    n_boot=args.n_boot,
                    seed=args.seed + int(k),
                )

    saturation = {
        "interaction_drop": _saturation(joined, "interaction_drop"),
        "feature_causal_gate": _saturation(joined, "feature_causal_gate"),
        "activation_gate_decoder_weighted": _saturation(joined, "activation_gate_decoder_weighted"),
    }
    coverage = {
        "max_upt_lit_reconstruction_error_rel": max(
            [v for v in (_finite(row.get("upt_lit_reconstruction_error_rel")) for row in joined) if v is not None],
            default=None,
        ),
        "mean_upt_lit_reconstruction_error_rel": _mean([row.get("upt_lit_reconstruction_error_rel") for row in joined]),
        "mean_uit_lit_reconstruction_error_rel": _mean([row.get("uit_lit_reconstruction_error_rel") for row in joined]),
        "mean_upt_lit_feature_l0": _mean([row.get("upt_lit_feature_l0") for row in joined]),
        "mean_uit_lit_feature_l0": _mean([row.get("uit_lit_feature_l0") for row in joined]),
    }

    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "feature_gating_event_rows.csv", joined)
    _write_csv(out_dir / "feature_gating_effects.csv", summary_rows)
    _write_csv(out_dir / "feature_gating_control_differences.csv", diff_rows)
    _write_csv(
        out_dir / "feature_gating_saturation.csv",
        [row for rows_ in saturation.values() for row in rows_],
    )
    _plot(args.run_root, summary_rows, diff_rows)
    report = _report(family_estimates, coverage, len(rows), len(joined), args.models)
    (out_dir / "feature_gating_report.md").write_text(report, encoding="utf-8")
    summary = {
        "experiment": "exp42_terminal_feature_upstream_conditioning",
        "run_root": str(args.run_root),
        "models": args.models,
        "n_raw_records": len(rows),
        "n_event_feature_rows": len(joined),
        "n_ablation_rows": len(ablation_events),
        "family_estimates": family_estimates,
        "saturation": saturation,
        "coverage": coverage,
    }
    (out_dir / "feature_gating_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _fmt_est(payload: dict[str, Any] | None) -> str:
    if not payload or payload.get("estimate") is None:
        return "n/a"
    est = float(payload["estimate"])
    lo = payload.get("ci_low")
    hi = payload.get("ci_high")
    if lo is None or hi is None:
        return f"{est:+.4f}"
    return f"{est:+.4f} [{float(lo):+.4f}, {float(hi):+.4f}]"


def _report(
    family_estimates: dict[str, Any],
    coverage: dict[str, Any],
    n_raw: int,
    n_joined: int,
    models: list[str],
) -> str:
    act = family_estimates.get("activation_gate_decoder_weighted__causal_minus_causal_matched_random__k200")
    causal = family_estimates.get("feature_causal_gate__causal_minus_causal_matched_random__k200")
    top_active = family_estimates.get("feature_causal_gate__causal_minus_top_active_noncausal__k200")
    return f"""# Exp42 Feature Upstream-Conditioning Report

Models: `{', '.join(models)}`

Raw records: `{n_raw}`

Joined event-feature rows: `{n_joined}`

## Primary k=200 Tests

- Activation gate, causal minus matched-random: `{_fmt_est(act)}`
- Causal gate, causal minus matched-random: `{_fmt_est(causal)}`
- Causal gate, causal minus top-active noncausal: `{_fmt_est(top_active)}`

## Sparse-Dictionary Coverage

- Mean `U_PT,L_IT` reconstruction error: `{coverage.get('mean_upt_lit_reconstruction_error_rel')}`
- Mean `U_IT,L_IT` reconstruction error: `{coverage.get('mean_uit_lit_reconstruction_error_rel')}`
- Mean `U_PT,L_IT` feature L0: `{coverage.get('mean_upt_lit_feature_l0')}`
- Mean `U_IT,L_IT` feature L0: `{coverage.get('mean_uit_lit_feature_l0')}`

## Interpretation Rule

Use this in the paper spine only if both the activation gate and causal gate beat matched-random controls, and top-active noncausal features do not reproduce the same causal gate.
"""


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS))
    parser.add_argument("--k-list", nargs="+", type=int, default=list(PRIMARY_KS))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)


def main(args: argparse.Namespace) -> None:
    analyze(args)
