"""Analyze Exp43 feature-rescue and middle-to-terminal handoff records."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PRIMARY_MODELS = ("gemma3_4b", "llama31_8b", "mistral_7b", "qwen3_4b")
RESCUE_METRICS = (
    "rescue_gain",
    "rescue_fraction",
    "interaction_drop_from_rescue",
    "target_sum_activation",
    "current_sum_activation",
    "target_decoder_margin_weighted_sum",
    "current_decoder_margin_weighted_sum",
)
MIDDLE_METRICS = (
    "margin_drop",
    "activation_drop",
    "decoder_margin_weighted_drop",
)


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


def _stable_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000_000


def _mean(values: list[Any]) -> float | None:
    vals = [v for v in (_finite(x) for x in values) if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _cluster_values(rows: list[dict[str, Any]], metric: str, cluster_key: str = "prompt_id") -> list[float]:
    """Collapse repeated rows from the same prompt before uncertainty estimates."""

    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(metric))
        if value is None:
            continue
        by_cluster[str(row.get(cluster_key))].append(value)
    return [float(np.mean(vals)) for vals in by_cluster.values() if vals]


def _percentile_ci(samples: list[float]) -> tuple[float | None, float | None]:
    if not samples:
        return None, None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _cluster_bootstrap_ci(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
    cluster_key: str = "prompt_id",
) -> tuple[float | None, float | None]:
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _finite(row.get(metric))
        if value is not None:
            by_cluster[str(row.get(cluster_key))].append(value)
    keys = sorted(key for key, vals in by_cluster.items() if vals)
    if len(keys) < 2 or n_boot <= 0:
        return None, None
    cluster_means = {key: float(np.mean(by_cluster[key])) for key in keys}
    rng = np.random.default_rng(int(seed))
    samples: list[float] = []
    for _ in range(int(n_boot)):
        draw = rng.choice(keys, size=len(keys), replace=True)
        samples.append(float(np.mean([cluster_means[str(key)] for key in draw])))
    return _percentile_ci(samples)


def _family_balanced_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
    model_key: str = "model",
    cluster_key: str = "prompt_id",
) -> dict[str, Any]:
    """Estimate equal-family mean with prompt-cluster bootstrap inside families."""

    by_model_cluster: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        value = _finite(row.get(metric))
        if value is None:
            continue
        by_model_cluster[str(row.get(model_key))][str(row.get(cluster_key))].append(value)

    family_means: dict[str, float] = {}
    cluster_means_by_model: dict[str, dict[str, float]] = {}
    for model, clusters in sorted(by_model_cluster.items()):
        cluster_means = {
            key: float(np.mean(vals))
            for key, vals in clusters.items()
            if vals
        }
        if cluster_means:
            cluster_means_by_model[model] = cluster_means
            family_means[model] = float(np.mean(list(cluster_means.values())))

    if not family_means:
        return {
            "estimate": None,
            "ci_low": None,
            "ci_high": None,
            "n_families": 0,
            "n_prompt_clusters": 0,
            "family_means": {},
        }

    estimate = float(np.mean(list(family_means.values())))
    samples: list[float] = []
    if n_boot > 0 and all(len(clusters) >= 2 for clusters in cluster_means_by_model.values()):
        rng = np.random.default_rng(int(seed))
        models = sorted(cluster_means_by_model)
        for _ in range(int(n_boot)):
            family_draws: list[float] = []
            for model in models:
                clusters = cluster_means_by_model[model]
                keys = sorted(clusters)
                draw = rng.choice(keys, size=len(keys), replace=True)
                family_draws.append(float(np.mean([clusters[str(key)] for key in draw])))
            samples.append(float(np.mean(family_draws)))
    lo, hi = _percentile_ci(samples)
    return {
        "estimate": estimate,
        "ci_low": lo,
        "ci_high": hi,
        "n_families": len(family_means),
        "n_prompt_clusters": int(sum(len(clusters) for clusters in cluster_means_by_model.values())),
        "family_means": family_means,
    }


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


def _event_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("model")), str(row.get("prompt_id")), str(row.get("event_kind"))


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


def _summarize_group(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    metrics: tuple[str, ...],
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        payload = {field: value for field, value in zip(group_keys, key, strict=False)}
        payload["n"] = len(group)
        payload["n_prompt_clusters"] = len({str(row.get("prompt_id")) for row in group})
        payload["n_pos3"] = sum(1 for row in group if row.get("position_ge_3"))
        for metric in metrics:
            vals = [row.get(metric) for row in group]
            mean = _mean(vals)
            lo, hi = _cluster_bootstrap_ci(
                group,
                metric,
                n_boot=n_boot,
                seed=seed + _stable_int(metric, key),
            )
            payload[f"{metric}_mean"] = mean
            payload[f"{metric}_ci_low"] = lo
            payload[f"{metric}_ci_high"] = hi
            payload[f"{metric}_pos3_mean"] = _mean([row.get(metric) for row in group if row.get("position_ge_3")])
        out.append(payload)
    return out


def _rescue_control_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rescue = [row for row in rows if row.get("record_type") == "rescue"]
    causal: dict[tuple[Any, ...], dict[str, Any]] = {}
    controls: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rescue:
        base = (*_event_key(row), int(row.get("k", 0)), float(row.get("alpha", 0.0)))
        if row.get("feature_set") == "causal_top" and row.get("control_mode") == "feature_delta":
            causal[base] = row
        elif row.get("feature_set") in {"causal_matched_random", "causal_same_delta_random"}:
            controls[(*base, str(row.get("feature_set")), str(row.get("control_mode")))].append(row)

    out: list[dict[str, Any]] = []
    for base, causal_row in causal.items():
        for control_name in ("causal_matched_random", "causal_same_delta_random"):
            control_groups = [
                group
                for key, group in controls.items()
                if key[:5] == base and key[5] == control_name
            ]
            if not control_groups:
                continue
            flat_controls = [row for group in control_groups for row in group]
            payload = {
                "model": causal_row.get("model"),
                "prompt_id": causal_row.get("prompt_id"),
                "event_kind": causal_row.get("event_kind"),
                "position_ge_3": bool(causal_row.get("position_ge_3")),
                "k": int(causal_row.get("k", 0)),
                "alpha": float(causal_row.get("alpha", 0.0)),
                "control": control_name,
                "n_control_rows": len(flat_controls),
            }
            for metric in ("rescue_gain", "rescue_fraction", "interaction_drop_from_rescue"):
                payload[f"{metric}_causal"] = _finite(causal_row.get(metric))
                payload[f"{metric}_control"] = _mean([row.get(metric) for row in flat_controls])
                if payload[f"{metric}_causal"] is not None and payload[f"{metric}_control"] is not None:
                    payload[f"{metric}_causal_minus_control"] = (
                        float(payload[f"{metric}_causal"]) - float(payload[f"{metric}_control"])
                    )
                else:
                    payload[f"{metric}_causal_minus_control"] = None
            out.append(payload)
    return out


def _overall_summary(
    rescue_summary: list[dict[str, Any]],
    control_summary: list[dict[str, Any]],
    middle_summary: list[dict[str, Any]],
    primary_family_balanced: list[dict[str, Any]],
    alpha0_sanity: list[dict[str, Any]],
    *,
    primary_k: int,
    primary_alpha: float,
) -> dict[str, Any]:
    def find(rows: list[dict[str, Any]], **query: Any) -> list[dict[str, Any]]:
        out = []
        for row in rows:
            ok = True
            for key, value in query.items():
                if key not in row:
                    ok = False
                    break
                if isinstance(value, float):
                    ok = abs(float(row[key]) - value) <= 1e-9
                else:
                    ok = row[key] == value
                if not ok:
                    break
            if ok:
                out.append(row)
        return out

    primary_causal = find(
        rescue_summary,
        feature_set="causal_top",
        control_mode="feature_delta",
        k=primary_k,
        alpha=primary_alpha,
    )
    primary_random = find(
        control_summary,
        control="causal_matched_random",
        k=primary_k,
        alpha=primary_alpha,
    )
    primary_same_delta = find(
        control_summary,
        control="causal_same_delta_random",
        k=primary_k,
        alpha=primary_alpha,
    )
    return {
        "experiment": "exp43_feature_rescue_handoff",
        "primary_k": primary_k,
        "primary_alpha": primary_alpha,
        "primary_causal_rows": primary_causal,
        "primary_causal_minus_matched_random_rows": primary_random,
        "primary_causal_minus_same_delta_random_rows": primary_same_delta,
        "primary_family_balanced_rows": primary_family_balanced,
        "alpha0_sanity_rows": alpha0_sanity,
        "middle_probe_rows": middle_summary,
    }


def _primary_family_balanced_rows(
    *,
    rescue_rows: list[dict[str, Any]],
    control_diffs: list[dict[str, Any]],
    primary_k: int,
    primary_alpha: float,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    specs: list[tuple[str, list[dict[str, Any]], tuple[str, ...]]] = []
    causal_rows = [
        row
        for row in rescue_rows
        if row.get("feature_set") == "causal_top"
        and row.get("control_mode") == "feature_delta"
        and int(row.get("k", -1)) == int(primary_k)
        and abs(float(row.get("alpha", -999.0)) - float(primary_alpha)) <= 1e-9
    ]
    specs.append(("causal_top", causal_rows, ("rescue_gain", "rescue_fraction", "interaction_drop_from_rescue")))
    for control in ("causal_matched_random", "causal_same_delta_random"):
        rows = [
            row
            for row in control_diffs
            if row.get("control") == control
            and int(row.get("k", -1)) == int(primary_k)
            and abs(float(row.get("alpha", -999.0)) - float(primary_alpha)) <= 1e-9
        ]
        specs.append(
            (
                f"causal_minus_{control}",
                rows,
                (
                    "rescue_gain_causal_minus_control",
                    "rescue_fraction_causal_minus_control",
                    "interaction_drop_from_rescue_causal_minus_control",
                ),
            )
        )

    out: list[dict[str, Any]] = []
    for effect, rows, metrics in specs:
        for metric in metrics:
            estimate = _family_balanced_bootstrap(
                rows,
                metric,
                n_boot=n_boot,
                seed=seed + _stable_int(effect, metric),
            )
            out.append(
                {
                    "effect": effect,
                    "metric": metric,
                    **estimate,
                }
            )
    return out


def _alpha0_sanity_rows(rescue_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rescue_rows:
        if abs(float(row.get("alpha", -999.0))) <= 1e-12:
            grouped[
                (
                    row.get("model"),
                    row.get("feature_set"),
                    row.get("control_mode"),
                    row.get("k"),
                    row.get("control_seed"),
                )
            ].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        vals = [abs(v) for v in (_finite(row.get("rescue_gain")) for row in group) if v is not None]
        out.append(
            {
                "model": key[0],
                "feature_set": key[1],
                "control_mode": key[2],
                "k": key[3],
                "control_seed": key[4],
                "n": len(group),
                "max_abs_rescue_gain": max(vals) if vals else None,
            }
        )
    return out


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Exp43 Feature Rescue And Handoff Report",
        "",
        f"Primary setting: k={summary['primary_k']}, alpha={summary['primary_alpha']}.",
        "",
        "## Feature Rescue",
        "",
        "| Model | Rescue gain | Rescue fraction | Interaction drop | n |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["primary_causal_rows"]:
        lines.append(
            "| {model} | {gain:.4g} | {frac:.4g} | {drop:.4g} | {n} |".format(
                model=row.get("model"),
                gain=row.get("rescue_gain_mean") or 0.0,
                frac=row.get("rescue_fraction_mean") or 0.0,
                drop=row.get("interaction_drop_from_rescue_mean") or 0.0,
                n=row.get("n"),
            )
        )
    lines.extend(
        [
            "",
            "## Causal Minus Matched-Random Rescue",
            "",
            "| Model | Rescue-gain diff | Rescue-fraction diff | Interaction-drop diff | n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["primary_causal_minus_matched_random_rows"]:
        lines.append(
            "| {model} | {gain:.4g} | {frac:.4g} | {drop:.4g} | {n} |".format(
                model=row.get("model"),
                gain=row.get("rescue_gain_causal_minus_control_mean") or 0.0,
                frac=row.get("rescue_fraction_causal_minus_control_mean") or 0.0,
                drop=row.get("interaction_drop_from_rescue_causal_minus_control_mean") or 0.0,
                n=row.get("n"),
            )
        )
    lines.extend(
        [
            "",
            "## Causal Minus Same-Delta-Random Rescue",
            "",
            "| Model | Rescue-gain diff | Rescue-fraction diff | Interaction-drop diff | n |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["primary_causal_minus_same_delta_random_rows"]:
        lines.append(
            "| {model} | {gain:.4g} | {frac:.4g} | {drop:.4g} | {n} |".format(
                model=row.get("model"),
                gain=row.get("rescue_gain_causal_minus_control_mean") or 0.0,
                frac=row.get("rescue_fraction_causal_minus_control_mean") or 0.0,
                drop=row.get("interaction_drop_from_rescue_causal_minus_control_mean") or 0.0,
                n=row.get("n"),
            )
        )
    lines.extend(
        [
            "",
            "## Family-Balanced Primary Estimates",
            "",
            "| Effect | Metric | Estimate | 95% CI | Families | Prompt clusters |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["primary_family_balanced_rows"]:
        est = row.get("estimate")
        lo = row.get("ci_low")
        hi = row.get("ci_high")
        ci = "" if lo is None or hi is None else f"[{lo:.4g}, {hi:.4g}]"
        lines.append(
            "| {effect} | {metric} | {est:.4g} | {ci} | {nf} | {np} |".format(
                effect=row.get("effect"),
                metric=row.get("metric"),
                est=est or 0.0,
                ci=ci,
                nf=row.get("n_families"),
                np=row.get("n_prompt_clusters"),
            )
        )
    lines.extend(
        [
            "",
            "## Middle-To-Terminal Probe",
            "",
            "| Model | Window | Terminal activation drop | Decoder-margin drop | Margin drop | n |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["middle_probe_rows"]:
        lines.append(
            "| {model} | {window} | {act:.4g} | {dec:.4g} | {margin:.4g} | {n} |".format(
                model=row.get("model"),
                window=row.get("window"),
                act=row.get("activation_drop_mean") or 0.0,
                dec=row.get("decoder_margin_weighted_drop_mean") or 0.0,
                margin=row.get("margin_drop_mean") or 0.0,
                n=row.get("n"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS))
    parser.add_argument("--primary-k", type=int, default=200)
    parser.add_argument("--primary-alpha", type=float, default=1.0)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)


def main(args: argparse.Namespace) -> None:
    rows = _load_records(args.run_root, list(args.models))
    rescue_rows = [row for row in rows if row.get("record_type") == "rescue"]
    middle_rows = [row for row in rows if row.get("record_type") == "middle_probe"]
    control_diffs = _rescue_control_differences(rows)

    rescue_summary = _summarize_group(
        rescue_rows,
        group_keys=("model", "feature_set", "control_mode", "k", "alpha"),
        metrics=RESCUE_METRICS,
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed),
    )
    control_summary = _summarize_group(
        control_diffs,
        group_keys=("model", "control", "k", "alpha"),
        metrics=(
            "rescue_gain_causal_minus_control",
            "rescue_fraction_causal_minus_control",
            "interaction_drop_from_rescue_causal_minus_control",
        ),
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed) + 10_000,
    )
    middle_summary = _summarize_group(
        middle_rows,
        group_keys=("model", "window", "feature_set", "k"),
        metrics=MIDDLE_METRICS,
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed) + 20_000,
    )
    primary_family_balanced = _primary_family_balanced_rows(
        rescue_rows=rescue_rows,
        control_diffs=control_diffs,
        primary_k=int(args.primary_k),
        primary_alpha=float(args.primary_alpha),
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed) + 30_000,
    )
    alpha0_sanity = _alpha0_sanity_rows(rescue_rows)

    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rescue_effects.csv", rescue_summary)
    _write_csv(out_dir / "rescue_control_differences_event_rows.csv", control_diffs)
    _write_csv(out_dir / "rescue_control_differences.csv", control_summary)
    _write_csv(out_dir / "middle_probe_effects.csv", middle_summary)
    _write_csv(out_dir / "primary_family_balanced_effects.csv", primary_family_balanced)
    _write_csv(out_dir / "alpha0_sanity.csv", alpha0_sanity)

    summary = _overall_summary(
        rescue_summary,
        control_summary,
        middle_summary,
        primary_family_balanced,
        alpha0_sanity,
        primary_k=int(args.primary_k),
        primary_alpha=float(args.primary_alpha),
    )
    (out_dir / "exp43_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "exp43_report.md").write_text(_report(summary), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
