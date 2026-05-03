"""Analyze Exp44 middle-to-terminal feature handoff records."""

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


PRIMARY_MODELS = ("llama31_8b", "mistral_7b", "qwen3_4b")
ALL_MODELS = ("llama31_8b", "mistral_7b", "qwen3_4b", "gemma3_4b")
HANDOFF_METRICS = (
    "expected_positive_margin_effect",
    "terminal_mediated_effect",
    "terminal_mediated_fraction",
    "activation_rescue_decoder_margin_weighted_sum",
    "activation_rescue_sum_activation",
    "base_reconstruction_error_rel",
    "perturbed_reconstruction_error_rel",
    "base_selected_union_mass_fraction",
    "perturbed_selected_union_mass_fraction",
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


def _mean(values: list[Any]) -> float | None:
    vals = [v for v in (_finite(x) for x in values) if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def _stable_int(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 1_000_000_000


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
    cluster_means = np.asarray([float(np.mean(by_cluster[key])) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(0, len(cluster_means), size=(int(n_boot), len(cluster_means)))
    samples = cluster_means[draw].mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _family_balanced_bootstrap(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    n_boot: int,
    seed: int,
    models: tuple[str, ...] | list[str] | None = None,
    cluster_key: str = "prompt_id",
) -> dict[str, Any]:
    allowed = set(models or [])
    by_model_cluster: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        model = str(row.get("model"))
        if allowed and model not in allowed:
            continue
        value = _finite(row.get(metric))
        if value is not None:
            by_model_cluster[model][str(row.get(cluster_key))].append(value)

    cluster_means_by_model: dict[str, dict[str, float]] = {}
    family_means: dict[str, float] = {}
    for model, clusters in sorted(by_model_cluster.items()):
        cluster_means = {key: float(np.mean(vals)) for key, vals in clusters.items() if vals}
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
    samples: np.ndarray | None = None
    if n_boot > 0 and all(len(clusters) >= 2 for clusters in cluster_means_by_model.values()):
        rng = np.random.default_rng(int(seed))
        family_samples = []
        for model in sorted(cluster_means_by_model):
            values = np.asarray([cluster_means_by_model[model][key] for key in sorted(cluster_means_by_model[model])])
            draw = rng.integers(0, len(values), size=(int(n_boot), len(values)))
            family_samples.append(values[draw].mean(axis=1))
        samples = np.vstack(family_samples).mean(axis=0)
    lo, hi = (
        (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        if samples is not None and len(samples) > 0
        else (None, None)
    )
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
            payload[f"{metric}_mean"] = _mean([row.get(metric) for row in group])
            lo, hi = _cluster_bootstrap_ci(
                group,
                metric,
                n_boot=n_boot,
                seed=seed + _stable_int(metric, key),
            )
            payload[f"{metric}_ci_low"] = lo
            payload[f"{metric}_ci_high"] = hi
            payload[f"{metric}_pos3_mean"] = _mean([row.get(metric) for row in group if row.get("position_ge_3")])
        out.append(payload)
    return out


def _event_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model"),
        row.get("prompt_id"),
        row.get("event_kind"),
        row.get("direction"),
        row.get("window"),
        int(row.get("k", 0)),
    )


def _control_differences(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    handoff = [row for row in rows if row.get("record_type") == "handoff"]
    causal: dict[tuple[Any, ...], dict[str, Any]] = {}
    controls: dict[tuple[tuple[Any, ...], str], list[dict[str, Any]]] = defaultdict(list)
    for row in handoff:
        key = _event_key(row)
        feature_set = str(row.get("feature_set"))
        control_mode = str(row.get("control_mode"))
        if feature_set == "causal_top" and control_mode == "feature_ablate":
            causal[key] = row
        elif feature_set in {"causal_matched_random", "top_active_noncausal", "causal_same_delta_random"}:
            controls[(key, feature_set)].append(row)

    out: list[dict[str, Any]] = []
    for key, causal_row in causal.items():
        for control_name in ("causal_matched_random", "top_active_noncausal", "causal_same_delta_random"):
            control_rows = controls.get((key, control_name), [])
            if not control_rows:
                continue
            payload = {
                "model": causal_row.get("model"),
                "prompt_id": causal_row.get("prompt_id"),
                "event_kind": causal_row.get("event_kind"),
                "position_ge_3": bool(causal_row.get("position_ge_3")),
                "direction": causal_row.get("direction"),
                "window": causal_row.get("window"),
                "k": int(causal_row.get("k", 0)),
                "control": control_name,
                "n_control_rows": len(control_rows),
            }
            for metric in (
                "expected_positive_margin_effect",
                "terminal_mediated_effect",
                "terminal_mediated_fraction",
                "activation_rescue_decoder_margin_weighted_sum",
                "activation_rescue_sum_activation",
            ):
                causal_val = _finite(causal_row.get(metric))
                control_val = _mean([row.get(metric) for row in control_rows])
                payload[f"{metric}_causal"] = causal_val
                payload[f"{metric}_control"] = control_val
                payload[f"{metric}_causal_minus_control"] = (
                    causal_val - control_val if causal_val is not None and control_val is not None else None
                )
            out.append(payload)
    return out


def _family_balanced_rows(
    *,
    handoff_rows: list[dict[str, Any]],
    control_diff_rows: list[dict[str, Any]],
    primary_models: list[str],
    primary_k: int,
    n_boot: int,
    seed: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    causal_rows = [
        row
        for row in handoff_rows
        if row.get("feature_set") == "causal_top"
        and row.get("control_mode") == "feature_ablate"
        and int(row.get("k", -1)) == int(primary_k)
    ]
    for direction in sorted({str(row.get("direction")) for row in causal_rows}):
        for window in sorted({str(row.get("window")) for row in causal_rows if row.get("direction") == direction}):
            subset = [row for row in causal_rows if row.get("direction") == direction and row.get("window") == window]
            for metric in (
                "expected_positive_margin_effect",
                "terminal_mediated_effect",
                "terminal_mediated_fraction",
                "activation_rescue_decoder_margin_weighted_sum",
            ):
                estimate = _family_balanced_bootstrap(
                    subset,
                    metric,
                    n_boot=n_boot,
                    seed=seed + _stable_int("causal", direction, window, metric),
                    models=primary_models,
                )
                out.append(
                    {
                        "effect": "causal_top",
                        "direction": direction,
                        "window": window,
                        "metric": metric,
                        "k": int(primary_k),
                        **estimate,
                    }
                )
    for control in sorted({str(row.get("control")) for row in control_diff_rows}):
        control_rows = [row for row in control_diff_rows if row.get("control") == control and int(row.get("k", -1)) == int(primary_k)]
        for direction in sorted({str(row.get("direction")) for row in control_rows}):
            for window in sorted({str(row.get("window")) for row in control_rows if row.get("direction") == direction}):
                subset = [row for row in control_rows if row.get("direction") == direction and row.get("window") == window]
                for metric in (
                    "terminal_mediated_effect_causal_minus_control",
                    "activation_rescue_decoder_margin_weighted_sum_causal_minus_control",
                ):
                    estimate = _family_balanced_bootstrap(
                        subset,
                        metric,
                        n_boot=n_boot,
                        seed=seed + _stable_int(control, direction, window, metric),
                        models=primary_models,
                    )
                    out.append(
                        {
                            "effect": f"causal_minus_{control}",
                            "direction": direction,
                            "window": window,
                            "metric": metric,
                            "k": int(primary_k),
                            **estimate,
                        }
                    )
    return out


def _event_permutation_null(
    rows: list[dict[str, Any]],
    *,
    primary_k: int,
    n_perm: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Permutation null for event-aligned activation/margin association.

    This is the paper-facing replacement for a generic wrong-prompt hidden-state
    donor control, which is usually invalid because prefix shapes and token
    positions do not align.  We keep model/window/direction fixed and permute
    activation-rescue values across prompt events.
    """

    causal_rows = [
        row
        for row in rows
        if row.get("record_type") == "handoff"
        and row.get("feature_set") == "causal_top"
        and row.get("control_mode") == "feature_ablate"
        and int(row.get("k", -1)) == int(primary_k)
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in causal_rows:
        grouped[(row.get("model"), row.get("direction"), row.get("window"))].append(row)

    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        xs = np.asarray(
            [
                _finite(row.get("activation_rescue_decoder_margin_weighted_sum"))
                for row in group
            ],
            dtype=object,
        )
        ys = np.asarray([_finite(row.get("expected_positive_margin_effect")) for row in group], dtype=object)
        pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False) if x is not None and y is not None]
        if len(pairs) < 4:
            continue
        x = np.asarray([p[0] for p in pairs], dtype=np.float64)
        y = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            obs = None
            null_lo = None
            null_hi = None
            p_two = None
        else:
            obs = float(np.corrcoef(x, y)[0, 1])
            samples = []
            for _ in range(int(n_perm)):
                xp = rng.permutation(x)
                samples.append(float(np.corrcoef(xp, y)[0, 1]))
            arr = np.asarray(samples, dtype=np.float64)
            null_lo = float(np.percentile(arr, 0.5))
            null_hi = float(np.percentile(arr, 99.5))
            p_two = float((np.sum(np.abs(arr) >= abs(obs)) + 1) / (len(arr) + 1))
        out.append(
            {
                "model": key[0],
                "direction": key[1],
                "window": key[2],
                "k": int(primary_k),
                "n": len(pairs),
                "observed_pearson": obs,
                "null_p005": null_lo,
                "null_p995": null_hi,
                "two_sided_p": p_two,
            }
        )
    return out


def _stage0_audit(run_root: Path, out_dir: Path) -> None:
    exp43_root = Path(
        "results/exp43_feature_rescue_handoff/exp43_full_h100x8_clean_20260503_182947/analysis"
    )
    lines = [
        "# Exp44 Stage 0: Exp43 Middle-Probe Audit",
        "",
        f"Checked expected Exp43 analysis root: `{exp43_root}`.",
        "",
    ]
    middle_csv = exp43_root / "middle_probe_effects.csv"
    if middle_csv.exists():
        lines.extend(
            [
                "Exp43 contains `middle_probe_effects.csv`, which is useful as a planning/debugging trace.",
                "It does not contain the Exp44-required mediation cell:",
                "`[Y(weak+IT_window)-Y(weak)] - [Y(weak+IT_window-ablate(F_k))-Y(weak-ablate(F_k))]`.",
                "So Exp44 inference is still necessary for the paper-facing handoff claim.",
            ]
        )
    else:
        lines.append("No usable Exp43 middle-probe table was found locally; Exp44 runs the required cells directly.")
    (out_dir / "stage0_exp43_middle_probe_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Exp44 Middle-To-Terminal Feature Handoff Report",
        "",
        f"Primary models: {', '.join(summary['primary_models'])}. Primary k={summary['primary_k']}.",
        "",
        "Design note: the original wrong-prompt hidden-state donor control was not used as a primary control,",
        "because full-prefix hidden-state patching requires shape/token-position alignment. The run instead",
        "uses matched-random, top-active noncausal, and same-delta-random feature controls plus an early-window",
        "source control.",
        "",
        "## Primary Family-Balanced Estimates",
        "",
        "| Effect | Direction | Window | Metric | Estimate | 95% CI | Families | Prompt clusters |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["primary_family_balanced_rows"]:
        est = row.get("estimate")
        lo = row.get("ci_low")
        hi = row.get("ci_high")
        ci = "" if lo is None or hi is None else f"[{lo:.4g}, {hi:.4g}]"
        lines.append(
            "| {effect} | {direction} | {window} | {metric} | {est:.4g} | {ci} | {nf} | {np} |".format(
                effect=row.get("effect"),
                direction=row.get("direction"),
                window=row.get("window"),
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
            "## Event-Permutation Alignment Null",
            "",
            "| Model | Direction | Window | Observed r | Null 99% interval | p | n |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary.get("event_permutation_null_rows", []):
        lo = row.get("null_p005")
        hi = row.get("null_p995")
        interval = "" if lo is None or hi is None else f"[{lo:.3g}, {hi:.3g}]"
        obs = row.get("observed_pearson")
        pval = row.get("two_sided_p")
        lines.append(
            "| {model} | {direction} | {window} | {obs:.3g} | {interval} | {p:.3g} | {n} |".format(
                model=row.get("model"),
                direction=row.get("direction"),
                window=row.get("window"),
                obs=obs or 0.0,
                interval=interval,
                p=pval or 0.0,
                n=row.get("n"),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _plot_primary(fb_rows: list[dict[str, Any]], out_dir: Path, *, primary_k: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    rows = [
        row
        for row in fb_rows
        if row.get("effect") == "causal_top"
        and row.get("metric") in {"expected_positive_margin_effect", "terminal_mediated_effect"}
        and int(row.get("k", -1)) == int(primary_k)
    ]
    if not rows:
        return
    windows = ["early", "mid", "late_preterminal", "midlate_preterminal", "terminal_entry"]
    directions = ["rescue", "degrade"]
    metrics = ["expected_positive_margin_effect", "terminal_mediated_effect"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, metric in zip(axes, metrics, strict=False):
        width = 0.35
        for idx, direction in enumerate(directions):
            vals = []
            lows = []
            highs = []
            for window in windows:
                match = next(
                    (
                        row
                        for row in rows
                        if row.get("direction") == direction and row.get("window") == window and row.get("metric") == metric
                    ),
                    None,
                )
                est = float(match.get("estimate") or 0.0) if match else 0.0
                lo = match.get("ci_low") if match else None
                hi = match.get("ci_high") if match else None
                vals.append(est)
                lows.append(max(0.0, est - float(lo)) if lo is not None else 0.0)
                highs.append(max(0.0, float(hi) - est) if hi is not None else 0.0)
            x = np.arange(len(windows)) + (idx - 0.5) * width
            ax.bar(x, vals, width=width, label=direction)
            ax.errorbar(x, vals, yerr=[lows, highs], fmt="none", color="black", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(np.arange(len(windows)))
        ax.set_xticklabels(windows, rotation=30, ha="right")
        ax.set_title(metric)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "exp44_primary_handoff_effects.png", dpi=180)
    plt.close(fig)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS))
    parser.add_argument("--primary-models", nargs="+", default=list(PRIMARY_MODELS))
    parser.add_argument("--primary-k", type=int, default=200)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--n-perm", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)


def main(args: argparse.Namespace) -> None:
    rows = _load_records(args.run_root, list(args.models))
    handoff_rows = [row for row in rows if row.get("record_type") == "handoff"]
    baseline_rows = [row for row in rows if row.get("record_type") == "event_baseline"]
    control_diffs = _control_differences(rows)

    handoff_summary = _summarize_group(
        handoff_rows,
        group_keys=("model", "direction", "window", "feature_set", "control_mode", "k"),
        metrics=HANDOFF_METRICS,
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed),
    )
    control_summary = _summarize_group(
        control_diffs,
        group_keys=("model", "direction", "window", "control", "k"),
        metrics=(
            "terminal_mediated_effect_causal_minus_control",
            "terminal_mediated_fraction_causal_minus_control",
            "activation_rescue_decoder_margin_weighted_sum_causal_minus_control",
            "activation_rescue_sum_activation_causal_minus_control",
        ),
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed) + 10_000,
    )
    primary_family_balanced = _family_balanced_rows(
        handoff_rows=handoff_rows,
        control_diff_rows=control_diffs,
        primary_models=list(args.primary_models),
        primary_k=int(args.primary_k),
        n_boot=int(args.n_boot),
        seed=int(args.bootstrap_seed) + 20_000,
    )
    event_permutation_null = _event_permutation_null(
        handoff_rows,
        primary_k=int(args.primary_k),
        n_perm=int(args.n_perm),
        seed=int(args.bootstrap_seed) + 30_000,
    )

    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "handoff_effects.csv", handoff_summary)
    _write_csv(out_dir / "handoff_control_differences_event_rows.csv", control_diffs)
    _write_csv(out_dir / "handoff_control_differences.csv", control_summary)
    _write_csv(out_dir / "primary_family_balanced_effects.csv", primary_family_balanced)
    _write_csv(out_dir / "event_permutation_null.csv", event_permutation_null)
    _stage0_audit(args.run_root, out_dir)
    _plot_primary(primary_family_balanced, out_dir, primary_k=int(args.primary_k))

    summary = {
        "experiment": "exp44_middle_terminal_feature_handoff",
        "run_root": str(args.run_root),
        "models": list(args.models),
        "primary_models": list(args.primary_models),
        "primary_k": int(args.primary_k),
        "n_records": len(rows),
        "n_baseline_records": len(baseline_rows),
        "n_handoff_records": len(handoff_rows),
        "n_control_difference_rows": len(control_diffs),
        "primary_family_balanced_rows": primary_family_balanced,
        "event_permutation_null_rows": event_permutation_null,
    }
    (out_dir / "exp44_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "exp44_report.md").write_text(_report(summary), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
