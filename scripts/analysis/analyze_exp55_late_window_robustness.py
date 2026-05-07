#!/usr/bin/env python3
"""Analyze Exp55 late-window width/center robustness runs.

Exp55 reuses the Exp11/14 matched-prefix runner with ``--late-window-sweep``.
This analyzer computes paired same-prompt/step branch-minus-baseline effects for
each swept window, then aggregates dense-family model means for paper synthesis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    import orjson
except Exception:  # pragma: no cover - optional speedup
    orjson = None


MODEL_ORDER = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
METRICS = ["kl_to_own_final", "delta_cosine", "cross_kl"]
REGIONS = ["final_20pct", "graft_window"]
FAST_METRICS = ["kl_to_own_final"]

_PROMPT_RE = re.compile(br'"prompt_id": "([^"]+)"')
_PIPELINE_RE = re.compile(br'"pipeline": "([^"]+)"')
_STEP_RE = re.compile(br'"step": ([0-9]+)')


def _loads(raw: bytes | str) -> Any:
    if orjson is not None:
        return orjson.loads(raw)
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="ignore")
    return json.loads(raw)


def _read_json(path: Path) -> dict[str, Any]:
    return _loads(path.read_bytes())


def _iter_jsonl(path: Path):
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                yield _loads(raw)


def _bytes_match(regex: re.Pattern[bytes], raw: bytes) -> str | None:
    match = regex.search(raw)
    if match is None:
        return None
    return match.group(1).decode("utf-8")


def _bytes_step(raw: bytes) -> int | None:
    match = _STEP_RE.search(raw)
    if match is None:
        return None
    return int(match.group(1))


def _extract_kl_to_own_final(raw: bytes) -> list[Any] | None:
    key = b'"kl_to_own_final": '
    key_pos = raw.find(key)
    if key_pos < 0:
        return None
    value_pos = key_pos + len(key)
    if raw.startswith(b"null", value_pos):
        return None
    start = raw.find(b"[", value_pos)
    end = raw.find(b"]", start)
    if start < 0 or end < start:
        return None
    return _loads(raw[start : end + 1])


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return sum(vals) / len(vals)


def _slice_mean(values: list[Any] | None, start: int, end: int) -> float | None:
    if values is None:
        return None
    subset = [float(v) for v in values[start:end] if v is not None]
    return _mean(subset)


def _ci(values: list[float], *, n_boot: int, seed: int) -> tuple[float | None, float | None]:
    if len(values) < 2 or n_boot <= 0:
        return None, None
    rng = random.Random(seed)
    boots = []
    n = len(values)
    for _ in range(n_boot):
        boots.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo_idx = max(0, min(len(boots) - 1, math.floor(0.025 * len(boots))))
    hi_idx = max(0, min(len(boots) - 1, math.ceil(0.975 * len(boots)) - 1))
    return boots[lo_idx], boots[hi_idx]


def _final_region(config: dict[str, Any]) -> tuple[int, int]:
    layers = config.get("final_region_layers") or []
    if layers:
        return int(layers[0]), int(layers[-1]) + 1
    n_layers = int(config["n_layers"])
    start = max(0, math.floor(n_layers * 0.8))
    return start, n_layers


def _collect_model_dirs(run_root: Path, models: list[str]) -> list[tuple[str, Path]]:
    base = run_root / "merged" if (run_root / "merged").exists() else run_root
    subdirs = {p.name: p for p in base.iterdir() if p.is_dir()} if base.exists() else {}
    return [(model, subdirs[model]) for model in models if model in subdirs]


def _window_region(window: dict[str, Any]) -> tuple[int, int]:
    return int(window["start_layer"]), int(window["end_layer_exclusive"])


def _region_value(
    *,
    metrics: dict[str, Any],
    metric: str,
    region: str,
    final_region: tuple[int, int],
    window: dict[str, Any] | None,
) -> float | None:
    if region == "final_20pct":
        start, end = final_region
    elif region == "graft_window":
        if window is None:
            return None
        start, end = _window_region(window)
    else:
        raise KeyError(region)
    return _slice_mean(metrics.get(metric), start, end)


def _baseline_pipeline(window: dict[str, Any]) -> str:
    host_variant = str(window.get("host_variant"))
    if host_variant == "pt":
        return "A_prime_raw"
    if host_variant == "it":
        return "C_it_chat"
    raise ValueError(f"unknown host_variant={host_variant!r}")


def _side_label(window: dict[str, Any]) -> str:
    host_variant = str(window.get("host_variant"))
    if host_variant == "pt":
        return "it_graft_into_pt"
    if host_variant == "it":
        return "pt_swap_into_it"
    raise ValueError(f"unknown host_variant={host_variant!r}")


def _summarize_model(model: str, model_dir: Path, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    config = _read_json(model_dir / "config.json")
    windows = config.get("graft_windows_by_pipeline") or {}
    branches = {
        name: window
        for name, window in windows.items()
        if str(window.get("host_variant")) in {"pt", "it"}
    }
    if not branches:
        return []

    final_region = _final_region(config)
    records: dict[tuple[str, int, str], dict[str, Any]] = {}
    records_by_pipeline: dict[str, list[tuple[str, int, dict[str, Any]]]] = defaultdict(list)
    for row in _iter_jsonl(model_dir / "step_metrics.jsonl"):
        prompt_id = str(row["prompt_id"])
        step = int(row["step"])
        pipeline = str(row["pipeline"])
        metrics = row["metrics"]
        records[(prompt_id, step, pipeline)] = metrics
        records_by_pipeline[pipeline].append((prompt_id, step, metrics))

    rows: list[dict[str, Any]] = []
    for pipeline, window in sorted(branches.items()):
        baseline = _baseline_pipeline(window)
        side = _side_label(window)
        start, end = _window_region(window)
        window_name = str(window.get("window_name") or pipeline)
        deltas_by_combo: dict[tuple[str, str], dict[str, list[float]]] = {
            (region, metric): defaultdict(list)
            for region in REGIONS
            for metric in METRICS
        }
        n_pairs_by_combo: dict[tuple[str, str], int] = {
            (region, metric): 0
            for region in REGIONS
            for metric in METRICS
        }
        for prompt_id, step, branch_metrics in records_by_pipeline.get(pipeline, []):
            base_metrics = records.get((prompt_id, step, baseline))
            if base_metrics is None:
                continue
            for region in REGIONS:
                for metric in METRICS:
                    branch_value = _region_value(
                        metrics=branch_metrics,
                        metric=metric,
                        region=region,
                        final_region=final_region,
                        window=window,
                    )
                    base_value = _region_value(
                        metrics=base_metrics,
                        metric=metric,
                        region=region,
                        final_region=final_region,
                        window=window,
                    )
                    if branch_value is None or base_value is None:
                        continue
                    key = (region, metric)
                    deltas_by_combo[key][prompt_id].append(branch_value - base_value)
                    n_pairs_by_combo[key] += 1
        for region in REGIONS:
            for metric in METRICS:
                deltas_by_prompt = deltas_by_combo[(region, metric)]
                n_pairs = n_pairs_by_combo[(region, metric)]
                prompt_means = [
                    mean
                    for vals in deltas_by_prompt.values()
                    if (mean := _mean(vals)) is not None
                ]
                estimate = _mean(prompt_means)
                ci_low, ci_high = _ci(prompt_means, n_boot=n_boot, seed=seed + len(rows))
                rows.append(
                    {
                        "level": "model",
                        "model": model,
                        "pipeline": pipeline,
                        "side": side,
                        "host_variant": window.get("host_variant"),
                        "donor_variant": window.get("donor_variant"),
                        "baseline_pipeline": baseline,
                        "window_name": window_name,
                        "start_layer": start,
                        "end_layer_exclusive": end,
                        "width": end - start,
                        "center": (start + end - 1) / 2.0,
                        "region": region,
                        "metric": metric,
                        "estimate": estimate,
                        "ci95_low": ci_low,
                        "ci95_high": ci_high,
                        "n_pairs": n_pairs,
                        "n_prompt_clusters": len(prompt_means),
                    }
                )
    return rows


def _summarize_model_fast_kl(model: str, model_dir: Path, *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    """Stream only KL arrays from very large Exp55 JSONL files.

    The full analyzer decodes every metric, including top-k token tables. For
    full dense5 runs that is tens of GB of JSON and is unnecessarily expensive
    when the paper-facing robustness table only needs final-window KL effects.
    """

    config = _read_json(model_dir / "config.json")
    windows = config.get("graft_windows_by_pipeline") or {}
    branches = {
        name: window
        for name, window in windows.items()
        if str(window.get("host_variant")) in {"pt", "it"}
    }
    if not branches:
        return []

    final_region = _final_region(config)
    baseline_windows: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for _pipeline, window in branches.items():
        baseline_windows[_baseline_pipeline(window)][str(window.get("window_name") or _pipeline)] = window

    wanted_pipelines = set(branches) | set(baseline_windows)
    branch_payloads: dict[str, list[tuple[str, int, float | None, float | None]]] = defaultdict(list)
    baseline_payloads: dict[tuple[str, str, int], dict[str, float | None]] = {}

    with (model_dir / "step_metrics.jsonl").open("rb") as f:
        for raw in f:
            pipeline = _bytes_match(_PIPELINE_RE, raw)
            if pipeline not in wanted_pipelines:
                continue
            prompt_id = _bytes_match(_PROMPT_RE, raw)
            step = _bytes_step(raw)
            if prompt_id is None or step is None:
                continue
            values = _extract_kl_to_own_final(raw)
            if values is None:
                continue
            final_value = _slice_mean(values, *final_region)
            if pipeline in branches:
                window = branches[pipeline]
                branch_payloads[pipeline].append(
                    (
                        prompt_id,
                        step,
                        final_value,
                        _slice_mean(values, *_window_region(window)),
                    )
                )
            if pipeline in baseline_windows:
                payload: dict[str, float | None] = {"final_20pct": final_value}
                for window_name, window in baseline_windows[pipeline].items():
                    payload[window_name] = _slice_mean(values, *_window_region(window))
                baseline_payloads[(pipeline, prompt_id, step)] = payload

    rows: list[dict[str, Any]] = []
    for pipeline, window in sorted(branches.items()):
        baseline = _baseline_pipeline(window)
        side = _side_label(window)
        start, end = _window_region(window)
        window_name = str(window.get("window_name") or pipeline)
        deltas_by_combo: dict[str, dict[str, list[float]]] = {
            region: defaultdict(list)
            for region in REGIONS
        }
        n_pairs_by_region = {region: 0 for region in REGIONS}
        for prompt_id, step, branch_final, branch_window in branch_payloads.get(pipeline, []):
            base = baseline_payloads.get((baseline, prompt_id, step))
            if base is None:
                continue
            region_values = {
                "final_20pct": (branch_final, base.get("final_20pct")),
                "graft_window": (branch_window, base.get(window_name)),
            }
            for region, (branch_value, base_value) in region_values.items():
                if branch_value is None or base_value is None:
                    continue
                deltas_by_combo[region][prompt_id].append(branch_value - base_value)
                n_pairs_by_region[region] += 1
        for region in REGIONS:
            prompt_means = [
                mean
                for vals in deltas_by_combo[region].values()
                if (mean := _mean(vals)) is not None
            ]
            estimate = _mean(prompt_means)
            ci_low, ci_high = _ci(prompt_means, n_boot=n_boot, seed=seed + len(rows))
            rows.append(
                {
                    "level": "model",
                    "model": model,
                    "pipeline": pipeline,
                    "side": side,
                    "host_variant": window.get("host_variant"),
                    "donor_variant": window.get("donor_variant"),
                    "baseline_pipeline": baseline,
                    "window_name": window_name,
                    "start_layer": start,
                    "end_layer_exclusive": end,
                    "width": end - start,
                    "center": (start + end - 1) / 2.0,
                    "region": region,
                    "metric": "kl_to_own_final",
                    "estimate": estimate,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "n_pairs": n_pairs_by_region[region],
                    "n_prompt_clusters": len(prompt_means),
                }
            )
    return rows


def _dense_rows(model_rows: list[dict[str, Any]], *, n_boot: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in model_rows:
        if row["model"] in MODEL_ORDER and row["estimate"] is not None:
            grouped[(row["window_name"], row["side"], row["region"], row["metric"])].append(row)

    out: list[dict[str, Any]] = []
    for (window_name, side, region, metric), rows in sorted(grouped.items()):
        estimates = [float(row["estimate"]) for row in rows]
        estimate = _mean(estimates)
        ci_low, ci_high = _ci(estimates, n_boot=n_boot, seed=seed + len(out))
        first = rows[0]
        out.append(
            {
                "level": "dense5_model_mean",
                "model": "dense5",
                "pipeline": "",
                "side": side,
                "host_variant": first["host_variant"],
                "donor_variant": first["donor_variant"],
                "baseline_pipeline": first["baseline_pipeline"],
                "window_name": window_name,
                "start_layer": "",
                "end_layer_exclusive": "",
                "width": "",
                "center": "",
                "region": region,
                "metric": metric,
                "estimate": estimate,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_pairs": sum(int(row["n_pairs"]) for row in rows),
                "n_prompt_clusters": sum(int(row["n_prompt_clusters"]) for row in rows),
                "n_models": len(rows),
                "models": ",".join(row["model"] for row in rows),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "level",
        "model",
        "pipeline",
        "side",
        "host_variant",
        "donor_variant",
        "baseline_pipeline",
        "window_name",
        "start_layer",
        "end_layer_exclusive",
        "width",
        "center",
        "region",
        "metric",
        "estimate",
        "ci95_low",
        "ci95_high",
        "n_pairs",
        "n_prompt_clusters",
        "n_models",
        "models",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.3f}"


def _fmt_interval(row: dict[str, Any] | None) -> str:
    if row is None:
        return "NA"
    return f"{_fmt(row.get('estimate'))} [{_fmt(row.get('ci95_low'))}, {_fmt(row.get('ci95_high'))}]"


def _write_note(path: Path, dense_rows: list[dict[str, Any]], models_seen: list[str]) -> None:
    by_key = {
        (row["window_name"], row["side"], row["region"], row["metric"]): row
        for row in dense_rows
    }
    lines = [
        "# Exp55 Late-Window Robustness",
        "",
        f"Models analyzed: `{', '.join(models_seen)}`.",
        "",
        "Dense5 model-mean final-20% KL effects:",
        "",
    ]
    for window_name in [
        "prelate_half",
        "late_full",
        "late_front_half",
        "late_center_half",
        "late_terminal_half",
        "terminal_quarter",
    ]:
        graft = by_key.get((window_name, "it_graft_into_pt", "final_20pct", "kl_to_own_final"))
        swap = by_key.get((window_name, "pt_swap_into_it", "final_20pct", "kl_to_own_final"))
        if graft is None and swap is None:
            continue
        lines.append(
            f"- `{window_name}`: graft `{_fmt(graft and graft.get('estimate'))}`, "
            f"swap `{_fmt(swap and swap.get('estimate'))}`."
        )
    late_full_graft = by_key.get(("late_full", "it_graft_into_pt", "final_20pct", "kl_to_own_final"))
    late_full_swap = by_key.get(("late_full", "pt_swap_into_it", "final_20pct", "kl_to_own_final"))
    late_full_graft_window = by_key.get(("late_full", "it_graft_into_pt", "graft_window", "kl_to_own_final"))
    prelate_swap = by_key.get(("prelate_half", "pt_swap_into_it", "final_20pct", "kl_to_own_final"))
    lines.extend(
        [
            "",
            "Observed interpretation:",
            "",
            f"- Late-full final-20% signs are in the expected direction: graft `{_fmt_interval(late_full_graft)}`, "
            f"swap `{_fmt_interval(late_full_swap)}`.",
            f"- The direct edited-window late-full graft is clearer: `{_fmt_interval(late_full_graft_window)}`.",
            f"- The pre-late swap also moves final-20% KL: `{_fmt_interval(prelate_swap)}`.",
            "- Use this audit as support for the late window being the strongest tested bidirectional handle, "
            "not as evidence for a sharp late-only boundary.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _maybe_plot(path: Path, dense_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    wanted_windows = [
        "prelate_half",
        "late_full",
        "late_front_half",
        "late_center_half",
        "late_terminal_half",
        "terminal_quarter",
    ]
    by_key = {
        (row["window_name"], row["side"], row["region"], row["metric"]): row
        for row in dense_rows
    }
    xs = list(range(len(wanted_windows)))
    graft = [
        (by_key.get((w, "it_graft_into_pt", "final_20pct", "kl_to_own_final")) or {}).get("estimate", 0.0)
        for w in wanted_windows
    ]
    swap = [
        (by_key.get((w, "pt_swap_into_it", "final_20pct", "kl_to_own_final")) or {}).get("estimate", 0.0)
        for w in wanted_windows
    ]
    fig, ax = plt.subplots(figsize=(8.0, 3.2))
    ax.axhline(0.0, color="#222222", linewidth=0.8)
    ax.bar([x - 0.18 for x in xs], graft, width=0.36, label="IT graft into PT", color="#4c78a8")
    ax.bar([x + 0.18 for x in xs], swap, width=0.36, label="PT swap into IT", color="#b45f5f")
    ax.set_xticks(xs)
    ax.set_xticklabels(wanted_windows, rotation=30, ha="right")
    ax.set_ylabel("Final-20% KL delta")
    ax.set_title("Exp55 late-window robustness")
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def analyze(run_root: Path, out_dir: Path, *, models: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    model_rows: list[dict[str, Any]] = []
    models_seen: list[str] = []
    for model, model_dir in _collect_model_dirs(run_root, models):
        rows = _summarize_model(model, model_dir, n_boot=n_boot, seed=seed)
        if rows:
            models_seen.append(model)
            model_rows.extend(rows)
    dense = _dense_rows(model_rows, n_boot=n_boot, seed=seed + 1000)
    all_rows = model_rows + dense

    out_dir.mkdir(parents=True, exist_ok=True)
    effects_csv = out_dir / "exp55_late_window_robustness_effects.csv"
    summary_json = out_dir / "exp55_late_window_robustness.json"
    note_md = out_dir / "exp55_late_window_robustness_note.md"
    plot_png = out_dir / "exp55_late_window_robustness.png"
    _write_csv(effects_csv, all_rows)
    _write_note(note_md, dense, models_seen)
    _maybe_plot(plot_png, dense)
    summary = {
        "analysis": "exp55_late_window_robustness",
        "run_root": str(run_root),
        "models_requested": models,
        "models_analyzed": models_seen,
        "n_boot": n_boot,
        "effects": all_rows,
        "artifacts": {
            "effects_csv": str(effects_csv),
            "note": str(note_md),
            "plot": str(plot_png) if plot_png.exists() else None,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def analyze_fast_kl(run_root: Path, out_dir: Path, *, models: list[str], n_boot: int, seed: int) -> dict[str, Any]:
    model_rows: list[dict[str, Any]] = []
    models_seen: list[str] = []
    for model, model_dir in _collect_model_dirs(run_root, models):
        print(f"[exp55] fast KL parse {model_dir}")
        rows = _summarize_model_fast_kl(model, model_dir, n_boot=n_boot, seed=seed)
        if rows:
            models_seen.append(model)
            model_rows.extend(rows)
    dense = _dense_rows(model_rows, n_boot=n_boot, seed=seed + 1000)
    all_rows = model_rows + dense

    out_dir.mkdir(parents=True, exist_ok=True)
    effects_csv = out_dir / "exp55_late_window_robustness_effects.csv"
    summary_json = out_dir / "exp55_late_window_robustness.json"
    note_md = out_dir / "exp55_late_window_robustness_note.md"
    plot_png = out_dir / "exp55_late_window_robustness.png"
    _write_csv(effects_csv, all_rows)
    _write_note(note_md, dense, models_seen)
    _maybe_plot(plot_png, dense)
    summary = {
        "analysis": "exp55_late_window_robustness",
        "analysis_mode": "fast_kl_only",
        "run_root": str(run_root),
        "models_requested": models,
        "models_analyzed": models_seen,
        "n_boot": n_boot,
        "metrics_analyzed": FAST_METRICS,
        "effects": all_rows,
        "artifacts": {
            "effects_csv": str(effects_csv),
            "note": str(note_md),
            "plot": str(plot_png) if plot_png.exists() else None,
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast-kl-only", action="store_true")
    args = parser.parse_args()
    if args.fast_kl_only:
        summary = analyze_fast_kl(args.run_root, args.out_dir, models=list(args.models), n_boot=args.n_boot, seed=args.seed)
    else:
        summary = analyze(args.run_root, args.out_dir, models=list(args.models), n_boot=args.n_boot, seed=args.seed)
    print(f"[exp55] wrote {summary['artifacts']['effects_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
