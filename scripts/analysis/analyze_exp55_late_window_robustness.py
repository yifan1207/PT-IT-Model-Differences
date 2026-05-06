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
from collections import defaultdict
from pathlib import Path
from typing import Any


MODEL_ORDER = ["gemma3_4b", "qwen3_4b", "llama31_8b", "mistral_7b", "olmo2_7b"]
METRICS = ["kl_to_own_final", "delta_cosine", "cross_kl"]
REGIONS = ["final_20pct", "graft_window"]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _iter_jsonl(path: Path):
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                yield json.loads(raw.decode("utf-8", errors="ignore"))


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
    for row in _iter_jsonl(model_dir / "step_metrics.jsonl"):
        records[(str(row["prompt_id"]), int(row["step"]), str(row["pipeline"]))] = row["metrics"]

    rows: list[dict[str, Any]] = []
    for pipeline, window in sorted(branches.items()):
        baseline = _baseline_pipeline(window)
        side = _side_label(window)
        start, end = _window_region(window)
        window_name = str(window.get("window_name") or pipeline)
        for region in REGIONS:
            for metric in METRICS:
                deltas_by_prompt: dict[str, list[float]] = defaultdict(list)
                n_pairs = 0
                for prompt_id, step, pipe in list(records):
                    if pipe != pipeline:
                        continue
                    branch_metrics = records[(prompt_id, step, pipeline)]
                    base_metrics = records.get((prompt_id, step, baseline))
                    if base_metrics is None:
                        continue
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
                    deltas_by_prompt[prompt_id].append(branch_value - base_value)
                    n_pairs += 1
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
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:+.3f}"


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
    lines.extend(
        [
            "",
            "Interpretation rule: this audit supports the paper only if late-centered or terminal windows "
            "preserve the expected graft-positive/swap-negative direction and pre-late controls are weaker.",
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    summary = analyze(args.run_root, args.out_dir, models=list(args.models), n_boot=args.n_boot, seed=args.seed)
    print(f"[exp55] wrote {summary['artifacts']['effects_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
