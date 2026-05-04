from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.poc.exp45_behavioral_bridge.judge_prompts import build_judge_requests
from src.poc.exp45_behavioral_bridge.metrics import (
    CELL_ORDER,
    family_balanced_ci,
    finite,
    mean,
    stable_int,
)


PRIMARY_READOUT = "common_it"


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_one_step(run_root: Path, models: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        path = run_root / "raw" / model / "one_step_records.jsonl.gz"
        if path.exists():
            rows.extend(_json_rows(path))
    return rows


def _load_rollouts(run_root: Path, models: list[str]) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        path = run_root / "raw" / model / "rollout_records.jsonl.gz"
        if path.exists():
            rows.extend(_json_rows(path))
    return rows


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
            writer.writerow(row)


def _flatten_one_step(rows: list[dict[str, Any]], *, readout: str = PRIMARY_READOUT) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        for cell in CELL_ORDER:
            payload = ((row.get("cells") or {}).get(cell) or {}).get(readout)
            if not isinstance(payload, dict):
                continue
            out.append(
                {
                    "model": row.get("model"),
                    "prompt_id": row.get("prompt_id"),
                    "event_kind": row.get("event_kind"),
                    "category": row.get("category"),
                    "position": row.get("position"),
                    "position_ge_3": bool(row.get("position_ge_3")),
                    "cell": cell,
                    "readout": readout,
                    "margin_it_minus_pt": payload.get("margin_it_minus_pt"),
                    "pairwise_it_win": float(bool(payload.get("pairwise_it_win"))),
                    "top1_it": float(bool(payload.get("top1_is_t_it"))),
                    "top1_pt": float(bool(payload.get("top1_is_t_pt"))),
                    "top5_it": float(bool(payload.get("t_it_top5"))),
                    "top5_pt": float(bool(payload.get("t_pt_top5"))),
                    "it_rank": payload.get("t_it_rank"),
                    "pt_rank": payload.get("t_pt_rank"),
                    "it_gap_to_top1": payload.get("t_it_gap_to_top1"),
                    "top1_class": payload.get("top1_class"),
                }
            )
    return out


def _one_step_effect_rows(flat: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metrics = ("margin_it_minus_pt", "pairwise_it_win", "top5_it", "top1_it", "it_rank", "it_gap_to_top1")
    for cell in CELL_ORDER:
        subset = [row for row in flat if row["cell"] == cell]
        for metric in metrics:
            out.append(
                {
                    "effect": "cell_mean",
                    "cell": cell,
                    "metric": metric,
                    **family_balanced_ci(
                        subset,
                        metric,
                        models=models,
                        n_boot=n_boot,
                        seed=stable_int("one_step", "cell", cell, metric),
                    ),
                }
            )

    by_event: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in flat:
        by_event[(str(row["model"]), str(row["prompt_id"]), str(row["event_kind"]))][str(row["cell"])] = row

    contrast_rows: list[dict[str, Any]] = []
    for (model, prompt_id, event_kind), cells in by_event.items():
        required = set(CELL_ORDER)
        if not required.issubset(cells):
            continue
        base = {"model": model, "prompt_id": prompt_id, "event_kind": event_kind}
        for metric in metrics:
            vals = {cell: finite(cells[cell].get(metric)) for cell in CELL_ORDER}
            if any(vals[cell] is None for cell in CELL_ORDER):
                continue
            # Higher rank is worse, so flip contrasts for rank to keep positive = more IT-favoring.
            sign = -1.0 if metric == "it_rank" else 1.0
            contrast_rows.append(
                {
                    **base,
                    "metric": metric,
                    "pairwise_portability_gap": sign * (vals["U_IT__L_IT"] - vals["U_PT__L_IT"]),
                    "late_effect_pt_upstream": sign * (vals["U_PT__L_IT"] - vals["U_PT__L_PT"]),
                    "late_effect_it_upstream": sign * (vals["U_IT__L_IT"] - vals["U_IT__L_PT"]),
                    "factorial_interaction": sign
                    * ((vals["U_IT__L_IT"] - vals["U_IT__L_PT"]) - (vals["U_PT__L_IT"] - vals["U_PT__L_PT"])),
                }
            )
    for metric in metrics:
        for key in ("pairwise_portability_gap", "late_effect_pt_upstream", "late_effect_it_upstream", "factorial_interaction"):
            subset = [row for row in contrast_rows if row["metric"] == metric]
            out.append(
                {
                    "effect": key,
                    "cell": "",
                    "metric": metric,
                    **family_balanced_ci(
                        subset,
                        key,
                        models=models,
                        n_boot=n_boot,
                        seed=stable_int("one_step", key, metric),
                    ),
                }
            )
    return out


def _rollout_effect_rows(rows: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metrics = (
        "full_lexical_it_like_score",
        "post_first_lexical_it_like_score",
        "full_word_count",
        "post_first_word_count",
        "first_generated_is_t_it",
        "first_generated_is_t_pt",
    )
    normalized = []
    for row in rows:
        new = dict(row)
        new["first_generated_is_t_it"] = float(bool(row.get("first_generated_is_t_it")))
        new["first_generated_is_t_pt"] = float(bool(row.get("first_generated_is_t_pt")))
        normalized.append(new)

    for cell in CELL_ORDER:
        subset = [row for row in normalized if row.get("cell") == cell]
        for metric in metrics:
            out.append(
                {
                    "effect": "cell_mean",
                    "cell": cell,
                    "metric": metric,
                    **family_balanced_ci(
                        subset,
                        metric,
                        models=models,
                        n_boot=n_boot,
                        seed=stable_int("rollout", "cell", cell, metric),
                    ),
                }
            )

    by_event: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in normalized:
        by_event[(str(row.get("model")), str(row.get("prompt_id")), str(row.get("event_kind")))][str(row.get("cell"))] = row
    contrast_rows: list[dict[str, Any]] = []
    for (model, prompt_id, event_kind), cells in by_event.items():
        if not set(CELL_ORDER).issubset(cells):
            continue
        base = {"model": model, "prompt_id": prompt_id, "event_kind": event_kind}
        for metric in metrics:
            vals = {cell: finite(cells[cell].get(metric)) for cell in CELL_ORDER}
            if any(vals[cell] is None for cell in CELL_ORDER):
                continue
            contrast_rows.append(
                {
                    **base,
                    "metric": metric,
                    "behavioral_portability_gap": vals["U_IT__L_IT"] - vals["U_PT__L_IT"],
                    "late_effect_pt_upstream": vals["U_PT__L_IT"] - vals["U_PT__L_PT"],
                    "late_effect_it_upstream": vals["U_IT__L_IT"] - vals["U_IT__L_PT"],
                    "behavioral_interaction": (vals["U_IT__L_IT"] - vals["U_IT__L_PT"])
                    - (vals["U_PT__L_IT"] - vals["U_PT__L_PT"]),
                }
            )
    for metric in metrics:
        subset = [row for row in contrast_rows if row["metric"] == metric]
        for key in ("behavioral_portability_gap", "late_effect_pt_upstream", "late_effect_it_upstream", "behavioral_interaction"):
            out.append(
                {
                    "effect": key,
                    "cell": "",
                    "metric": metric,
                    **family_balanced_ci(
                        subset,
                        key,
                        models=models,
                        n_boot=n_boot,
                        seed=stable_int("rollout", key, metric),
                    ),
                }
            )
    return out


def _plot_one_step(rows: list[dict[str, Any]], out_dir: Path) -> None:
    cell_rows = [r for r in rows if r["effect"] == "cell_mean" and r["metric"] in {"pairwise_it_win", "top5_it", "top1_it"}]
    if not cell_rows:
        return
    metrics = ["pairwise_it_win", "top5_it", "top1_it"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 3.8), sharey=True)
    for ax, metric in zip(axes, metrics):
        vals = [next((r for r in cell_rows if r["cell"] == cell and r["metric"] == metric), None) for cell in CELL_ORDER]
        y = [float(r["estimate"]) if r and r["estimate"] is not None else 0.0 for r in vals]
        lo = [float(r["ci_low"]) if r and r["ci_low"] is not None else y[i] for i, r in enumerate(vals)]
        hi = [float(r["ci_high"]) if r and r["ci_high"] is not None else y[i] for i, r in enumerate(vals)]
        err = [[max(0.0, y[i] - lo[i]) for i in range(len(y))], [max(0.0, hi[i] - y[i]) for i in range(len(y))]]
        ax.bar(range(len(CELL_ORDER)), y, yerr=err, capsize=3)
        ax.set_title(metric)
        ax.set_xticks(range(len(CELL_ORDER)), [c.replace("__", "\n") for c in CELL_ORDER], rotation=0)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_dir / "one_step_token_choice.png", dpi=180)
    plt.close(fig)


def _plot_rollout(rows: list[dict[str, Any]], out_dir: Path) -> None:
    cell_rows = [
        r
        for r in rows
        if r["effect"] == "cell_mean"
        and r["metric"] in {"full_lexical_it_like_score", "post_first_lexical_it_like_score"}
    ]
    if not cell_rows:
        return
    metrics = ["full_lexical_it_like_score", "post_first_lexical_it_like_score"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(10, 3.8))
    for ax, metric in zip(axes, metrics):
        vals = [next((r for r in cell_rows if r["cell"] == cell and r["metric"] == metric), None) for cell in CELL_ORDER]
        y = [float(r["estimate"]) if r and r["estimate"] is not None else 0.0 for r in vals]
        lo = [float(r["ci_low"]) if r and r["ci_low"] is not None else y[i] for i, r in enumerate(vals)]
        hi = [float(r["ci_high"]) if r and r["ci_high"] is not None else y[i] for i, r in enumerate(vals)]
        err = [[max(0.0, y[i] - lo[i]) for i in range(len(y))], [max(0.0, hi[i] - y[i]) for i in range(len(y))]]
        ax.bar(range(len(CELL_ORDER)), y, yerr=err, capsize=3)
        ax.set_title(metric)
        ax.set_xticks(range(len(CELL_ORDER)), [c.replace("__", "\n") for c in CELL_ORDER], rotation=0)
    fig.tight_layout()
    fig.savefig(out_dir / "hybrid_completion_scores.png", dpi=180)
    plt.close(fig)


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Exp45 Behavioral Bridge Report",
        "",
        f"Models: {', '.join(summary['models'])}.",
        f"One-step records: {summary['n_one_step_records']}. Rollout records: {summary['n_rollout_records']}.",
        "",
        "## Primary One-Step Effects",
        "",
    ]
    for row in summary["one_step_effects"]:
        if row["effect"] in {"pairwise_portability_gap", "factorial_interaction"} and row["metric"] in {"pairwise_it_win", "top5_it", "top1_it"}:
            lines.append(
                f"- {row['effect']} / {row['metric']}: {row['estimate']:.4g} "
                f"[{row['ci_low']:.4g}, {row['ci_high']:.4g}]"
            )
    lines += ["", "## Primary Rollout Effects", ""]
    for row in summary["rollout_effects"]:
        if row["effect"] in {"behavioral_portability_gap", "behavioral_interaction"} and row["metric"] in {
            "full_lexical_it_like_score",
            "post_first_lexical_it_like_score",
            "first_generated_is_t_it",
        }:
            lines.append(
                f"- {row['effect']} / {row['metric']}: {row['estimate']:.4g} "
                f"[{row['ci_low']:.4g}, {row['ci_high']:.4g}]"
            )
    lines += [
        "",
        "Interpretation guardrail: one-step pairwise/top-k effects are the primary behavioral bridge. "
        "Lexical rollout metrics are deterministic sanity checks; LLM judge requests should be scored before "
        "making strong completion-level claims.",
        "",
    ]
    return "\n".join(lines)


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--primary-models", nargs="+", default=None)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--include-absolute-judge", action="store_true")


def main(args: argparse.Namespace) -> None:
    out_dir = args.run_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    models = args.primary_models or args.models
    one = _load_one_step(args.run_root, args.models)
    rollouts = _load_rollouts(args.run_root, args.models)
    flat = _flatten_one_step(one, readout=PRIMARY_READOUT)
    one_effects = _one_step_effect_rows(flat, models=models, n_boot=args.n_boot)
    rollout_effects = _rollout_effect_rows(rollouts, models=models, n_boot=args.n_boot)
    judge_requests = build_judge_requests(rollouts, include_absolute=args.include_absolute_judge)

    _write_csv(out_dir / "one_step_effects.csv", one_effects)
    _write_csv(out_dir / "behavioral_effects.csv", rollout_effects)
    with (out_dir / "llm_judge_requests.jsonl").open("w", encoding="utf-8") as handle:
        for req in judge_requests:
            handle.write(json.dumps(req, separators=(",", ":")) + "\n")

    summary = {
        "experiment": "exp45_behavioral_bridge",
        "run_root": str(args.run_root),
        "models": args.models,
        "primary_models": models,
        "n_one_step_records": len(one),
        "n_flat_one_step_rows": len(flat),
        "n_rollout_records": len(rollouts),
        "n_judge_requests": len(judge_requests),
        "one_step_effects": one_effects,
        "rollout_effects": rollout_effects,
    }
    (out_dir / "one_step_summary.json").write_text(json.dumps({"effects": one_effects}, indent=2))
    (out_dir / "behavioral_summary.json").write_text(json.dumps({"effects": rollout_effects}, indent=2))
    (out_dir / "exp45_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "report.md").write_text(_report(summary))
    _plot_one_step(one_effects, out_dir)
    _plot_rollout(rollout_effects, out_dir)

