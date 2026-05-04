#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from src.poc.exp45_behavioral_bridge.metrics import CELL_ORDER, family_balanced_ci


PRIMARY_MODELS = ("llama31_8b", "mistral_7b", "qwen3_4b", "olmo2_7b")
ABS_METRICS = ("it_likeness", "instruction_following", "answer_structure", "pt_style_raw_continuation")


def _json_rows(path: Path) -> list[dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    fallback_rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                request_id = row.get("request_id")
                if request_id is None:
                    fallback_rows.append(row)
                else:
                    # Retry passes append replacement rows; the latest row is the authoritative result.
                    rows_by_id[str(request_id)] = row
    return [*rows_by_id.values(), *fallback_rows]


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _winner_scores(row: dict[str, Any]) -> dict[str, float | None]:
    result = row.get("result") or {}
    if "error" in result:
        return {
            "left_win_tie_half": None,
            "left_win_resolved": None,
            "right_win": None,
            "tie_or_unclear": None,
        }
    winner = str(result.get("winner", "")).strip().lower()
    a_cell = row.get("a_cell")
    b_cell = row.get("b_cell")
    left_cell = row.get("left_cell")
    if winner == "a":
        winning_cell = a_cell
    elif winner == "b":
        winning_cell = b_cell
    elif winner in {"tie", "unclear"}:
        winning_cell = None
    else:
        return {
            "left_win_tie_half": None,
            "left_win_resolved": None,
            "right_win": None,
            "tie_or_unclear": None,
        }
    if winning_cell is None:
        return {
            "left_win_tie_half": 0.5,
            "left_win_resolved": None,
            "right_win": 0.0,
            "tie_or_unclear": 1.0,
        }
    left_win = float(winning_cell == left_cell)
    return {
        "left_win_tie_half": left_win,
        "left_win_resolved": left_win,
        "right_win": 1.0 - left_win,
        "tie_or_unclear": 0.0,
    }


def _pairwise_effects(rows: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    pair_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != "pairwise":
            continue
        scores = _winner_scores(row)
        pair_rows.append(
            {
                "model": row.get("model"),
                "prompt_id": row.get("prompt_id"),
                "comparison": row.get("comparison"),
                **scores,
            }
        )
    out: list[dict[str, Any]] = []
    for comparison in sorted({str(row.get("comparison")) for row in pair_rows}):
        subset = [row for row in pair_rows if row.get("comparison") == comparison]
        for metric in ("left_win_tie_half", "left_win_resolved", "right_win", "tie_or_unclear"):
            usable = [row for row in subset if row.get(metric) is not None]
            ci = family_balanced_ci(usable, metric, models=models, n_boot=n_boot, seed=9200 + len(out))
            out.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "n": len(usable),
                    **ci,
                }
            )
    return out


def _absolute_cell_means(rows: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    abs_rows: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != "absolute":
            continue
        result = row.get("result") or {}
        if "error" in result:
            continue
        payload = {
            "model": row.get("model"),
            "prompt_id": row.get("prompt_id"),
            "cell": row.get("cell"),
        }
        for metric in ABS_METRICS:
            value = _float(result.get(metric))
            if value is not None:
                payload[metric] = value
        if "pt_style_raw_continuation" in payload:
            payload["pt_style_raw_continuation_reversed"] = -float(payload["pt_style_raw_continuation"])
        abs_rows.append(payload)

    out: list[dict[str, Any]] = []
    for cell in CELL_ORDER:
        subset = [row for row in abs_rows if row.get("cell") == cell]
        for metric in (*ABS_METRICS, "pt_style_raw_continuation_reversed"):
            usable = [row for row in subset if row.get(metric) is not None]
            ci = family_balanced_ci(usable, metric, models=models, n_boot=n_boot, seed=9300 + len(out))
            out.append({"cell": cell, "metric": metric, "n": len(usable), **ci})
    return out


def _absolute_contrasts(rows: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    by_event: dict[tuple[str, str], dict[str, dict[str, float]]] = defaultdict(dict)
    for row in rows:
        if row.get("kind") != "absolute":
            continue
        result = row.get("result") or {}
        if "error" in result:
            continue
        values: dict[str, float] = {}
        for metric in ABS_METRICS:
            value = _float(result.get(metric))
            if value is not None:
                values[metric] = value
        if "pt_style_raw_continuation" in values:
            values["pt_style_raw_continuation_reversed"] = -values["pt_style_raw_continuation"]
        by_event[(str(row.get("model")), str(row.get("prompt_id")))][str(row.get("cell"))] = values

    contrast_rows: list[dict[str, Any]] = []
    for (model, prompt_id), cells in by_event.items():
        if not set(CELL_ORDER).issubset(cells):
            continue
        for metric in ("it_likeness", "instruction_following", "answer_structure", "pt_style_raw_continuation_reversed"):
            vals = {cell: cells[cell].get(metric) for cell in CELL_ORDER}
            if any(value is None for value in vals.values()):
                continue
            contrast_rows.append(
                {
                    "model": model,
                    "prompt_id": prompt_id,
                    "metric": metric,
                    "absolute_portability_gap": vals["U_IT__L_IT"] - vals["U_PT__L_IT"],
                    "late_effect_pt_upstream": vals["U_PT__L_IT"] - vals["U_PT__L_PT"],
                    "late_effect_it_upstream": vals["U_IT__L_IT"] - vals["U_IT__L_PT"],
                    "absolute_interaction": (vals["U_IT__L_IT"] - vals["U_IT__L_PT"])
                    - (vals["U_PT__L_IT"] - vals["U_PT__L_PT"]),
                }
            )
    out: list[dict[str, Any]] = []
    for metric in sorted({row["metric"] for row in contrast_rows}):
        subset = [row for row in contrast_rows if row["metric"] == metric]
        for effect in ("absolute_portability_gap", "late_effect_pt_upstream", "late_effect_it_upstream", "absolute_interaction"):
            ci = family_balanced_ci(subset, effect, models=models, n_boot=n_boot, seed=9400 + len(out))
            out.append({"effect": effect, "metric": metric, "n": len(subset), **ci})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plot_pairwise(rows: list[dict[str, Any]], out_path: Path) -> None:
    primary = [row for row in rows if row["metric"] == "left_win_tie_half"]
    labels = [row["comparison"].replace("_", "\n") for row in primary]
    vals = [float(row["estimate"]) for row in primary]
    lows = [float(row["ci_low"]) for row in primary]
    highs = [float(row["ci_high"]) for row in primary]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(vals))
    ax.bar(x, vals, color="#4f6f52")
    ax.errorbar(x, vals, yerr=[[v - l for v, l in zip(vals, lows)], [h - v for v, h in zip(vals, highs)]], fmt="none", color="black", capsize=3)
    ax.axhline(0.5, color="0.35", linewidth=1, linestyle="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Judge prefers left cell (tie=0.5)")
    ax.set_xticks(list(x), labels)
    ax.set_title("Exp45 LLM Judge Pairwise Behavioral Bridge")
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(PRIMARY_MODELS))
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()

    rows = _json_rows(args.results)
    out_dir = args.out_dir or args.results.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    errors = [row for row in rows if "error" in (row.get("result") or {})]
    pairwise = _pairwise_effects(rows, models=args.models, n_boot=args.n_boot)
    cell_means = _absolute_cell_means(rows, models=args.models, n_boot=args.n_boot)
    abs_contrasts = _absolute_contrasts(rows, models=args.models, n_boot=args.n_boot)

    _write_csv(out_dir / "llm_judge_pairwise_effects.csv", pairwise)
    _write_csv(out_dir / "llm_judge_absolute_cell_means.csv", cell_means)
    _write_csv(out_dir / "llm_judge_absolute_contrasts.csv", abs_contrasts)
    _plot_pairwise(pairwise, out_dir / "llm_judge_pairwise.png")

    summary = {
        "results": str(args.results),
        "n_rows": len(rows),
        "n_errors": len(errors),
        "models": args.models,
        "pairwise_effects": pairwise,
        "absolute_cell_means": cell_means,
        "absolute_contrasts": abs_contrasts,
    }
    (out_dir / "llm_judge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def find_pair(comparison: str, metric: str = "left_win_tie_half") -> dict[str, Any]:
        return next(row for row in pairwise if row["comparison"] == comparison and row["metric"] == metric)

    report = [
        "# Exp45 LLM Judge Summary",
        "",
        f"Rows: {len(rows)}. Errors: {len(errors)}.",
        "",
        "## Pairwise Effects",
    ]
    for comparison in ("IT_late_portability", "PT_upstream_late_effect", "IT_upstream_late_effect"):
        row = find_pair(comparison)
        report.append(
            f"- `{comparison}`: {float(row['estimate']):.3f} "
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}], n={row['n']}"
        )
    report.extend(
        [
            "",
            "Interpretation: values above 0.5 mean the judge prefers the left cell named by the comparison "
            "(`U_IT__L_IT` for portability and IT-upstream late effect, `U_PT__L_IT` for PT-upstream late effect).",
        ]
    )
    (out_dir / "llm_judge_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
