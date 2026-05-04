from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np

from src.poc.exp45_behavioral_bridge.metrics import family_balanced_ci, stable_int


PRIMARY_COMPARISONS = ("native_it_necessity", "pt_upstream_sufficiency")
POSITIVE_CONTROL = "positive_control_full_pt_to_it"
NEUTRAL_WINNERS = {"TIE", "BOTH_BAD", "UNCLEAR"}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _result(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    return result if isinstance(result, dict) else {}


def _winner_cell(row: dict[str, Any], winner: str) -> str | None:
    winner = winner.upper()
    if winner == "A":
        return str(row.get("a_cell") or "")
    if winner == "B":
        return str(row.get("b_cell") or "")
    return None


def _score_row(row: dict[str, Any]) -> dict[str, Any]:
    result = _result(row)
    error = result.get("error")
    winner = str(result.get("winner") or "").upper()
    valid = bool(winner in {"A", "B", "TIE", "BOTH_BAD", "UNCLEAR"} and not error)
    target_cell = str(row.get("target_cell") or "")
    winning_cell = _winner_cell(row, winner) if valid else None
    neutral = winner in NEUTRAL_WINNERS if valid else False
    target_score = None
    target_score_resolved = None
    tie_or_neutral = None
    if valid:
        tie_or_neutral = float(neutral)
        if neutral:
            target_score = 0.5
        else:
            target_score = float(winning_cell == target_cell)
            target_score_resolved = target_score
    same_text_tie_success = None
    if row.get("comparison") == "same_text_tie_control" and valid:
        same_text_tie_success = float(winner in {"TIE", "BOTH_BAD", "UNCLEAR"})
    return {
        **{key: row.get(key) for key in row if key not in {"result", "usage"}},
        "winner": winner if valid else "",
        "valid": float(valid),
        "error": str(error) if error else "",
        "target_win_tie_half": target_score,
        "target_win_resolved": target_score_resolved,
        "tie_or_neutral": tie_or_neutral,
        "same_text_tie_success": same_text_tie_success,
        "confidence": _float(result.get("confidence")),
        "judge_margin": result.get("margin"),
        "primary_reason_tag": result.get("primary_reason_tag"),
        "tie_reason": result.get("tie_reason"),
        "length_bias_flag": float(bool(result.get("length_bias_flag"))) if valid else None,
        "short_rationale": result.get("short_rationale"),
        "prompt_cluster": f"{row.get('source_run')}::{row.get('model')}::{row.get('prompt_id')}",
    }


def _aggregate_orders(score_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_logical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_logical[str(row.get("logical_id") or row.get("request_id"))].append(row)
    out: list[dict[str, Any]] = []
    for logical_id, rows in sorted(by_logical.items()):
        rows = sorted(rows, key=lambda row: int(row.get("order_index") or 0))
        base = dict(rows[0])
        base["logical_id"] = logical_id
        base["n_orders"] = len(rows)
        base["valid_orders"] = sum(1 for row in rows if row.get("valid") == 1.0)
        for metric in (
            "target_win_tie_half",
            "target_win_resolved",
            "tie_or_neutral",
            "same_text_tie_success",
            "confidence",
            "length_bias_flag",
        ):
            vals = [_float(row.get(metric)) for row in rows]
            vals = [v for v in vals if v is not None]
            base[metric] = float(np.mean(vals)) if vals else None
        order_scores = [_float(row.get("target_win_tie_half")) for row in rows]
        order_scores = [v for v in order_scores if v is not None]
        base["order_abs_delta"] = abs(order_scores[0] - order_scores[1]) if len(order_scores) >= 2 else None
        resolved_scores = [_float(row.get("target_win_resolved")) for row in rows]
        resolved_scores = [v for v in resolved_scores if v is not None]
        base["order_resolved_flip"] = float(len(resolved_scores) >= 2 and abs(resolved_scores[0] - resolved_scores[1]) > 0.5)
        base["winners_by_order"] = "|".join(str(row.get("winner") or "") for row in rows)
        out.append(base)
    return out


def _effect_rows(rows: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metrics = ("target_win_tie_half", "target_win_resolved", "tie_or_neutral", "confidence", "length_bias_flag")
    groups: list[tuple[str, str, list[dict[str, Any]]]] = []
    for comparison in sorted({str(row.get("comparison")) for row in rows}):
        groups.append(("comparison", comparison, [row for row in rows if row.get("comparison") == comparison]))
    for recipe_group in sorted({str(row.get("recipe_group")) for row in rows}):
        for comparison in sorted({str(row.get("comparison")) for row in rows}):
            subset = [row for row in rows if row.get("recipe_group") == recipe_group and row.get("comparison") == comparison]
            groups.append((f"recipe_group={recipe_group}", comparison, subset))
    for category in sorted({str(row.get("category")) for row in rows}):
        for comparison in sorted({str(row.get("comparison")) for row in rows}):
            subset = [row for row in rows if row.get("category") == category and row.get("comparison") == comparison]
            groups.append((f"category={category}", comparison, subset))
    for model in sorted({str(row.get("model")) for row in rows}):
        for comparison in sorted({str(row.get("comparison")) for row in rows}):
            subset = [row for row in rows if row.get("model") == model and row.get("comparison") == comparison]
            groups.append((f"model={model}", comparison, subset))

    for group_name, comparison, subset in groups:
        if not subset:
            continue
        for metric in metrics:
            usable = [row for row in subset if _float(row.get(metric)) is not None]
            if not usable:
                continue
            ci = family_balanced_ci(
                usable,
                metric,
                models=models,
                n_boot=n_boot,
                seed=stable_int("exp50", "effect", group_name, comparison, metric),
            )
            out.append(
                {
                    "group": group_name,
                    "comparison": comparison,
                    "metric": metric,
                    "n": len(usable),
                    **ci,
                }
            )
    return out


def _interaction_rows(rows: list[dict[str, Any]], *, models: list[str], n_boot: int) -> list[dict[str, Any]]:
    eligible = [row for row in rows if row.get("comparison") in PRIMARY_COMPARISONS]
    by_event: dict[tuple[str, str, str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in eligible:
        key = (
            str(row.get("source_run")),
            str(row.get("model")),
            str(row.get("prompt_id")),
            str(row.get("event_kind")),
            str(row.get("position")),
        )
        by_event[key][str(row.get("comparison"))] = row

    contrast_rows: list[dict[str, Any]] = []
    for (_source, model, prompt_id, event_kind, position), comps in by_event.items():
        if not all(name in comps for name in PRIMARY_COMPARISONS):
            continue
        native = _float(comps["native_it_necessity"].get("target_win_tie_half"))
        suff = _float(comps["pt_upstream_sufficiency"].get("target_win_tie_half"))
        if native is None or suff is None:
            continue
        base = {
            "source_run": _source,
            "model": model,
            "prompt_id": prompt_id,
            "event_kind": event_kind,
            "position": position,
            "category": comps["native_it_necessity"].get("category"),
            "recipe_group": comps["native_it_necessity"].get("recipe_group"),
            "prompt_cluster": comps["native_it_necessity"].get("prompt_cluster"),
        }
        contrast_rows.append(
            {
                **base,
                "native_it_necessity": native,
                "pt_upstream_sufficiency": suff,
                "behavioral_interaction": native - suff,
            }
        )

    out: list[dict[str, Any]] = []
    group_specs: list[tuple[str, list[dict[str, Any]]]] = [("all", contrast_rows)]
    for group in sorted({str(row.get("recipe_group")) for row in contrast_rows}):
        group_specs.append((f"recipe_group={group}", [row for row in contrast_rows if row.get("recipe_group") == group]))
    for category in sorted({str(row.get("category")) for row in contrast_rows}):
        group_specs.append((f"category={category}", [row for row in contrast_rows if row.get("category") == category]))
    for model in sorted({str(row.get("model")) for row in contrast_rows}):
        group_specs.append((f"model={model}", [row for row in contrast_rows if row.get("model") == model]))

    for group_name, subset in group_specs:
        if not subset:
            continue
        for metric in ("native_it_necessity", "pt_upstream_sufficiency", "behavioral_interaction"):
            ci = family_balanced_ci(
                subset,
                metric,
                models=models,
                n_boot=n_boot,
                seed=stable_int("exp50", "interaction", group_name, metric),
            )
            out.append({"group": group_name, "metric": metric, "n": len(subset), **ci})
    return out


def _length_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_name, group_rows in _analysis_groups(rows):
        for comparison in sorted({str(row.get("comparison")) for row in group_rows}):
            subset = [
                row
                for row in group_rows
                if row.get("comparison") == comparison
                and _float(row.get("target_win_tie_half")) is not None
                and _float(row.get("length_delta_target_minus_other")) is not None
            ]
            if len(subset) < 3:
                continue
            y = np.asarray([float(row["target_win_tie_half"]) for row in subset], dtype=float)
            x = np.asarray([float(row["length_delta_target_minus_other"]) for row in subset], dtype=float)
            x = x - float(np.mean(x))
            design = np.column_stack([np.ones_like(x), x])
            beta, *_ = np.linalg.lstsq(design, y, rcond=None)
            pred_at_equal_length = float(beta[0] - beta[1] * float(np.mean([row["length_delta_target_minus_other"] for row in subset])))
            out.append(
                {
                    "group": group_name,
                    "comparison": comparison,
                    "n": len(subset),
                    "raw_mean": float(np.mean(y)),
                    "length_slope": float(beta[1]),
                    "equal_length_estimate": pred_at_equal_length,
                    "mean_length_delta_target_minus_other": float(
                        np.mean([row["length_delta_target_minus_other"] for row in subset])
                    ),
                }
            )
    return out


def _analysis_groups(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups = [("all", rows)]
    for recipe_group in sorted({str(row.get("recipe_group")) for row in rows}):
        groups.append((f"recipe_group={recipe_group}", [row for row in rows if row.get("recipe_group") == recipe_group]))
    for category in sorted({str(row.get("category")) for row in rows}):
        groups.append((f"category={category}", [row for row in rows if row.get("category") == category]))
    return groups


def _order_bias_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for group_name, subset in _analysis_groups(rows):
        if not subset:
            continue
        primary = [row for row in subset if row.get("comparison") in PRIMARY_COMPARISONS]
        controls = [row for row in subset if row.get("comparison") == "same_text_tie_control"]
        for label, label_rows in (("primary", primary), ("same_text_control", controls)):
            if not label_rows:
                continue
            out.append(
                {
                    "group": group_name,
                    "kind": label,
                    "n_logical": len(label_rows),
                    "mean_valid_orders": float(np.mean([_float(row.get("valid_orders")) or 0.0 for row in label_rows])),
                    "mean_order_abs_delta": _mean(row.get("order_abs_delta") for row in label_rows),
                    "order_resolved_flip_rate": _mean(row.get("order_resolved_flip") for row in label_rows),
                    "same_text_tie_success": _mean(row.get("same_text_tie_success") for row in label_rows),
                }
            )
    return out


def _mean(values: Iterable[Any]) -> float | None:
    vals = [_float(value) for value in values]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _plot_interactions(rows: list[dict[str, Any]], out_path: Path) -> None:
    primary = [row for row in rows if row["metric"] == "behavioral_interaction" and row["group"].startswith("recipe_group=")]
    if not primary:
        return
    labels = [row["group"].split("=", 1)[1] for row in primary]
    vals = [float(row["estimate"]) for row in primary]
    lows = [float(row["ci_low"]) for row in primary]
    highs = [float(row["ci_high"]) for row in primary]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(vals))
    ax.bar(x, vals, color="#465f8c")
    ax.errorbar(x, vals, yerr=[np.asarray(vals) - np.asarray(lows), np.asarray(highs) - np.asarray(vals)], fmt="none", color="black", capsize=3)
    ax.axhline(0, color="0.3", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("Judge behavioral interaction")
    ax.set_title("Exp50 order-balanced LLM judge bridge")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def analyze(*, responses: Path, out_dir: Path, models: list[str], n_boot: int) -> dict[str, Any]:
    raw_rows = list(_iter_jsonl(responses))
    score_rows = [_score_row(row) for row in raw_rows]
    aggregate_rows = _aggregate_orders(score_rows)
    effect_rows = _effect_rows(aggregate_rows, models=models, n_boot=n_boot)
    interaction_rows = _interaction_rows(aggregate_rows, models=models, n_boot=n_boot)
    length_rows = _length_control_rows(aggregate_rows)
    order_rows = _order_bias_rows(aggregate_rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "judge_scores.csv", score_rows)
    _write_csv(out_dir / "judge_order_aggregated_scores.csv", aggregate_rows)
    _write_csv(out_dir / "pairwise_winrates.csv", effect_rows)
    _write_csv(out_dir / "behavioral_interactions.csv", interaction_rows)
    _write_csv(out_dir / "length_controlled_effects.csv", length_rows)
    _write_csv(out_dir / "order_bias_audit.csv", order_rows)
    _plot_interactions(interaction_rows, out_dir / "exp50_behavioral_interaction_by_recipe.png")

    invalid = [row for row in score_rows if row.get("valid") != 1.0]
    primary_order = [row for row in order_rows if row["group"] == "all" and row["kind"] == "primary"]
    same_text = [row for row in order_rows if row["group"] == "all" and row["kind"] == "same_text_control"]
    positive = [
        row
        for row in effect_rows
        if row["group"] == "comparison"
        and row["comparison"] == POSITIVE_CONTROL
        and row["metric"] == "target_win_tie_half"
    ]
    paper_grade = {
        "invalid_rate_ok": (len(invalid) / max(1, len(score_rows))) < 0.01,
        "positive_control_above_chance": bool(positive and _float(positive[0].get("ci_low")) is not None and float(positive[0]["ci_low"]) > 0.5),
        "order_bias_reported": bool(primary_order),
        "same_text_control_reported": bool(same_text),
    }
    paper_grade["ok"] = all(paper_grade.values())

    summary = {
        "responses": str(responses),
        "models": models,
        "n_response_rows": len(raw_rows),
        "n_score_rows": len(score_rows),
        "n_logical_rows": len(aggregate_rows),
        "n_invalid": len(invalid),
        "invalid_rate": len(invalid) / max(1, len(score_rows)),
        "paper_grade_checks": paper_grade,
        "primary_interactions": [
            row
            for row in interaction_rows
            if row["metric"] == "behavioral_interaction" and row["group"] in {"all", "recipe_group=instruction_like", "recipe_group=math_reasoning"}
        ],
        "positive_control": positive,
        "order_bias_audit": order_rows,
    }
    (out_dir / "judge_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_paper_claims(out_dir / "paper_claims_exp50.md", summary)
    return summary


def _write_paper_claims(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Exp50 Paper-Facing Claims",
        "",
        "Exp50 is a secondary behavioral bridge. It should not replace the fixed-prefix logit and likelihood estimands.",
        "",
        f"- Response rows: `{summary['n_response_rows']}`.",
        f"- Invalid/schema-error rate: `{summary['invalid_rate']:.4f}`.",
        f"- Paper-grade checks pass: `{summary['paper_grade_checks']['ok']}`.",
        "",
        "Primary behavioral interaction rows:",
    ]
    for row in summary.get("primary_interactions", []):
        lines.append(
            f"- `{row['group']}`: `{row['estimate']}` "
            f"[`{row['ci_low']}`, `{row['ci_high']}`], n=`{row['n']}`."
        )
    lines.extend(
        [
            "",
            "Safe wording if positive:",
            "",
            "> Blind pairwise judging provides secondary evidence that the fixed-prefix late-stack interaction transfers to short natural completions for instruction-like prompts.",
            "",
            "Safe wording if weak or category-specific:",
            "",
            "> The constrained-likelihood bridge remains the primary sequence-level evidence; blind completion judging is category-dependent and is treated as a behavioral sanity check.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Exp50 OpenAI LLM judge responses.")
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=["llama31_8b", "mistral_7b", "qwen3_4b", "olmo2_7b"])
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    out_dir = args.out_dir or args.responses.parent
    summary = analyze(responses=args.responses, out_dir=out_dir, models=args.models, n_boot=args.n_boot)
    print(json.dumps({"ok": True, "out_dir": str(out_dir), "paper_grade": summary["paper_grade_checks"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

