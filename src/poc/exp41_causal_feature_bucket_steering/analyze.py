"""Analyze Exp41 logit replay outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MATCH_TOKEN_BUCKETS = {
    "structure_readout": {"newline_or_blankline", "list_or_bullet_marker", "field_label_or_colon"},
    "format_control": {"punctuation_boundary", "quote_or_parenthesis", "surface_subword", "field_label_or_colon"},
    "mcq_scaffold": {"mcq_option_marker", "field_label_or_colon", "quote_or_parenthesis"},
    "surface_punctuation": {"punctuation_boundary", "quote_or_parenthesis", "surface_subword"},
    "safety_advice_boundary": {"refusal_or_safety_phrase"},
    "artifact_repetition": {"rare_unicode_or_artifact"},
}

MATCH_PROMPT_CATEGORIES = {
    "structure_readout": {"GOV-CONV", "GOV-REGISTER"},
    "format_control": {"GOV-FORMAT"},
    "mcq_scaffold": {"CONTENT-FACT", "CONTENT-REASON"},
    "surface_punctuation": {"GOV-FORMAT", "GOV-CONV"},
    "safety_advice_boundary": {"SAFETY"},
    "artifact_repetition": set(),
}

METRICS = [
    "drop_native",
    "drop_pt_upstream",
    "interaction_drop",
    "native_nll_it_shift",
    "it_it_mean_delta_norm_frac",
]


def _read_jsonl_many(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def _float(row: dict[str, Any], key: str) -> float:
    try:
        value = row.get(key)
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    vals = [x for x in values if not math.isnan(x)]
    return sum(vals) / len(vals) if vals else float("nan")


def _se(values: list[float]) -> float:
    vals = [x for x in values if not math.isnan(x)]
    if len(vals) < 2:
        return float("nan")
    mu = _mean(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)
    return math.sqrt(var / len(vals))


def _token_match(row: dict[str, Any]) -> bool:
    expected = MATCH_TOKEN_BUCKETS.get(str(row.get("source_bucket")), set())
    return str(row.get("it_token_bucket")) in expected or str(row.get("pt_token_bucket")) in expected


def _prompt_match(row: dict[str, Any]) -> bool:
    expected = MATCH_PROMPT_CATEGORIES.get(str(row.get("source_bucket")), set())
    return bool(expected) and str(row.get("prompt_category")) in expected


def _summarize(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "") for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        payload = {name: value for name, value in zip(keys, key, strict=False)}
        payload["n"] = len(group)
        for metric in METRICS:
            values = [_float(row, metric) for row in group]
            mean = _mean(values)
            se = _se(values)
            payload[f"{metric}_mean"] = mean
            payload[f"{metric}_se"] = se
            payload[f"{metric}_ci95_lo"] = mean - 1.96 * se if not math.isnan(se) else float("nan")
            payload[f"{metric}_ci95_hi"] = mean + 1.96 * se if not math.isnan(se) else float("nan")
        out.append(payload)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def analyze(*, run_dir: Path) -> dict[str, Any]:
    rows = _read_jsonl_many(sorted((run_dir / "logit_replay").glob("logit_records_*.jsonl")))
    for row in rows:
        row["token_match"] = _token_match(row)
        row["prompt_match"] = _prompt_match(row)
    analysis_dir = run_dir / "analysis"
    by_model = _summarize(
        rows,
        ["model", "source_bucket", "edit_name", "condition_kind", "alpha"],
    )
    by_token = _summarize(
        rows,
        ["model", "source_bucket", "condition_kind", "alpha", "token_match"],
    )
    by_prompt = _summarize(
        rows,
        ["model", "source_bucket", "condition_kind", "alpha", "prompt_match"],
    )
    by_category = _summarize(
        rows,
        ["source_bucket", "condition_kind", "alpha", "prompt_category", "it_token_bucket"],
    )
    _write_csv(analysis_dir / "bucket_effects_by_model.csv", by_model)
    _write_csv(analysis_dir / "bucket_effects_by_token_category.csv", by_token)
    _write_csv(analysis_dir / "bucket_effects_by_prompt_category.csv", by_prompt)
    _write_csv(analysis_dir / "bucket_alpha_dose_response.csv", by_category)

    primary = [
        row
        for row in by_model
        if row.get("condition_kind") == "feature_bucket" and float(row.get("alpha", 0.0)) in {1.0, 2.0}
    ]
    top = sorted(
        primary,
        key=lambda row: float(row.get("drop_native_mean", float("nan"))),
        reverse=True,
    )[:20]
    summary = {
        "run_dir": str(run_dir),
        "n_rows": len(rows),
        "n_prompt_events": len({(row.get("model"), row.get("prompt_id")) for row in rows}),
        "top_feature_bucket_rows": top,
    }
    _write_json(analysis_dir / "exp41_logit_replay_summary.json", summary)

    lines = [
        "# Exp41 Logit Replay Summary",
        "",
        f"Rows: `{len(rows)}`",
        f"Prompt/model events: `{summary['n_prompt_events']}`",
        "",
        "Top feature-bucket rows by native IT margin drop:",
        "",
    ]
    for row in top[:12]:
        lines.append(
            "- "
            f"{row['model']} `{row['source_bucket']}` alpha={row['alpha']}: "
            f"drop_native={float(row['drop_native_mean']):+.4f}, "
            f"interaction_drop={float(row['interaction_drop_mean']):+.4f}, n={row['n']}"
        )
    (analysis_dir / "exp41_logit_replay_note.md").write_text("\n".join(lines).rstrip() + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze(run_dir=args.run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

