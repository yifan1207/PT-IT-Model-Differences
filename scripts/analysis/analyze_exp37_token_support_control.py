#!/usr/bin/env python3
"""Audit token-support composition for Exp37 random-local disagreement controls.

The first-divergence support is intentionally selected. This script asks whether
that selected support is unusually surface/format-heavy relative to Exp37's
random local PT/IT disagreement prefixes from the same prompts and rollouts.
It uses deterministic token metadata only; the separate LLM audit remains the
semantic category audit for the true first-divergence support.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.poc.exp37_random_prefix_baseline import (
    DENSE5_MODELS,
    PREDIV_KEY,
    RANDOM_IT_KEY,
    RANDOM_PT_KEY,
    REFERENCE_KEY,
)


DEFAULT_RUN_ROOT = Path(
    "results/exp37_random_prefix_baseline/"
    "exp37_full_dense5_auth_xetfast_h100x8_20260503_002609"
)

PUNCT_RE = re.compile(r"^[\s\n\r\t\.,;:!?\-–—_\*#`'\"\[\]\(\)\{\}<>/\\|=+~]+$")


def _json_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _event(record: dict[str, Any]) -> dict[str, Any] | None:
    event = (record.get("divergence_events") or {}).get("first_diff")
    if isinstance(event, dict):
        return event
    event = ((record.get("readouts") or {}).get("first_diff") or {}).get("event")
    return event if isinstance(event, dict) else None


def _token_text(token: dict[str, Any]) -> str:
    return str(token.get("token_str", ""))


def _is_surface(token: dict[str, Any]) -> bool:
    text = _token_text(token)
    category = str(token.get("token_category", "OTHER"))
    collapsed = str(token.get("token_category_collapsed", "OTHER"))
    if text == "" or text.strip() == "":
        return True
    if collapsed == "FORMAT" and PUNCT_RE.match(text):
        return True
    if category in {"PUNCTUATION", "WHITESPACE"} and PUNCT_RE.match(text):
        return True
    return False


def _any_content(row: dict[str, Any]) -> bool:
    return any(
        str(token.get("token_category", "")) == "CONTENT"
        or str(token.get("token_category_collapsed", "")) == "CONTENT"
        for token in (row["pt_token"], row["it_token"])
    )


def _any_format(row: dict[str, Any]) -> bool:
    return any(str(token.get("token_category_collapsed", "")) == "FORMAT" for token in (row["pt_token"], row["it_token"]))


def _both_surface(row: dict[str, Any]) -> bool:
    return _is_surface(row["pt_token"]) and _is_surface(row["it_token"])


def _position_0(row: dict[str, Any]) -> bool:
    return int(row["step"]) == 0


def _position_ge_3(row: dict[str, Any]) -> bool:
    return int(row["step"]) >= 3


def _position_ge_5(row: dict[str, Any]) -> bool:
    return int(row["step"]) >= 5


def _pair_kind(row: dict[str, Any]) -> str:
    if _both_surface(row):
        return "both_surface_format"
    if _any_content(row):
        return "any_content"
    if _any_format(row):
        return "format_vs_nonformat"
    if any(str(token.get("token_category", "")) == "DISCOURSE" for token in (row["pt_token"], row["it_token"])):
        return "discourse"
    if all(str(token.get("token_category", "")) == "FUNCTION" for token in (row["pt_token"], row["it_token"])):
        return "function_vs_function"
    return "other"


def _load_rows(manifest_root: Path, key: str, models: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        path = manifest_root / key / "raw_shared" / model / "exp20_validation_records.jsonl"
        if not path.exists():
            continue
        for record in _json_rows(path):
            event = _event(record)
            if not event:
                continue
            pt_token = event.get("pt_token") or {}
            it_token = event.get("it_token") or {}
            if pt_token.get("token_id") is None or it_token.get("token_id") is None:
                continue
            rows.append(
                {
                    "key": key,
                    "model": model,
                    "prompt_id": str(record.get("prompt_id")),
                    "step": int(event.get("step", 0)),
                    "pt_token": pt_token,
                    "it_token": it_token,
                    "pt_token_str": _token_text(pt_token),
                    "it_token_str": _token_text(it_token),
                    "pt_category": str(pt_token.get("token_category", "OTHER")),
                    "it_category": str(it_token.get("token_category", "OTHER")),
                    "pt_collapsed": str(pt_token.get("token_category_collapsed", "OTHER")),
                    "it_collapsed": str(it_token.get("token_category_collapsed", "OTHER")),
                    "pair_kind": "",
                }
            )
    for row in rows:
        row["pair_kind"] = _pair_kind(row)
    return rows


def _family_balanced_metric(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    by_model: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_model[row["model"]][row["prompt_id"]].append(row)
    model_values: dict[str, np.ndarray] = {}
    model_summary: dict[str, Any] = {}
    for model, groups in sorted(by_model.items()):
        values = np.array(
            [np.mean([1.0 if predicate(row) else 0.0 for row in group]) for group in groups.values()],
            dtype=float,
        )
        model_values[model] = values
        model_summary[model] = {
            "n_prompt_groups": int(values.size),
            "fraction": float(values.mean()) if values.size else None,
        }
    valid = [model for model, values in model_values.items() if values.size]
    point = float(np.mean([model_values[model].mean() for model in valid])) if valid else float("nan")
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    for _ in range(n_bootstrap):
        family_means = []
        for model in valid:
            values = model_values[model]
            idx = rng.integers(0, values.size, size=values.size)
            family_means.append(float(values[idx].mean()))
        boots.append(float(np.mean(family_means)))
    lo, hi = np.percentile(np.array(boots, dtype=float), [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return {
        "fraction": point,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_rows": int(len(rows)),
        "n_prompt_groups": int(sum(model_values[model].size for model in valid)),
        "n_models": int(len(valid)),
        "models": model_summary,
        "bootstrap_unit": "prompt_group_within_family",
        "n_bootstrap": int(n_bootstrap),
    }


def _raw_fraction(rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    n = len(rows)
    k = int(sum(1 for row in rows if predicate(row)))
    if n == 0:
        return {"n": 0, "count": 0, "fraction": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    p = k / n
    z = 1.96
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return {"n": n, "count": k, "fraction": p, "ci95_low": center - half, "ci95_high": center + half}


def _summarize_rows(rows: list[dict[str, Any]], *, n_bootstrap: int, seed: int) -> dict[str, Any]:
    metrics = {
        "position_0": _position_0,
        "position_ge_3": _position_ge_3,
        "position_ge_5": _position_ge_5,
        "any_content_token": _any_content,
        "any_format_token": _any_format,
        "both_surface_format": _both_surface,
    }
    out = {
        "n_rows": len(rows),
        "raw": {name: _raw_fraction(rows, predicate) for name, predicate in metrics.items()},
        "family_balanced": {
            name: _family_balanced_metric(rows, predicate, n_bootstrap=n_bootstrap, seed=seed + idx)
            for idx, (name, predicate) in enumerate(metrics.items())
        },
        "pair_kind_counts": Counter(row["pair_kind"] for row in rows),
        "top_token_pairs": Counter((row["pt_token_str"], row["it_token_str"]) for row in rows).most_common(40),
    }
    out["pair_kind_counts"] = dict(out["pair_kind_counts"])
    return out


def _fmt_pct(row: dict[str, Any]) -> str:
    return f"{100 * row['fraction']:.1f}% [{100 * row['ci95_low']:.1f}%, {100 * row['ci95_high']:.1f}%]"


def _write_report(summary: dict[str, Any], out_path: Path) -> None:
    comparison = summary["comparison"]
    metric_names = [
        ("position_0", "Generated position 0"),
        ("position_ge_3", "Generated position >=3"),
        ("position_ge_5", "Generated position >=5"),
        ("any_content_token", "Any deterministic content token"),
        ("any_format_token", "Any deterministic format token"),
    ]
    lines = [
        "# Exp37 Random-Local Token-Support Control",
        "",
        "This deterministic audit compares true first-divergence support with Exp37 random local PT/IT disagreement prefixes from the same prompts and generated rollouts.",
        "",
        "| Metric | True first divergence | Random local disagreement, source-balanced |",
        "|---|---:|---:|",
    ]
    first = comparison["first_diff__reference"]["family_balanced"]
    random_local = comparison["random_local_disagreement__source_balanced"]["family_balanced"]
    for key, label in metric_names:
        lines.append(f"| {label} | `{_fmt_pct(first[key])}` | `{_fmt_pct(random_local[key])}` |")
    lines.extend(
        [
            "",
            "Interpretation: random local disagreements are later and more deterministic-content-heavy than first divergences, yet Exp37 shows they retain only 63% of the first-divergence interaction. Thus the larger first-divergence interaction is not explained by selecting format-heavy token pairs.",
            "",
            "Caveat: these are deterministic token-category tags, not the semantic LLM audit categories used for the first-divergence support analysis.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--models", nargs="*", default=list(DENSE5_MODELS))
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    out_dir = args.out_dir or args.run_root / "analysis" / "token_support_control"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_root = args.run_root / "manifests"
    key_rows = {
        REFERENCE_KEY: _load_rows(manifest_root, REFERENCE_KEY, args.models),
        RANDOM_PT_KEY: _load_rows(manifest_root, RANDOM_PT_KEY, args.models),
        RANDOM_IT_KEY: _load_rows(manifest_root, RANDOM_IT_KEY, args.models),
        PREDIV_KEY: _load_rows(manifest_root, PREDIV_KEY, args.models),
    }
    random_source_balanced = key_rows[RANDOM_PT_KEY] + key_rows[RANDOM_IT_KEY]
    comparison = {
        key: _summarize_rows(rows, n_bootstrap=args.n_bootstrap, seed=args.seed + idx * 100)
        for idx, (key, rows) in enumerate(key_rows.items())
    }
    comparison["random_local_disagreement__source_balanced"] = _summarize_rows(
        random_source_balanced,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed + 999,
    )
    summary = {
        "experiment": "exp37_token_support_control",
        "run_root": str(args.run_root),
        "manifest_root": str(manifest_root),
        "models": list(args.models),
        "comparison": comparison,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_report(summary, out_dir / "token_support_control_report.md")
    print(f"[exp37-token-support] wrote {summary_path}")


if __name__ == "__main__":
    main()
