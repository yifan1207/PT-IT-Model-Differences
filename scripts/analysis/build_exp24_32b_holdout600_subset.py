#!/usr/bin/env python3
"""Build the Qwen2.5-32B holdout-600 support synthesis.

The original Exp24 32B run used the full 1400-prompt support.  For the paper's
Core-5 headline we use the same governance/safety holdout support as the four
smaller families: 300 GOV-CONV, 150 GOV-FORMAT, and 150 SAFETY prompts.  The
32B raw run has one SAFETY prompt without a valid first-divergence event, so the
paper-facing subset contains 599 valid events.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_RECORDS = Path(
    "results/paper_synthesis/exp24_32b_external_validity/"
    "qwen25_32b_residual_factorial_records.jsonl.gz"
)
DEFAULT_FULL_SUMMARY = Path(
    "results/exp24_32b_external_validity/"
    "exp24_qwen25_32b_full_eval_v21_20260427_194839/"
    "analysis/exp23_midlate_interaction_suite/exp23_summary.json"
)
DEFAULT_OUT_DIR = Path("results/paper_synthesis/exp24_32b_external_validity")

MODEL = "qwen25_32b"
READOUTS = ("common_it", "common_pt")
EFFECTS = (
    "late_it_given_pt_upstream",
    "late_it_given_it_upstream",
    "late_weight_effect",
    "upstream_context_effect",
    "interaction",
)
POSITION_FILTERS = (
    ("all", "all positions", lambda step: True),
    ("step_0", "position 0", lambda step: step == 0),
    ("step_ge1", "positions >=1", lambda step: step >= 1),
    ("step_ge3", "positions >=3", lambda step: step >= 3),
    ("step_ge5", "position >=5", lambda step: step >= 5),
    ("step_ge10", "position >=10", lambda step: step >= 10),
)
TARGET_COUNTS = {"GOV-CONV": 300, "GOV-FORMAT": 150, "SAFETY": 150}


def _prompt_category(prompt_id: str) -> str:
    parts = prompt_id.split("_")
    if len(parts) < 3:
        return "UNKNOWN"
    return "_".join(parts[1:-1])


def _load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with gzip.open(path, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            first_diff = row.get("events", {}).get("first_diff", {})
            if first_diff.get("valid"):
                row["prompt_category"] = _prompt_category(row["prompt_id"])
                row["step"] = int(first_diff["event"]["step"])
                records.append(row)
    return records


def _select_holdout(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_category[row["prompt_category"]].append(row)
    selected = []
    for category, n_target in TARGET_COUNTS.items():
        rows = sorted(by_category.get(category, []), key=lambda row: row["prompt_id"])
        selected.extend(rows[:n_target])
    return sorted(selected, key=lambda row: row["prompt_id"])


def _margin(row: dict[str, Any], cell: str, readout: str) -> float:
    return float(row["events"]["first_diff"]["cells"][cell][readout]["it_vs_pt_margin"])


def _event_effects(row: dict[str, Any], readout: str) -> dict[str, float]:
    a = _margin(row, "U_PT__L_PT", readout)
    b = _margin(row, "U_PT__L_IT", readout)
    c = _margin(row, "U_IT__L_PT", readout)
    d = _margin(row, "U_IT__L_IT", readout)
    late_pt = b - a
    late_it = d - c
    return {
        "late_it_given_pt_upstream": late_pt,
        "late_it_given_it_upstream": late_it,
        "late_weight_effect": 0.5 * (late_pt + late_it),
        "upstream_context_effect": 0.5 * ((c - a) + (d - b)),
        "interaction": late_it - late_pt,
    }


def _summarize(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "estimate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "n_units": 0,
            "n_prompt_clusters": 0,
            "n_boot": 0,
        }
    estimate = float(np.mean(values))
    if values.size == 1 or n_boot <= 0:
        lo = hi = estimate
        n_used = 0
    else:
        idx = rng.integers(0, values.size, size=(n_boot, values.size))
        means = np.mean(values[idx], axis=1)
        lo, hi = np.quantile(means, [0.025, 0.975])
        n_used = n_boot
    return {
        "estimate": estimate,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "n_units": int(values.size),
        "n_prompt_clusters": int(values.size),
        "n_boot": int(n_used),
    }


def _effect_matrix(records: list[dict[str, Any]], readout: str) -> dict[str, np.ndarray]:
    rows = [_event_effects(row, readout) for row in records]
    return {effect: np.array([row[effect] for row in rows], dtype=float) for effect in EFFECTS}


def _summary_json(records: list[dict[str, Any]], full_summary_path: Path, n_boot: int) -> dict[str, Any]:
    template = json.loads(full_summary_path.read_text())
    rng = np.random.default_rng(20260504)
    effects: dict[str, dict[str, Any]] = {}
    for readout in READOUTS:
        matrices = _effect_matrix(records, readout)
        effects[readout] = {}
        for effect, values in matrices.items():
            stats = _summarize(values, rng, n_boot)
            effects[readout][effect] = {
                "effect": effect,
                "readout": readout,
                "estimate": stats["estimate"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
                "n_models": 1,
                "n_units": stats["n_units"],
                "n_prompt_clusters": stats["n_prompt_clusters"],
                "n_boot": stats["n_boot"],
                "bootstrap_unit": "prompt_cluster",
                "models": {MODEL: stats["estimate"]},
                "model_cis": {MODEL: stats},
                "leave_one_out": {},
            }
    template["residual_factorial"]["effects"] = effects
    template["residual_factorial"]["n_units_by_model"] = {MODEL: len(records)}
    template["residual_factorial"]["n_units_total_primary_readout"] = len(records)
    template["residual_factorial"]["quality"] = {
        "subset": "holdout600_governance_safety",
        "target_counts": TARGET_COUNTS,
        "valid_counts": dict(Counter(row["prompt_category"] for row in records)),
        "n_valid_events": len(records),
        "note": "One SAFETY prompt lacks a valid first-divergence event, so the subset has 599 valid events.",
    }
    return template


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _position_rows(records: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    rows = []
    for filter_name, label, predicate in POSITION_FILTERS:
        subset = [row for row in records if predicate(row["step"])]
        matrices = _effect_matrix(subset, "common_it") if subset else {}
        rng = np.random.default_rng(20260504 + len(rows))
        for metric in ("interaction", "late_it_given_pt_upstream", "late_it_given_it_upstream"):
            stats = _summarize(matrices.get(metric, np.array([], dtype=float)), rng, n_boot)
            rows.append(
                {
                    "position_filter": filter_name,
                    "position_label": label,
                    "metric": metric.replace("late_it_given_pt_upstream", "late_given_pt").replace(
                        "late_it_given_it_upstream", "late_given_it"
                    ),
                    "estimate": stats["estimate"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "n_records": stats["n_units"],
                    "n_prompt_clusters": stats["n_prompt_clusters"],
                }
            )
    return rows


def _category_rows(records: list[dict[str, Any]], n_boot: int) -> list[dict[str, Any]]:
    rows = []
    for category in sorted(set(row["prompt_category"] for row in records)):
        subset = [row for row in records if row["prompt_category"] == category]
        matrices = _effect_matrix(subset, "common_it")
        rng = np.random.default_rng(20260504 + len(rows))
        for metric in ("interaction", "late_it_given_pt_upstream", "late_it_given_it_upstream"):
            stats = _summarize(matrices[metric], rng, n_boot)
            rows.append(
                {
                    "prompt_category": category,
                    "metric": metric.replace("late_it_given_pt_upstream", "late_given_pt").replace(
                        "late_it_given_it_upstream", "late_given_it"
                    ),
                    "estimate": stats["estimate"],
                    "ci95_low": stats["ci95_low"],
                    "ci95_high": stats["ci95_high"],
                    "n_records": stats["n_units"],
                    "n_prompt_clusters": stats["n_prompt_clusters"],
                }
            )
    return rows


def _summary_csv(summary: dict[str, Any]) -> list[dict[str, Any]]:
    common_it = summary["residual_factorial"]["effects"]["common_it"]
    return [
        {
            "model": MODEL,
            "subset": "holdout600_governance_safety",
            "n_first_divergence_records": common_it["interaction"]["n_units"],
            "pt_upstream_late_effect": common_it["late_it_given_pt_upstream"]["estimate"],
            "pt_upstream_ci_low": common_it["late_it_given_pt_upstream"]["ci95_low"],
            "pt_upstream_ci_high": common_it["late_it_given_pt_upstream"]["ci95_high"],
            "it_upstream_late_effect": common_it["late_it_given_it_upstream"]["estimate"],
            "it_upstream_ci_low": common_it["late_it_given_it_upstream"]["ci95_low"],
            "it_upstream_ci_high": common_it["late_it_given_it_upstream"]["ci95_high"],
            "interaction": common_it["interaction"]["estimate"],
            "interaction_ci_low": common_it["interaction"]["ci95_low"],
            "interaction_ci_high": common_it["interaction"]["ci95_high"],
            "ratio_descriptive": common_it["late_it_given_it_upstream"]["estimate"]
            / common_it["late_it_given_pt_upstream"]["estimate"],
            "portable_share": common_it["late_it_given_pt_upstream"]["estimate"]
            / common_it["late_it_given_it_upstream"]["estimate"],
        }
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--full-summary", type=Path, default=DEFAULT_FULL_SUMMARY)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-boot", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = _select_holdout(_load_records(args.records))
    summary = _summary_json(records, args.full_summary, args.n_boot)
    (args.out_dir / "exp24_32b_holdout600_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(args.out_dir / "exp24_32b_holdout600_summary.csv", _summary_csv(summary))
    _write_csv(args.out_dir / "exp24_32b_holdout600_position_sensitivity.csv", _position_rows(records, args.n_boot))
    _write_csv(args.out_dir / "exp24_32b_holdout600_prompt_category_effects.csv", _category_rows(records, args.n_boot))
    counts = Counter(row["prompt_category"] for row in records)
    (args.out_dir / "exp24_32b_holdout600_note.md").write_text(
        "# Exp24 Qwen2.5-32B Holdout-600 Subset\n\n"
        "This CPU-only synthesis restricts the 32B residual-factorial run to the paper's holdout support: "
        "`GOV-CONV/GOV-FORMAT/SAFETY = 300/150/150`. One SAFETY prompt has no valid first-divergence event, "
        f"so the exact analyzed subset is `{len(records)}` events: `{dict(counts)}`.\n\n"
        "The full-1400 run remains an audit artifact, but the paper-facing Core-5 synthesis uses this matched support.\n"
    )
    print(f"[exp24-holdout600] wrote {args.out_dir} with {len(records)} valid events")


if __name__ == "__main__":
    main()
