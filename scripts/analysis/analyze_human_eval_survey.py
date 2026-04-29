#!/usr/bin/env python3
"""Analyze blinded human survey returns for the Exp15 behavioral sanity check.

Outputs are intentionally compact and paper-facing:
  - pairwise human resolved win rates with prompt-cluster bootstrap CIs;
  - human inter-rater agreement / Cohen kappa;
  - LLM-judge rates on the same sampled pairwise items;
  - pointwise inter-rater agreement and human-vs-judge agreement.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
TARGET_CONDITION = {
    "pt_late_vs_a": "B_late_raw",
    "it_c_vs_dlate": "C_it_chat",
}
OTHER_CONDITION = {
    "pt_late_vs_a": "A_pt_raw",
    "it_c_vs_dlate": "D_late_ptswap",
}
COMPARISON_LABEL = {
    "pt_late_vs_a": "PT late graft vs PT baseline",
    "it_c_vs_dlate": "IT baseline vs late PT swap",
}
PAIRWISE_LABELS = ["target", "other", "tie", "both_bad"]
PAIRWISE_COLLAPSED_LABELS = ["target", "other", "unresolved"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--survey-dir",
        type=Path,
        default=Path("paper_draft/human_eval_survey"),
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("results/exp15_symmetric_behavioral_causality/data"),
    )
    parser.add_argument("--run-prefix", default="exp15_eval_core_600_t512_dense5")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/exp15_symmetric_behavioral_causality/human_eval"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def model_dir(run_root: Path, run_prefix: str, model: str) -> Path:
    return run_root / f"{run_prefix}_{model}" / f"{run_prefix}_{model}"


def cohen_kappa(
    a: list[str],
    b: list[str],
    *,
    labels: list[str],
    quadratic_weights: bool = False,
) -> float:
    pairs = [(x, y) for x, y in zip(a, b) if x in labels and y in labels]
    if not pairs:
        return float("nan")
    idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    observed = np.zeros((n_labels, n_labels), dtype=float)
    for x, y in pairs:
        observed[idx[x], idx[y]] += 1.0
    total = observed.sum()
    if total == 0:
        return float("nan")
    observed /= total
    row_marg = observed.sum(axis=1)
    col_marg = observed.sum(axis=0)
    expected = np.outer(row_marg, col_marg)

    if quadratic_weights:
        weights = np.zeros((n_labels, n_labels), dtype=float)
        denom = max(1, n_labels - 1)
        for i in range(n_labels):
            for j in range(n_labels):
                weights[i, j] = ((i - j) / denom) ** 2
    else:
        weights = np.ones((n_labels, n_labels), dtype=float) - np.eye(n_labels)

    expected_disagreement = float((weights * expected).sum())
    observed_disagreement = float((weights * observed).sum())
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def agreement(a: list[str], b: list[str]) -> float:
    if not a:
        return float("nan")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def _mean_pairwise_metric(
    by_item: dict[str, dict[str, str]],
    *,
    raters: list[str],
    labels: list[str],
    metric: str,
    quadratic_weights: bool = False,
) -> float:
    values = []
    for left, right in itertools.combinations(raters, 2):
        left_values = []
        right_values = []
        for item in by_item.values():
            if left in item and right in item:
                left_values.append(item[left])
                right_values.append(item[right])
        if not left_values:
            continue
        if metric == "agreement":
            values.append(agreement(left_values, right_values))
        elif metric == "kappa":
            values.append(
                cohen_kappa(
                    left_values,
                    right_values,
                    labels=labels,
                    quadratic_weights=quadratic_weights,
                )
            )
        else:
            raise ValueError(f"unknown metric: {metric}")
    kept = [value for value in values if math.isfinite(value)]
    return float(np.mean(kept)) if kept else float("nan")


def fleiss_kappa(
    by_item: dict[str, dict[str, str]],
    *,
    raters: list[str],
    labels: list[str],
) -> float:
    """Nominal Fleiss' kappa over items with all requested raters present."""
    rows = []
    for item in by_item.values():
        if not all(rater in item and item[rater] in labels for rater in raters):
            continue
        counts = Counter(item[rater] for rater in raters)
        rows.append([counts[label] for label in labels])
    if not rows:
        return float("nan")
    counts_arr = np.asarray(rows, dtype=float)
    n_raters = counts_arr.sum(axis=1)
    if not np.all(n_raters == n_raters[0]) or n_raters[0] <= 1:
        return float("nan")
    n = float(n_raters[0])
    p_i = ((counts_arr * counts_arr).sum(axis=1) - n) / (n * (n - 1.0))
    p_bar = float(p_i.mean())
    p_j = counts_arr.sum(axis=0) / counts_arr.sum()
    p_e = float((p_j * p_j).sum())
    if p_e == 1.0:
        return 1.0 if p_bar == 1.0 else float("nan")
    return (p_bar - p_e) / (1.0 - p_e)


def _rater_id_from_path(path: Path, prefix: str) -> str:
    stem = path.stem
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def normalize_pairwise_choice(row: dict[str, str], key: dict[str, str]) -> str:
    choice = (row.get("choice") or "").strip().upper()
    if choice == "A":
        return key["condition_a"]
    if choice == "B":
        return key["condition_b"]
    if choice == "TIE":
        return "TIE"
    if choice == "BOTH_BAD":
        return "BOTH_BAD"
    return "MISSING"


def directional_label(condition_label: str, comparison: str) -> str:
    if condition_label == TARGET_CONDITION[comparison]:
        return "target"
    if condition_label == OTHER_CONDITION[comparison]:
        return "other"
    if condition_label == "TIE":
        return "tie"
    if condition_label == "BOTH_BAD":
        return "both_bad"
    return "missing"


def bootstrap_resolved_rate(
    item_votes: list[list[dict[str, str]]],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap over pair_ids; each sampled item contributes all completed rater votes."""
    if not item_votes:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values = []
    n_items = len(item_votes)
    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        target = 0
        resolved = 0
        for i in idx:
            for vote in item_votes[i]:
                if vote["dir"] in {"target", "other"}:
                    resolved += 1
                    target += int(vote["dir"] == "target")
        values.append(target / resolved if resolved else np.nan)
    arr = np.asarray(values, dtype=float)
    return tuple(float(x) for x in np.nanpercentile(arr, [2.5, 97.5]))


def summarize_pairwise(
    *,
    survey_dir: Path,
    run_root: Path,
    run_prefix: str,
    n_boot: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    key_rows = read_csv(survey_dir / "keys" / "pairwise_hidden_key.csv")
    key_by_rater_pair = {(row["rater_id"], row["pair_id"]): row for row in key_rows}
    key_by_pair_signature: dict[tuple[str, str, str], dict[str, str]] = {}

    pairwise_paths = sorted((survey_dir / "pairwise").glob("pairwise_primary_R*.csv"))
    survey_rows_by_rater: dict[str, list[dict[str, str]]] = {}
    for path in pairwise_paths:
        rows = read_csv(path)
        if not rows:
            continue
        rater_id = (rows[0].get("rater_id") or _rater_id_from_path(path, "pairwise_primary_")).strip()
        survey_rows_by_rater[rater_id] = rows

    # R3/R4 packets may be generated from an existing visible packet without
    # adding new hidden-key rows.  Decode choices by the actual A/B response
    # order, not by pair_id alone, because A/B order is rater-randomized.
    for rater_id, rows in survey_rows_by_rater.items():
        by_pair = {row["pair_id"]: row for row in rows}
        for pair_id in by_pair:
            key = key_by_rater_pair.get((rater_id, pair_id))
            if key is None:
                continue
            row = by_pair[pair_id]
            signature = (pair_id, row["response_a"], row["response_b"])
            key_by_pair_signature[signature] = key

    human_rows: list[dict[str, Any]] = []
    raters = sorted(survey_rows_by_rater)
    for rater_id in raters:
        for row in survey_rows_by_rater[rater_id]:
            key = key_by_rater_pair.get((rater_id, row["pair_id"]))
            if key is None:
                key = key_by_pair_signature.get((row["pair_id"], row["response_a"], row["response_b"]))
            if key is None:
                raise KeyError(
                    f"No hidden key for rater={rater_id} pair={row['pair_id']} "
                    "with this response_a/response_b order."
                )
            norm = normalize_pairwise_choice(row, key)
            direction = directional_label(norm, key["comparison"])
            human_rows.append(
                {
                    **key,
                    "key_rater_id": key.get("rater_id", ""),
                    "rater_id": rater_id,
                    "choice": (row.get("choice") or "").strip().upper(),
                    "confidence": row.get("confidence", ""),
                    "notes": row.get("notes", ""),
                    "normalized_label": norm,
                    "dir": direction,
                }
            )

    judge_by_pair_id: dict[str, dict[str, Any]] = {}
    for model in MODELS:
        for row in load_jsonl(model_dir(run_root, run_prefix, model) / "judge_pairwise.jsonl"):
            pair_id = safe_id(f"pair_{model}_{row['item_id']}")
            judge_by_pair_id[pair_id] = row

    summary_rows: list[dict[str, Any]] = []
    for comparison in ["it_c_vs_dlate", "pt_late_vs_a"]:
        for criterion in ["G2", "S2"]:
            subset = [
                row
                for row in human_rows
                if row["comparison"] == comparison and row["criterion"] == criterion
            ]
            counts = Counter(row["dir"] for row in subset)
            resolved_n = counts["target"] + counts["other"]
            resolved_rate = counts["target"] / resolved_n if resolved_n else float("nan")

            by_pair: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in subset:
                by_pair[row["pair_id"]].append({"dir": row["dir"]})
            ci_low, ci_high = bootstrap_resolved_rate(
                list(by_pair.values()),
                n_boot=n_boot,
                seed=1000 + 17 * len(summary_rows),
            )

            rater_rates = {}
            for rater_id in raters:
                rater_subset = [row for row in subset if row["rater_id"] == rater_id]
                rater_counts = Counter(row["dir"] for row in rater_subset)
                rater_resolved = rater_counts["target"] + rater_counts["other"]
                rater_rates[f"{rater_id}_resolved_rate"] = (
                    rater_counts["target"] / rater_resolved if rater_resolved else float("nan")
                )
                rater_rates[f"{rater_id}_counts"] = dict(rater_counts)

            sampled_pair_ids = sorted({row["pair_id"] for row in subset})
            judge_dirs = []
            for pair_id in sampled_pair_ids:
                judge = judge_by_pair_id.get(pair_id)
                preferred = judge.get("preferred_condition", "MISSING") if judge else "MISSING"
                judge_dirs.append(directional_label(preferred, comparison))
            judge_counts = Counter(judge_dirs)
            judge_resolved = judge_counts["target"] + judge_counts["other"]

            summary_rows.append(
                {
                    "comparison": comparison,
                    "comparison_label": COMPARISON_LABEL[comparison],
                    "criterion": criterion,
                    "target_condition": TARGET_CONDITION[comparison],
                    "other_condition": OTHER_CONDITION[comparison],
                    "n_items": len(sampled_pair_ids),
                    "n_votes": len(subset),
                    "n_raters": len(raters),
                    "raters": raters,
                    "human_counts": dict(counts),
                    "human_resolved_n": resolved_n,
                    "human_resolved_target_rate": resolved_rate,
                    "human_resolved_target_ci95": [ci_low, ci_high],
                    "unresolved_rate": (counts["tie"] + counts["both_bad"]) / len(subset),
                    "llm_same_sample_counts": dict(judge_counts),
                    "llm_same_sample_resolved_target_rate": (
                        judge_counts["target"] / judge_resolved if judge_resolved else float("nan")
                    ),
                    **rater_rates,
                }
            )

    agreement_rows: list[dict[str, Any]] = []
    by_pair_all: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in human_rows:
        by_pair_all[row["pair_id"]][row["rater_id"]] = row

    for comparison in ["it_c_vs_dlate", "pt_late_vs_a"]:
        for criterion in ["G2", "S2"]:
            paired = {}
            collapsed = {}
            for pair_id, value in by_pair_all.items():
                any_row = next(iter(value.values()))
                if any_row["comparison"] != comparison or any_row["criterion"] != criterion:
                    continue
                paired[pair_id] = {rater: row["dir"] for rater, row in value.items()}
                collapsed[pair_id] = {
                    rater: ("unresolved" if row["dir"] in {"tie", "both_bad"} else row["dir"])
                    for rater, row in value.items()
                }
            agreement_rows.append(
                {
                    "comparison": comparison,
                    "criterion": criterion,
                    "n_items": len(paired),
                    "n_raters": len(raters),
                    "mean_pairwise_raw_agreement_4label": _mean_pairwise_metric(
                        paired, raters=raters, labels=PAIRWISE_LABELS, metric="agreement"
                    ),
                    "mean_pairwise_kappa_4label": _mean_pairwise_metric(
                        paired, raters=raters, labels=PAIRWISE_LABELS, metric="kappa"
                    ),
                    "mean_pairwise_raw_agreement_collapsed": _mean_pairwise_metric(
                        collapsed, raters=raters, labels=PAIRWISE_COLLAPSED_LABELS, metric="agreement"
                    ),
                    "mean_pairwise_kappa_collapsed": _mean_pairwise_metric(
                        collapsed, raters=raters, labels=PAIRWISE_COLLAPSED_LABELS, metric="kappa"
                    ),
                    "fleiss_kappa_4label": fleiss_kappa(
                        paired, raters=raters, labels=PAIRWISE_LABELS
                    ),
                    "fleiss_kappa_collapsed": fleiss_kappa(
                        collapsed, raters=raters, labels=PAIRWISE_COLLAPSED_LABELS
                    ),
                }
            )

    return summary_rows, {"rows": human_rows, "raters": raters}, agreement_rows


def summarize_pointwise(
    *,
    survey_dir: Path,
    run_root: Path,
    run_prefix: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    key_rows = read_csv(survey_dir / "keys" / "pointwise_hidden_key.csv")
    key_by_survey = {row["survey_id"]: row for row in key_rows}

    judge_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for model in MODELS:
        for row in load_jsonl(model_dir(run_root, run_prefix, model) / "judge_pointwise_final.jsonl"):
            judge_by_key[(model, row["condition"], row["record_id"], row["task"])] = row

    ratings: list[dict[str, Any]] = []
    pointwise_paths = sorted((survey_dir / "pointwise").glob("pointwise_R*.csv"))
    raters: list[str] = []
    for path in pointwise_paths:
        rows = read_csv(path)
        if not rows:
            continue
        rater_id = (rows[0].get("rater_id") or _rater_id_from_path(path, "pointwise_")).strip()
        raters.append(rater_id)
        for row in rows:
            meta = key_by_survey[row["survey_id"]]
            for task in ["g1", "g2", "s1", "s2"]:
                value = (row.get(task) or "").strip().upper()
                if not value:
                    continue
                judge = judge_by_key.get((meta["model"], meta["condition"], meta["record_id"], task))
                ratings.append(
                    {
                        **meta,
                        "rater_id": rater_id,
                        "task": task,
                        "value": value,
                        "confidence": row.get("confidence", ""),
                        "judge_value": str(judge.get("label", judge.get("numeric_score", ""))).upper()
                        if judge
                        else "",
                        "judge_numeric": judge.get("numeric_score") if judge else None,
                    }
                )

    by_item_task: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for row in ratings:
        by_item_task[(row["survey_id"], row["task"])][row["rater_id"]] = row["value"]

    summary_rows: list[dict[str, Any]] = []
    for task in ["g1", "g2", "s1", "s2"]:
        paired = {
            key[0]: value
            for key, value in by_item_task.items()
            if key[1] == task and sum(rater in value for rater in raters) >= 2
        }
        if task in {"g1", "g2"}:
            labels = ["1", "2", "3", "4", "5"]
            weighted = True
        elif task == "s1":
            labels = ["COMPLY", "INCOHERENT", "REFUSE"]
            weighted = False
        else:
            labels = ["0", "1"]
            weighted = False

        human_llm_rows = [row for row in ratings if row["task"] == task and row["judge_value"]]
        human_values = [row["value"] for row in human_llm_rows]
        judge_values = [
            str(int(row["judge_numeric"])) if task in {"g1", "g2", "s2"} and row["judge_numeric"] is not None else row["judge_value"]
            for row in human_llm_rows
        ]

        summary_rows.append(
            {
                "task": task,
                "n_items": len(paired),
                "n_raters": len(raters),
                "mean_pairwise_agreement": _mean_pairwise_metric(
                    paired, raters=raters, labels=labels, metric="agreement"
                ),
                "mean_pairwise_kappa": _mean_pairwise_metric(
                    paired,
                    raters=raters,
                    labels=labels,
                    metric="kappa",
                    quadratic_weights=weighted,
                ),
                "fleiss_kappa": float("nan") if weighted else fleiss_kappa(
                    paired, raters=raters, labels=labels
                ),
                "n_human_llm_ratings": len(human_llm_rows),
                "human_llm_agreement": agreement(human_values, judge_values),
                "human_llm_kappa": cohen_kappa(
                    human_values,
                    judge_values,
                    labels=labels,
                    quadratic_weights=weighted,
                ),
            }
        )

    return summary_rows, ratings


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairwise_summary, pairwise_detail, pairwise_agreement = summarize_pairwise(
        survey_dir=args.survey_dir,
        run_root=args.run_root,
        run_prefix=args.run_prefix,
        n_boot=args.bootstrap_samples,
    )
    pointwise_summary, pointwise_ratings = summarize_pointwise(
        survey_dir=args.survey_dir,
        run_root=args.run_root,
        run_prefix=args.run_prefix,
    )

    summary = {
        "pairwise_primary": pairwise_summary,
        "pairwise_interrater": pairwise_agreement,
        "pointwise_interrater_and_judge": pointwise_summary,
        "notes": {
            "pairwise_rate_definition": "Resolved rate excludes TIE and BOTH_BAD votes; CIs bootstrap over pair_id and include all completed rater votes per sampled item.",
            "pointwise_role": "Pointwise labels are used for agreement and judge-validation diagnostics; pairwise labels are the primary diagnostic human behavioral readout.",
        },
    }

    (args.out_dir / "human_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(
        args.out_dir / "pairwise_primary_summary.csv",
        pairwise_summary,
        [
            "comparison",
            "comparison_label",
            "criterion",
            "target_condition",
            "other_condition",
            "n_items",
            "n_votes",
            "n_raters",
            "raters",
            "human_resolved_n",
            "human_resolved_target_rate",
            "human_resolved_target_ci95",
            "unresolved_rate",
            "llm_same_sample_resolved_target_rate",
            "human_counts",
            "llm_same_sample_counts",
            "R1_resolved_rate",
            "R2_resolved_rate",
            "R3_resolved_rate",
            "R4_resolved_rate",
        ],
    )
    write_csv(
        args.out_dir / "pairwise_interrater.csv",
        pairwise_agreement,
        [
            "comparison",
            "criterion",
            "n_items",
            "n_raters",
            "mean_pairwise_raw_agreement_4label",
            "mean_pairwise_kappa_4label",
            "mean_pairwise_raw_agreement_collapsed",
            "mean_pairwise_kappa_collapsed",
            "fleiss_kappa_4label",
            "fleiss_kappa_collapsed",
        ],
    )
    write_csv(
        args.out_dir / "pointwise_agreement.csv",
        pointwise_summary,
        [
            "task",
            "n_items",
            "n_raters",
            "mean_pairwise_agreement",
            "mean_pairwise_kappa",
            "fleiss_kappa",
            "n_human_llm_ratings",
            "human_llm_agreement",
            "human_llm_kappa",
        ],
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
