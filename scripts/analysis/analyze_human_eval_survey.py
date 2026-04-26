#!/usr/bin/env python3
"""Analyze blinded human survey returns for the Exp15 behavioral bridge.

Outputs are intentionally compact and paper-facing:
  - pairwise human resolved win rates with prompt-cluster bootstrap CIs;
  - human inter-rater agreement / Cohen kappa;
  - LLM-judge rates on the same sampled pairwise items;
  - pointwise inter-rater agreement and human-vs-judge agreement.
"""

from __future__ import annotations

import argparse
import csv
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
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
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
    """Bootstrap over pair_ids; each sampled item contributes both rater votes."""
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

    human_rows: list[dict[str, Any]] = []
    for rater_id in ["R1", "R2"]:
        for row in read_csv(survey_dir / "pairwise" / f"pairwise_primary_{rater_id}.csv"):
            key = key_by_rater_pair[(rater_id, row["pair_id"])]
            norm = normalize_pairwise_choice(row, key)
            direction = directional_label(norm, key["comparison"])
            human_rows.append(
                {
                    **key,
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
            for rater_id in ["R1", "R2"]:
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
            paired = [
                value
                for value in by_pair_all.values()
                if "R1" in value
                and "R2" in value
                and value["R1"]["comparison"] == comparison
                and value["R1"]["criterion"] == criterion
            ]
            r1 = [value["R1"]["dir"] for value in paired]
            r2 = [value["R2"]["dir"] for value in paired]
            r1_collapsed = ["unresolved" if x in {"tie", "both_bad"} else x for x in r1]
            r2_collapsed = ["unresolved" if x in {"tie", "both_bad"} else x for x in r2]
            agreement_rows.append(
                {
                    "comparison": comparison,
                    "criterion": criterion,
                    "n_items": len(paired),
                    "raw_agreement_4label": agreement(r1, r2),
                    "kappa_4label": cohen_kappa(
                        r1,
                        r2,
                        labels=["target", "other", "tie", "both_bad"],
                    ),
                    "raw_agreement_collapsed": agreement(r1_collapsed, r2_collapsed),
                    "kappa_collapsed": cohen_kappa(
                        r1_collapsed,
                        r2_collapsed,
                        labels=["target", "other", "unresolved"],
                    ),
                }
            )

    return summary_rows, {"rows": human_rows}, agreement_rows


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
    for rater_id in ["R1", "R2"]:
        for row in read_csv(survey_dir / "pointwise" / f"pointwise_{rater_id}.csv"):
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
        paired = [value for key, value in by_item_task.items() if key[1] == task and "R1" in value and "R2" in value]
        r1 = [value["R1"] for value in paired]
        r2 = [value["R2"] for value in paired]
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
                "n_common_r1_r2": len(paired),
                "r1_r2_agreement": agreement(r1, r2),
                "r1_r2_kappa": cohen_kappa(r1, r2, labels=labels, quadratic_weights=weighted),
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
            "pairwise_rate_definition": "Resolved rate excludes TIE and BOTH_BAD votes; CIs bootstrap over pair_id and include both raters per sampled item.",
            "pointwise_role": "Pointwise labels are used for agreement and judge-validation diagnostics; pairwise labels are the primary confirmatory human behavioral readout.",
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
            "human_resolved_n",
            "human_resolved_target_rate",
            "human_resolved_target_ci95",
            "unresolved_rate",
            "llm_same_sample_resolved_target_rate",
            "human_counts",
            "llm_same_sample_counts",
            "R1_resolved_rate",
            "R2_resolved_rate",
        ],
    )
    write_csv(
        args.out_dir / "pairwise_interrater.csv",
        pairwise_agreement,
        [
            "comparison",
            "criterion",
            "n_items",
            "raw_agreement_4label",
            "kappa_4label",
            "raw_agreement_collapsed",
            "kappa_collapsed",
        ],
    )
    write_csv(
        args.out_dir / "pointwise_agreement.csv",
        pointwise_summary,
        [
            "task",
            "n_common_r1_r2",
            "r1_r2_agreement",
            "r1_r2_kappa",
            "n_human_llm_ratings",
            "human_llm_agreement",
            "human_llm_kappa",
        ],
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
