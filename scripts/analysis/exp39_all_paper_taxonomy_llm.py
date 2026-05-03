#!/usr/bin/env python3
"""Role-blind paper taxonomy for all Exp39 causal and control features."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAPER_CATEGORIES = [
    "assistant_register_or_user_facing_style",
    "alignment_safety_or_advice_boundary",
    "instruction_following_or_format_control",
    "response_structure_or_answer_readout",
    "evaluation_or_multiple_choice_scaffold",
    "factual_explanation_or_correction_readout",
    "code_or_tool_syntax_readout",
    "surface_punctuation_or_tokenization",
    "dataset_or_repetition_artifact",
    "rare_unicode_or_web_artifact",
    "generic_frequency_or_unclear",
]

BEHAVIOR_CATEGORIES = {
    "assistant_register_or_user_facing_style",
    "alignment_safety_or_advice_boundary",
    "instruction_following_or_format_control",
    "response_structure_or_answer_readout",
    "evaluation_or_multiple_choice_scaffold",
    "factual_explanation_or_correction_readout",
    "code_or_tool_syntax_readout",
}

ARTIFACT_CATEGORIES = {
    "dataset_or_repetition_artifact",
    "rare_unicode_or_web_artifact",
    "generic_frequency_or_unclear",
}

SAFETY_RE = re.compile(
    r"\b("
    r"safety|safe|unsafe|refus\w*|harm\w*|risk\w*|policy|legal|medical|doctor|lawyer|"
    r"advice|caution\w*|disclaimer\w*|danger\w*|weapon|bomb|suicide|self-harm|"
    r"identity theft|illegal|criminal|violence|abuse|exploit|hotline"
    r")\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    def log_choose(n: int, k: int) -> float:
        if k < 0 or k > n:
            return float("-inf")
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)

    row1 = a + b
    row2 = c + d
    col1 = a + c
    total = row1 + row2

    def hypergeom(x: int) -> float:
        return math.exp(log_choose(col1, x) + log_choose(total - col1, row1 - x) - log_choose(total, row1))

    max_x = min(row1, col1)
    p = sum(hypergeom(x) for x in range(a, max_x + 1))
    odds = float("inf") if b * c == 0 and a * d > 0 else ((a * d) / (b * c) if b * c else 0.0)
    return odds, min(1.0, p)


def mean(values: Any) -> float | None:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else None


def chat_json(client: Any, *, model: str, messages: list[dict[str, str]], max_retries: int = 6) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
    }
    if not model.startswith("gpt-5.5"):
        kwargs["temperature"] = 0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**kwargs)
            return json.loads(response.choices[0].message.content or "{}")
        except Exception as exc:
            message = str(exc)
            if "temperature" in kwargs and "Unsupported value: 'temperature'" in message:
                kwargs.pop("temperature", None)
                continue
            transient = any(
                marker in message.lower()
                for marker in ("rate limit", "server error", "temporarily unavailable", "timeout", "connection")
            )
            if attempt + 1 >= max_retries or not transient:
                raise
            time.sleep(min(45.0, 2.0 * (2**attempt)) + random.random())
    raise RuntimeError("unreachable retry state")


def top_examples(row: dict[str, Any], *, per_section: int = 2) -> list[dict[str, str]]:
    out = []
    examples = row.get("examples") or {}
    for section in ("top_it", "contrast_it_over_pt", "top_pt", "contrast_pt_over_it", "near_miss"):
        for ex in examples.get(section, [])[:per_section]:
            context = str(ex.get("context_window") or ex.get("text") or "")
            out.append(
                {
                    "section": section,
                    "token_text": str(ex.get("token_text", "")),
                    "context_excerpt": context[:650],
                }
            )
    safety_hits = []
    for section, items in examples.items():
        if not isinstance(items, list):
            continue
        for ex in items[:20]:
            context = str(ex.get("context_window") or ex.get("text") or "")
            if SAFETY_RE.search(context):
                safety_hits.append(
                    {
                        "section": section,
                        "token_text": str(ex.get("token_text", "")),
                        "context_excerpt": context[:650],
                    }
                )
            if len(safety_hits) >= 4:
                break
        if len(safety_hits) >= 4:
            break
    return out[:12] + safety_hits


def projection_summary(row: dict[str, Any]) -> dict[str, Any]:
    projection = row.get("output_projection") or {}
    return {
        "top_promoted": (projection.get("top_promoted") or [])[:12],
        "top_suppressed": (projection.get("top_suppressed") or [])[:12],
    }


def make_records(labels: list[dict[str, Any]], dashboards: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    records = []
    id_map = {}
    for idx, label in enumerate(labels):
        record_id = f"R{idx:04d}"
        feature_id = str(label["feature_id"])
        id_map[record_id] = feature_id
        dash = dashboards.get(feature_id, {})
        records.append(
            {
                "record_id": record_id,
                "model": label.get("model"),
                "current_specific_label": label.get("specific_label") or label.get("label"),
                "current_domain": label.get("niche_v2_paper_domain") or label.get("coarse_category"),
                "current_niche": label.get("niche_v2_category"),
                "fires_on": label.get("fires_on"),
                "output_effect": label.get("output_effect"),
                "mechanism_hypothesis": label.get("mechanism_hypothesis"),
                "uncertainty_allocation": label.get("uncertainty_allocation"),
                "artifact_risk": label.get("artifact_risk"),
                "confidence": label.get("confidence"),
                "score_mean": dash.get("score_mean"),
                "decoder_norm_ratio_it_pt": dash.get("decoder_norm_ratio_it_pt"),
                "examples": top_examples(dash),
                "output_projection": projection_summary(dash),
            }
        )
    return records, id_map


def classify_batch(client: Any, *, model: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = (
        "You are a senior mechanistic-interpretability human reviewer. You classify feature evidence "
        "for a paper-facing taxonomy. You are role-blind: records may be causal or controls, and you "
        "must not infer role from inclusion. Return only JSON."
    )
    user = json.dumps(
        {
            "task": (
                "Classify each feature into the most paper-useful category. Be favorable to legitimate "
                "instruction-tuned behavior when activation examples or output projection support it, "
                "but do not call rare-token, web-scrape, common-token, or repeated-template artifacts "
                "alignment behavior. Broaden assistant/alignment behavior to include user-facing advice, "
                "refusal/disclaimer/safety/legal/medical boundaries, instruction constraints, response "
                "openers, and answer-structure readout. Split artifact/unclear features into specific "
                "artifact categories instead of leaving them generic."
            ),
            "allowed_categories": PAPER_CATEGORIES,
            "category_guidance": {
                "assistant_register_or_user_facing_style": "Sure/certainly/openers, conversational stance, polite/user-facing response style, advice tone.",
                "alignment_safety_or_advice_boundary": "Refusal, safety disclaimer, unsafe/harm/legal/medical advice caution, risk mitigation, abuse/violence/self-harm/illegal content boundaries.",
                "instruction_following_or_format_control": "Explicit constraints: no commas, all caps/lowercase, language constraints, exact sections, markdown, answer-only, length/count constraints.",
                "response_structure_or_answer_readout": "Q/A fields, answer/explanation delimiters, response paragraph/list/newline boundaries, prompt-to-answer readout.",
                "evaluation_or_multiple_choice_scaffold": "Multiple choice options, Answer: (A), benchmark answer blocks, option delimiters.",
                "factual_explanation_or_correction_readout": "Factual correction, explanation steps, reasoning/rationale, not-answerable, factual answer stance.",
                "code_or_tool_syntax_readout": "Code blocks, Python/docstrings, HTML/markdown syntax with code-like structure, tool/data scaffold.",
                "surface_punctuation_or_tokenization": "Apostrophes, commas, periods, subword/suffix/letter-token boundaries that are mostly surface form.",
                "dataset_or_repetition_artifact": "Copied prompt loops, repeated synthetic Q/A, duplicated templates, degenerate generated text.",
                "rare_unicode_or_web_artifact": "Rare Unicode, mojibake, web scrape, URL, EOT, malformed markup, unused-token projection.",
                "generic_frequency_or_unclear": "Common-token frequency or genuinely mixed/incoherent feature after artifact subtype checks.",
            },
            "return_schema": {
                "assignments": [
                    {
                        "record_id": "opaque id",
                        "paper_category": "one allowed category",
                        "niche_label": "5-12 word paper-facing label",
                        "behavior_score": 0,
                        "paper_use": "showcase | support | weak_support | diagnostic_artifact | reject",
                        "safety_alignment_score": 0,
                        "human_review_confidence": 0.0,
                        "rationale": "one sentence",
                    }
                ]
            },
            "records": records,
        },
        ensure_ascii=False,
    )
    payload = chat_json(
        client,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return list(payload.get("assignments", []))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    causal = [row for row in rows if row.get("role") == "causal"]
    controls = [row for row in rows if row.get("role") == "control"]
    matched = [row for row in controls if row.get("control_kind") == "matched_noncausal"]
    random_active = [row for row in controls if row.get("control_kind") == "random_active_noncausal"]

    category_rows = []
    for cat in PAPER_CATEGORIES:
        c_rows = [row for row in causal if row.get("paper_category") == cat]
        ctrl_rows = [row for row in controls if row.get("paper_category") == cat]
        m_rows = [row for row in matched if row.get("paper_category") == cat]
        r_rows = [row for row in random_active if row.get("paper_category") == cat]
        odds_all, p_all = fisher_exact_greater(len(c_rows), len(causal) - len(c_rows), len(ctrl_rows), len(controls) - len(ctrl_rows))
        odds_matched, p_matched = fisher_exact_greater(len(c_rows), len(causal) - len(c_rows), len(m_rows), len(matched) - len(m_rows))
        category_rows.append(
            {
                "paper_category": cat,
                "n_causal": len(c_rows),
                "n_control": len(ctrl_rows),
                "n_matched_control": len(m_rows),
                "n_random_active_control": len(r_rows),
                "causal_share": len(c_rows) / len(causal),
                "control_share": len(ctrl_rows) / len(controls),
                "matched_control_share": len(m_rows) / len(matched),
                "odds_ratio_vs_all_controls": odds_all,
                "fisher_p_vs_all_controls": p_all,
                "odds_ratio_vs_matched_controls": odds_matched,
                "fisher_p_vs_matched_controls": p_matched,
                "causal_examples": " | ".join(str(row.get("paper_niche_label")) for row in c_rows[:5]),
                "control_examples": " | ".join(str(row.get("paper_niche_label")) for row in ctrl_rows[:5]),
            }
        )

    def set_row(name: str, cats: set[str]) -> dict[str, Any]:
        c_n = sum(row.get("paper_category") in cats for row in causal)
        ctrl_n = sum(row.get("paper_category") in cats for row in controls)
        m_n = sum(row.get("paper_category") in cats for row in matched)
        odds_all, p_all = fisher_exact_greater(c_n, len(causal) - c_n, ctrl_n, len(controls) - ctrl_n)
        odds_matched, p_matched = fisher_exact_greater(c_n, len(causal) - c_n, m_n, len(matched) - m_n)
        return {
            "set": name,
            "n_causal": c_n,
            "n_control": ctrl_n,
            "n_matched_control": m_n,
            "causal_share": c_n / len(causal),
            "control_share": ctrl_n / len(controls),
            "matched_control_share": m_n / len(matched),
            "odds_ratio_vs_all_controls": odds_all,
            "fisher_p_vs_all_controls": p_all,
            "odds_ratio_vs_matched_controls": odds_matched,
            "fisher_p_vs_matched_controls": p_matched,
        }

    set_rows = [
        set_row("behavior_categories", BEHAVIOR_CATEGORIES),
        set_row("artifact_categories", ARTIFACT_CATEGORIES),
        set_row("alignment_safety_or_advice_boundary", {"alignment_safety_or_advice_boundary"}),
        set_row("instruction_or_response_readout", {"instruction_following_or_format_control", "response_structure_or_answer_readout", "evaluation_or_multiple_choice_scaffold"}),
    ]

    model_rows = []
    for model in sorted({str(row.get("model")) for row in rows}):
        for role in ("causal", "control"):
            subset = [row for row in rows if str(row.get("model")) == model and row.get("role") == role]
            if not subset:
                continue
            behavior = sum(row.get("paper_category") in BEHAVIOR_CATEGORIES for row in subset)
            artifact = sum(row.get("paper_category") in ARTIFACT_CATEGORIES for row in subset)
            safety = sum(row.get("paper_category") == "alignment_safety_or_advice_boundary" for row in subset)
            model_rows.append(
                {
                    "model": model,
                    "role": role,
                    "n": len(subset),
                    "behavior_aligned_n": behavior,
                    "behavior_aligned_share": behavior / len(subset),
                    "artifact_or_unclear_n": artifact,
                    "artifact_or_unclear_share": artifact / len(subset),
                    "safety_alignment_n": safety,
                    "safety_alignment_share": safety / len(subset),
                    "category_counts": json.dumps(Counter(row.get("paper_category") for row in subset), sort_keys=True),
                }
            )

    return {
        "n_features": len(rows),
        "n_causal": len(causal),
        "n_control": len(controls),
        "category_rows": category_rows,
        "set_rows": set_rows,
        "model_rows": model_rows,
        "overall_counts_by_role": {
            "causal": dict(Counter(row.get("paper_category") for row in causal)),
            "control": dict(Counter(row.get("paper_category") for row in controls)),
            "matched_noncausal": dict(Counter(row.get("paper_category") for row in matched)),
            "random_active_noncausal": dict(Counter(row.get("paper_category") for row in random_active)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Role-blind Exp39 paper taxonomy for all features")
    parser.add_argument("--out-root", type=Path, default=Path("results/exp39_causal_feature_interpretation"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")

    from openai import OpenAI

    run_dir = args.out_root / args.run_name
    labels_path = run_dir / "autointerp" / "llm_feature_labels_niche_v2.jsonl"
    if not labels_path.exists():
        labels_path = run_dir / "autointerp" / "llm_feature_labels_grouped.jsonl"
    labels = read_jsonl(labels_path)
    labels.sort(key=lambda row: (str(row.get("model")), str(row.get("role")), str(row.get("control_kind")), str(row.get("feature_id"))))
    label_by_feature = {str(row["feature_id"]): row for row in labels}
    dashboards = {str(row.get("feature_id")): row for row in read_jsonl(run_dir / "dashboards" / "feature_dashboards.jsonl")}
    records, id_map = make_records(labels, dashboards)

    out_dir = run_dir / "autointerp"
    partial_path = out_dir / "all_paper_taxonomy_v4.partial.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        for row in read_jsonl(partial_path):
            done[str(row.get("record_id"))] = row

    client = OpenAI()
    pending = [record for record in records if record["record_id"] not in done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        assignments = classify_batch(client, model=args.model, records=batch)
        by_id = {str(row.get("record_id")): row for row in assignments}
        missing = [row["record_id"] for row in batch if row["record_id"] not in by_id]
        if missing:
            raise RuntimeError(f"Missing assignments for {missing[:10]}")
        with partial_path.open("a", encoding="utf-8") as handle:
            for record in batch:
                assignment = by_id[record["record_id"]]
                handle.write(json.dumps(assignment, ensure_ascii=False) + "\n")
                done[record["record_id"]] = assignment
        print(f"[all-taxonomy] classified {len(done)}/{len(records)}")

    rows = []
    for record_id, feature_id in id_map.items():
        label = label_by_feature[feature_id]
        assignment = dict(done[record_id])
        cat = assignment.get("paper_category")
        if cat not in PAPER_CATEGORIES:
            cat = "generic_frequency_or_unclear"
        enriched = dict(label)
        enriched.update(
            {
                "paper_taxonomy_v4_model": args.model,
                "paper_taxonomy_v4_record_id": record_id,
                "paper_category": cat,
                "paper_niche_label": assignment.get("niche_label") or label.get("specific_label") or label.get("label"),
                "paper_behavior_score": assignment.get("behavior_score"),
                "paper_use": assignment.get("paper_use"),
                "paper_safety_alignment_score": assignment.get("safety_alignment_score"),
                "paper_human_review_confidence": assignment.get("human_review_confidence"),
                "paper_taxonomy_rationale": assignment.get("rationale"),
            }
        )
        rows.append(enriched)

    summary = summarize(rows)
    summary.update({"run_name": args.run_name, "model": args.model, "paper_categories": PAPER_CATEGORIES})
    analysis_dir = run_dir / "analysis"
    write_jsonl(out_dir / "all_paper_taxonomy_v4.jsonl", rows)
    write_json(analysis_dir / "exp39_all_paper_taxonomy_summary_v4.json", summary)
    write_csv(analysis_dir / "all_paper_taxonomy_category_by_role_v4.csv", summary["category_rows"])
    write_csv(analysis_dir / "all_paper_taxonomy_set_tests_v4.csv", summary["set_rows"])
    write_csv(analysis_dir / "all_paper_taxonomy_model_role_summary_v4.csv", summary["model_rows"])

    lines = [
        "# Exp39 All-Feature Paper Taxonomy V4",
        "",
        "Role-blind classification over causal and control features using the same paper-facing taxonomy.",
        "",
        "## Set Tests",
    ]
    for row in summary["set_rows"]:
        lines.append(
            f"- {row['set']}: causal={row['n_causal']}/100, controls={row['n_control']}/200, "
            f"matched={row['n_matched_control']}/100, p_all={row['fisher_p_vs_all_controls']:.4g}, "
            f"p_matched={row['fisher_p_vs_matched_controls']:.4g}"
        )
    lines.extend(["", "## Categories"])
    for row in summary["category_rows"]:
        lines.append(
            f"- {row['paper_category']}: causal={row['n_causal']}/100, controls={row['n_control']}/200, "
            f"matched={row['n_matched_control']}/100"
        )
    (analysis_dir / "exp39_all_paper_taxonomy_note_v4.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
