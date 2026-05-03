#!/usr/bin/env python3
"""Causal-only paper-facing taxonomy for Exp39.

This is a descriptive pass over causally selected features only. It is intended
for paper-facing taxonomy tables, not for category-enrichment claims.
"""

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

HUMAN_SAFETY_ADVICE_OVERRIDES = {
    "gemma3_4b:L33:F80933:causal": "adversarial child-exploitation prompt plus safety-disclaimer text",
    "qwen3_4b:L35:F8401:causal": "suicide-method refusal and hotline-support answer",
    "qwen3_4b:L35:F38615:causal": "explicit not-safe-to-consume-chlorine answer",
    "gemma3_4b:L33:F3693:causal": "family-violence danger/help and legal-help examples",
    "llama31_8b:L31:F80640:causal": "weapon/bomb/advice examples around instruction boundary",
    "mistral_7b:L30:F10437:causal": "weapon-building list/answer boundary",
    "mistral_7b:L31:F82495:causal": "suicide, trafficking, and euthanasia definitions",
    "qwen3_4b:L35:F18377:causal": "legal/lawyer and medical-regulatory answer blocks",
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


def top_examples(row: dict[str, Any], *, per_section: int = 3) -> list[dict[str, str]]:
    out = []
    examples = row.get("examples") or {}
    for section in ("top_it", "contrast_it_over_pt", "top_pt", "contrast_pt_over_it", "near_miss"):
        for ex in examples.get(section, [])[:per_section]:
            context = str(ex.get("context_window") or ex.get("text") or "")
            out.append(
                {
                    "section": section,
                    "token_text": str(ex.get("token_text", "")),
                    "context_excerpt": context[:700],
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
                        "context_excerpt": context[:700],
                    }
                )
            if len(safety_hits) >= 4:
                break
        if len(safety_hits) >= 4:
            break
    return out[:15] + safety_hits


def projection_summary(row: dict[str, Any]) -> dict[str, Any]:
    projection = row.get("output_projection") or {}
    promoted = projection.get("top_promoted") or []
    suppressed = projection.get("top_suppressed") or []
    return {
        "top_promoted": promoted[:15],
        "top_suppressed": suppressed[:15],
    }


def classify_batch(client: Any, *, model: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = (
        "You are a senior mechanistic-interpretability human reviewer. You classify causally selected "
        "PT-vs-IT terminal crosscoder features for a paper-facing descriptive taxonomy. Return only JSON."
    )
    user = json.dumps(
        {
            "task": (
                "Classify each causal feature into the most paper-useful category. These are all causally "
                "selected features, so do not compare to controls. Be favorable to legitimate instruction-tuned "
                "behavior when the activation examples or output projection support it, but do not call rare-token, "
                "web-scrape, or repeated-template artifacts alignment behavior. Broaden assistant/alignment behavior "
                "to include user-facing advice, refusal/disclaimer/safety/legal/medical boundaries, instruction "
                "constraints, response openers, and answer-structure readout. Split artifact/unclear features into "
                "specific artifact categories instead of leaving them generic."
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
                        "feature_id": "id",
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


def make_records(labels: list[dict[str, Any]], dashboards: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for label in labels:
        dash = dashboards.get(label["feature_id"], {})
        records.append(
            {
                "feature_id": label["feature_id"],
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
                "causal_score_mean": dash.get("score_mean"),
                "causal_top200_interaction_drop": dash.get("causal_top200_interaction_drop_mean"),
                "decoder_norm_ratio_it_pt": dash.get("decoder_norm_ratio_it_pt"),
                "examples": top_examples(dash),
                "output_projection": projection_summary(dash),
            }
        )
    return records


def summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_model_cat: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        model = str(row.get("model"))
        cat = str(row.get("paper_category"))
        by_model_cat[(model, cat)].append(row)
        by_cat[cat].append(row)
        by_model[model].append(row)

    per_model = []
    for model, model_rows in sorted(by_model.items()):
        denom = len(model_rows)
        for cat in PAPER_CATEGORIES:
            group = by_model_cat.get((model, cat), [])
            per_model.append(
                {
                    "model": model,
                    "paper_category": cat,
                    "n": len(group),
                    "share": len(group) / denom if denom else 0,
                    "n_showcase_or_support": sum(row.get("paper_use") in {"showcase", "support"} for row in group),
                    "mean_behavior_score": mean(row.get("paper_behavior_score") for row in group),
                    "mean_safety_alignment_score": mean(row.get("paper_safety_alignment_score") for row in group),
                    "example_features": " | ".join(str(row.get("feature_id")) for row in group[:4]),
                    "example_labels": " | ".join(str(row.get("paper_niche_label")) for row in group[:4]),
                }
            )

    overall = []
    for cat in PAPER_CATEGORIES:
        group = by_cat.get(cat, [])
        overall.append(
            {
                "paper_category": cat,
                "n": len(group),
                "share": len(group) / len(rows) if rows else 0,
                "n_showcase_or_support": sum(row.get("paper_use") in {"showcase", "support"} for row in group),
                "mean_behavior_score": mean(row.get("paper_behavior_score") for row in group),
                "mean_safety_alignment_score": mean(row.get("paper_safety_alignment_score") for row in group),
                "models": ",".join(sorted({str(row.get("model")) for row in group})),
                "example_labels": " | ".join(str(row.get("paper_niche_label")) for row in group[:6]),
            }
        )

    model_summary = []
    for model, model_rows in sorted(by_model.items()):
        behavior = [row for row in model_rows if row.get("paper_category") in BEHAVIOR_CATEGORIES]
        artifact = [row for row in model_rows if row.get("paper_category") in ARTIFACT_CATEGORIES]
        safety = [row for row in model_rows if row.get("paper_category") == "alignment_safety_or_advice_boundary"]
        model_summary.append(
            {
                "model": model,
                "n_features": len(model_rows),
                "behavior_aligned_n": len(behavior),
                "behavior_aligned_share": len(behavior) / len(model_rows) if model_rows else 0,
                "artifact_or_unclear_n": len(artifact),
                "artifact_or_unclear_share": len(artifact) / len(model_rows) if model_rows else 0,
                "safety_alignment_n": len(safety),
                "safety_alignment_share": len(safety) / len(model_rows) if model_rows else 0,
                "mean_behavior_score": mean(row.get("paper_behavior_score") for row in model_rows),
                "category_counts": json.dumps(Counter(row.get("paper_category") for row in model_rows), sort_keys=True),
            }
        )
    return overall, per_model, model_summary


def mean(values: Any) -> float | None:
    nums = []
    for value in values:
        try:
            nums.append(float(value))
        except (TypeError, ValueError):
            pass
    return sum(nums) / len(nums) if nums else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal-only Exp39 paper taxonomy")
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
    labels = [row for row in read_jsonl(labels_path) if row.get("role") == "causal"]
    dashboards = {row.get("feature_id"): row for row in read_jsonl(run_dir / "dashboards" / "feature_dashboards.jsonl")}
    records = make_records(labels, dashboards)
    label_by_id = {row["feature_id"]: row for row in labels}

    out_dir = run_dir / "autointerp"
    partial_path = out_dir / "causal_paper_taxonomy_v3.partial.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        for row in read_jsonl(partial_path):
            done[str(row.get("feature_id"))] = row

    client = OpenAI()
    pending = [record for record in records if record["feature_id"] not in done]
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        assignments = classify_batch(client, model=args.model, records=batch)
        by_id = {str(row.get("feature_id")): row for row in assignments}
        missing = [row["feature_id"] for row in batch if row["feature_id"] not in by_id]
        if missing:
            raise RuntimeError(f"Missing assignments for {missing[:10]}")
        with partial_path.open("a", encoding="utf-8") as handle:
            for record in batch:
                assignment = by_id[record["feature_id"]]
                handle.write(json.dumps(assignment, ensure_ascii=False) + "\n")
                done[record["feature_id"]] = assignment
        print(f"[causal-taxonomy] classified {len(done)}/{len(records)}")

    rows = []
    for record in records:
        feature_id = record["feature_id"]
        label = label_by_id[feature_id]
        assignment = dict(done[feature_id])
        cat = assignment.get("paper_category")
        if cat not in PAPER_CATEGORIES:
            cat = "generic_frequency_or_unclear"
        assignment["paper_category"] = cat
        human_override_note = ""
        if feature_id in HUMAN_SAFETY_ADVICE_OVERRIDES:
            human_override_note = HUMAN_SAFETY_ADVICE_OVERRIDES[feature_id]
            assignment["auto_paper_category"] = cat
            assignment["paper_category"] = "alignment_safety_or_advice_boundary"
            assignment["niche_label"] = f"safety/advice-adjacent readout: {human_override_note}"
            assignment["paper_use"] = "support" if cat not in ARTIFACT_CATEGORIES else "weak_support"
            assignment["safety_alignment_score"] = max(float(assignment.get("safety_alignment_score") or 0), 2.0)
            assignment["behavior_score"] = max(float(assignment.get("behavior_score") or 0), 1.0)
        enriched = dict(label)
        enriched.update(
            {
                "paper_taxonomy_v3_model": args.model,
                "paper_auto_category": assignment.get("auto_paper_category", cat),
                "paper_category": assignment["paper_category"],
                "paper_niche_label": assignment.get("niche_label") or label.get("specific_label") or label.get("label"),
                "paper_behavior_score": assignment.get("behavior_score"),
                "paper_use": assignment.get("paper_use"),
                "paper_safety_alignment_score": assignment.get("safety_alignment_score"),
                "paper_human_review_confidence": assignment.get("human_review_confidence"),
                "paper_taxonomy_rationale": assignment.get("rationale"),
                "paper_human_override_note": human_override_note,
            }
        )
        rows.append(enriched)

    overall, per_model, model_summary = summarize(rows)
    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "n_causal_features": len(rows),
        "paper_categories": PAPER_CATEGORIES,
        "overall_counts": dict(Counter(row.get("paper_category") for row in rows)),
        "model_summary": model_summary,
        "overall_distribution": overall,
    }

    analysis_dir = run_dir / "analysis"
    write_jsonl(out_dir / "causal_paper_taxonomy_v3.jsonl", rows)
    write_json(analysis_dir / "exp39_causal_paper_taxonomy_summary_v3.json", summary)
    write_csv(analysis_dir / "causal_paper_taxonomy_overall_v3.csv", overall)
    write_csv(analysis_dir / "causal_paper_taxonomy_by_model_v3.csv", per_model)
    write_csv(analysis_dir / "causal_paper_taxonomy_model_summary_v3.csv", model_summary)

    lines = [
        "# Exp39 Causal Paper Taxonomy V3",
        "",
        "Causal features only. This table supports descriptive claims about what the selected PT-vs-IT features represent.",
        "",
        "## Model Summary",
    ]
    for row in model_summary:
        lines.append(
            f"- {row['model']}: behavior-aligned {row['behavior_aligned_n']}/{row['n_features']} "
            f"({100*row['behavior_aligned_share']:.1f}%), artifact/unclear {row['artifact_or_unclear_n']}/{row['n_features']} "
            f"({100*row['artifact_or_unclear_share']:.1f}%), safety/alignment {row['safety_alignment_n']}/{row['n_features']}"
        )
    lines.extend(["", "## Overall Categories"])
    for row in overall:
        lines.append(f"- {row['paper_category']}: {row['n']}/100 ({100*row['share']:.1f}%)")
    (analysis_dir / "exp39_causal_paper_taxonomy_note_v3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
