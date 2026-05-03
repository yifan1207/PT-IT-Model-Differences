#!/usr/bin/env python3
"""Second-pass niche grouping for Exp39 feature labels.

This pass is intentionally role-blind: the LLM sees opaque record IDs and label
evidence, but not whether a feature is causal or a control. We join the true
roles only after classification to compute enrichment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PAPER_DOMAINS = [
    "assistant_register",
    "safety_refusal_or_advice_boundary",
    "instruction_compliance_or_format_constraint",
    "response_structure_readout",
    "factual_correction_or_reasoning",
    "punctuation_or_surface_tokenization",
    "code_or_tool_syntax",
    "content_topic_or_named_entity",
    "data_or_tokenization_artifact",
    "generic_frequency_or_unclear",
]

CORE_DOMAINS = {
    "assistant_register",
    "safety_refusal_or_advice_boundary",
    "instruction_compliance_or_format_constraint",
    "response_structure_readout",
    "factual_correction_or_reasoning",
}

SURFACE_DOMAINS = {"punctuation_or_surface_tokenization", "code_or_tool_syntax"}
ARTIFACT_DOMAINS = {"data_or_tokenization_artifact", "generic_frequency_or_unclear"}


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
    """One-sided Fisher exact test for [[a,b],[c,d]], alternative='greater'."""

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


def compact_record(record_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "record_id": record_id,
        "model_family": row.get("model"),
        "specific_label": row.get("specific_label") or row.get("label"),
        "coarse_category_previous": row.get("coarse_category") or row.get("category"),
        "fires_on": row.get("fires_on"),
        "output_effect": row.get("output_effect"),
        "mechanism_hypothesis": row.get("mechanism_hypothesis"),
        "label_distribution": row.get("label_distribution"),
        "uncertainty_allocation": row.get("uncertainty_allocation"),
        "evidence": row.get("evidence"),
        "ambiguity": row.get("counterexamples_or_ambiguity"),
        "artifact_risk": row.get("artifact_risk"),
        "confidence": row.get("confidence"),
        "paper_example_quality": row.get("paper_example_quality"),
    }


def classify_batch(client: Any, *, model: str, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    system = (
        "You are doing role-blind mechanistic feature taxonomy for a paper. "
        "You do not know whether any record is causal or a control. Classify evidence accurately; "
        "do not make categories look favorable. Return only JSON."
    )
    user = json.dumps(
        {
            "task": (
                "Assign each feature to a niche paper-facing category. The aim is to reduce lazy "
                "'generic unclear' labels by separating real artifact subtypes, structural readout "
                "subtypes, instruction/compliance subtypes, safety/refusal/advice subtypes, and "
                "surface tokenization subtypes. Use safety/refusal only when examples or output "
                "effects directly indicate refusal, harm avoidance, policy/safety language, legal/"
                "medical advice caution, or risk disclaimers. If a feature is a data artifact, name "
                "the artifact subtype instead of leaving it generic. If evidence is truly mixed, say so."
            ),
            "allowed_paper_domains": PAPER_DOMAINS,
            "niche_category_guidance": [
                "assistant response opener / acknowledgement / conversational register",
                "refusal phrase / safety disclaimer / legal-medical advice boundary",
                "explicit format constraint such as no comma, all caps, answer-only, length constraint",
                "Q/A field label, answer/explanation delimiter, chat-template boundary",
                "markdown/list/table/heading/newline/paragraph boundary",
                "multiple-choice option marker or answer-choice delimiter",
                "factual correction, negation, rationale, calculation, or explanation step marker",
                "code syntax, docstring, indentation, stack trace, or tool-like scaffold",
                "sentence punctuation, comma/conjunction, apostrophe/suffix, quotation delimiter",
                "URL/web-scrape/post-EOT artifact, rare Unicode/mojibake, subword/collocation",
                "degenerate repetition loop, copied template, common-token frequency feature",
                "topic/entity/domain feature with weak structural admixture",
            ],
            "classification_schema": {
                "record_id": "opaque ID from input",
                "paper_domain": "exactly one allowed_paper_domains value",
                "niche_category": "5-12 word concrete subgroup, not a broad ontology label",
                "paper_relevance": "core_support | surface_adjacent | diagnostic_artifact | reject_unclear",
                "safety_refusal_subtype": "specific subtype or none",
                "is_post_training_readout_relevant": True,
                "artifact_or_surface_risk": "low | medium | high",
                "confidence": 0.0,
                "reason": "one sentence grounded in evidence",
            },
            "records": batch,
            "return_schema": {"assignments": []},
        },
        ensure_ascii=False,
    )
    payload = chat_json(
        client,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return list(payload.get("assignments", []))


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_key[str(row.get(key) or "")].append(row)
    out = []
    for value, group in by_key.items():
        causal = [row for row in group if row.get("role") == "causal"]
        controls = [row for row in group if row.get("role") == "control"]
        matched = [row for row in controls if row.get("control_kind") == "matched_noncausal"]
        random_active = [row for row in controls if row.get("control_kind") == "random_active_noncausal"]
        a = len(causal)
        c = len(controls)
        odds_all, p_all = fisher_exact_greater(a, 100 - a, c, 200 - c)
        odds_matched, p_matched = fisher_exact_greater(a, 100 - a, len(matched), 100 - len(matched))
        out.append(
            {
                key: value,
                "n_total": len(group),
                "n_causal": len(causal),
                "n_control": len(controls),
                "n_control_matched": len(matched),
                "n_control_random_active": len(random_active),
                "causal_share": a / 100,
                "control_share": c / 200,
                "matched_control_share": len(matched) / 100,
                "odds_ratio_vs_all_controls": odds_all,
                "fisher_p_vs_all_controls": p_all,
                "odds_ratio_vs_matched_controls": odds_matched,
                "fisher_p_vs_matched_controls": p_matched,
                "models": ",".join(sorted({str(row.get("model")) for row in group})),
                "example_labels": " | ".join(str(row.get("specific_label") or row.get("label")) for row in group[:5]),
            }
        )
    out.sort(key=lambda row: (-int(row["n_causal"]), float(row["fisher_p_vs_all_controls"]), str(row[key])))
    return out


def set_test(rows: list[dict[str, Any]], name: str, predicate: Any) -> dict[str, Any]:
    causal_n = sum(1 for row in rows if row.get("role") == "causal" and predicate(row))
    control_n = sum(1 for row in rows if row.get("role") == "control" and predicate(row))
    matched_n = sum(
        1
        for row in rows
        if row.get("role") == "control" and row.get("control_kind") == "matched_noncausal" and predicate(row)
    )
    odds_all, p_all = fisher_exact_greater(causal_n, 100 - causal_n, control_n, 200 - control_n)
    odds_matched, p_matched = fisher_exact_greater(causal_n, 100 - causal_n, matched_n, 100 - matched_n)
    return {
        "set": name,
        "n_causal": causal_n,
        "n_control": control_n,
        "n_control_matched": matched_n,
        "causal_share": causal_n / 100,
        "control_share": control_n / 200,
        "matched_control_share": matched_n / 100,
        "odds_ratio_vs_all_controls": odds_all,
        "fisher_p_vs_all_controls": p_all,
        "odds_ratio_vs_matched_controls": odds_matched,
        "fisher_p_vs_matched_controls": p_matched,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Role-blind niche regrouping for Exp39 labels")
    parser.add_argument("--out-root", type=Path, default=Path("results/exp39_causal_feature_interpretation"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for LLM regrouping")

    from openai import OpenAI

    run_dir = args.out_root / args.run_name
    labels = read_jsonl(run_dir / "autointerp" / "llm_feature_labels.jsonl")
    out_dir = run_dir / "autointerp"
    partial_path = out_dir / "niche_group_assignments_v2.partial.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if args.resume and partial_path.exists():
        for row in read_jsonl(partial_path):
            done[str(row.get("record_id"))] = row

    id_to_feature: dict[str, dict[str, Any]] = {}
    compact = []
    for idx, label in enumerate(labels):
        record_id = f"R{idx:04d}"
        id_to_feature[record_id] = label
        if record_id not in done:
            compact.append(compact_record(record_id, label))

    client = OpenAI()
    for start in range(0, len(compact), args.batch_size):
        batch = compact[start : start + args.batch_size]
        assignments = classify_batch(client, model=args.model, batch=batch)
        by_id = {str(row.get("record_id")): row for row in assignments}
        missing = [row["record_id"] for row in batch if row["record_id"] not in by_id]
        if missing:
            raise RuntimeError(f"Missing assignments for {missing[:10]}")
        with partial_path.open("a", encoding="utf-8") as handle:
            for row in batch:
                assignment = by_id[row["record_id"]]
                handle.write(json.dumps(assignment, ensure_ascii=False) + "\n")
                done[row["record_id"]] = assignment
        print(f"[niche-regroup] classified {len(done)}/{len(labels)}")

    rows = []
    for record_id, label in id_to_feature.items():
        assignment = done[record_id]
        domain = assignment.get("paper_domain")
        if domain not in PAPER_DOMAINS:
            domain = "generic_frequency_or_unclear"
        enriched = dict(label)
        enriched.update(
            {
                "niche_v2_record_id": record_id,
                "niche_v2_model": args.model,
                "niche_v2_paper_domain": domain,
                "niche_v2_category": assignment.get("niche_category") or domain,
                "niche_v2_paper_relevance": assignment.get("paper_relevance") or "reject_unclear",
                "niche_v2_safety_refusal_subtype": assignment.get("safety_refusal_subtype") or "none",
                "niche_v2_post_training_readout_relevant": assignment.get("is_post_training_readout_relevant"),
                "niche_v2_artifact_or_surface_risk": assignment.get("artifact_or_surface_risk"),
                "niche_v2_confidence": assignment.get("confidence"),
                "niche_v2_reason": assignment.get("reason"),
            }
        )
        rows.append(enriched)

    domain_table = aggregate(rows, "niche_v2_paper_domain")
    category_table = aggregate(rows, "niche_v2_category")
    relevance_table = aggregate(rows, "niche_v2_paper_relevance")
    set_tests = [
        set_test(rows, "core_post_training_readout_domains", lambda row: row.get("niche_v2_paper_domain") in CORE_DOMAINS),
        set_test(
            rows,
            "core_domains_excluding_low_confidence",
            lambda row: row.get("niche_v2_paper_domain") in CORE_DOMAINS
            and float(row.get("niche_v2_confidence") or 0) >= 0.55,
        ),
        set_test(
            rows,
            "core_or_surface_domains",
            lambda row: row.get("niche_v2_paper_domain") in CORE_DOMAINS | SURFACE_DOMAINS,
        ),
        set_test(rows, "artifact_or_unclear_domains", lambda row: row.get("niche_v2_paper_domain") in ARTIFACT_DOMAINS),
        set_test(
            rows,
            "safety_refusal_or_advice_boundary",
            lambda row: row.get("niche_v2_paper_domain") == "safety_refusal_or_advice_boundary",
        ),
    ]

    summary = {
        "run_name": args.run_name,
        "model": args.model,
        "n_features": len(rows),
        "domain_counts": dict(Counter(row.get("niche_v2_paper_domain") for row in rows)),
        "relevance_counts": dict(Counter(row.get("niche_v2_paper_relevance") for row in rows)),
        "safety_refusal_counts": dict(Counter(row.get("niche_v2_safety_refusal_subtype") for row in rows)),
        "set_tests": set_tests,
        "top_domain_rows": domain_table,
        "recommendation": (
            "candidate_supporting_result"
            if any(
                row["set"] == "core_post_training_readout_domains"
                and row["fisher_p_vs_matched_controls"] < 0.05
                and row["fisher_p_vs_all_controls"] < 0.05
                for row in set_tests
            )
            else "diagnostic_or_appendix_unless_manual_review_identifies_clean_subset"
        ),
    }

    write_jsonl(out_dir / "llm_feature_labels_niche_v2.jsonl", rows)
    write_json(out_dir / "niche_group_summary_v2.json", summary)
    analysis_dir = run_dir / "analysis"
    write_csv(analysis_dir / "feature_niche_domain_table_v2.csv", domain_table)
    write_csv(analysis_dir / "feature_niche_category_table_v2.csv", category_table)
    write_csv(analysis_dir / "feature_niche_relevance_table_v2.csv", relevance_table)
    write_csv(analysis_dir / "feature_niche_set_tests_v2.csv", set_tests)
    write_json(analysis_dir / "exp39_niche_regroup_summary_v2.json", summary)

    lines = [
        "# Exp39 Niche Regrouping V2",
        "",
        f"Model: `{args.model}`",
        f"Recommendation: `{summary['recommendation']}`",
        "",
        "## Set Tests",
    ]
    for row in set_tests:
        lines.append(
            f"- {row['set']}: causal={row['n_causal']}/100, controls={row['n_control']}/200, "
            f"matched={row['n_control_matched']}/100, p_all={row['fisher_p_vs_all_controls']:.4g}, "
            f"p_matched={row['fisher_p_vs_matched_controls']:.4g}"
        )
    lines.append("")
    lines.append("## Domains")
    for row in domain_table:
        lines.append(
            f"- {row['niche_v2_paper_domain']}: causal={row['n_causal']}/100, "
            f"controls={row['n_control']}/200, matched={row['n_control_matched']}/100, "
            f"p_all={float(row['fisher_p_vs_all_controls']):.4g}"
        )
    (analysis_dir / "exp39_niche_regroup_note_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
