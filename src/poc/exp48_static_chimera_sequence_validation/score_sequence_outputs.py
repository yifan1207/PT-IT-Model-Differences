"""Automatic scoring for Exp48 sequence-chimera outputs."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

_NUMBERED_LIST_RE = re.compile(r"^\s*\d+[\.\)]\s", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+\S", re.MULTILINE)
_MD_HEADER_RE = re.compile(r"^#{1,4}\s+\S", re.MULTILINE)
_CODE_BLOCK_RE = re.compile(r"```")
_BOLD_RE = re.compile(r"\*\*\S.*?\*\*", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def _count_sentences(text: str) -> int:
    return len([p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()])


def _check_compliance_v2(text: str, criteria: dict) -> float | None:
    """Local copy of the eval_dataset_v2 criteria checker.

    Importing the original scorer pulls in the steering runtime and ``nnsight``;
    Exp48 scoring only needs this pure deterministic helper.
    """
    if not criteria:
        return None
    ctype = criteria.get("type", "other")
    params = criteria.get("params", {})
    t = text.strip()
    if ctype == "other" or not t:
        return None
    if ctype == "bullet_list":
        return 1.0 if len(_BULLET_RE.findall(t)) >= max(params.get("min_count", 1), 1) else 0.0
    if ctype == "numbered_list":
        return 1.0 if len(_NUMBERED_LIST_RE.findall(t)) >= max(params.get("min_count", 1), 1) else 0.0
    if ctype == "json":
        try:
            json.loads(t)
            return 1.0
        except (json.JSONDecodeError, ValueError):
            inner = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.DOTALL).strip()
            try:
                json.loads(inner)
                return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0
    if ctype == "markdown_sections":
        return 1.0 if len(_MD_HEADER_RE.findall(t)) >= params.get("min_count", 2) else 0.0
    if ctype == "highlighted_sections":
        return 1.0 if len(_BOLD_RE.findall(t)) >= params.get("min_count", 1) else 0.0
    if ctype == "code_block":
        return 1.0 if _CODE_BLOCK_RE.search(t) else 0.0
    if ctype == "no_commas":
        return 1.0 if "," not in t else 0.0
    if ctype == "all_lowercase":
        alpha = [c for c in t if c.isalpha()]
        return None if not alpha else (1.0 if all(c.islower() for c in alpha) else 0.0)
    if ctype == "all_uppercase":
        alpha = [c for c in t if c.isalpha()]
        return None if not alpha else (1.0 if all(c.isupper() for c in alpha) else 0.0)
    if ctype == "word_count_min":
        min_w = params.get("min", 50)
        return None if min_w > 150 else (1.0 if len(t.split()) >= min_w else 0.0)
    if ctype == "word_count_max":
        return 1.0 if len(t.split()) <= params.get("max", 200) else 0.0
    if ctype == "sentence_count_min":
        return 1.0 if _count_sentences(t) >= params.get("min", 3) else 0.0
    if ctype == "sentence_count_max":
        return 1.0 if _count_sentences(t) <= params.get("max", 5) else 0.0
    if ctype == "wrapped_in_quotes":
        return 1.0 if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")) else 0.0
    if ctype == "starts_with":
        prefix = params.get("prefix", "").strip().lower()
        return None if not prefix else (1.0 if t.lower().startswith(prefix) else 0.0)
    if ctype == "ends_with":
        suffix = params.get("suffix", "").strip().lower()
        return None if not suffix else (1.0 if t.lower().endswith(suffix) else 0.0)
    if ctype == "contains_all_keywords":
        keywords = params.get("keywords", [])
        if not keywords:
            return None
        lower = t.lower()
        return 1.0 if all(str(kw).lower() in lower for kw in keywords) else 0.0
    if ctype == "contains_phrase":
        phrase = params.get("phrase", "").strip()
        return None if not phrase else (1.0 if phrase.lower() in t.lower() else 0.0)
    if ctype == "has_placeholders":
        return 1.0 if len(re.findall(r"\[[^\[\]\n]{1,50}\]", t)) >= params.get("min_count", 1) else 0.0
    return None


def _json_rows(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _dataset_lookup(dataset: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with dataset.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("id", row.get("record_id")))] = row
    return out


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _score_gsm8k(record: dict[str, Any], text: str) -> float | None:
    if record.get("category") != "CONTENT-REASON":
        return None
    expected = str(record.get("expected_answer", "")).replace(",", "").strip()
    if not expected:
        return None
    nums = re.findall(r"-?\d+(?:,\d+)*", text.replace(",", ""))
    return float(bool(nums) and nums[-1].replace(",", "") == expected)


def _score_mc(record: dict[str, Any], text: str) -> float | None:
    expected = record.get("answer") or record.get("expected_answer")
    if expected not in {"A", "B", "C", "D"}:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    return float(match is not None and match.group(1) == expected)


def _score_governance(record: dict[str, Any], text: str) -> tuple[str | None, float | None]:
    category = record.get("category")
    if category == "GOV-FORMAT":
        return "ifeval_format", float(_check_compliance_v2(text, record.get("compliance_criteria") or {}))
    if category in {"GOV-CONV", "GOV-REGISTER"}:
        # Cheap deterministic proxy for structured assistant-style behavior.
        has_structure = bool(re.search(r"(?m)^\s*(?:[-*]|\d+[.)])\s+", text)) or "\n" in text
        has_answer_opening = bool(re.match(r"^\s*(?:sure|certainly|here|to|in)\b", text.strip().lower()))
        return "governance_structure_proxy", float(has_structure or has_answer_opening)
    return None, None


def score_row(row: dict[str, Any], dataset_record: dict[str, Any]) -> list[dict[str, Any]]:
    text = str(row.get("continuation_text", ""))
    prompt_id = str(row.get("prompt_id"))
    base = {
        "model": row.get("model"),
        "wrong_model": row.get("wrong_model"),
        "prompt_id": prompt_id,
        "prompt_split": row.get("prompt_split"),
        "category": dataset_record.get("category", row.get("category")),
        "source": dataset_record.get("source", row.get("source")),
        "boundary": row.get("boundary"),
        "cell": row.get("cell"),
        "scenario": row.get("scenario"),
        "component": row.get("component"),
        "interpolation_alpha": row.get("interpolation_alpha"),
        "late_parent": row.get("late_parent"),
        "generated_tokens_count": row.get("generated_tokens_count"),
        "eos_emitted": float(bool(row.get("eos_emitted"))),
        "invalid_output": float(bool(row.get("invalid_output"))),
        "has_long_loop": float(bool(row.get("has_long_loop"))),
        "unique_token_fraction": row.get("unique_token_fraction"),
        "mean_step_entropy": row.get("mean_step_entropy"),
        "full_lexical_it_like_score": row.get("full_lexical_it_like_score"),
        "post_first_lexical_it_like_score": row.get("post_first_lexical_it_like_score"),
    }
    health = row.get("health") or {}
    for key in (
        "prompt_nll",
        "prompt_perplexity",
        "boundary_last_norm",
        "boundary_last_rms",
        "final_last_norm",
        "kl_to_base_last",
        "kl_to_ft_last",
    ):
        base[key] = health.get(key)

    out: list[dict[str, Any]] = []
    task, score = _score_governance(dataset_record, text)
    if score is not None and task is not None:
        out.append({**base, "task": task, "score": float(score)})
    gsm = _score_gsm8k(dataset_record, text)
    if gsm is not None:
        out.append({**base, "task": "gsm8k_exact", "score": float(gsm)})
    mc = _score_mc(dataset_record, text)
    if mc is not None:
        out.append({**base, "task": "multiple_choice_exact", "score": float(mc)})
    if not out:
        out.append({**base, "task": "health_only", "score": None})
    return out


def score_file(args: argparse.Namespace) -> Path:
    dataset = _dataset_lookup(args.dataset)
    out_path = args.out_dir / "sequence_scores.jsonl.gz"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out_path, "wt", encoding="utf-8") as fout:
        for path in args.inputs:
            for row in _json_rows(path):
                record = dataset.get(str(row.get("prompt_id")), {})
                for scored in score_row(row, record):
                    fout.write(json.dumps(scored, separators=(",", ":")) + "\n")
                    n += 1
    print(json.dumps({"ok": True, "out": str(out_path), "rows": n}, indent=2))
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/eval_dataset_v2.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("inputs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    score_file(parse_args())


if __name__ == "__main__":
    main()
