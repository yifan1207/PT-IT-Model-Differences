from __future__ import annotations

import json
import random
from typing import Any

from src.poc.exp45_behavioral_bridge.metrics import stable_int


PAIRWISE_COMPARISONS = (
    ("IT_late_portability", "U_IT__L_IT", "U_PT__L_IT"),
    ("PT_upstream_late_effect", "U_PT__L_IT", "U_PT__L_PT"),
    ("IT_upstream_late_effect", "U_IT__L_IT", "U_IT__L_PT"),
)


def absolute_prompt(prompt: str, continuation: str) -> str:
    return f"""You are comparing short continuations from a pretrained base model and an instruction-following descendant.
Given the user prompt and a continuation, rate how instruction-following / assistant-like the continuation is.
Focus on helpful answer structure, directness, formatting appropriate to the instruction, and refusal/safety behavior when relevant.
Do not reward verbosity alone.

User prompt:
{prompt[:2000]}

Continuation:
{continuation[:2000]}

Return strict JSON:
{{"it_likeness": 0-5, "instruction_following": 0-5, "answer_structure": 0-5, "pt_style_raw_continuation": 0-5, "notes": "short reason"}}"""


def pairwise_prompt(prompt: str, continuation_a: str, continuation_b: str) -> str:
    return f"""You are judging which short continuation is more instruction-following / assistant-like.
Focus on helpful answer structure, directness, formatting appropriate to the instruction, and refusal/safety behavior when relevant.
Do not reward verbosity alone. If both are bad or indistinguishable, choose tie or unclear.

User prompt:
{prompt[:2000]}

Continuation A:
{continuation_a[:2000]}

Continuation B:
{continuation_b[:2000]}

Return strict JSON:
{{"winner": "A|B|tie|unclear", "confidence": 0-1, "notes": "short reason"}}"""


def build_judge_requests(rollout_records: list[dict[str, Any]], *, include_absolute: bool = True) -> list[dict[str, Any]]:
    by_event: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in rollout_records:
        key = (str(row.get("model")), str(row.get("prompt_id")), str(row.get("event_kind")))
        by_event.setdefault(key, {})[str(row.get("cell"))] = row

    requests: list[dict[str, Any]] = []
    for key, cells in sorted(by_event.items()):
        model, prompt_id, event_kind = key
        prompt = next(iter(cells.values())).get("raw_prompt", "")
        if include_absolute:
            for cell, row in sorted(cells.items()):
                requests.append(
                    {
                        "request_id": f"abs::{model}::{prompt_id}::{event_kind}::{cell}",
                        "kind": "absolute",
                        "model": model,
                        "prompt_id": prompt_id,
                        "event_kind": event_kind,
                        "cell": cell,
                        "prompt": absolute_prompt(prompt, row.get("continuation_text", "")),
                    }
                )
        for comparison, left, right in PAIRWISE_COMPARISONS:
            if left not in cells or right not in cells:
                continue
            seed = stable_int(model, prompt_id, event_kind, comparison)
            rng = random.Random(seed)
            left_is_a = rng.random() < 0.5
            a_cell, b_cell = (left, right) if left_is_a else (right, left)
            requests.append(
                {
                    "request_id": f"pair::{model}::{prompt_id}::{event_kind}::{comparison}",
                    "kind": "pairwise",
                    "comparison": comparison,
                    "model": model,
                    "prompt_id": prompt_id,
                    "event_kind": event_kind,
                    "left_cell": left,
                    "right_cell": right,
                    "a_cell": a_cell,
                    "b_cell": b_cell,
                    "left_is_a": left_is_a,
                    "prompt": pairwise_prompt(
                        prompt,
                        cells[a_cell].get("continuation_text", ""),
                        cells[b_cell].get("continuation_text", ""),
                    ),
                }
            )
    return requests


def parse_jsonish(text: str) -> dict[str, Any]:
    blob = text.strip()
    if blob.startswith("```"):
        parts = blob.split("```")
        if len(parts) >= 2:
            blob = parts[1].strip()
            if blob.startswith("json"):
                blob = blob[4:].strip()
    if not blob.startswith("{"):
        start = blob.find("{")
        end = blob.rfind("}")
        if start >= 0 and end > start:
            blob = blob[start : end + 1]
    return json.loads(blob)
