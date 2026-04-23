from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Sequence


CONDITION_ORDER = [
    "A_pt_raw",
    "B_early_raw",
    "B_mid_raw",
    "B_late_raw",
    "C_it_chat",
    "D_early_ptswap",
    "D_mid_ptswap",
    "D_late_ptswap",
]

DIVERGENCE_KINDS = ["first_diff", "first_nonformat_diff", "first_assistant_marker_diff"]

ASSISTANT_MARKER_RE = re.compile(
    r"^\s*(sure|certainly|absolutely|course|happy|glad|help|assist|"
    r"here'?s|here|let'?s|let|answer|first|step|sorry|cannot|can't|"
    r"unable|i|we)\b",
    re.IGNORECASE,
)


def classify_assistant_marker(token_str: str) -> bool:
    return bool(ASSISTANT_MARKER_RE.search(token_str or ""))


def _collapsed(step: dict[str, Any] | None) -> str | None:
    if step is None:
        return None
    return step.get("token_category_collapsed")


def _token_id(step: dict[str, Any] | None) -> int | None:
    if step is None:
        return None
    value = step.get("token_id")
    return int(value) if value is not None else None


def _event_payload(kind: str, idx: int, pt_step: dict[str, Any] | None, it_step: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "kind": kind,
        "step": idx,
        "pt_token": _compact_step(pt_step),
        "it_token": _compact_step(it_step),
    }


def _compact_step(step: dict[str, Any] | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "token_id": int(step["token_id"]),
        "token_str": step.get("token_str", ""),
        "token_category": step.get("token_category", "OTHER"),
        "token_category_collapsed": step.get("token_category_collapsed", "FUNCTION_OTHER"),
        "assistant_marker": classify_assistant_marker(step.get("token_str", "")),
    }


def find_divergence_events(pt_steps: Sequence[dict[str, Any]], it_steps: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    """Find PT/IT divergence events while their generated prefix is still shared."""

    out: dict[str, dict[str, Any] | None] = {kind: None for kind in DIVERGENCE_KINDS}
    max_len = max(len(pt_steps), len(it_steps))
    for idx in range(max_len):
        pt_step = pt_steps[idx] if idx < len(pt_steps) else None
        it_step = it_steps[idx] if idx < len(it_steps) else None
        if _token_id(pt_step) == _token_id(it_step):
            continue
        if out["first_diff"] is None:
            out["first_diff"] = _event_payload("first_diff", idx, pt_step, it_step)
        if (
            out["first_nonformat_diff"] is None
            and _collapsed(pt_step) != "FORMAT"
            and _collapsed(it_step) != "FORMAT"
        ):
            out["first_nonformat_diff"] = _event_payload("first_nonformat_diff", idx, pt_step, it_step)
        if (
            out["first_assistant_marker_diff"] is None
            and (
                classify_assistant_marker((pt_step or {}).get("token_str", ""))
                or classify_assistant_marker((it_step or {}).get("token_str", ""))
            )
        ):
            out["first_assistant_marker_diff"] = _event_payload(
                "first_assistant_marker_diff",
                idx,
                pt_step,
                it_step,
            )
        if all(out.values()):
            break
    return out


def pairwise_agreement(tokens_a: Sequence[int], tokens_b: Sequence[int], max_len: int | None = None) -> dict[str, Any]:
    limit = max_len if max_len is not None else max(len(tokens_a), len(tokens_b))
    compared = 0
    same = 0
    first_divergence = None
    for idx in range(limit):
        a = tokens_a[idx] if idx < len(tokens_a) else None
        b = tokens_b[idx] if idx < len(tokens_b) else None
        if a is None and b is None:
            break
        compared += 1
        if a == b:
            same += 1
        elif first_divergence is None:
            first_divergence = idx
    return {
        "compared": compared,
        "same": same,
        "agreement_fraction": (same / compared if compared else None),
        "first_divergence_step": first_divergence,
    }


def summarize_token_clusters(condition_tokens: dict[str, Sequence[int]], max_len: int) -> dict[str, Any]:
    cluster_entropy: list[float] = []
    n_unique: list[int] = []
    majority_size: list[int] = []
    leaves_majority_step = {condition: None for condition in condition_tokens}
    unique_token_step = {condition: None for condition in condition_tokens}

    for idx in range(max_len):
        ids = {
            condition: (tokens[idx] if idx < len(tokens) else None)
            for condition, tokens in condition_tokens.items()
        }
        counts = Counter(ids.values())
        total = sum(counts.values())
        if total <= 0:
            continue
        majority_token, majority_count = counts.most_common(1)[0]
        n_unique.append(len(counts))
        majority_size.append(int(majority_count))
        ent = 0.0
        for count in counts.values():
            p = count / total
            ent -= p * math.log(p)
        cluster_entropy.append(ent)
        unique_tokens = {tok for tok, count in counts.items() if count == 1}
        for condition, token_id in ids.items():
            if leaves_majority_step[condition] is None and token_id != majority_token:
                leaves_majority_step[condition] = idx
            if unique_token_step[condition] is None and token_id in unique_tokens:
                unique_token_step[condition] = idx

    return {
        "mean_cluster_entropy": _mean(cluster_entropy),
        "mean_unique_token_count": _mean([float(x) for x in n_unique]),
        "mean_majority_size": _mean([float(x) for x in majority_size]),
        "leaves_majority_step": leaves_majority_step,
        "unique_token_step": unique_token_step,
    }


def window_logit_summary(values: Sequence[float], window: tuple[int, int]) -> dict[str, float | None]:
    start, end = window
    start = max(0, int(start))
    end = min(len(values), int(end))
    if end <= start or not values:
        return {"mean_step_delta": None, "total_delta": None, "start_value": None, "end_value": None}
    step_deltas = [
        float(values[idx]) - float(values[idx - 1])
        for idx in range(max(start, 1), end)
        if _finite(values[idx]) and _finite(values[idx - 1])
    ]
    baseline_idx = start - 1 if start > 0 else start
    total_delta = None
    if 0 <= baseline_idx < len(values) and _finite(values[baseline_idx]) and _finite(values[end - 1]):
        total_delta = float(values[end - 1]) - float(values[baseline_idx])
    return {
        "mean_step_delta": _mean(step_deltas),
        "total_delta": total_delta,
        "start_value": float(values[start]) if _finite(values[start]) else None,
        "end_value": float(values[end - 1]) if _finite(values[end - 1]) else None,
    }


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _mean(values: Iterable[float]) -> float | None:
    kept = [float(v) for v in values if _finite(v)]
    if not kept:
        return None
    return sum(kept) / len(kept)
