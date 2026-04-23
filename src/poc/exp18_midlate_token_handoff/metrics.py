"""Shared metric helpers for Exp18.

The helpers in this file are intentionally small and side-effect free so they
can be tested with toy logits/trace rows before any GPU collection runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

RAW_CATEGORIES = [
    "CONTENT",
    "FUNCTION",
    "DISCOURSE",
    "STRUCTURAL",
    "PUNCTUATION",
    "OTHER",
]
COLLAPSED_CATEGORIES = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    start_layer: int
    end_layer_exclusive: int
    kind: str = "disjoint"

    @property
    def layers(self) -> range:
        return range(self.start_layer, self.end_layer_exclusive)

    @property
    def display_range(self) -> str:
        if self.end_layer_exclusive <= self.start_layer:
            return "empty"
        return f"{self.start_layer}-{self.end_layer_exclusive - 1}"

    def contains(self, layer_idx: int | None) -> bool:
        if layer_idx is None:
            return False
        return self.start_layer <= layer_idx < self.end_layer_exclusive

    def to_json(self) -> dict[str, int | str]:
        return {
            "start_layer": self.start_layer,
            "end_layer_exclusive": self.end_layer_exclusive,
            "display_range": self.display_range,
            "kind": self.kind,
        }


def collapse_category(raw: str) -> str:
    if raw in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FORMAT"}:
        return "FORMAT"
    if raw == "CONTENT":
        return "CONTENT"
    return "FUNCTION_OTHER"


def disjoint_windows(
    *,
    n_layers: int,
    phase_boundary: int,
    corrective_onset: int,
) -> dict[str, WindowSpec]:
    """Return the primary Exp18 windows.

    These windows are non-overlapping: early proposal, mid policy selection, and
    late reconciliation. Clamp boundaries to keep tests robust for toy models.
    """

    phase_boundary = max(0, min(phase_boundary, n_layers))
    corrective_onset = max(phase_boundary, min(corrective_onset, n_layers))
    return {
        "early": WindowSpec("early", 0, phase_boundary, "disjoint"),
        "mid_policy": WindowSpec(
            "mid_policy",
            phase_boundary,
            corrective_onset,
            "disjoint",
        ),
        "late_reconciliation": WindowSpec(
            "late_reconciliation",
            corrective_onset,
            n_layers,
            "disjoint",
        ),
    }


def overlapping_windows(
    windows: dict[str, tuple[int, int]],
) -> dict[str, WindowSpec]:
    return {
        name: WindowSpec(name, int(start), int(end), "overlapping")
        for name, (start, end) in windows.items()
    }


def finite_mean(values: Iterable[float | int | None]) -> float | None:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not kept:
        return None
    return sum(kept) / len(kept)


def mean_over_layers(values: Sequence[float | int], layers: range) -> float | None:
    return finite_mean(values[layer] for layer in layers if 0 <= layer < len(values))


def rank_gain(row_a: dict, row_b: dict, layers: range) -> float | None:
    """Positive means row_b improves target rank over row_a."""

    ranks_a = row_a.get("metrics", {}).get("next_token_rank", [])
    ranks_b = row_b.get("metrics", {}).get("next_token_rank", [])
    diffs = []
    for layer in layers:
        if layer < len(ranks_a) and layer < len(ranks_b):
            diffs.append(float(ranks_a[layer]) - float(ranks_b[layer]))
    return finite_mean(diffs)


def first_layer_in_topk(
    top20_ids_by_layer: Sequence[Sequence[int]],
    token_id: int,
    *,
    k: int,
) -> int | None:
    """Return first layer where token_id is in the first k top-k entries."""

    for layer_idx, ids in enumerate(top20_ids_by_layer):
        if token_id in [int(x) for x in ids[:k]]:
            return layer_idx
    return None


def top1_top20_delta(
    row_a: dict,
    row_b: dict,
    layers: range,
) -> dict[str, float | int]:
    """Compare two trace rows over a layer window.

    `row_a` is the baseline and `row_b` is the intervention/current row.
    """

    top1_a = row_a.get("metrics", {}).get("top1_token", [])
    top1_b = row_b.get("metrics", {}).get("top1_token", [])
    top20_a = row_a.get("metrics", {}).get("top20_ids", [])
    top20_b = row_b.get("metrics", {}).get("top20_ids", [])
    total = 0
    changed = 0
    entries = 0
    exits = 0
    jaccard_sum = 0.0
    jaccard_count = 0

    for layer in layers:
        if layer >= len(top1_a) or layer >= len(top1_b):
            continue
        total += 1
        if int(top1_a[layer]) != int(top1_b[layer]):
            changed += 1
        if layer < len(top20_a) and layer < len(top20_b):
            set_a = {int(x) for x in top20_a[layer]}
            set_b = {int(x) for x in top20_b[layer]}
            entries += len(set_b - set_a)
            exits += len(set_a - set_b)
            union = set_a | set_b
            if union:
                jaccard_sum += len(set_a & set_b) / len(union)
                jaccard_count += 1

    return {
        "n_layer_observations": total,
        "top1_changed": changed,
        "top1_change_fraction": changed / total if total else math.nan,
        "top20_entries": entries,
        "top20_exits": exits,
        "mean_top20_entries": entries / total if total else math.nan,
        "mean_top20_exits": exits / total if total else math.nan,
        "mean_top20_jaccard": jaccard_sum / jaccard_count if jaccard_count else math.nan,
    }


def summarize_numbers(values: Sequence[float | int | None]) -> dict[str, float | int | None]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"count": 0, "mean": None}
    return {"count": len(finite), "mean": sum(finite) / len(finite)}


def promote_suppress_transition(
    prev_logits: Sequence[float | int],
    curr_logits: Sequence[float | int],
    *,
    target_id: int,
    alternative_ids: Sequence[int],
) -> dict[str, float | None]:
    """Return support, repulsion, and margin deltas for one layer transition.

    `repulsion_top10_delta` keeps the direct sign of alternative-token logit
    change: negative means alternatives were suppressed.
    """

    if target_id >= len(prev_logits) or target_id >= len(curr_logits):
        return {
            "support_target_delta": None,
            "repulsion_top10_delta": None,
            "margin_delta": None,
        }
    support = float(curr_logits[target_id]) - float(prev_logits[target_id])
    alt_deltas = [
        float(curr_logits[idx]) - float(prev_logits[idx])
        for idx in alternative_ids
        if idx < len(prev_logits) and idx < len(curr_logits) and idx != target_id
    ]
    repulsion = sum(alt_deltas) / len(alt_deltas) if alt_deltas else None
    return {
        "support_target_delta": support,
        "repulsion_top10_delta": repulsion,
        "margin_delta": support - repulsion if repulsion is not None else None,
    }


def make_category_template() -> dict[str, dict[str, float | int | None]]:
    return {cat: {"sum": 0.0, "count": 0, "positive": 0} for cat in RAW_CATEGORIES + COLLAPSED_CATEGORIES}


def add_category_value(
    stats: dict[str, dict[str, float | int | None]],
    raw_category: str,
    value: float | int | None,
) -> None:
    if value is None or not math.isfinite(float(value)):
        return
    for cat in (raw_category, collapse_category(raw_category)):
        if cat not in stats:
            stats[cat] = {"sum": 0.0, "count": 0, "positive": 0}
        stats[cat]["sum"] = float(stats[cat]["sum"]) + float(value)
        stats[cat]["count"] = int(stats[cat]["count"]) + 1
        if float(value) > 0:
            stats[cat]["positive"] = int(stats[cat]["positive"]) + 1


def finalize_category_values(
    stats: dict[str, dict[str, float | int | None]],
    *,
    mean_key: str,
    positive_key: str = "fraction_positive",
) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for cat, values in stats.items():
        count = int(values["count"])
        total = float(values["sum"])
        positive = int(values["positive"])
        out[cat] = {
            "count": count,
            mean_key: total / count if count else None,
            positive_key: positive / count if count else None,
        }
    return out
