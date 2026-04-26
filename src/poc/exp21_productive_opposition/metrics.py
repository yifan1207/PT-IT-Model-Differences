"""Small, testable metric helpers for Exp21 productive opposition."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F


COLLAPSED_CATEGORIES = ["FORMAT", "CONTENT", "FUNCTION_OTHER"]


def finite_mean(values: Iterable[float | int | None]) -> float | None:
    kept = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not kept:
        return None
    return sum(kept) / len(kept)


def finite_rate(values: Iterable[bool | None]) -> float | None:
    kept = [bool(v) for v in values if v is not None]
    if not kept:
        return None
    return sum(1.0 for v in kept if v) / len(kept)


def as_float(value: float | int | None) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def mlp_delta_cosine(update: torch.Tensor, residual: torch.Tensor) -> float | None:
    if update.numel() == 0 or residual.numel() == 0:
        return None
    update_f = update.float().flatten()
    residual_f = residual.float().flatten()
    if float(update_f.norm().item()) == 0.0 or float(residual_f.norm().item()) == 0.0:
        return None
    return float(F.cosine_similarity(update_f, residual_f, dim=0).item())


def negative_parallel_component(update: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """Return the component of `update` that points opposite `residual`.

    If the update has non-negative projection on the residual, the returned
    component is all zeros.
    """

    update_f = update.float()
    residual_f = residual.float()
    denom = torch.dot(residual_f.flatten(), residual_f.flatten()).clamp_min(1e-12)
    coeff = torch.dot(update_f.flatten(), residual_f.flatten()) / denom
    if float(coeff.item()) >= 0.0:
        return torch.zeros_like(update_f)
    return coeff * residual_f


def negative_parallel_norm(update: torch.Tensor, residual: torch.Tensor) -> float:
    return float(negative_parallel_component(update, residual).norm().item())


def productive_opposition(delta_cosine: float | None, margin_writein: float | None) -> bool | None:
    delta = as_float(delta_cosine)
    margin = as_float(margin_writein)
    if delta is None or margin is None:
        return None
    return delta < 0.0 and margin > 0.0


def collapse_category(raw: str | None) -> str:
    if raw in {"STRUCTURAL", "DISCOURSE", "PUNCTUATION", "FORMAT"}:
        return "FORMAT"
    if raw == "CONTENT":
        return "CONTENT"
    return "FUNCTION_OTHER"


def summarize_categories(categories: Sequence[str | None]) -> dict[str, int]:
    counts = Counter(collapse_category(cat) for cat in categories if cat is not None)
    return {name: int(counts.get(name, 0)) for name in COLLAPSED_CATEGORIES}


def summarize_metric_rows(rows: Sequence[dict], key: str) -> float | None:
    return finite_mean(row.get(key) for row in rows)


def summarize_bool_rows(rows: Sequence[dict], key: str) -> float | None:
    return finite_rate(row.get(key) for row in rows)

