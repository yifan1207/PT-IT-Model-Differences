"""Metric primitives for Exp22 endpoint-deconfounded convergence-gap analysis.

The collector computes full-vocabulary distributions transiently on GPU, then
stores only compact scalar arrays and top-k summaries.  The analyzer reuses the
endpoint-free helpers here to avoid silently changing conventions between
collection and post-processing.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import torch


def exact_kl_from_logprobs(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Return KL(P || Q) for log-probability vectors with shared support."""

    finite = torch.isfinite(logp)
    p = torch.exp(torch.where(finite, logp, torch.zeros_like(logp)))
    terms = p * (torch.where(finite, logp, torch.zeros_like(logp)) - torch.where(finite, logq, torch.zeros_like(logq)))
    return torch.sum(torch.where(finite, terms, torch.zeros_like(terms)))


def exact_js_from_logprobs(logp: torch.Tensor, logq: torch.Tensor) -> torch.Tensor:
    """Return Jensen-Shannon divergence between two log-probability vectors."""

    finite = torch.isfinite(logp) | torch.isfinite(logq)
    p = torch.exp(torch.where(torch.isfinite(logp), logp, torch.full_like(logp, float("-inf"))))
    q = torch.exp(torch.where(torch.isfinite(logq), logq, torch.full_like(logq, float("-inf"))))
    m = 0.5 * (p + q)
    logm = torch.where(m > 0, torch.log(m), torch.zeros_like(m))
    kl_pm = p * (torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp)) - logm)
    kl_qm = q * (torch.where(torch.isfinite(logq), logq, torch.zeros_like(logq)) - logm)
    return 0.5 * torch.sum(torch.where(finite, kl_pm, torch.zeros_like(kl_pm))) + 0.5 * torch.sum(
        torch.where(finite, kl_qm, torch.zeros_like(kl_qm))
    )


def entropy_from_logprobs(logp: torch.Tensor) -> torch.Tensor:
    """Return exact entropy from a log-probability vector."""

    finite = torch.isfinite(logp)
    p = torch.exp(torch.where(finite, logp, torch.zeros_like(logp)))
    return -torch.sum(torch.where(finite, p * logp, torch.zeros_like(logp)))


def top1_confidence_and_margin(logp: torch.Tensor) -> tuple[float, float]:
    """Return top-1 probability and top1-top2 log-probability margin."""

    k = min(2, int(logp.numel()))
    vals = torch.topk(logp, k=k).values
    confidence = float(torch.exp(vals[0]).item())
    margin = float((vals[0] - vals[1]).item()) if k > 1 else float("nan")
    return confidence, margin


def distribution_arrays_from_logits(logits_by_layer: torch.Tensor, top_k: int = 5) -> dict[str, Any]:
    """Compute compact per-layer distribution summaries from full logits.

    Args:
        logits_by_layer: Tensor ``[n_layers, vocab]``. Invalid tokens should
            already be set to ``-inf``.
        top_k: Number of top tokens/logprobs to retain.

    Returns:
        JSON-serialisable arrays for KL-to-final, entropy, confidence,
        top1-top2 margin, top-k IDs/logprobs, adjacent KL, and adjacent JS.
    """

    if logits_by_layer.ndim != 2:
        raise ValueError(f"expected [n_layers, vocab] logits, got {tuple(logits_by_layer.shape)}")
    n_layers = int(logits_by_layer.shape[0])
    k = min(top_k, int(logits_by_layer.shape[1]))
    logp = torch.log_softmax(logits_by_layer.float(), dim=-1)
    final_logp = logp[-1]

    kl_to_final: list[float] = []
    entropy: list[float] = []
    confidence: list[float] = []
    top1_margin: list[float] = []
    adjacent_kl: list[float | None] = []
    adjacent_js: list[float | None] = []

    top_vals, top_ids = torch.topk(logp, k=k, dim=-1)
    for layer in range(n_layers):
        lp = logp[layer]
        kl_to_final.append(float(exact_kl_from_logprobs(lp, final_logp).item()))
        entropy.append(float(entropy_from_logprobs(lp).item()))
        conf, margin = top1_confidence_and_margin(lp)
        confidence.append(conf)
        top1_margin.append(margin)
        if layer < n_layers - 1:
            adjacent_kl.append(float(exact_kl_from_logprobs(lp, logp[layer + 1]).item()))
            adjacent_js.append(float(exact_js_from_logprobs(lp, logp[layer + 1]).item()))
        else:
            adjacent_kl.append(None)
            adjacent_js.append(None)

    return {
        "kl_to_final": finite_or_none_nested(kl_to_final),
        "entropy": finite_or_none_nested(entropy),
        "confidence": finite_or_none_nested(confidence),
        "top1_margin": finite_or_none_nested(top1_margin),
        "top1_ids": [[int(x[0])] for x in top_ids.detach().cpu().tolist()],
        "top5_ids": [[int(y) for y in row] for row in top_ids.detach().cpu().tolist()],
        "top5_logprobs": finite_or_none_nested(top_vals.detach().cpu().tolist()),
        "adjacent_kl": finite_or_none_nested(adjacent_kl),
        "adjacent_js": finite_or_none_nested(adjacent_js),
    }


def finite_or_none(value: float | int | None) -> float | None:
    """Convert non-finite numeric values to ``None`` for strict JSON records."""

    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        return None
    return out


def finite_or_none_nested(value: Any) -> Any:
    """Recursively convert non-finite numeric values to ``None``."""

    if isinstance(value, (list, tuple)):
        return [finite_or_none_nested(v) for v in value]
    if isinstance(value, (float, int)) or value is None:
        return finite_or_none(value)
    return value


def as_float(value: Any) -> float:
    """Parse JSON scalar values, mapping missing/non-finite values to NaN."""

    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def late_start_layer(n_layers: int, frac: float = 0.8) -> int:
    """Start index of the final-depth window used for Exp22 summaries."""

    return max(0, min(n_layers - 1, math.floor(float(n_layers) * frac)))


def finite_mean(values: list[float]) -> float:
    """Mean over finite values, returning NaN for an empty finite set."""

    kept = [float(v) for v in values if math.isfinite(float(v))]
    if not kept:
        return float("nan")
    return float(sum(kept) / len(kept))


def late_kl_mean(kl_to_final: list[Any], *, include_final: bool = True) -> float:
    """Mean KL-to-final in the final 20% of layers."""

    n_layers = len(kl_to_final)
    start = late_start_layer(n_layers)
    end = n_layers if include_final else max(start, n_layers - 1)
    return finite_mean([as_float(v) for v in kl_to_final[start:end]])


def remaining_adjacent_path(adjacent_values: list[Any], *, divergence: str = "js") -> float:
    """Mean remaining adjacent path length from each late pre-final layer.

    The returned scalar is endpoint-free: it sums adjacent layer-to-layer
    movement from a late layer to the end rather than comparing that layer
    directly to the final distribution.
    """

    if divergence not in {"js", "kl"}:
        raise ValueError("divergence must be 'js' or 'kl'")
    vals = [as_float(v) for v in adjacent_values]
    n_layers = len(vals)
    start = late_start_layer(n_layers)
    totals: list[float] = []
    for layer in range(start, max(start, n_layers - 1)):
        totals.append(sum(v for v in vals[layer : n_layers - 1] if math.isfinite(v)))
    return finite_mean(totals)


def future_top1_flips(top1_ids: list[Any]) -> float:
    """Mean number of future top-1 changes from late pre-final layers."""

    ids = [_first_id(x) for x in top1_ids]
    n_layers = len(ids)
    start = late_start_layer(n_layers)
    counts: list[float] = []
    for layer in range(start, max(start, n_layers - 1)):
        count = 0
        for j in range(layer, n_layers - 1):
            if ids[j] is not None and ids[j + 1] is not None and ids[j] != ids[j + 1]:
                count += 1
        counts.append(float(count))
    return finite_mean(counts)


def top5_churn(top5_ids: list[list[int]]) -> float:
    """Mean adjacent top-5 Jaccard distance in the late window."""

    n_layers = len(top5_ids)
    start = late_start_layer(n_layers)
    churns: list[float] = []
    for layer in range(start, max(start, n_layers - 1)):
        a = {int(x) for x in top5_ids[layer] if x is not None}
        b = {int(x) for x in top5_ids[layer + 1] if x is not None}
        if not a and not b:
            continue
        union = a | b
        churns.append(1.0 - (len(a & b) / len(union)))
    return finite_mean(churns)


def stable_top5_entry_layer(top5_ids: list[list[int]], target_id: int | None) -> int | None:
    """Earliest layer where ``target_id`` enters top-5 and never leaves."""

    if target_id is None:
        return None
    n_layers = len(top5_ids)
    for layer in range(n_layers):
        if all(target_id in {int(x) for x in row if x is not None} for row in top5_ids[layer:]):
            return layer
    return None


def final_top1_stable_top5_entry(top1_ids: list[Any], top5_ids: list[list[int]]) -> float:
    """Normalised stable top-5 entry depth for the final top-1 token."""

    target = _first_id(top1_ids[-1]) if top1_ids else None
    entry = stable_top5_entry_layer(top5_ids, target)
    if entry is None or len(top5_ids) <= 1:
        return float("nan")
    return float(entry / (len(top5_ids) - 1))


def late_consensus_stable_top5_entry(top1_ids: list[Any], top5_ids: list[list[int]]) -> float:
    """Normalised stable top-5 entry depth for the late-window consensus top-1."""

    n_layers = len(top1_ids)
    start = late_start_layer(n_layers)
    late_ids = [_first_id(x) for x in top1_ids[start:] if _first_id(x) is not None]
    if not late_ids:
        return float("nan")
    target = Counter(late_ids).most_common(1)[0][0]
    entry = stable_top5_entry_layer(top5_ids, target)
    if entry is None or n_layers <= 1:
        return float("nan")
    return float(entry / (n_layers - 1))


def _first_id(value: Any) -> int | None:
    if isinstance(value, list):
        if not value:
            return None
        value = value[0]
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def derived_step_metrics(probe_payload: dict[str, Any], step_idx: int) -> dict[str, float]:
    """Compute analyzer-side scalar metrics for one generated token step."""

    kl = probe_payload["kl_to_final"][step_idx]
    adjacent_js = probe_payload["adjacent_js"][step_idx]
    adjacent_kl = probe_payload["adjacent_kl"][step_idx]
    top1_ids = probe_payload["top1_ids"][step_idx]
    top5_ids = probe_payload["top5_ids"][step_idx]
    entropy = probe_payload["entropy"][step_idx]
    confidence = probe_payload["confidence"][step_idx]
    margin = probe_payload["top1_margin"][step_idx]
    final_idx = len(kl) - 1
    return {
        "late_kl_mean": late_kl_mean(kl, include_final=True),
        "prefinal_late_kl_mean": late_kl_mean(kl, include_final=False),
        "remaining_adj_js": remaining_adjacent_path(adjacent_js, divergence="js"),
        "remaining_adj_kl": remaining_adjacent_path(adjacent_kl, divergence="kl"),
        "future_top1_flips": future_top1_flips(top1_ids),
        "top5_churn": top5_churn(top5_ids),
        "final_top1_stable_top5_entry": final_top1_stable_top5_entry(top1_ids, top5_ids),
        "late_consensus_stable_top5_entry": late_consensus_stable_top5_entry(top1_ids, top5_ids),
        "final_entropy": as_float(entropy[final_idx]),
        "final_confidence": as_float(confidence[final_idx]),
        "final_top1_margin": as_float(margin[final_idx]),
    }

