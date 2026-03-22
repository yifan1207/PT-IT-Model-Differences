from __future__ import annotations

import math

import torch


def perplexity_from_nlls(nlls: list[float]) -> float:
    if not nlls:
        return float("nan")
    return float(math.exp(sum(nlls) / len(nlls)))


def token_nll_from_logits(logits: torch.Tensor, token_id: int) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return float(-log_probs[token_id].item())

