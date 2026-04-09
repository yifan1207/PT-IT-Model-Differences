from __future__ import annotations

import re
from dataclasses import dataclass

import torch


_NUMBERED_LIST_RE = re.compile(r"^\d+\.$")
_HEADER_RE = re.compile(r"^#+$")
_RULE_RE = re.compile(r"^(?:---|===|\*\*\*|-{4,}|={4,}|\*{4,})$")

_TIER2_WORDS = {
    "sure",
    "here",
    "certainly",
    "absolutely",
    "let",
    "i'll",
    "i’d",
    "i'd",
    "however",
    "additionally",
    "furthermore",
    "moreover",
    "therefore",
    "first",
    "second",
    "third",
    "finally",
    "next",
    "step",
    "note",
    "summary",
    "example",
    "conclusion",
    "key",
}


@dataclass(frozen=True)
class StructuralTokenMasks:
    tier1: torch.Tensor
    tier12: torch.Tensor


def _is_tier1(decoded: str) -> bool:
    stripped = decoded.strip()
    if not stripped:
        return False
    if stripped in {"-", "*", "+", "•", "`", "```", ">", "\n\n", "\n- ", "\n* ", "\n#"}:
        return True
    if _HEADER_RE.match(stripped):
        return True
    if _NUMBERED_LIST_RE.match(stripped):
        return True
    if _RULE_RE.match(stripped):
        return True
    if decoded.startswith("\n#") or decoded.startswith("\n-") or decoded.startswith("\n*"):
        return True
    return False


def _is_tier2(decoded: str) -> bool:
    stripped = decoded.strip().lower()
    if not stripped:
        return False
    return stripped in _TIER2_WORDS


def build_structural_token_masks(
    tokenizer,
    logit_dim: int,
    real_token_mask: torch.Tensor,
    device: torch.device,
) -> StructuralTokenMasks:
    """Build model-specific structural token masks on the true logits dimension."""
    tok_vocab_size = len(tokenizer)
    tier1 = torch.zeros(logit_dim, dtype=torch.bool, device=device)
    tier12 = torch.zeros(logit_dim, dtype=torch.bool, device=device)

    for token_id in range(min(tok_vocab_size, logit_dim)):
        if not bool(real_token_mask[token_id].item()):
            continue
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        is_t1 = _is_tier1(decoded)
        is_t2 = _is_tier2(decoded)
        tier1[token_id] = is_t1
        tier12[token_id] = is_t1 or is_t2

    tier1 &= real_token_mask
    tier12 &= real_token_mask
    return StructuralTokenMasks(tier1=tier1, tier12=tier12)

