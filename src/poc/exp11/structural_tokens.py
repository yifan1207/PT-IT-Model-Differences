from __future__ import annotations

import re
from dataclasses import dataclass

import torch


_NUMBERED_LIST_RE = re.compile(r"^\d+\.$")
_HEADER_RE = re.compile(r"^#+$")
_RULE_RE = re.compile(r"^(?:---|===|\*\*\*|-{4,}|={4,}|\*{4,})$")


@dataclass(frozen=True)
class StructuralTokenMasks:
    tier1: torch.Tensor


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


def build_structural_token_masks(
    tokenizer,
    logit_dim: int,
    real_token_mask: torch.Tensor,
    device: torch.device,
) -> StructuralTokenMasks:
    """Build model-specific structural token masks on the true logits dimension."""
    tok_vocab_size = len(tokenizer)
    tier1 = torch.zeros(logit_dim, dtype=torch.bool, device=device)

    for token_id in range(min(tok_vocab_size, logit_dim)):
        if not bool(real_token_mask[token_id].item()):
            continue
        decoded = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        is_t1 = _is_tier1(decoded)
        tier1[token_id] = is_t1

    tier1 &= real_token_mask
    return StructuralTokenMasks(tier1=tier1)
