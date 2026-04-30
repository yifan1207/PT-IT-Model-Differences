"""Generation, masking, and replay scoring helpers for Exp27."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


POSITION_BUCKETS = ("gen_02_04", "gen_05_16", "gen_17_plus")


@dataclass(frozen=True)
class Rollout:
    prompt_id: str
    category: str
    prompt_text: str
    prompt_ids: list[int]
    generated_ids: list[int]
    generated_text: str

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def generated_len(self) -> int:
        return len(self.generated_ids)

    @property
    def full_ids(self) -> list[int]:
        return self.prompt_ids + self.generated_ids


def ensure_padding_token(tokenizer: Any) -> int:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
    return int(tokenizer.pad_token_id)


def first_stop_index(token_ids: list[int], stop_ids: set[int], pad_id: int) -> int:
    for idx, token_id in enumerate(token_ids):
        if int(token_id) == pad_id or int(token_id) in stop_ids:
            return idx
    return len(token_ids)


@torch.no_grad()
def generate_rollouts(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[tuple[str, str, str]],
    device: torch.device,
    max_new_tokens: int,
    max_prompt_tokens: int,
    stop_token_ids: list[int],
) -> list[Rollout]:
    """Greedily generate one native continuation per prompt."""

    pad_id = ensure_padding_token(tokenizer)
    old_padding_side = getattr(tokenizer, "padding_side", "right")
    tokenizer.padding_side = "left"
    try:
        texts = [prompt_text for _, _, prompt_text in prompts]
        encoded = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
        )
    finally:
        tokenizer.padding_side = old_padding_side
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
        eos_token_id=stop_token_ids or tokenizer.eos_token_id,
        use_cache=True,
    )
    new_tokens = outputs[:, input_ids.shape[1] :]
    stop_ids = {int(token_id) for token_id in stop_token_ids if token_id is not None}
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    rollouts: list[Rollout] = []
    for row_idx, (prompt_id, category, prompt_text) in enumerate(prompts):
        prompt_ids = input_ids[row_idx][attention_mask[row_idx].bool()].detach().cpu().tolist()
        generated_raw = [int(token_id) for token_id in new_tokens[row_idx].detach().cpu().tolist()]
        generated_ids = generated_raw[: first_stop_index(generated_raw, stop_ids, pad_id)]
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        rollouts.append(
            Rollout(
                prompt_id=prompt_id,
                category=category,
                prompt_text=prompt_text,
                prompt_ids=[int(token_id) for token_id in prompt_ids],
                generated_ids=generated_ids,
                generated_text=generated_text,
            )
        )
    return rollouts


def pad_rollout_ids(
    rollouts: list[Rollout],
    *,
    pad_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    max_len = max(len(rollout.full_ids) for rollout in rollouts)
    input_rows = []
    mask_rows = []
    prompt_lens = []
    generated_lens = []
    for rollout in rollouts:
        ids = rollout.full_ids
        pad_n = max_len - len(ids)
        input_rows.append(ids + [pad_id] * pad_n)
        mask_rows.append([1] * len(ids) + [0] * pad_n)
        prompt_lens.append(rollout.prompt_len)
        generated_lens.append(rollout.generated_len)
    return (
        torch.tensor(input_rows, dtype=torch.long, device=device),
        torch.tensor(mask_rows, dtype=torch.long, device=device),
        prompt_lens,
        generated_lens,
    )


def build_generated_source_mask(
    *,
    input_ids: torch.Tensor,
    prompt_lens: list[int],
    generated_lens: list[int],
    include_boundary_source: bool = False,
) -> torch.Tensor:
    """Mask source positions whose logits predict measured generated tokens.

    By default this excludes the prompt-boundary source position, so the first
    generated token is not part of the main metric. Set ``include_boundary`` to
    include the prompt's final source position as a secondary check.
    """

    mask = torch.zeros(input_ids.shape, dtype=torch.bool, device=input_ids.device)
    for row_idx, (prompt_len, generated_len) in enumerate(zip(prompt_lens, generated_lens, strict=True)):
        if generated_len <= 0:
            continue
        start = prompt_len - 1 if include_boundary_source else prompt_len
        stop = prompt_len + generated_len - 1
        if start < 0:
            start = 0
        if stop > start:
            mask[row_idx, start:stop] = True
    return mask


def _bucket_for_target_generated_index(target_generated_index: int) -> str:
    if target_generated_index <= 3:
        return "gen_02_04"
    if target_generated_index <= 15:
        return "gen_05_16"
    return "gen_17_plus"


def _mean(values: list[float]) -> float | None:
    kept = [float(value) for value in values if math.isfinite(float(value))]
    if not kept:
        return None
    return sum(kept) / len(kept)


def _score_selected(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    source_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = source_mask.nonzero(as_tuple=False)
    row_idx = indices[:, 0]
    src_idx = indices[:, 1]
    selected_logits = logits[row_idx, src_idx, :].float()
    target_ids = input_ids[row_idx, src_idx + 1]
    target_logits = selected_logits.gather(1, target_ids[:, None]).squeeze(1)
    nll = torch.logsumexp(selected_logits, dim=-1) - target_logits
    return row_idx, nll, target_logits


def score_variant_against_full(
    *,
    full_logits: torch.Tensor,
    variant_logits: torch.Tensor,
    input_ids: torch.Tensor,
    source_mask: torch.Tensor,
    row_ids: list[str],
    prompt_lens: list[int],
) -> dict[str, dict[str, Any]]:
    """Return per-row NLL/logit-drop summaries for one ablation variant."""

    if not bool(source_mask.any().item()):
        return {
            row_id: {
                "n_positions": 0,
                "nll_full": None,
                "nll_variant": None,
                "nll_delta": None,
                "true_logit_full": None,
                "true_logit_variant": None,
                "true_logit_drop": None,
                "position_buckets": {},
            }
            for row_id in row_ids
        }
    full_rows, full_nll, full_target_logits = _score_selected(full_logits, input_ids, source_mask)
    variant_rows, variant_nll, variant_target_logits = _score_selected(variant_logits, input_ids, source_mask)
    if not torch.equal(full_rows, variant_rows):
        raise RuntimeError("Full and variant scoring row indices differ")
    source_indices = source_mask.nonzero(as_tuple=False)[:, 1].detach().cpu().tolist()
    rows_cpu = full_rows.detach().cpu().tolist()
    out: dict[str, dict[str, Any]] = {}
    for row_pos, row_id in enumerate(row_ids):
        selected = [idx for idx, value in enumerate(rows_cpu) if int(value) == row_pos]
        if not selected:
            out[row_id] = {
                "n_positions": 0,
                "nll_full": None,
                "nll_variant": None,
                "nll_delta": None,
                "true_logit_full": None,
                "true_logit_variant": None,
                "true_logit_drop": None,
                "position_buckets": {},
            }
            continue
        full_nll_vals = [float(full_nll[idx].item()) for idx in selected]
        variant_nll_vals = [float(variant_nll[idx].item()) for idx in selected]
        full_logit_vals = [float(full_target_logits[idx].item()) for idx in selected]
        variant_logit_vals = [float(variant_target_logits[idx].item()) for idx in selected]
        bucket_payload: dict[str, dict[str, Any]] = {}
        for bucket in POSITION_BUCKETS:
            bucket_indices = [
                idx
                for idx in selected
                if _bucket_for_target_generated_index(
                    int(source_indices[idx]) + 1 - int(prompt_lens[row_pos])
                )
                == bucket
            ]
            if not bucket_indices:
                continue
            b_full_nll = [float(full_nll[idx].item()) for idx in bucket_indices]
            b_variant_nll = [float(variant_nll[idx].item()) for idx in bucket_indices]
            b_full_logit = [float(full_target_logits[idx].item()) for idx in bucket_indices]
            b_variant_logit = [float(variant_target_logits[idx].item()) for idx in bucket_indices]
            bucket_payload[bucket] = {
                "n_positions": len(bucket_indices),
                "nll_delta": _mean(b_variant_nll) - _mean(b_full_nll),
                "true_logit_drop": _mean(b_full_logit) - _mean(b_variant_logit),
            }
        out[row_id] = {
            "n_positions": len(selected),
            "nll_full": _mean(full_nll_vals),
            "nll_variant": _mean(variant_nll_vals),
            "nll_delta": _mean(variant_nll_vals) - _mean(full_nll_vals),
            "true_logit_full": _mean(full_logit_vals),
            "true_logit_variant": _mean(variant_logit_vals),
            "true_logit_drop": _mean(full_logit_vals) - _mean(variant_logit_vals),
            "position_buckets": bucket_payload,
        }
    return out
