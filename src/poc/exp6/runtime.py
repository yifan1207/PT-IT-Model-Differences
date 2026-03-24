"""Per-record generation for Exp6 steering experiments.

Two generation paths:
  A — nnsight trace with Exp6InterventionSpec (directional_remove / directional_add / controls).
      Same architecture as Exp5 runtime.py.
  B — plain autoregressive generation with register_forward_hook on layernorm/mlp modules
      (transcoder encode/decode cannot run inside nnsight trace proxies).

Both paths return a GeneratedSample6 dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from src.poc.exp6.config import Exp6Config
from src.poc.exp6.interventions import Exp6InterventionSpec


class _RealTokenMaskProcessor(LogitsProcessor):
    """Suppress <unusedXXXX> placeholder tokens during generation."""
    def __init__(self, bad_token_mask: torch.Tensor) -> None:
        self.bad_token_mask = bad_token_mask  # True = suppress

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores[:, self.bad_token_mask.to(scores.device)] = float("-inf")
        return scores


@dataclass
class GeneratedSample6:
    record_id: str
    prompt: str
    generated_text: str
    generated_tokens: list[dict]
    category: str              # GOV-FORMAT | GOV-CONV | ... for stratified analysis


# ── Prompt preparation ────────────────────────────────────────────────────────

def _tokenize_prompt(record: dict, tokenizer: Any, cfg: Exp6Config, device: torch.device) -> torch.Tensor:
    prompt = record["formats"][cfg.prompt_format]
    if cfg.model_variant == "it" and cfg.apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
    return tokenizer.encode(prompt, return_tensors="pt").to(device)


def _eos_token_ids(tokenizer: Any, cfg: Exp6Config) -> list[int]:
    eos = [tokenizer.eos_token_id]
    if cfg.model_variant == "it":
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos.append(eot)
    return eos


# ── Path A: batched hook-based generation ────────────────────────────────────

_BATCH_SIZE_A = 8   # records per GPU batch; H100 80 GB handles this comfortably


def generate_records_A_batch(
    records: list[dict],
    loaded: Any,
    cfg: Exp6Config,
    intervention: Exp6InterventionSpec,
    batch_size: int = _BATCH_SIZE_A,
) -> list[GeneratedSample6]:
    """Batch-generate for a list of records using register_forward_hook + model.generate().

    Groups records into mini-batches (left-padded), runs model.generate() with direction
    hooks active. ~8x faster than sequential single-record generation.

    Benchmarked: batch=8 → ~212 tok/s vs ~34 tok/s sequential (6x+ improvement).
    """
    model_raw = loaded.model._model   # Gemma3ForConditionalGeneration
    tokenizer = loaded.tokenizer
    device = next(model_raw.parameters()).device

    intervention.validate(n_layers=cfg.n_layers)

    # Left-pad so all seqs in a batch are aligned to the right
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    bad_mask = ~loaded.real_token_mask.to(device)
    logits_processor = LogitsProcessorList([_RealTokenMaskProcessor(bad_mask)])
    eos_ids = _eos_token_ids(tokenizer, cfg)

    results: list[GeneratedSample6] = []

    try:
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start: batch_start + batch_size]

            # Tokenize batch
            prompts = [r["formats"][cfg.prompt_format] for r in batch]
            if cfg.model_variant == "it" and cfg.apply_chat_template:
                encoded = tokenizer.apply_chat_template(
                    [[{"role": "user", "content": p}] for p in prompts],
                    return_tensors="pt",
                    add_generation_prompt=True,
                    padding=True,
                ).to(device)
            else:
                encoded = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to(device)

            attn_mask = (encoded != tokenizer.pad_token_id).long()

            handles = intervention.register_hooks(model_raw, cfg)
            try:
                with torch.no_grad():
                    out_ids = model_raw.generate(
                        encoded,
                        attention_mask=attn_mask,
                        max_new_tokens=cfg.max_gen_tokens,
                        do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        logits_processor=logits_processor,
                        use_cache=True,
                    )
            finally:
                for h in handles:
                    h.remove()

            prompt_len = encoded.shape[1]
            for i, rec in enumerate(batch):
                new_ids = [
                    tid for tid in out_ids[i, prompt_len:].tolist()
                    if tid != tokenizer.pad_token_id
                ]
                gen_tokens = [{"token_id": tid, "token_str": tokenizer.decode([tid])} for tid in new_ids]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                results.append(GeneratedSample6(
                    record_id=rec["id"],
                    prompt=rec["formats"][cfg.prompt_format],
                    generated_text=text,
                    generated_tokens=gen_tokens,
                    category=rec.get("category", ""),
                ))
    finally:
        tokenizer.padding_side = orig_padding_side

    return results


# Thin single-record wrapper (used by _select_records path for small subsets)
def generate_record_A(
    record: dict,
    loaded: Any,
    cfg: Exp6Config,
    intervention: Exp6InterventionSpec,
) -> GeneratedSample6:
    return generate_records_A_batch([record], loaded, cfg, intervention, batch_size=1)[0]


# ── Path B: plain-hook generation ────────────────────────────────────────────

def _make_feature_clamp_hooks(
    model_raw: Any,
    transcoders: list,
    governance_features: dict[str, list[int]],
    mean_acts: dict[int, np.ndarray],
    cfg: Exp6Config,
) -> list:
    """Register forward hooks to clamp governance features at γ × mean_activation.

    Strategy:
      1. Hook `pre_feedforward_layernorm.output` to capture the MLP input.
      2. Hook `mlp` (forward hook, replaces output) to run transcoder encode →
         clamp governance feature values → transcoder decode → return as new output.

    This approach avoids nnsight entirely and uses the underlying HuggingFace model
    directly, following the precedent in scripts/precompute_extra_directions.py.
    """
    handles = []
    gamma = cfg.gamma
    layer_range = set(cfg.corrective_layers)

    for l_idx in layer_range:
        pre_ln = model_raw.language_model.layers[l_idx].pre_feedforward_layernorm
        mlp_mod = model_raw.language_model.layers[l_idx].mlp
        tc = transcoders[l_idx]
        feat_key = f"layer_{l_idx}"
        target_feats = governance_features.get(feat_key, [])
        mean_acts_l = torch.tensor(mean_acts.get(l_idx, np.zeros(tc.W_dec.shape[0])),
                                   dtype=torch.float32)

        # Closure captures layer-specific variables
        def make_hooks(layer_idx: int, transcoder, feats: list[int], means: torch.Tensor):
            buf: list[torch.Tensor | None] = [None]

            def pre_hook(mod: Any, inp: tuple, out: torch.Tensor) -> None:
                buf[0] = out  # save layernorm output [B, T, d_model]

            def mlp_hook(mod: Any, inp: tuple, out: torch.Tensor) -> torch.Tensor:
                x = buf[0]
                if x is None or not feats:
                    return out
                orig_shape = out.shape
                with torch.no_grad():
                    x_flat = x.reshape(-1, x.shape[-1]).to(transcoder.W_enc.dtype)
                    # Encode → [B*T, n_features]
                    encoded = transcoder.encode(x_flat)
                    # Clamp governance features
                    means_dev = means.to(device=encoded.device, dtype=encoded.dtype)
                    for f in feats:
                        if f < encoded.shape[-1]:
                            encoded[:, f] = gamma * means_dev[f]
                    # Decode back to residual stream
                    decoded = transcoder.decode(encoded)
                    return decoded.reshape(orig_shape).to(out.dtype)

            return pre_hook, mlp_hook

        ph, mh = make_hooks(l_idx, tc, target_feats, mean_acts_l)
        handles.append(pre_ln.register_forward_hook(ph))
        handles.append(mlp_mod.register_forward_hook(mh))

    return handles


def _make_wdec_inject_hooks(
    model_raw: Any,
    governance_direction: dict[int, torch.Tensor],
    cfg: Exp6Config,
) -> list:
    """Register hooks to inject W_dec-projected governance direction into MLP output.

    Uses the same directional_inject_tensor formula (h' = h + β × d̂ × ‖h‖) but
    with a direction constructed bottom-up from governance features' decoder vectors
    rather than top-down from the full IT-PT mean difference.
    """
    from src.poc.exp6.interventions import directional_inject_tensor
    handles = []
    beta = cfg.directional_beta
    layer_range = set(cfg.corrective_layers)

    for l_idx in layer_range:
        if l_idx not in governance_direction:
            continue
        mlp_mod = model_raw.language_model.layers[l_idx].mlp
        direction = governance_direction[l_idx]

        def make_hook(d: torch.Tensor):
            def hook(mod: Any, inp: tuple, out: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    d_dev = d.to(device=out.device, dtype=out.dtype)
                    return directional_inject_tensor(out, d_dev, beta)
            return hook

        handles.append(mlp_mod.register_forward_hook(make_hook(direction)))

    return handles


_BATCH_SIZE_B = 8


def generate_records_B_batch(
    records: list[dict],
    model_raw: Any,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    cfg: Exp6Config,
    hooks_config: dict,
    batch_size: int = _BATCH_SIZE_B,
) -> list[GeneratedSample6]:
    """Batched generation for B experiments using register_forward_hook + model.generate().

    Same approach as generate_records_A_batch but registers B-specific hooks
    (feature_clamp or wdec_inject) instead of direction hooks.  ~20× faster than
    the original token-by-token loop because model.generate() uses KV cache.
    """
    device = next(model_raw.parameters()).device
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    bad_mask = ~real_token_mask.to(device)
    logits_processor = LogitsProcessorList([_RealTokenMaskProcessor(bad_mask)])
    eos_ids = _eos_token_ids(tokenizer, cfg)

    results: list[GeneratedSample6] = []
    try:
        for batch_start in range(0, len(records), batch_size):
            batch = records[batch_start: batch_start + batch_size]
            prompts = [r["formats"][cfg.prompt_format] for r in batch]
            if cfg.model_variant == "it" and cfg.apply_chat_template:
                encoded = tokenizer.apply_chat_template(
                    [[{"role": "user", "content": p}] for p in prompts],
                    return_tensors="pt",
                    add_generation_prompt=True,
                    padding=True,
                ).to(device)
            else:
                encoded = tokenizer(prompts, return_tensors="pt", padding=True)["input_ids"].to(device)

            attn_mask = (encoded != tokenizer.pad_token_id).long()

            # Register B-specific hooks
            hook_handles: list = []
            if cfg.method == "feature_clamp":
                hook_handles = _make_feature_clamp_hooks(
                    model_raw,
                    hooks_config["transcoders"],
                    hooks_config["governance_features"],
                    hooks_config["mean_acts"],
                    cfg,
                )
            elif cfg.method == "wdec_inject":
                hook_handles = _make_wdec_inject_hooks(
                    model_raw,
                    hooks_config["governance_direction"],
                    cfg,
                )

            try:
                with torch.no_grad():
                    out_ids = model_raw.generate(
                        encoded,
                        attention_mask=attn_mask,
                        max_new_tokens=cfg.max_gen_tokens,
                        do_sample=False,
                        eos_token_id=eos_ids,
                        pad_token_id=tokenizer.pad_token_id,
                        logits_processor=logits_processor,
                        use_cache=True,
                    )
            finally:
                for h in hook_handles:
                    h.remove()

            prompt_len = encoded.shape[1]
            for i, rec in enumerate(batch):
                new_ids = [
                    tid for tid in out_ids[i, prompt_len:].tolist()
                    if tid != tokenizer.pad_token_id
                ]
                gen_tokens = [{"token_id": tid, "token_str": tokenizer.decode([tid])} for tid in new_ids]
                text = tokenizer.decode(new_ids, skip_special_tokens=True)
                results.append(GeneratedSample6(
                    record_id=rec["id"],
                    prompt=rec["formats"][cfg.prompt_format],
                    generated_text=text,
                    generated_tokens=gen_tokens,
                    category=rec.get("category", ""),
                ))
    finally:
        tokenizer.padding_side = orig_padding_side

    return results


def generate_record_B(
    record: dict,
    model_raw: Any,
    tokenizer: Any,
    real_token_mask: torch.Tensor,
    cfg: Exp6Config,
    hooks_config: dict,  # {"transcoders": ..., "governance_features": ..., "mean_acts": ..., "governance_direction": ...}
) -> GeneratedSample6:
    """Autoregressive generation with plain forward hooks (Approach B).

    Uses the raw HuggingFace model (not nnsight-wrapped) with register_forward_hook
    to intercept MLP layers. This is necessary because transcoder.encode/decode
    cannot be called on nnsight proxy tensors.

    Args:
        model_raw: The underlying HuggingFace CausalLM model.
        tokenizer: HuggingFace tokenizer.
        real_token_mask: [vocab_size] bool mask for valid tokens.
        hooks_config: dict with intervention-specific tensors:
            For feature_clamp: "transcoders", "governance_features", "mean_acts"
            For wdec_inject: "governance_direction"
    """
    device = next(model_raw.parameters()).device

    # Prepare input
    prompt = record["formats"][cfg.prompt_format]
    if cfg.model_variant == "it" and cfg.apply_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    eos = {tokenizer.eos_token_id}
    if cfg.model_variant == "it":
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos.add(eot)

    # Register hooks based on method
    hook_handles: list = []
    if cfg.method == "feature_clamp":
        hook_handles = _make_feature_clamp_hooks(
            model_raw,
            hooks_config["transcoders"],
            hooks_config["governance_features"],
            hooks_config["mean_acts"],
            cfg,
        )
    elif cfg.method == "wdec_inject":
        hook_handles = _make_wdec_inject_hooks(
            model_raw,
            hooks_config["governance_direction"],
            cfg,
        )

    generated_tokens: list[dict] = []
    current_ids = input_ids.clone()

    try:
        for _ in range(cfg.max_gen_tokens):
            with torch.no_grad():
                outputs = model_raw(current_ids)
            logits = outputs.logits[0, -1, :].float()
            masked = logits.clone()
            masked[~real_token_mask.to(device)] = float("-inf")
            next_id = int(masked.argmax().item())
            generated_tokens.append({"token_id": next_id, "token_str": tokenizer.decode([next_id])})
            current_ids = torch.cat([current_ids, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id in eos:
                break
    finally:
        for h in hook_handles:
            h.remove()

    text = tokenizer.decode([t["token_id"] for t in generated_tokens], skip_special_tokens=True)
    return GeneratedSample6(
        record_id=record["id"],
        prompt=prompt,
        generated_text=text,
        generated_tokens=generated_tokens,
        category=record.get("category", ""),
    )
