from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from nnsight import save as nnsave

from src.poc.exp5.config import Exp5Config
from src.poc.exp5.interventions import InterventionSpec, PrecomputedAblationStats
from src.poc.exp5.utils import navigate_path
from src.poc.shared.collect_config import hooks_for_model


@dataclass
class GeneratedSample:
    record_id: str
    prompt: str
    generated_text: str
    generated_tokens: list[dict]
    hidden_states: dict[int, np.ndarray]
    logit_lens_entropy: list[list[float]]
    top1_token_per_layer: list[list[int]]


def _entropy(logits: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
    probs = torch.softmax(logits.float(), dim=-1)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


def _prepare_input_ids(record: dict, loaded: Any, cfg: Exp5Config) -> torch.Tensor:
    tokenizer = loaded.tokenizer
    device = loaded.W_U.device
    prompt = record["formats"][cfg.prompt_format]
    if cfg.model_variant == "it" and cfg.apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Complete the following: {prompt}"}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
    return tokenizer.encode(prompt, return_tensors="pt").to(device)


def generate_record(
    record: dict,
    loaded: Any,
    cfg: Exp5Config,
    intervention: InterventionSpec,
) -> GeneratedSample:
    tokenizer = loaded.tokenizer
    device = loaded.W_U.device
    hooks = hooks_for_model(cfg.model_id)
    input_ids = _prepare_input_ids(record, loaded, cfg)
    current_ids = input_ids.clone()

    eos_ids = {tokenizer.eos_token_id}
    if cfg.model_variant == "it":
        eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_ids.add(eot)

    generated_tokens: list[dict] = []
    hidden_states: dict[int, list[np.ndarray]] = {li: [] for li in cfg.checkpoint_layers}
    ll_entropy: list[list[float]] = []
    top1_by_layer: list[list[int]] = []

    intervention.validate()

    for _ in range(cfg.max_gen_tokens):
        with loaded.model.trace(current_ids):
            residual_saves = []
            layers_root = navigate_path(loaded.model, hooks.layers_root)
            for layer_i in range(cfg.n_layers):
                layer = layers_root[layer_i]
                intervention.apply_in_trace(layer, layer_i, hooks)
                residual_saves.append(nnsave(navigate_path(layer, hooks.layer_residual)))
            logits_save = nnsave(navigate_path(loaded.model, hooks.lm_head).output)

        residuals = [r[0, -1, :].float() for r in residual_saves]
        logits = logits_save[0, -1, :].float()
        masked_logits = logits.clone()
        masked_logits[~loaded.real_token_mask] = float("-inf")
        next_token_id = int(masked_logits.argmax().item())
        generated_tokens.append(
            {"token_id": next_token_id, "token_str": tokenizer.decode([next_token_id])}
        )

        final_norm = navigate_path(loaded.model, hooks.final_norm)
        step_ent: list[float] = []
        step_top1: list[int] = []
        with torch.inference_mode():
            for layer_i, h in enumerate(residuals):
                ll = final_norm(h.to(device)).float() @ loaded.W_U
                step_ent.append(_entropy(ll, mask=loaded.real_token_mask))
                masked_ll = ll.clone()
                masked_ll[~loaded.real_token_mask] = float("-inf")
                step_top1.append(int(masked_ll.argmax().item()))
            for checkpoint_layer in cfg.checkpoint_layers:
                hidden_states[checkpoint_layer].append(
                    residuals[checkpoint_layer].cpu().to(torch.float32).numpy()
                )

        ll_entropy.append(step_ent)
        top1_by_layer.append(step_top1)

        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_token_id]], device=device)], dim=1
        )
        if next_token_id in eos_ids:
            break

    generated_text = tokenizer.decode(
        [t["token_id"] for t in generated_tokens], skip_special_tokens=True
    )
    hidden_arrays = {
        layer_i: np.stack(steps, axis=0) if steps else np.zeros((0, cfg.d_model), dtype=np.float32)
        for layer_i, steps in hidden_states.items()
    }
    return GeneratedSample(
        record_id=record["id"],
        prompt=record["formats"][cfg.prompt_format],
        generated_text=generated_text,
        generated_tokens=generated_tokens,
        hidden_states=hidden_arrays,
        logit_lens_entropy=ll_entropy,
        top1_token_per_layer=top1_by_layer,
    )


def build_intervention(cfg: Exp5Config) -> InterventionSpec:
    stats = PrecomputedAblationStats.load(
        mean_acts_path=cfg.mean_acts_path,
        corrective_direction_path=cfg.corrective_direction_path,
        resample_bank_path=cfg.resample_bank_path,
        device=cfg.device,
    )
    return InterventionSpec(
        method=cfg.method,
        layers=cfg.ablation_layers,
        alpha=cfg.directional_alpha,
        stats=stats,
        proposal_boundary=cfg.proposal_boundary,
    )

