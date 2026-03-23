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
    logit_lens_entropy: list[list[float]]   # empty in exp5 — kept for schema compat
    top1_token_per_layer: list[list[int]]   # empty in exp5 — kept for schema compat


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
    """Generate one record under the given intervention.

    Speed optimisations vs the original implementation:
    - Only nnsave at checkpoint_layers (4 saves vs 34) — eliminates 30 unnecessary
      GPU→CPU tensor copies per step.
    - Logit lens (34 × W_U matmuls per step) is skipped entirely — it was computed
      but never used by any exp5 scorer or written to the output JSONL.
    - hooks_for_model is called once per record (unchanged) but the result is
      looked up by key, not recomputed inside the token loop.
    """
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
    checkpoint_set = set(cfg.checkpoint_layers)

    intervention.validate(n_layers=cfg.n_layers)

    for _ in range(cfg.max_gen_tokens):
        checkpoint_saves: dict[int, Any] = {}
        with loaded.model.trace(current_ids):
            layers_root = navigate_path(loaded.model, hooks.layers_root)
            for layer_i in range(cfg.n_layers):
                layer = layers_root[layer_i]
                # Apply ablation intervention (all layers, as before).
                intervention.apply_in_trace(layer, layer_i, hooks)
                # Only nnsave at checkpoint layers — 4 saves instead of 34.
                # The other 30 residuals were transferred to CPU and discarded;
                # skipping them eliminates ~88% of per-step GPU→CPU copies.
                if layer_i in checkpoint_set:
                    checkpoint_saves[layer_i] = nnsave(
                        navigate_path(layer, hooks.layer_residual)
                    )
            logits_save = nnsave(navigate_path(loaded.model, hooks.lm_head).output)

        logits = logits_save[0, -1, :].float()
        masked_logits = logits.clone()
        masked_logits[~loaded.real_token_mask] = float("-inf")
        next_token_id = int(masked_logits.argmax().item())
        generated_tokens.append(
            {"token_id": next_token_id, "token_str": tokenizer.decode([next_token_id])}
        )

        # Save checkpoint hidden states (last-token position only).
        for li, saved in checkpoint_saves.items():
            hidden_states[li].append(saved[0, -1, :].detach().float().cpu().numpy())

        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_token_id]], device=device)], dim=1
        )
        if next_token_id in eos_ids:
            break

    generated_text = tokenizer.decode(
        [t["token_id"] for t in generated_tokens], skip_special_tokens=True
    )
    hidden_arrays = {
        li: np.stack(steps, axis=0) if steps else np.zeros((0, cfg.d_model), dtype=np.float32)
        for li, steps in hidden_states.items()
    }
    return GeneratedSample(
        record_id=record["id"],
        prompt=record["formats"][cfg.prompt_format],
        generated_text=generated_text,
        generated_tokens=generated_tokens,
        hidden_states=hidden_arrays,
        logit_lens_entropy=[],      # not used in exp5; omitted for speed
        top1_token_per_layer=[],    # not used in exp5; omitted for speed
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
