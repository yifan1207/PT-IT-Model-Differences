"""Steering-pipeline model adapter for Exp6.

Bridges the cross-model adapter system (src/poc/cross_model/adapters/) with the
exp6 steering pipeline.  Provides:
  - Layer/MLP/attention module access for hook registration
  - EOS/EOT token ID collection
  - Real-token mask construction (generalised from Gemma's <unusedXXXX> pattern)
  - Logit-lens components (final_norm, lm_head)

All architecture-specific path resolution is delegated to the cross-model
ModelAdapter subclasses; this module just wraps them with convenience methods.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.poc.cross_model.adapters import ModelAdapter, get_adapter
from src.poc.cross_model.config import MODEL_REGISTRY, ModelSpec, get_spec, model_id_for_variant


@dataclass
class SteeringAdapter:
    """Exp6 steering adapter — wraps a cross-model ModelAdapter with steering-specific helpers."""

    model_name: str         # short name, e.g. "llama31_8b"
    spec: ModelSpec          # from MODEL_REGISTRY
    adapter: ModelAdapter    # cross-model adapter instance

    # ── Layer/module access (for hook registration) ─────────────────────────

    def get_layers(self, model_raw: nn.Module) -> list[nn.Module]:
        """Return transformer block modules in forward order."""
        return self.adapter.layers(model_raw)

    def get_layer(self, model_raw: nn.Module, layer_idx: int) -> nn.Module:
        """Return a single transformer block module."""
        return self.adapter.layers(model_raw)[layer_idx]

    def get_mlp(self, model_raw: nn.Module, layer_idx: int) -> nn.Module:
        """Return the MLP sub-module of a layer."""
        return self.get_layer(model_raw, layer_idx).mlp

    def get_attn(self, model_raw: nn.Module, layer_idx: int) -> nn.Module:
        """Return the self-attention sub-module of a layer."""
        return self.get_layer(model_raw, layer_idx).self_attn

    # ── Logit lens ──────────────────────────────────────────────────────────

    def get_final_norm(self, model_raw: nn.Module) -> nn.Module:
        """Final LayerNorm applied before lm_head."""
        return self.adapter.final_norm(model_raw)

    def get_lm_head(self, model_raw: nn.Module) -> nn.Module:
        """Language model head (unembedding)."""
        return self.adapter.lm_head(model_raw)

    # ── EOS tokens ──────────────────────────────────────────────────────────

    def eos_token_ids(self, tokenizer: Any) -> list[int]:
        """All token IDs that should terminate generation."""
        return sorted(self.adapter.stop_token_ids(tokenizer))

    # ── Real-token mask ─────────────────────────────────────────────────────

    def real_token_mask(self, tokenizer: Any, device: torch.device) -> torch.Tensor:
        """Boolean mask [vocab_size] — True for real tokens, False for placeholders.

        Gemma has ~100k <unusedXXXX> placeholder tokens whose W_U columns have
        high norms from random init; generating them produces garbage.  Other
        models typically don't have this issue, so we return all-True for them.

        We also suppress any token that the tokenizer maps to UNK (id == unk_token_id
        AND the token string is a known placeholder pattern).
        """
        vocab_size = len(tokenizer)
        all_token_strs = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))

        # Patterns for tokens to suppress during generation
        _suppress_patterns = [
            re.compile(r"^<unused\d+>$"),          # Gemma
            re.compile(r"^<\|reserved_special_token_\d+\|>$"),  # Llama 3
        ]

        mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        for i, tok_str in enumerate(all_token_strs):
            if tok_str is None:
                mask[i] = False
                continue
            for pat in _suppress_patterns:
                if pat.match(tok_str):
                    mask[i] = False
                    break

        n_real = int(mask.sum().item())
        n_suppressed = vocab_size - n_real
        if n_suppressed > 0:
            print(f"  Vocabulary: {vocab_size} total, {n_real} real tokens, "
                  f"{n_suppressed} placeholder tokens filtered")
        return mask

    # ── Architecture properties ─────────────────────────────────────────────

    @property
    def n_layers(self) -> int:
        return self.spec.n_layers

    @property
    def d_model(self) -> int:
        return self.spec.d_model

    @property
    def corrective_onset(self) -> int:
        return self.spec.corrective_onset

    @property
    def max_gen_tokens(self) -> int:
        """Model-specific max generation length (DeepSeek needs 64)."""
        if self.model_name == "deepseek_v2_lite":
            return 64
        return 200

    def model_id(self, variant: str) -> str:
        """HuggingFace model ID for given variant."""
        return model_id_for_variant(self.spec, variant)


def get_steering_adapter(model_name: str) -> SteeringAdapter:
    """Build a SteeringAdapter from a MODEL_REGISTRY key."""
    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    return SteeringAdapter(model_name=model_name, spec=spec, adapter=adapter)
