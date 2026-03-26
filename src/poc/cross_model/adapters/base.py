"""
Abstract base class for model-architecture-specific adapters.

All 6 target models follow the standard HuggingFace CausalLM convention:
  - Transformer layers at model.model.layers[i]
  - Layer output is a tuple (hidden_states, ...)
  - Final LayerNorm at model.model.norm
  - LM head at model.lm_head

Each adapter subclass only needs to override apply_template() and
optionally stop_token_ids() for IT-specific stop tokens.
"""
from __future__ import annotations
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ModelAdapter(ABC):
    """Provides model-architecture-specific access for cross-model analysis."""

    def layers(self, model: nn.Module) -> list[nn.Module]:
        """Return the list of transformer block modules in forward order."""
        return list(model.model.layers)

    def residual_from_output(self, layer_output) -> torch.Tensor:
        """Extract the residual stream [batch, seq, d_model] from a layer's output.

        All HF decoder layers return a tuple whose first element is hidden_states.
        """
        if isinstance(layer_output, tuple):
            return layer_output[0]
        return layer_output

    def final_norm(self, model: nn.Module) -> nn.Module:
        """The final LayerNorm applied before the LM head."""
        return model.model.norm

    def lm_head(self, model: nn.Module) -> nn.Module:
        """The language model head (Linear layer, [d_model, vocab_size])."""
        return model.lm_head

    @abstractmethod
    def apply_template(self, tokenizer, prompt: str, is_it: bool) -> str:
        """Return the (optionally template-wrapped) prompt string.

        For PT models, always return the raw prompt.
        For IT models, apply the model's chat template.
        """
        ...

    def stop_token_ids(self, tokenizer) -> set[int]:
        """Token IDs that should terminate generation."""
        ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            ids.add(tokenizer.eos_token_id)
        return ids
