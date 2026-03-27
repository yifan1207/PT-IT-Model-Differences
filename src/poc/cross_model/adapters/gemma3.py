"""Adapter for Gemma 3 4B (google/gemma-3-4b-pt / -it).

CRITICAL path note:
  AutoModelForCausalLM maps model_type="gemma3" to Gemma3ForConditionalGeneration
  (the multimodal class, even for the text-only -pt variant).

  Actual verified attribute structure:
    model                              Gemma3ForConditionalGeneration
    model.model                        Gemma3Model  (multimodal wrapper)
    model.model.vision_tower           SiglipVisionModel
    model.model.language_model         Gemma3TextModel  ← text backbone
    model.model.language_model.layers  ModuleList[34]   ← transformer blocks
    model.model.language_model.norm    Gemma3RMSNorm    ← final norm
    model.lm_head                      Linear           ← unembedding (top-level)

  Note: model.language_model is a property that returns model.model.language_model
  (a Gemma3TextModel), NOT a Gemma3ForCausalLM wrapper.  Calling .model on it fails.

  model.lm_head is at the TOP LEVEL of Gemma3ForConditionalGeneration, so the
  base-class lm_head() and the direct model.lm_head.weight access in collect_L1L2
  are both correct without override.
"""
import torch
from .base import ModelAdapter


class Gemma3Adapter(ModelAdapter):

    # ── layer / norm / head access ───────────────────────────────────────────

    def layers(self, model) -> list:
        """Transformer blocks at model.model.language_model.layers."""
        return list(model.model.language_model.layers)

    def final_norm(self, model) -> torch.nn.Module:
        return model.model.language_model.norm

    # lm_head: NOT overridden — base class returns model.lm_head which is correct.

    # ── template (NOT used in collection scripts — raw text only) ────────────

    def apply_template(self, tokenizer, prompt: str, is_it: bool) -> str:
        if not is_it:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def stop_token_ids(self, tokenizer) -> set[int]:
        ids = super().stop_token_ids(tokenizer)
        # <end_of_turn> token (id=106) used by Gemma IT models
        end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn, int):
            ids.add(end_of_turn)
        return ids
