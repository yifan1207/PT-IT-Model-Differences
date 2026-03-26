"""Adapter for OLMo 2 7B (allenai/OLMo-2-0325-7B / -Instruct).

OLMo 2 uses a non-standard internal architecture (no pre-norm, post-norm after
each sublayer). However, the standard HF CausalLM interface is the same:
  model.model.layers[i] returns (hidden_states, ...)

OLMo is fully open: weights + data + training recipe (Tülu 3 IT via DPO).
This makes it the most reproducible model in the study.
"""
from .base import ModelAdapter


class OLMo2Adapter(ModelAdapter):

    def apply_template(self, tokenizer, prompt: str, is_it: bool) -> str:
        if not is_it:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
