"""Adapter for OLMo 2 7B (allenai/OLMo-2-1124-7B / -Instruct).

OLMo 2 7B was released 2024-11 (model ID suffix "1124").
Architecture: standard decoder-only transformer with post-norm (RMSNorm after
each sub-layer, unlike typical pre-norm). Standard HF CausalLM interface:
  model.model.layers[i] returns (hidden_states, ...)
Uses full MHA (n_kv_heads == n_heads == 32), not GQA.

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
