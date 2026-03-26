"""Adapter for DeepSeek-V2-Lite (deepseek-ai/DeepSeek-V2-Lite / -Chat).

DeepSeek-V2-Lite is an MoE model (16B total, 2.4B active per token).
Architecture details:
  - 27 transformer layers (0-26)
  - Not all layers are MoE — some have dense MLP (check config.json moe_layer_freq)
  - Multi-Head Latent Attention (MLA) compresses KV into a low-rank latent space
  - output_attentions=True returns decompressed [batch, n_heads, T, T] weights

The residual stream analysis works identically to dense models:
  model.model.layers[i] output[0] is the post-residual hidden state.
"""
from .base import ModelAdapter


class DeepseekV2Adapter(ModelAdapter):

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
        # DeepSeek Chat uses <|end_of_sentence|> as stop token
        eos_candidates = ["<|end_of_sentence|>", "<｜end▁of▁sentence｜>"]
        for tok in eos_candidates:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int):
                ids.add(tid)
        return ids
