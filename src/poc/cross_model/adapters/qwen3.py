"""Adapter for Qwen 3 4B (Qwen/Qwen3-4B-Base / Qwen3-4B)."""
from .base import ModelAdapter


class Qwen3Adapter(ModelAdapter):

    def apply_template(self, tokenizer, prompt: str, is_it: bool) -> str:
        if not is_it:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        try:
            return tokenizer.apply_chat_template(
                messages,
                # Qwen 3 supports thinking mode; disable for comparability with other models.
                # Qwen 2.5 ignores this template kwarg on current Transformers.
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
