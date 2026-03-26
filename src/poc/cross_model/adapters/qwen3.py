"""Adapter for Qwen 3 4B (Qwen/Qwen3-4B-Base / Qwen3-4B)."""
from .base import ModelAdapter


class Qwen3Adapter(ModelAdapter):

    def apply_template(self, tokenizer, prompt: str, is_it: bool) -> str:
        if not is_it:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen 3 supports thinking mode; disable for comparability with other models
            enable_thinking=False,
        )
