"""Adapter for Mistral 7B v0.3 (mistralai/Mistral-7B-v0.3 / -Instruct-v0.3)."""
from .base import ModelAdapter


class MistralAdapter(ModelAdapter):

    def apply_template(self, tokenizer, prompt: str, is_it: bool) -> str:
        if not is_it:
            return prompt
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
