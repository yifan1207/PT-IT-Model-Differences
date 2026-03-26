"""Adapter for Llama 3.1 8B (meta-llama/Llama-3.1-8B / -Instruct)."""
from .base import ModelAdapter


class Llama31Adapter(ModelAdapter):

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
        # <|eot_id|> token used by Llama 3 IT to end turns
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int):
            ids.add(eot)
        return ids
