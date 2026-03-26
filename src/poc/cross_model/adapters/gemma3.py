"""Adapter for Gemma 3 4B (google/gemma-3-4b-pt / -it).

Note: In the cross-model framework we load Gemma 3 via AutoModelForCausalLM,
which gives Gemma3ForCausalLM with the standard HF path model.model.layers[i].
This is DIFFERENT from the nnsight/circuit-tracer path used in exp2/exp4
(model.language_model.layers[i]), which wraps the model in
Gemma3ForConditionalGeneration.
"""
from .base import ModelAdapter


class Gemma3Adapter(ModelAdapter):

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
        # <end_of_turn> token (107) used by Gemma IT
        end_of_turn = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn, int):
            ids.add(end_of_turn)
        return ids
