"""
Token-type façade used by exp3 plotting.

Classification now runs at the word/span level and broadcasts categories back to
the member tokens. This keeps the existing plotting interface stable while
avoiding brittle token-piece heuristics.
"""
from typing import Literal

from src.poc.exp03_corrective_stage_characterization.analysis.word_categories import classify_generated_tokens_by_word

TokenType = Literal["CONTENT", "FUNCTION", "PUNCTUATION", "DISCOURSE", "STRUCTURAL", "OTHER"]


def classify_generated_tokens(generated_tokens: list[dict]) -> list[TokenType]:
    """Classify generated output at the word/span level and return token labels."""
    return classify_generated_tokens_by_word(generated_tokens)


def stratify_by_token_type(
    result: dict,
    metric_key: str,
) -> dict[TokenType, list[list[float]]]:
    """Group per-step layer metrics by the token type generated at each step.

    Parameters
    ----------
    result : dict
        A single prompt result containing 'generated_tokens' and metric_key.
        metric_key should be a [step][layer] list-of-lists (e.g. 'layer_delta_cosine',
        'logit_delta_contrib', 'kl_to_final').

    Returns
    -------
    dict mapping TokenType → list of per-step layer-value lists for that type.
    """
    token_types = classify_generated_tokens(result.get("generated_tokens", []))
    metric_data = result.get(metric_key, [])

    if len(token_types) != len(metric_data):
        import warnings
        warnings.warn(
            f"stratify_by_token_type: token_types length ({len(token_types)}) != "
            f"metric '{metric_key}' length ({len(metric_data)}) for prompt "
            f"'{result.get('prompt_id', '?')}'. Extra steps will be silently dropped.",
            stacklevel=2,
        )

    grouped: dict[TokenType, list[list[float]]] = {
        "CONTENT": [], "FUNCTION": [], "PUNCTUATION": [],
        "DISCOURSE": [], "STRUCTURAL": [], "OTHER": [],
    }

    for step_idx, (tok_type, step_vals) in enumerate(zip(token_types, metric_data)):
        grouped[tok_type].append(step_vals)

    return grouped
