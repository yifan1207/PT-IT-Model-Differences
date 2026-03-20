"""
Token type classification for stratified corrective-stage analysis (Exp 2a).

Works on 'generated_tokens' from existing exp2 or exp3 results — no new inference.

Token categories
----------------
  CONTENT      First content-bearing token of a response (the "answer token").
               Usually the token right after any discourse preamble.
               Mostly semantic — the model's actual answer.

  PUNCTUATION  Punctuation marks, whitespace, newlines.
               Classified by token string matching.

  DISCOURSE    Discourse markers and filler phrases:
               "Well,", "Let me", "I think", "Sure,", "Of course," etc.
               Classified by token-list lookup — no LLM judge needed.
               These are format/persona tokens, not content.

  STRUCTURAL   Formatting tokens: "**", "##", "```", "- ", "1.", "\n\n".
               Appear disproportionately in IT (format enforcement hypothesis).

  OTHER        Everything else (ordinary content words, function words).

The hypothesis: the corrective stage (late-layer delta norm / cos) will be
significantly larger for DISCOURSE and STRUCTURAL tokens than for CONTENT or
OTHER tokens in IT, but not in PT.
"""
import re
from typing import Literal

TokenType = Literal["CONTENT", "PUNCTUATION", "DISCOURSE", "STRUCTURAL", "OTHER"]


# ── lookup tables ──────────────────────────────────────────────────────────────

# Token strings that indicate discourse markers / filler (case-insensitive prefix match).
DISCOURSE_PREFIXES: list[str] = [
    "well,", "well ", "let me", "i think", "i believe", "i'd say",
    "sure,", "sure!", "of course", "certainly", "absolutely",
    "great,", "great!", "happy to", "glad to",
    "to answer", "in short", "in summary", "to summarize",
    "first,", "firstly", "secondly", "finally,", "lastly",
    "note that", "keep in mind", "it's worth",
    "so,", "now,", "okay,", "ok,",
]

# Token strings that indicate structural/formatting tokens.
STRUCTURAL_TOKENS: set[str] = {
    "**", "*", "##", "###", "####", "#",
    "```", "``", "`",
    "- ", "* ", "+ ",
    "\n\n", "\n", "\r\n",
    "1.", "2.", "3.", "4.", "5.",
    "---", "===", "...",
}

# Punctuation regex: matches tokens that are purely punctuation/whitespace.
_PUNCT_RE = re.compile(r"^[\s\.,;:!?\-\(\)\[\]\{\}\"\'\/\\@#\$%\^&\*+=<>|~`]+$")


def classify_token(token_str: str) -> TokenType:
    """Classify a single token string into a TokenType category.

    The classification is deterministic and fast (no model calls).
    Used to stratify layer-wise statistics by what kind of token is being predicted.
    """
    s = token_str.strip()
    if not s:
        return "PUNCTUATION"

    # Structural formatting tokens (exact match first).
    if token_str in STRUCTURAL_TOKENS:
        return "STRUCTURAL"

    # Pure punctuation / whitespace.
    if _PUNCT_RE.match(token_str):
        return "PUNCTUATION"

    # Discourse markers (case-insensitive prefix).
    s_lower = s.lower()
    for prefix in DISCOURSE_PREFIXES:
        if s_lower.startswith(prefix):
            return "DISCOURSE"

    return "OTHER"


def classify_generated_tokens(generated_tokens: list[dict]) -> list[TokenType]:
    """Classify all generated tokens for one prompt result.

    The FIRST non-DISCOURSE/non-PUNCTUATION/non-STRUCTURAL token is relabelled CONTENT
    (the actual answer onset).

    Returns a list of TokenType, same length as generated_tokens.
    """
    types = [classify_token(t["token_str"]) for t in generated_tokens]

    # Override DISCOURSE for multi-word markers
    text_so_far = ""
    discourse_tokens = []
    for i, t in enumerate(generated_tokens):
        # Allow skipping structural and punctuation at the start
        if types[i] in ("STRUCTURAL", "PUNCTUATION"):
            continue
            
        # Accumulate substantive tokens to check against discourse prefixes
        text_so_far += t["token_str"].lower()
        discourse_tokens.append(i)
        
        test_str = text_so_far.strip()
        is_partial_match = any(p.startswith(test_str) for p in DISCOURSE_PREFIXES)
        is_full_match = any(test_str.startswith(p) for p in DISCOURSE_PREFIXES)
        
        if is_full_match:
            for idx in discourse_tokens:
                types[idx] = "DISCOURSE"
            # It's possible there are more discourse markers consecutively, but typically one serves the prefix.
            break
            
        if not is_partial_match:
            break

    # Relabel the first substantive token as CONTENT.
    for i, t in enumerate(types):
        # If it's a structural/punct/discourse, keep skipping
        if t not in ("STRUCTURAL", "PUNCTUATION", "DISCOURSE"):
            types[i] = "CONTENT"
            break

    return types


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
        "CONTENT": [], "PUNCTUATION": [], "DISCOURSE": [],
        "STRUCTURAL": [], "OTHER": [],
    }

    for step_idx, (tok_type, step_vals) in enumerate(zip(token_types, metric_data)):
        grouped[tok_type].append(step_vals)

    return grouped
