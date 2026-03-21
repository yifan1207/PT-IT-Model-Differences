"""
Deterministic word-level categorization for generated output.

This replaces token-piece heuristics with categories derived from reconstructed
word spans. Tokens inherit the category of the word/span they participate in.
"""
from dataclasses import dataclass
import re
from typing import Literal


WordCategory = Literal["CONTENT", "FUNCTION", "DISCOURSE", "STRUCTURAL", "PUNCTUATION", "OTHER"]

_STRUCTURAL_TOKEN_RE = re.compile(
    r"^(?:#+|\*+|`+|-{2,}|={2,}|>{1,}|[-+•]|\d+\.)$"
)
_PUNCT_ONLY_RE = re.compile(r"^[\s\.,;:!?\(\)\[\]\{\}\"'\/\\@#\$%\^&\*+=<>|~`-]+$")
_WORD_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*|\d+(?:\.\d+)?")

_FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his", "her", "its",
    "our", "their", "all", "some", "any", "each", "every", "either", "neither",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them",
    "myself", "yourself", "ourselves", "themselves",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "doing", "have", "has", "had",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "to", "of", "in", "on", "at", "by", "for", "from", "with", "about", "into", "over",
    "under", "between", "through", "during", "before", "after", "without", "within",
    "and", "or", "but", "if", "because", "while", "although", "though", "than",
    "as", "so", "nor", "yet", "not", "no", "yes", "very", "more", "most", "less", "least",
    "here", "there", "then", "now",
}

_DISCOURSE_SINGLE = {
    "however", "therefore", "thus", "meanwhile", "overall", "finally", "first", "second",
    "third", "next", "alternatively", "instead", "indeed", "anyway", "anyhow", "basically",
    "similarly", "specifically", "notably", "frankly", "honestly", "actually",
}
_DISCOURSE_PHRASES = {
    ("of", "course"),
    ("for", "example"),
    ("for", "instance"),
    ("in", "summary"),
    ("in", "short"),
    ("in", "other", "words"),
    ("on", "the", "other", "hand"),
    ("as", "a", "result"),
    ("for", "that", "reason"),
    ("to", "be", "clear"),
    ("that", "said"),
    ("in", "fact"),
    ("by", "contrast"),
    ("for", "now"),
}


@dataclass(frozen=True)
class WordSpan:
    start: int
    end: int
    text: str
    category: WordCategory


def _concat_tokens(generated_tokens: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in generated_tokens:
        token_str = token.get("token_str", "")
        start = cursor
        text_parts.append(token_str)
        cursor += len(token_str)
        spans.append((start, cursor))
    return "".join(text_parts), spans


def _token_level_structural_or_punct(token_str: str) -> WordCategory | None:
    stripped = token_str.strip()
    if not stripped:
        return "PUNCTUATION"
    if _STRUCTURAL_TOKEN_RE.match(stripped):
        return "STRUCTURAL"
    if _PUNCT_ONLY_RE.match(token_str):
        return "PUNCTUATION"
    return None


def _base_word_category(word: str) -> WordCategory:
    lower = word.lower()
    if lower in _DISCOURSE_SINGLE:
        return "DISCOURSE"
    if lower in _FUNCTION_WORDS:
        return "FUNCTION"
    if any(ch.isalpha() for ch in lower) or any(ch.isdigit() for ch in lower):
        return "CONTENT"
    return "OTHER"


def classify_generated_tokens_by_word(generated_tokens: list[dict]) -> list[WordCategory]:
    if not generated_tokens:
        return []

    text, token_spans = _concat_tokens(generated_tokens)
    categories: list[WordCategory | None] = [None] * len(generated_tokens)

    for i, token in enumerate(generated_tokens):
        direct = _token_level_structural_or_punct(token.get("token_str", ""))
        if direct is not None:
            categories[i] = direct

    word_matches = list(_WORD_RE.finditer(text))
    words = [m.group(0) for m in word_matches]
    lowers = [w.lower() for w in words]
    word_cats = [_base_word_category(word) for word in words]

    for span_len in range(4, 1, -1):
        for start in range(0, len(words) - span_len + 1):
            phrase = tuple(lowers[start:start + span_len])
            if phrase in _DISCOURSE_PHRASES:
                for j in range(start, start + span_len):
                    word_cats[j] = "DISCOURSE"

    for match, category in zip(word_matches, word_cats, strict=False):
        span_start, span_end = match.span()
        for i, (tok_start, tok_end) in enumerate(token_spans):
            if tok_end <= span_start or tok_start >= span_end:
                continue
            if categories[i] is None:
                categories[i] = category

    return [cat if cat is not None else "OTHER" for cat in categories]
