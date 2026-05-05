"""Exp52 forced divergence-token consequence bridge."""

from __future__ import annotations

DEFAULT_MODELS = (
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_openmath2",
)

DEFAULT_BRANCHES = (
    "it_branch",
    "pt_branch",
    "it_rank_matched_alt",
    "token_class_matched_alt",
)

DEFAULT_LENGTHS = {
    "GOV-FORMAT": 96,
    "SAFETY": 128,
    "GOV-CONV": 128,
    "CONTENT-REASON": 256,
}

