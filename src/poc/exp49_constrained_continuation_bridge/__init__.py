"""Exp49 constrained continuation bridge for Exp47 first-divergence events."""

from __future__ import annotations

DEFAULT_MODELS = (
    "llama31_meta_instruct",
    "llama31_tulu3_sft",
    "llama31_tulu3_dpo",
    "llama31_tulu3_final",
    "llama31_openmath2",
)

DEFAULT_HORIZONS = (0, 1, 2, 4, 8)
DEFAULT_MAX_TAIL = 8
DEFAULT_EVENT_KIND = "first_diff"
DEFAULT_READOUTS = ("common_it", "common_pt")
DEFAULT_CANDIDATE_KINDS = ("desc_primary", "base_primary", "base_forced_desc")

