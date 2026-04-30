"""Exp26: residual-opposition mediation of the upstream x late interaction."""

from __future__ import annotations


EXPERIMENT = "exp26_residual_opposition_mediation"

DENSE5_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
PILOT_MODELS = ["gemma3_4b", "llama31_8b", "olmo2_7b"]

DEFAULT_VARIANTS = [
    "opp_scale_0p5",
    "noopp",
    "flipopp",
    "normpres_noopp",
    "ptlevel_opp",
    "randorth",
]

