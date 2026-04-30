"""Exp27: residual-opposition importance on natural greedy rollouts."""

from __future__ import annotations


EXPERIMENT = "exp27_natural_rollout_residual_opposition_ntp"

DENSE5_MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b"]
PILOT_MODELS = ["gemma3_4b", "llama31_8b", "olmo2_7b"]

DEFAULT_VARIANTS = [
    "full",
    "noopp",
    "normpres_noopp",
    "flipopp",
    "randorth",
    "randremove",
    "randremove_resnorm",
]

RANDOM_VARIANTS = {"randorth", "randremove", "randremove_resnorm"}
