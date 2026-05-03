"""Exp37: matched-prefix selection baselines for first-divergence factorial diffing."""

from __future__ import annotations

DENSE5_MODELS = ("gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "olmo2_7b")

REFERENCE_KEY = "first_diff__reference"
RANDOM_PT_KEY = "random_local_disagreement__pt_rollout"
RANDOM_IT_KEY = "random_local_disagreement__it_rollout"
PREDIV_KEY = "prediv_future_pair__shared_prediv"

ARM_SOURCE_KEYS = (REFERENCE_KEY, RANDOM_PT_KEY, RANDOM_IT_KEY, PREDIV_KEY)

