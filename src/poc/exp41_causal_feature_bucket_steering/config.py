"""Shared configuration for Exp41 bucket steering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


EXP39_RUN = Path(
    "results/exp39_causal_feature_interpretation/"
    "exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345"
)
EXP20_ROOT = Path(
    "results/exp20_divergence_token_counterfactual/"
    "factorial_validation_holdout_fast_20260425_2009_with_early"
)
DATASET_PATH = Path("data/eval_dataset_v2_holdout_0600_1199.jsonl")
RESULT_ROOT = Path("results/exp41_causal_feature_bucket_steering")

PRIMARY_MODELS = ("gemma3_4b", "llama31_8b", "mistral_7b", "qwen3_4b")
TERMINAL_LAYERS = {
    "gemma3_4b": (31, 32, 33),
    "llama31_8b": (29, 30, 31),
    "mistral_7b": (29, 30, 31),
    "qwen3_4b": (34, 35),
}

BUCKET_TO_CATEGORIES = {
    "structure_readout": {"response_structure_or_answer_readout"},
    "format_control": {"instruction_following_or_format_control"},
    "mcq_scaffold": {"evaluation_or_multiple_choice_scaffold"},
    "surface_punctuation": {"surface_punctuation_or_tokenization"},
    "safety_advice_boundary": {"alignment_safety_or_advice_boundary"},
    "artifact_repetition": {
        "dataset_or_repetition_artifact",
        "rare_unicode_or_web_artifact",
        "generic_frequency_or_unclear",
    },
}

MAIN_SMOKE_BUCKETS = (
    "structure_readout",
    "format_control",
    "surface_punctuation",
    "artifact_repetition",
)
FULL_BUCKETS = tuple(BUCKET_TO_CATEGORIES)
DEFAULT_ALPHAS_SMOKE = (0.0, 1.0, 2.0)
DEFAULT_ALPHAS_FULL = (0.0, 0.5, 1.0, 1.5, 2.0)


@dataclass(frozen=True)
class ManifestMode:
    name: str
    min_confidence: float
    allow_weak_support: bool
    max_features_per_model_bucket: int


STRICT_PRIMARY = ManifestMode(
    name="strict_primary",
    min_confidence=0.60,
    allow_weak_support=True,
    max_features_per_model_bucket=10,
)
EXPANDED_SENSITIVITY = ManifestMode(
    name="expanded_sensitivity",
    min_confidence=0.45,
    allow_weak_support=True,
    max_features_per_model_bucket=20,
)

