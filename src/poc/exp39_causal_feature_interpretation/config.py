"""Configuration for Exp39 causal feature interpretation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Exp39Family:
    model: str
    result_root: Path
    gcs_uri: str
    layers: tuple[int, ...]
    status: str = "strict_pass"


PRIMARY_FAMILIES: dict[str, Exp39Family] = {
    "gemma3_4b": Exp39Family(
        model="gemma3_4b",
        result_root=Path(
            "results/exp34_dense5_final_readout_crosscoder/"
            "exp34_gemma3_4b_full_20260502_2110_a100x8_bs16/"
            "gemma3_4b/selected_d81920_k64"
        ),
        gcs_uri=(
            "gs://pt-vs-it-results/results/exp34_dense5_final_readout_crosscoder/"
            "exp34_gemma3_4b_full_20260502_2110_a100x8_bs16/"
            "gemma3_4b/selected_d81920_k64/"
        ),
        layers=(31, 32, 33),
    ),
    "llama31_8b": Exp39Family(
        model="llama31_8b",
        result_root=Path(
            "results/exp30_final_readout_crosscoder_mediation/"
            "exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/"
            "selected_d131072_k64"
        ),
        gcs_uri=(
            "gs://pt-vs-it-results/results/exp30_final_readout_crosscoder_mediation/"
            "exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/"
            "selected_d131072_k64/"
        ),
        layers=(29, 30, 31),
    ),
    "mistral_7b": Exp39Family(
        model="mistral_7b",
        result_root=Path(
            "results/exp34_dense5_final_readout_crosscoder/"
            "exp34_mistral_7b_full_20260502_1124/"
            "mistral_7b/selected_d131072_k64"
        ),
        gcs_uri=(
            "gs://pt-vs-it-results/results/exp34_dense5_final_readout_crosscoder/"
            "exp34_mistral_7b_full_20260502_1124/"
            "mistral_7b/selected_d131072_k64/"
        ),
        layers=(29, 30, 31),
    ),
    "qwen3_4b": Exp39Family(
        model="qwen3_4b",
        result_root=Path(
            "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_qwen3_4b_final2_d81920_k64_20260503_0451_a100x2/"
            "selected_d81920_k64"
        ),
        gcs_uri=(
            "gs://pt-vs-it-results/results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_qwen3_4b_final2_d81920_k64_20260503_0451_a100x2/"
            "selected_d81920_k64/"
        ),
        layers=(34, 35),
    ),
}

DIAGNOSTIC_FAMILIES: dict[str, Exp39Family] = {
    "olmo2_7b": Exp39Family(
        model="olmo2_7b",
        result_root=Path(
            "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_olmo2_7b_final2_d131072_k64_20260503_0441_a100x2/"
            "selected_d131072_k64"
        ),
        gcs_uri=(
            "gs://pt-vs-it-results/results/exp38_qwen_olmo_final_layer_crosscoder_hardening/"
            "exp38_olmo2_7b_final2_d131072_k64_20260503_0441_a100x2/"
            "selected_d131072_k64/"
        ),
        layers=(30, 31),
        status="diagnostic_quality_fail",
    ),
}

DEFAULT_FAMILIES: dict[str, Exp39Family] = dict(PRIMARY_FAMILIES)
ALL_FAMILIES: dict[str, Exp39Family] = {**PRIMARY_FAMILIES, **DIAGNOSTIC_FAMILIES}


DEFAULT_OUT_ROOT = Path("results/exp39_causal_feature_interpretation")
DEFAULT_DASHBOARD_DATASETS = (
    Path("data/eval_dataset_v2.jsonl"),
    Path("data/exp3_dataset.jsonl"),
    Path("data/exp6_dataset.jsonl"),
)
DEFAULT_EXCLUDE_DATASETS = (Path("data/eval_dataset_v2_holdout_0600_1199.jsonl"),)
