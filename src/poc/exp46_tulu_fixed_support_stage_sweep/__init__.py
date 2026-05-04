"""Exp46: Tulu Base-anchored fixed-support stage decomposition.

The stage labels are released cumulative checkpoints, not claims about the
isolated training algorithm that produced the transition.
"""

from __future__ import annotations

from dataclasses import dataclass

LLAMA31_8B_BASE_SHA = "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"
TULU3_8B_SFT_SHA = "f2a0b46b0cfda21003c6141b1ff837b7e165524d"
TULU3_8B_DPO_SHA = "a7beb67e33ffd01cc87ac3b46cadc1000985b8db"
TULU3_8B_FINAL_SHA = "666943798adbde0b1aff34626007e26986a3c107"


@dataclass(frozen=True)
class StageSpec:
    key: str
    label: str
    repo_id: str
    revision: str
    cumulative_model_name: str | None = None


STAGES: dict[str, StageSpec] = {
    "B": StageSpec(
        key="B",
        label="Base",
        repo_id="meta-llama/Llama-3.1-8B",
        revision=LLAMA31_8B_BASE_SHA,
    ),
    "S": StageSpec(
        key="S",
        label="SFT",
        repo_id="allenai/Llama-3.1-Tulu-3-8B-SFT",
        revision=TULU3_8B_SFT_SHA,
        cumulative_model_name="tulu3_8b_sft",
    ),
    "D": StageSpec(
        key="D",
        label="DPO",
        repo_id="allenai/Llama-3.1-Tulu-3-8B-DPO",
        revision=TULU3_8B_DPO_SHA,
        cumulative_model_name="tulu3_8b_dpo",
    ),
    "R": StageSpec(
        key="R",
        label="Final/RLVR",
        repo_id="allenai/Llama-3.1-Tulu-3-8B",
        revision=TULU3_8B_FINAL_SHA,
        cumulative_model_name="tulu3_8b_final",
    ),
}

STAGE_ORDER = ("B", "S", "D", "R")
NON_BASE_STAGES = ("S", "D", "R")
MODEL_NAME = "llama31_8b"
DEFAULT_BOUNDARY_LAYER = 19
SUPPORT_FILE_NAME = "base_final_first_diff.jsonl.gz"
PROMPT_MODES = ("raw_shared", "tulu_shared_template")
PRIMARY_PROMPT_MODE = "raw_shared"
