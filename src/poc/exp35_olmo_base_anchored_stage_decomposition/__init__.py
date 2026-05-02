"""Exp35: OLMo Base-anchored fixed-support stage decomposition."""

from __future__ import annotations

from dataclasses import dataclass

from src.poc.cross_model.config import (
    OLMO2_7B_DPO_SHA,
    OLMO2_7B_PT_SHA,
    OLMO2_7B_RLVR_SHA,
    OLMO2_7B_SFT_SHA,
)


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
        repo_id="allenai/OLMo-2-1124-7B",
        revision=OLMO2_7B_PT_SHA,
    ),
    "S": StageSpec(
        key="S",
        label="SFT",
        repo_id="allenai/OLMo-2-1124-7B-SFT",
        revision=OLMO2_7B_SFT_SHA,
        cumulative_model_name="olmo2_7b_pt_sft",
    ),
    "D": StageSpec(
        key="D",
        label="DPO",
        repo_id="allenai/OLMo-2-1124-7B-DPO",
        revision=OLMO2_7B_DPO_SHA,
        cumulative_model_name="olmo2_7b_pt_dpo",
    ),
    "R": StageSpec(
        key="R",
        label="RLVR",
        repo_id="allenai/OLMo-2-1124-7B-Instruct",
        revision=OLMO2_7B_RLVR_SHA,
        cumulative_model_name="olmo2_7b",
    ),
}

STAGE_ORDER = ("B", "S", "D", "R")
NON_BASE_STAGES = ("S", "D", "R")
MODEL_NAME = "olmo2_7b"
DEFAULT_BOUNDARY_LAYER = 19

