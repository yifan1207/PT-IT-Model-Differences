"""Exp25: OLMo 2 7B stage-progression first-divergence case study."""

STAGE_MODELS = [
    "olmo2_7b_pt_sft",
    "olmo2_7b_sft_dpo",
    "olmo2_7b_dpo_rlvr",
    "olmo2_7b",
]

STAGE_LABELS = {
    "olmo2_7b_pt_sft": "PT->SFT",
    "olmo2_7b_sft_dpo": "SFT->DPO",
    "olmo2_7b_dpo_rlvr": "DPO->RLVR",
    "olmo2_7b": "PT->RLVR",
}
