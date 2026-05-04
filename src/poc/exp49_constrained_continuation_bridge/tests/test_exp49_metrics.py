from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.poc.exp49_constrained_continuation_bridge.analyze import row_effects
from src.poc.exp49_constrained_continuation_bridge.common import cumulative_sum, suffix_sum


def test_cumulative_and_suffix_sum() -> None:
    values = [-1.0, -2.0, -3.0]
    assert cumulative_sum(values, 0) == -1.0
    assert cumulative_sum(values, 2) == -6.0
    assert suffix_sum(values, 0) == 0.0
    assert suffix_sum(values, 2) == -5.0


def test_primary_decomposition_and_tail_metrics() -> None:
    row = {
        "model": "m",
        "prompt_id": "p",
        "valid": True,
        "scored": True,
        "slices": ["full_1400"],
        "candidates": {
            "desc_primary": {"token_ids": [10, 11, 12], "first_eos_position": None, "non_eos_special_positions": []},
            "base_primary": {"token_ids": [20, 21, 22], "first_eos_position": None, "non_eos_special_positions": []},
            "base_forced_desc": {"token_ids": [10, 23, 24], "first_eos_position": None, "non_eos_special_positions": []},
            "desc_shuffled": {"token_ids": [10, 99, 98], "first_eos_position": None, "non_eos_special_positions": []},
        },
        "cells": {
            "U_PT__L_PT": {"readouts": {"common_it": {
                "desc_primary": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "base_primary": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "base_forced_desc": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "desc_shuffled": {"target_logprobs": [-1.0, -1.0, -1.0]},
            }}},
            "U_PT__L_IT": {"readouts": {"common_it": {
                "desc_primary": {"target_logprobs": [-0.5, -0.5, -0.5]},
                "base_primary": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "base_forced_desc": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "desc_shuffled": {"target_logprobs": [-0.5, -1.0, -1.0]},
            }}},
            "U_IT__L_PT": {"readouts": {"common_it": {
                "desc_primary": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "base_primary": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "base_forced_desc": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "desc_shuffled": {"target_logprobs": [-1.0, -1.0, -1.0]},
            }}},
            "U_IT__L_IT": {"readouts": {"common_it": {
                "desc_primary": {"target_logprobs": [0.0, 0.0, 0.0]},
                "base_primary": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "base_forced_desc": {"target_logprobs": [-1.0, -1.0, -1.0]},
                "desc_shuffled": {"target_logprobs": [0.0, -1.0, -1.0]},
            }}},
        },
    }
    effects = row_effects(row, horizons=[0, 2], readouts=("common_it",))
    primary_n0 = [e for e in effects if e["comparison"] == "primary_desc_vs_base" and e["horizon"] == 0][0]
    primary_n2 = [e for e in effects if e["comparison"] == "primary_desc_vs_base" and e["horizon"] == 2][0]
    assert primary_n0["P"] == 0.5
    assert primary_n0["M"] == 1.0
    assert primary_n0["C"] == 0.5
    assert primary_n2["P"] == 1.5
    assert primary_n2["M"] == 3.0
    assert primary_n2["C_tail"] == 1.0
    assert primary_n2["Cbar_tail"] == 0.5


def test_eos_horizon_filter() -> None:
    row = {
        "model": "m",
        "prompt_id": "p",
        "valid": True,
        "scored": True,
        "slices": ["full_1400"],
        "candidates": {
            "desc_primary": {"token_ids": [10, 2], "first_eos_position": 1, "non_eos_special_positions": []},
            "base_primary": {"token_ids": [20, 21], "first_eos_position": None, "non_eos_special_positions": []},
            "base_forced_desc": {"token_ids": [10, 23], "first_eos_position": None, "non_eos_special_positions": []},
            "desc_shuffled": {"token_ids": [10, 99], "first_eos_position": None, "non_eos_special_positions": []},
        },
        "cells": {
            cell: {"readouts": {"common_it": {
                "desc_primary": {"target_logprobs": [-1.0, -1.0]},
                "base_primary": {"target_logprobs": [-1.0, -1.0]},
                "base_forced_desc": {"target_logprobs": [-1.0, -1.0]},
                "desc_shuffled": {"target_logprobs": [-1.0, -1.0]},
            }}}
            for cell in ("U_PT__L_PT", "U_PT__L_IT", "U_IT__L_PT", "U_IT__L_IT")
        },
    }
    effects = row_effects(row, horizons=[1, 2], readouts=("common_it",))
    valid = [e for e in effects if e["comparison"] == "primary_desc_vs_base" and e["horizon"] == 1][0]
    invalid = [e for e in effects if e["comparison"] == "primary_desc_vs_base" and e["horizon"] == 2][0]
    assert valid["valid_horizon"] is True
    assert invalid["valid_horizon"] is False
