from __future__ import annotations

import torch
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp35_olmo_base_anchored_stage_decomposition.common import validate_full_prefix_boundary
from src.poc.exp35_olmo_base_anchored_stage_decomposition.analyze import _cluster_bootstrap, _unit_effects


def test_validate_full_prefix_boundary_rejects_last_token_state() -> None:
    with pytest.raises(RuntimeError, match="full-prefix"):
        validate_full_prefix_boundary(torch.zeros(4096))


def test_validate_full_prefix_boundary_checks_sequence_length() -> None:
    state = torch.zeros(1, 7, 4)
    validate_full_prefix_boundary(state, expected_seq_len=7)
    with pytest.raises(RuntimeError, match="sequence-length"):
        validate_full_prefix_boundary(state, expected_seq_len=8)


def test_fixed_stage_interaction_algebra() -> None:
    scored = []
    margins = {
        "U_B__L_B": 0.0,
        "U_B__L_R": 1.0,
        "U_R__L_B": 2.0,
        "U_R__L_R": 6.0,
    }
    for cell, margin in margins.items():
        upstream, late = cell.split("__")
        scored.append(
            {
                "readout": "common_r",
                "prompt_id": "p0",
                "event_kind": "first_diff",
                "cell": cell,
                "upstream_stage": upstream.removeprefix("U_"),
                "late_stage": late.removeprefix("L_"),
                "margin_rlvr_minus_base": margin,
            }
        )
    units = _unit_effects(scored, "common_r", "R")
    assert len(units) == 1
    assert units[0]["interaction"] == 3.0
    assert units[0]["late_effect_from_base"] == 1.0
    assert units[0]["late_effect_from_matched"] == 4.0


def test_cluster_bootstrap_uses_prompt_clusters() -> None:
    out = _cluster_bootstrap([("p0", 1.0), ("p0", 3.0), ("p1", 5.0)], n_boot=20, seed=0)
    assert out["estimate"] == 3.0
    assert out["n_units"] == 3
    assert out["n_prompt_clusters"] == 2
