from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp19_late_mlp_specificity_controls.controls import (
    CONTROL_MODE_RANDOM_NORM,
    CONTROL_MODE_RANDOM_RESPROJ,
    layer_permutation_map,
    matched_random_delta,
    stable_int_seed,
)


def test_layer_permutation_reverse_is_bijective():
    mapping = layer_permutation_map(20, 24)
    assert mapping == {20: 23, 21: 22, 22: 21, 23: 20}
    assert sorted(mapping) == sorted(mapping.values())


def test_stable_seed_is_deterministic():
    assert stable_int_seed("gemma", "B", 0, 3) == stable_int_seed("gemma", "B", 0, 3)
    assert stable_int_seed("gemma", "B", 0, 3) != stable_int_seed("gemma", "B", 0, 4)


def test_random_norm_control_preserves_delta_norm():
    true_delta = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 2.0]])
    residual = torch.ones_like(true_delta)
    control = matched_random_delta(
        true_delta=true_delta,
        residual_reference=residual,
        mode=CONTROL_MODE_RANDOM_NORM,
        seed=123,
    )
    assert torch.allclose(control.norm(dim=-1), true_delta.norm(dim=-1), atol=1e-5)


def test_residual_projection_control_preserves_signed_projection_and_orth_norm():
    true_delta = torch.tensor([[3.0, 4.0, 0.0], [-2.0, 1.0, 2.0]])
    residual = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    control = matched_random_delta(
        true_delta=true_delta,
        residual_reference=residual,
        mode=CONTROL_MODE_RANDOM_RESPROJ,
        seed=456,
    )
    residual_unit = residual / residual.norm(dim=-1, keepdim=True)
    true_proj = (true_delta * residual_unit).sum(dim=-1)
    control_proj = (control * residual_unit).sum(dim=-1)
    true_orth = true_delta - true_proj.unsqueeze(-1) * residual_unit
    control_orth = control - control_proj.unsqueeze(-1) * residual_unit
    assert torch.allclose(control_proj, true_proj, atol=1e-5)
    assert torch.allclose(control_orth.norm(dim=-1), true_orth.norm(dim=-1), atol=1e-5)
