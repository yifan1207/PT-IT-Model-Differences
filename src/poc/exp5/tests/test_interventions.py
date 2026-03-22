from __future__ import annotations

import pytest
import torch

from src.poc.exp5.interventions import InterventionSpec, directional_ablate_tensor


# Alpha convention per exp5 design Section C.2 and Arditi et al. (2024):
#   alpha = 1.0  → baseline (no change, full corrective direction retained)
#   alpha = 0.0  → corrective direction fully removed  (= Arditi abliteration)
#   alpha = 0.5  → half-strength correction
#   alpha = 2.0  → double correction (direction amplified)
#   alpha = -1.0 → direction reversed (anti-correction)
# Formula: h' = h - (1 - alpha) * proj(h, v̂)


def test_directional_ablation_is_baseline_at_alpha_one():
    """alpha=1.0 → no intervention; output equals input."""
    mlp = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
    out = directional_ablate_tensor(mlp, direction, alpha=1.0)
    assert torch.allclose(out, mlp, atol=1e-5), (
        "alpha=1.0 should be a no-op (baseline — full correction retained)"
    )


def test_directional_ablation_removes_projection_at_alpha_zero():
    """alpha=0.0 → direction fully removed; matches Arditi et al. abliteration."""
    mlp = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
    out = directional_ablate_tensor(mlp, direction, alpha=0.0)
    # proj onto [1,0] is 3; removing it leaves [0, 4]
    assert torch.allclose(out, torch.tensor([[0.0, 4.0]]), atol=1e-5), (
        "alpha=0.0 should fully remove the projected component"
    )


def test_directional_ablation_amplifies_at_alpha_two():
    """alpha=2.0 → double correction; proj is added back rather than removed."""
    mlp = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
    out = directional_ablate_tensor(mlp, direction, alpha=2.0)
    # (1 - 2.0) = -1.0 → h - (-1)*proj = h + proj → [3+3, 4] = [6, 4]
    assert torch.allclose(out, torch.tensor([[6.0, 4.0]]), atol=1e-5), (
        "alpha=2.0 should double the direction component"
    )


def test_directional_ablation_reverses_at_alpha_minus_one():
    """alpha=-1.0 → anti-correction; direction doubled in removal direction."""
    mlp = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
    out = directional_ablate_tensor(mlp, direction, alpha=-1.0)
    # (1 - (-1)) = 2.0 → h - 2*proj → [3-6, 4] = [-3, 4]
    assert torch.allclose(out, torch.tensor([[-3.0, 4.0]]), atol=1e-5), (
        "alpha=-1.0 should reverse (double-remove) the direction"
    )


def test_directional_ablation_half_strength_at_alpha_half():
    """alpha=0.5 → half the direction removed."""
    mlp = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    direction = torch.tensor([1.0, 0.0], dtype=torch.float32)
    out = directional_ablate_tensor(mlp, direction, alpha=0.5)
    # (1 - 0.5) = 0.5 → h - 0.5*proj → [4-2, 0] = [2, 0]
    assert torch.allclose(out, torch.tensor([[2.0, 0.0]]), atol=1e-5), (
        "alpha=0.5 should remove half the directional component"
    )


def test_directional_ablation_orthogonal_direction_unchanged():
    """Direction orthogonal to mlp_out has zero projection; output unchanged."""
    mlp = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    direction = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        out = directional_ablate_tensor(mlp, direction, alpha=alpha)
        assert torch.allclose(out, mlp, atol=1e-6), (
            f"Orthogonal direction at alpha={alpha} should not change output"
        )


def test_intervention_resample_validate_raises():
    """Resample ablation is not implemented end-to-end; validate() must reject it."""
    spec = InterventionSpec(method="resample", layers=[20])
    with pytest.raises(NotImplementedError):
        spec.validate()


def test_intervention_directional_below_boundary_raises():
    """Directional ablation on proposal-stage layers (< boundary) must be rejected."""
    spec = InterventionSpec(method="directional", layers=[5], proposal_boundary=20)
    with pytest.raises(ValueError, match="corrective layers"):
        spec.validate()


def test_intervention_none_is_inactive():
    """method='none' with layers set should still report inactive."""
    spec = InterventionSpec(method="none", layers=[10, 20])
    assert not spec.active


def test_intervention_mean_without_stats_raises():
    """mean ablation with no precomputed stats must raise ValueError."""
    spec = InterventionSpec(method="mean", layers=[5])
    with pytest.raises(ValueError, match="Missing mean MLP outputs"):
        spec.validate()
