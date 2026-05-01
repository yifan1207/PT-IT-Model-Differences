from __future__ import annotations

import torch

from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import (
    BatchTopKCrossCoder,
    CrosscoderConfig,
    batch_topk_mask,
)
from src.poc.exp28_late_mlp_crosscoder_mediation.mediation_hooks import (
    coverage_branch_contribution,
)


def test_batch_topk_budget() -> None:
    values = torch.arange(40, dtype=torch.float32).view(4, 10)
    mask = batch_topk_mask(values, k=3)
    assert int(mask.sum().item()) == 12


def test_crosscoder_shapes_and_branch_contribution() -> None:
    model = BatchTopKCrossCoder(CrosscoderConfig(activation_dim=8, dict_size=16, k=4))
    x = torch.randn(5, 2, 8)
    recon, features, preacts = model(x, output_features=True, return_preacts=True)
    assert recon.shape == x.shape
    assert features.shape == (5, 16)
    assert preacts.shape == (5, 16)
    assert int((features > 0).sum().item()) <= 5 * 4

    latents = torch.tensor([1, 3, 5])
    contrib, branch_features = model.selected_branch_contribution(
        x[:, 1, :],
        branch=1,
        latent_ids=latents,
        use_threshold=False,
    )
    assert contrib.shape == (5, 8)
    assert branch_features.shape == (5, 3)


def test_coverage_branch_contribution_selects_active_mass() -> None:
    torch.manual_seed(0)
    model = BatchTopKCrossCoder(CrosscoderConfig(activation_dim=8, dict_size=16, k=4))
    x = torch.randn(2, 8)
    contrib, features, diag = coverage_branch_contribution(
        model,
        x,
        branch=1,
        coverage_fraction=0.5,
        coverage_metric="norm",
        use_threshold=False,
    )
    assert contrib.shape == x.shape
    assert features.shape == (2, 16)
    assert diag["active_l0"] > 0
    assert diag["n_selected"] > 0
    assert 0.5 <= diag["removed_mass_frac"] <= 1.0


def test_margin_coverage_requires_margin_vector() -> None:
    model = BatchTopKCrossCoder(CrosscoderConfig(activation_dim=8, dict_size=16, k=4))
    x = torch.randn(1, 8)
    try:
        coverage_branch_contribution(
            model,
            x,
            branch=1,
            coverage_fraction=0.5,
            coverage_metric="margin_pos",
            use_threshold=False,
        )
    except ValueError as exc:
        assert "requires margin_vector" in str(exc)
    else:
        raise AssertionError("margin_pos coverage should require a margin vector")
