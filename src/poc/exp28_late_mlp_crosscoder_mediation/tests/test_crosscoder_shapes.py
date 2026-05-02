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


def test_crosscoder_same_init_and_decoder_norm_target() -> None:
    torch.manual_seed(0)
    model = BatchTopKCrossCoder(
        CrosscoderConfig(
            activation_dim=8,
            dict_size=16,
            k=4,
            same_init_for_all_branches=True,
            norm_init_scale=0.5,
            decoder_norm_target=0.25,
        )
    )
    assert torch.allclose(model.decoder[:, 0, :], model.decoder[:, 1, :])
    flat_norms = model.decoder.detach().flatten(start_dim=1).norm(dim=1)
    assert torch.allclose(flat_norms, torch.full_like(flat_norms, 0.5), atol=1e-5)

    model.normalize_decoder_()
    flat_norms = model.decoder.detach().flatten(start_dim=1).norm(dim=1)
    assert torch.allclose(flat_norms, torch.full_like(flat_norms, 0.25), atol=1e-5)


def test_norm_scaled_topk_keeps_raw_activations() -> None:
    model = BatchTopKCrossCoder(
        CrosscoderConfig(
            activation_dim=4,
            dict_size=3,
            k=1,
            scale_topk_by_decoder_norm=True,
        )
    )
    acts = torch.tensor([[5.0, 4.0, 1.0]])
    with torch.no_grad():
        model.decoder.fill_(0.0)
        model.decoder[0].fill_(0.01)
        model.decoder[1].fill_(1.0)
        model.decoder[2].fill_(0.01)
    kept = acts * batch_topk_mask(model._topk_scores(acts), k=1).to(acts.dtype)
    assert kept.tolist() == [[0.0, 4.0, 0.0]]


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
