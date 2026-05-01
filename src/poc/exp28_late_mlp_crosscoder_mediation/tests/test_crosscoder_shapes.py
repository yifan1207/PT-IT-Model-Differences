from __future__ import annotations

import torch

from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import (
    BatchTopKCrossCoder,
    CrosscoderConfig,
    batch_topk_mask,
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
