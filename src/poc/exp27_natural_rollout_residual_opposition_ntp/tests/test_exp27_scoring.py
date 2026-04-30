from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp27_natural_rollout_residual_opposition_ntp.scoring import (
    build_generated_source_mask,
    score_variant_against_full,
)
from src.poc.exp27_natural_rollout_residual_opposition_ntp.variants import apply_variant


def test_generated_source_mask_excludes_prompt_boundary_by_default() -> None:
    input_ids = torch.zeros((2, 10), dtype=torch.long)
    mask = build_generated_source_mask(
        input_ids=input_ids,
        prompt_lens=[4, 6],
        generated_lens=[5, 2],
    )
    assert mask[0].nonzero(as_tuple=False).squeeze(1).tolist() == [4, 5, 6, 7]
    assert mask[1].nonzero(as_tuple=False).squeeze(1).tolist() == [6]


def test_generated_source_mask_can_include_prompt_boundary() -> None:
    input_ids = torch.zeros((1, 8), dtype=torch.long)
    mask = build_generated_source_mask(
        input_ids=input_ids,
        prompt_lens=[3],
        generated_lens=[4],
        include_boundary_source=True,
    )
    assert mask[0].nonzero(as_tuple=False).squeeze(1).tolist() == [2, 3, 4, 5]


def test_score_variant_against_full_positive_drop() -> None:
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    source_mask = torch.tensor([[False, True, True, False, False]])
    full = torch.zeros((1, 5, 8), dtype=torch.float32)
    variant = torch.zeros_like(full)
    full[0, 1, 3] = 5.0
    full[0, 2, 4] = 5.0
    variant[0, 1, 3] = 3.0
    variant[0, 2, 4] = 3.0
    scores = score_variant_against_full(
        full_logits=full,
        variant_logits=variant,
        input_ids=input_ids,
        source_mask=source_mask,
        row_ids=["p0"],
        prompt_lens=[1],
    )
    assert scores["p0"]["n_positions"] == 2
    assert scores["p0"]["nll_delta"] > 0
    assert math.isclose(scores["p0"]["true_logit_drop"], 2.0, rel_tol=1e-6)


def test_randremove_resnorm_matches_noopp_postres_norm() -> None:
    residual = torch.tensor([[2.0, 0.0, 0.0]])
    update = torch.tensor([[-1.0, 0.3, 0.4]])
    noopp = apply_variant(update, residual, variant="noopp")
    control = apply_variant(update, residual, variant="randremove_resnorm", rand_seed=17)
    noopp_post_norm = (residual + noopp.update).norm(dim=-1)
    control_post_norm = (residual + control.update).norm(dim=-1)
    assert torch.allclose(noopp_post_norm, control_post_norm, atol=1e-6)
    assert control.diagnostics["postres_norm_ratio_after_hook"] is not None
