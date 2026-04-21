from __future__ import annotations

import numpy as np

from src.poc.exp05_corrective_direction_ablation_cartography.analysis.subspace import linear_cka, participation_ratio, summarise_checkpoint_shift


def test_participation_ratio_is_finite_for_full_rank_matrix():
    x = np.eye(4, dtype=np.float32)
    assert participation_ratio(x) > 0


def test_linear_cka_identity_is_one():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert abs(linear_cka(x, x) - 1.0) < 1e-6


def test_checkpoint_shift_summary_contains_expected_metrics():
    baseline = {11: np.ones((3, 4), dtype=np.float32)}
    ablated = {11: np.ones((3, 4), dtype=np.float32)}
    summary = summarise_checkpoint_shift(baseline, ablated)
    assert 11 in summary
    assert "cka_to_baseline" in summary[11]
