"""
Unit tests for weight-shift metric functions (src/poc/shared/metrics.py).

Tests cover:
  frob_shift       — identity, known exact value, scale-normalisation,
                     small perturbation, zero-PT edge cases, 1-D vectors
  cosine_distance  — identity, opposite direction, orthogonal vectors/matrices,
                     scale-invariance, zero-norm edge cases, numerical bounds
  independence     — proof that the two metrics capture different phenomena
"""
import math

import torch
import pytest

from src.poc.shared.metrics import frob_shift, cosine_distance


# ── frob_shift ─────────────────────────────────────────────────────────────────

def test_frob_shift_identity():
    """Δ_rel = 0 when W_it is a clone of W_pt."""
    W = torch.randn(8, 8)
    assert frob_shift(W, W.clone()) == pytest.approx(0.0, abs=1e-6)


def test_frob_shift_double():
    """W_it = 2·W_pt  →  diff = W_pt  →  Δ_rel = 1.0."""
    W = torch.randn(8, 8)
    assert frob_shift(W, 2.0 * W) == pytest.approx(1.0, rel=1e-5)


def test_frob_shift_small_perturbation():
    """A tiny additive perturbation gives a small positive shift."""
    W = torch.ones(4, 4)          # ||W||_F = 4
    delta = torch.zeros(4, 4)
    delta[0, 0] = 0.01
    result = frob_shift(W, W + delta)
    assert 0.0 < result < 0.01    # ||delta||_F / 4  ≈ 0.0025


def test_frob_shift_scale_normalised():
    """Multiplying both matrices by any scalar k doesn't change Δ_rel."""
    W_pt = torch.randn(6, 6)
    W_it = W_pt + 0.1 * torch.randn(6, 6)
    expected = frob_shift(W_pt, W_it)
    for k in (0.5, 2.0, 10.0):
        assert frob_shift(k * W_pt, k * W_it) == pytest.approx(expected, rel=1e-4)


def test_frob_shift_zero_pt_returns_nan():
    """Returns nan — not inf, not an exception — when ‖W_pt‖_F == 0."""
    assert math.isnan(frob_shift(torch.zeros(4, 4), torch.randn(4, 4)))


def test_frob_shift_both_zero_returns_nan():
    """Both-zero case: pt_norm check fires first → nan."""
    assert math.isnan(frob_shift(torch.zeros(3, 3), torch.zeros(3, 3)))


def test_frob_shift_1d_bias_vector():
    """Works on 1-D tensors (bias / layer-norm weight vectors).

    W_pt = [1, 2, 3],  W_it = [1, 2, 4]
    diff = [0, 0, 1],  ||diff||_F = 1,  ||W_pt||_F = sqrt(14)
    Δ_rel = 1 / sqrt(14)
    """
    W_pt = torch.tensor([1.0, 2.0, 3.0])
    W_it = torch.tensor([1.0, 2.0, 4.0])
    assert frob_shift(W_pt, W_it) == pytest.approx(1.0 / math.sqrt(14.0), rel=1e-5)


# ── cosine_distance ────────────────────────────────────────────────────────────

def test_cosine_distance_identity():
    """d_cos = 0 when W_it is a clone of W_pt."""
    W = torch.randn(8, 8)
    assert cosine_distance(W, W.clone()) == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_opposite():
    """d_cos = 2.0 when W_it = −W_pt (fully anti-parallel)."""
    W = torch.randn(6, 6)
    assert cosine_distance(W, -W) == pytest.approx(2.0, abs=1e-5)


def test_cosine_distance_orthogonal_vectors():
    """d_cos = 1.0 for exactly orthogonal flattened vectors."""
    a = torch.tensor([[1.0, 0.0]])
    b = torch.tensor([[0.0, 1.0]])
    assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_orthogonal_matrices():
    """d_cos = 1.0 for 2×2 matrices whose flattened forms are orthogonal."""
    a = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_scale_invariant():
    """Scaling both matrices by any k > 0 doesn't change d_cos."""
    W_pt = torch.randn(6, 6)
    W_it = W_pt + 0.1 * torch.randn(6, 6)
    expected = cosine_distance(W_pt, W_it)
    for k in (0.1, 3.0, 100.0):
        assert cosine_distance(k * W_pt, k * W_it) == pytest.approx(expected, rel=1e-4)


def test_cosine_distance_zero_pt_returns_nan():
    assert math.isnan(cosine_distance(torch.zeros(4, 4), torch.randn(4, 4)))


def test_cosine_distance_zero_it_returns_nan():
    assert math.isnan(cosine_distance(torch.randn(4, 4), torch.zeros(4, 4)))


def test_cosine_distance_bounded_zero_to_two():
    """d_cos is always in [0, 2] for any pair of non-zero matrices."""
    for _ in range(30):
        W_pt = torch.randn(8, 8)
        W_it = torch.randn(8, 8)
        d = cosine_distance(W_pt, W_it)
        assert 0.0 <= d <= 2.0 + 1e-6


def test_cosine_distance_1d_vector_known_value():
    """Works on 1-D tensors with a hand-computed expected value.

    a = [3, 4], b = [4, 3]
    dot = 12 + 12 = 24,  |a| = |b| = 5
    cos = 24/25,  d_cos = 1 - 24/25 = 1/25 = 0.04
    """
    a = torch.tensor([3.0, 4.0])
    b = torch.tensor([4.0, 3.0])
    assert cosine_distance(a, b) == pytest.approx(1.0 - 24.0 / 25.0, rel=1e-5)


# ── independence of the two metrics ───────────────────────────────────────────

def test_metrics_are_independent():
    """frob_shift and cosine_distance capture genuinely different properties.

    Case 1 — scale change only:
        W_it = 2·W_pt  →  Δ_rel > 0,  d_cos ≈ 0
        (matrix grew but kept the same direction)

    Case 2 — direction change only (norm-preserving permutation):
        Construct W_it so ‖W_it‖_F = ‖W_pt‖_F but the directions differ.
        d_cos > 0 even though magnitudes match (Δ_rel may be small).
    """
    W = torch.tensor([[3.0, 0.0], [0.0, 4.0]])

    # Case 1: scale up
    W_scaled = 2.0 * W
    assert frob_shift(W, W_scaled) > 0.0
    assert cosine_distance(W, W_scaled) == pytest.approx(0.0, abs=1e-6)

    # Case 2: direction change — swap columns
    W_rotated = torch.tensor([[0.0, 3.0], [0.0, 4.0]])
    assert cosine_distance(W, W_rotated) > 0.0
