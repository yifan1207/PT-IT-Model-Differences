"""
Reusable metric functions for weight and activation analysis.

All functions operate on float32 torch.Tensor objects on a single device.
They are pure (no side-effects) and NaN-safe: if the inputs have zero norm
the result is float('nan') rather than inf or an exception.
"""
import torch


def frob_shift(w_pt: torch.Tensor, w_it: torch.Tensor) -> float:
    """Relative Frobenius shift: ||W_it − W_pt||_F / ||W_pt||_F.

    Measures how much the matrix changed in absolute magnitude during
    fine-tuning, normalised by the pre-training scale so large matrices do
    not dominate.

    Returns float('nan') if ||W_pt||_F == 0.
    """
    pt_norm = w_pt.norm().item()
    if pt_norm == 0.0:
        return float("nan")
    return ((w_it - w_pt).norm() / pt_norm).item()


def cosine_distance(w_pt: torch.Tensor, w_it: torch.Tensor) -> float:
    """Cosine distance on flattened weights: 1 − cos(vec(W_pt), vec(W_it)).

    Measures whether fine-tuning changed the matrix's orientation / subspace
    independently of its magnitude.

      d_cos ≈ 0   →  same direction (scale may differ)
      d_cos ≈ 1   →  orthogonal (entirely new subspace)
      d_cos ≈ 2   →  opposite direction

    Returns float('nan') if either norm is zero.
    The cosine similarity is clamped to [−1, 1] before inversion to guard
    against floating-point overshoot.
    """
    pt_flat = w_pt.reshape(-1)
    it_flat = w_it.reshape(-1)
    pt_norm = pt_flat.norm().item()
    it_norm = it_flat.norm().item()
    denom = pt_norm * it_norm
    if denom == 0.0:
        return float("nan")
    cos_sim = torch.dot(pt_flat, it_flat).item() / denom
    cos_sim = max(-1.0, min(1.0, cos_sim))
    return 1.0 - cos_sim
