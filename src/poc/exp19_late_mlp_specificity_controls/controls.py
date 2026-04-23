"""Control-intervention helpers for Exp19.

These helpers are intentionally tensor-local and deterministic so the GPU
runner can resume safely without relying on global RNG state.
"""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch

CONTROL_MODE_TRUE = "true"
CONTROL_MODE_RANDOM_NORM = "random_norm"
CONTROL_MODE_RANDOM_RESPROJ = "random_residual_projection"
VALID_CONTROL_MODES = {
    CONTROL_MODE_TRUE,
    CONTROL_MODE_RANDOM_NORM,
    CONTROL_MODE_RANDOM_RESPROJ,
}


def stable_int_seed(*parts: object) -> int:
    """Return a deterministic positive seed from arbitrary identifying parts."""

    payload = "::".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Torch generators accept signed 64-bit seeds. Stay comfortably below that.
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def layer_permutation_map(start_layer: int, end_layer_exclusive: int, *, mode: str = "reverse") -> dict[int, int]:
    """Return a deterministic donor-layer permutation inside a contiguous window."""

    if start_layer < 0 or end_layer_exclusive <= start_layer:
        raise ValueError(f"Invalid layer window [{start_layer}, {end_layer_exclusive})")
    layers = list(range(start_layer, end_layer_exclusive))
    if mode == "reverse":
        donors = list(reversed(layers))
    else:
        raise ValueError(f"Unsupported layer permutation mode: {mode}")
    return dict(zip(layers, donors, strict=True))


def _make_noise_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(int(seed) & ((1 << 63) - 1))
    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=torch.float32,
        generator=generator,
    )


def _unit_vector(vectors: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    norms = vectors.norm(dim=-1, keepdim=True)
    return torch.where(norms > eps, vectors / norms.clamp_min(eps), torch.zeros_like(vectors))


def _orthogonalize(vectors: torch.Tensor, direction_unit: torch.Tensor) -> torch.Tensor:
    projection = (vectors * direction_unit).sum(dim=-1, keepdim=True) * direction_unit
    return vectors - projection


def matched_random_delta(
    *,
    true_delta: torch.Tensor,
    residual_reference: torch.Tensor,
    mode: str,
    seed: int,
) -> torch.Tensor:
    """Construct a deterministic random control delta.

    `true_delta` and `residual_reference` have shape [..., d_model].

    - `random_norm` preserves only the total norm of `true_delta`.
    - `random_residual_projection` preserves the signed projection of
      `true_delta` onto `residual_reference` and the orthogonal norm, while
      randomizing the orthogonal direction.
    """

    if mode not in VALID_CONTROL_MODES:
        raise ValueError(f"Unsupported control mode: {mode}")
    true_delta_f = true_delta.float()
    residual_f = residual_reference.float()
    if mode == CONTROL_MODE_TRUE:
        return true_delta

    noise = _make_noise_like(true_delta_f, seed)
    eps = 1e-12

    if mode == CONTROL_MODE_RANDOM_NORM:
        noise_unit = _unit_vector(noise, eps=eps)
        delta_norm = true_delta_f.norm(dim=-1, keepdim=True)
        return (delta_norm * noise_unit).to(dtype=true_delta.dtype)

    residual_norm = residual_f.norm(dim=-1, keepdim=True)
    residual_unit = _unit_vector(residual_f, eps=eps)
    true_parallel_coeff = (true_delta_f * residual_unit).sum(dim=-1, keepdim=True)
    true_parallel = true_parallel_coeff * residual_unit
    true_orth = true_delta_f - true_parallel
    true_orth_norm = true_orth.norm(dim=-1, keepdim=True)

    noise_orth = _orthogonalize(noise, residual_unit)
    noise_orth_unit = _unit_vector(noise_orth, eps=eps)
    matched = true_parallel + true_orth_norm * noise_orth_unit

    # If the reference is degenerate, fall back to norm matching. This should
    # rarely happen for real residuals but makes toy tests well-defined.
    fallback = true_delta_f.norm(dim=-1, keepdim=True) * _unit_vector(noise, eps=eps)
    matched = torch.where(residual_norm > eps, matched, fallback)
    return matched.to(dtype=true_delta.dtype)


def mean_abs_projection(values: Iterable[float]) -> float:
    kept = [abs(float(value)) for value in values]
    return sum(kept) / len(kept) if kept else 0.0
