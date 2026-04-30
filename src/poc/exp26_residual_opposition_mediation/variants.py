"""Residual-opposition MLP-output transformations for Exp26.

The transformations operate on the current-token MLP update ``m`` and the
pre-MLP residual stream ``r``. They intentionally do not interpret the
residual-opposing component as a direct token-write vector; Exp26 tests whether
editing that component mediates the Exp23 upstream-state x late-stack
interaction.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch


SUPPORTED_VARIANTS = {
    "full",
    "opp_scale_0p5",
    "noopp",
    "flipopp",
    "randorth",
    "normpres_noopp",
    "ptlevel_opp",
    "opponly",
}


@dataclass(frozen=True)
class VariantResult:
    update: torch.Tensor
    diagnostics: dict[str, float | str | int | None]


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def decompose_negative_parallel(
    update: torch.Tensor,
    residual: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(m_opp, m_rem, coeff)`` for the component of update opposing residual.

    ``update`` and ``residual`` may be shaped ``[d]`` or ``[..., d]``. ``coeff`` has
    shape ``[..., 1]`` and is zeroed when the projection is non-negative.
    """

    update_f = update.float()
    residual_f = residual.float()
    denom = (residual_f * residual_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    coeff_raw = (update_f * residual_f).sum(dim=-1, keepdim=True) / denom
    coeff = torch.where(coeff_raw < 0, coeff_raw, torch.zeros_like(coeff_raw))
    opposing = coeff * residual_f
    remainder = update_f - opposing
    return opposing, remainder, coeff


def _orthogonal_random_like(
    residual: torch.Tensor,
    target_norm: torch.Tensor,
    *,
    seed: int,
    eps: float = 1e-12,
) -> torch.Tensor:
    residual_f = residual.float()
    generator = torch.Generator(device=residual_f.device)
    generator.manual_seed(int(seed))
    noise = torch.randn(residual_f.shape, device=residual_f.device, dtype=torch.float32, generator=generator)
    denom = (residual_f * residual_f).sum(dim=-1, keepdim=True).clamp_min(eps)
    noise = noise - ((noise * residual_f).sum(dim=-1, keepdim=True) / denom) * residual_f
    norm = noise.norm(dim=-1, keepdim=True).clamp_min(eps)
    return noise * (target_norm.float() / norm)


def _norm_ratio(after: torch.Tensor, before: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return after.float().norm(dim=-1) / before.float().norm(dim=-1).clamp_min(eps)


def _mean_finite(tensor: torch.Tensor) -> float | None:
    flat = tensor.detach().float().flatten()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return None
    return float(finite.mean().item())


def apply_variant(
    update: torch.Tensor,
    residual: torch.Tensor,
    *,
    variant: str,
    rand_seed: int | None = None,
    alpha: float | None = None,
) -> VariantResult:
    """Transform one MLP output update according to an Exp26 variant."""

    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported Exp26 variant: {variant}")

    original_dtype = update.dtype
    update_f = update.float()
    residual_f = residual.float()
    opposing, remainder, coeff = decompose_negative_parallel(update_f, residual_f)
    opp_norm = opposing.norm(dim=-1, keepdim=True)
    update_norm = update_f.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    if variant == "full":
        transformed = update_f
    elif variant == "opp_scale_0p5":
        transformed = remainder + 0.5 * opposing
    elif variant == "noopp":
        transformed = remainder
    elif variant == "flipopp":
        transformed = remainder - opposing
    elif variant == "randorth":
        if rand_seed is None:
            raise ValueError("randorth requires rand_seed")
        transformed = remainder + _orthogonal_random_like(residual_f, opp_norm, seed=rand_seed)
    elif variant == "normpres_noopp":
        rem_norm = remainder.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        transformed = remainder * (update_norm / rem_norm)
    elif variant == "ptlevel_opp":
        if alpha is None:
            raise ValueError("ptlevel_opp requires alpha")
        transformed = remainder + float(alpha) * opposing
    elif variant == "opponly":
        transformed = opposing
    else:  # pragma: no cover - kept for exhaustiveness if variants expand.
        raise AssertionError(variant)

    diagnostics = {
        "variant": variant,
        "alpha": None if alpha is None else float(alpha),
        "projection_coeff_mean": _mean_finite(coeff),
        "opp_norm_frac": _mean_finite(opp_norm.squeeze(-1) / update_norm.squeeze(-1)),
        "update_norm_ratio_after_hook": _mean_finite(_norm_ratio(transformed, update_f)),
        "postres_norm_ratio_after_hook": _mean_finite(
            _norm_ratio(residual_f + transformed, residual_f + update_f)
        ),
    }
    return VariantResult(update=transformed.to(dtype=original_dtype), diagnostics=diagnostics)


def expand_variants(variants: list[str], rand_seeds: list[int]) -> list[tuple[str, int | None]]:
    expanded: list[tuple[str, int | None]] = []
    for variant in variants:
        if variant.startswith("randorth_s"):
            expanded.append(("randorth", int(variant.removeprefix("randorth_s"))))
        elif variant == "randorth":
            for seed in rand_seeds:
                expanded.append(("randorth", int(seed)))
        else:
            expanded.append((variant, None))
    return expanded


def finite_mean(values: list[float | int | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not kept:
        return None
    return sum(kept) / len(kept)

