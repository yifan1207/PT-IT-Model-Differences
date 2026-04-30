"""Residual-opposition variants for Exp27 natural-rollout replay."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.poc.exp26_residual_opposition_mediation.variants import (
    apply_variant as apply_exp26_variant,
    decompose_negative_parallel,
    finite_mean,
    stable_seed,
)


SUPPORTED_VARIANTS = {
    "full",
    "noopp",
    "normpres_noopp",
    "flipopp",
    "randorth",
    "randremove",
    "randremove_resnorm",
}


@dataclass(frozen=True)
class VariantResult:
    update: torch.Tensor
    diagnostics: dict[str, float | str | int | None]


def _mean_finite(tensor: torch.Tensor) -> float | None:
    flat = tensor.detach().float().flatten()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return None
    return float(finite.mean().item())


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
    return noise * (target_norm.float() / noise.norm(dim=-1, keepdim=True).clamp_min(eps))


def _diagnostics(
    *,
    variant: str,
    alpha: float | None,
    coeff: torch.Tensor,
    opposing: torch.Tensor,
    update: torch.Tensor,
    transformed: torch.Tensor,
    residual: torch.Tensor,
) -> dict[str, float | str | int | None]:
    update_norm = update.float().norm(dim=-1).clamp_min(1e-12)
    opp_norm = opposing.float().norm(dim=-1)
    post_before = residual.float() + update.float()
    post_after = residual.float() + transformed.float()
    return {
        "variant": variant,
        "alpha": None if alpha is None else float(alpha),
        "projection_coeff_mean": _mean_finite(coeff),
        "opp_norm_frac": _mean_finite(opp_norm / update_norm),
        "update_norm_ratio_after_hook": _mean_finite(transformed.float().norm(dim=-1) / update_norm),
        "postres_norm_ratio_after_hook": _mean_finite(
            post_after.norm(dim=-1) / post_before.norm(dim=-1).clamp_min(1e-12)
        ),
    }


def apply_variant(
    update: torch.Tensor,
    residual: torch.Tensor,
    *,
    variant: str,
    rand_seed: int | None = None,
) -> VariantResult:
    """Apply an Exp27 MLP-output variant.

    ``randremove`` subtracts a random orthogonal vector with the same norm as the
    learned residual-opposing component. ``randremove_resnorm`` then rescales the
    post-MLP residual to match the post-residual norm produced by ``noopp``.
    """

    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f"Unsupported Exp27 variant: {variant}")
    if variant in {"full", "noopp", "normpres_noopp", "flipopp", "randorth"}:
        result = apply_exp26_variant(update, residual, variant=variant, rand_seed=rand_seed)
        return VariantResult(update=result.update, diagnostics=dict(result.diagnostics))
    if rand_seed is None:
        raise ValueError(f"{variant} requires rand_seed")

    original_dtype = update.dtype
    update_f = update.float()
    residual_f = residual.float()
    opposing, remainder, coeff = decompose_negative_parallel(update_f, residual_f)
    random_vec = _orthogonal_random_like(
        residual_f,
        opposing.norm(dim=-1, keepdim=True),
        seed=int(rand_seed),
    )
    transformed = update_f - random_vec
    if variant == "randremove_resnorm":
        target_post_norm = (residual_f + remainder).norm(dim=-1, keepdim=True)
        control_post = residual_f + transformed
        transformed = control_post * (target_post_norm / control_post.norm(dim=-1, keepdim=True).clamp_min(1e-12)) - residual_f
    diagnostics = _diagnostics(
        variant=variant,
        alpha=None,
        coeff=coeff,
        opposing=opposing,
        update=update_f,
        transformed=transformed,
        residual=residual_f,
    )
    return VariantResult(update=transformed.to(dtype=original_dtype), diagnostics=diagnostics)


def expand_variants(variants: list[str], rand_seeds: list[int]) -> list[tuple[str, int | None]]:
    expanded: list[tuple[str, int | None]] = []
    for variant in variants:
        if variant in {"randorth", "randremove", "randremove_resnorm"}:
            for seed in rand_seeds:
                expanded.append((variant, int(seed)))
        elif variant.startswith("randorth_s"):
            expanded.append(("randorth", int(variant.removeprefix("randorth_s"))))
        elif variant.startswith("randremove_resnorm_s"):
            expanded.append(("randremove_resnorm", int(variant.removeprefix("randremove_resnorm_s"))))
        elif variant.startswith("randremove_s"):
            expanded.append(("randremove", int(variant.removeprefix("randremove_s"))))
        else:
            expanded.append((variant, None))
    return expanded
