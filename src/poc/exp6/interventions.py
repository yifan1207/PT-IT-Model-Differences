"""Intervention primitives for Exp6 steering experiments.

Approach A — Direction-level steering (used inside nnsight trace):
  directional_remove: h' = h - (1-α) × proj(h, v̂)   [existing Exp5 formula]
  directional_add:    h' = h + β × d × ‖h_flat‖      [inject at natural magnitude]
  directional_random: same as add but with a seeded random unit vector
  directional_rotated: same as add but with a vector orthogonal to d_corr

Approach B — Feature-level steering (used with plain register_forward_hook):
  Implemented in runtime.py as _make_feature_clamp_hooks() — transcoder operations
  cannot run inside nnsight trace proxies, so they use standard PyTorch hooks.

All intervention objects hold preloaded tensors so they can be applied
inside a hot generation loop without any file I/O or dict lookups.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.poc.exp5.interventions import (
    PrecomputedAblationStats,
    _broadcast_like,
    directional_ablate_tensor,
)
from src.poc.exp5.utils import navigate_path


# ── New primitive: direction injection ────────────────────────────────────────

def directional_inject_tensor(
    mlp_out: torch.Tensor,
    direction: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Inject a direction into mlp_out, scaled by the per-token hidden-state norm.

    Formula:  h' = h + β × d̂ × ‖h‖

    Scaling by ‖h‖ ensures the injection has the same relative magnitude as the
    natural hidden-state activations, following Turner et al. 2023 (ActAdd)
    magnitude-matching convention.  β=0 → identity; β>0 → amplify; β<0 → suppress.

    Args:
        mlp_out: [..., d_model] MLP output tensor (any batch/sequence dims).
        direction: [d_model] unit vector (will be normalised internally).
        beta: injection magnitude.
    Returns:
        Modified tensor, same shape as mlp_out.
    """
    direction = direction.to(device=mlp_out.device, dtype=mlp_out.dtype)
    direction = direction / (direction.norm() + 1e-12)
    flat = mlp_out.reshape(-1, mlp_out.shape[-1])           # [B*T, d_model]
    norms = flat.norm(dim=-1, keepdim=True)                  # [B*T, 1]
    injected = flat + beta * norms * direction.unsqueeze(0)  # broadcast
    return injected.reshape_as(mlp_out)


def _random_unit_vector(d_model: int, seed: int, layer_idx: int) -> torch.Tensor:
    """Generate a reproducible random unit vector for a given layer."""
    rng = torch.Generator()
    rng.manual_seed(seed + layer_idx * 7919)  # layer-specific seed offset
    v = torch.randn(d_model, generator=rng)
    return v / (v.norm() + 1e-12)


def _rotated_direction(direction: torch.Tensor, seed: int, layer_idx: int) -> torch.Tensor:
    """Compute a unit vector orthogonal to `direction` in a random plane.

    Gramm-Schmidt: pick a random vector, subtract its projection onto direction.
    This gives a vector that is exactly orthogonal to direction (dot product = 0).
    """
    rng = torch.Generator()  # CPU generator — generate on CPU, then move to target device
    rng.manual_seed(seed + layer_idx * 1031 + 999983)
    random_v = torch.randn(direction.shape, generator=rng, dtype=direction.dtype).to(device=direction.device)
    random_v = random_v / (random_v.norm() + 1e-12)
    d_norm = direction / (direction.norm() + 1e-12)
    # Remove component along direction
    ortho = random_v - (random_v @ d_norm) * d_norm
    return ortho / (ortho.norm() + 1e-12)


# ── Intervention spec for Approach A ─────────────────────────────────────────

class Exp6InterventionSpec:
    """Runtime intervention applied inside an nnsight trace (Approach A only).

    For Approach B (feature_clamp, wdec_inject) use _make_feature_clamp_hooks()
    in runtime.py — those operate outside the nnsight trace context.
    """

    def __init__(
        self,
        method: str = "none",
        layers: list[int] | None = None,
        alpha: float = 1.0,         # for directional_remove
        beta: float = 0.0,          # for directional_add / random / rotated / content
        corrective_directions: dict[int, torch.Tensor] | None = None,
        content_directions: dict[int, torch.Tensor] | None = None,
        random_seed: int = 42,
    ) -> None:
        self.method = method
        self.layers = set(layers or [])
        self.alpha = alpha
        self.beta = beta
        self.corrective_directions = corrective_directions or {}
        self.content_directions = content_directions or {}
        self.random_seed = random_seed
        # Precompute random/rotated directions once per layer
        self._random_dirs: dict[int, torch.Tensor] = {}
        self._rotated_dirs: dict[int, torch.Tensor] = {}

    @property
    def active(self) -> bool:
        return self.method != "none" and bool(self.layers)

    def validate(self, n_layers: int = 34) -> None:
        out_of_range = sorted(i for i in self.layers if not (0 <= i < n_layers))
        if out_of_range:
            raise ValueError(f"Intervention layer indices out of range [0, {n_layers}): {out_of_range}")
        if self.method in ("directional_remove", "directional_add"):
            missing = sorted(i for i in self.layers if i not in self.corrective_directions)
            if missing:
                raise ValueError(f"Missing corrective directions for layers: {missing}")
            degenerate = sorted(
                i for i in self.layers
                if i in self.corrective_directions
                and float(self.corrective_directions[i].norm()) < 1e-4
            )
            if degenerate:
                raise ValueError(
                    f"Corrective directions at layers {degenerate} have near-zero norm."
                )
        if self.method == "content_direction":
            missing = sorted(i for i in self.layers if i not in self.content_directions)
            if missing:
                raise ValueError(f"Missing content directions for layers: {missing}")

    def _get_random_dir(self, layer_idx: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if layer_idx not in self._random_dirs:
            v = _random_unit_vector(d_model, self.random_seed, layer_idx)
            self._random_dirs[layer_idx] = v.to(device=device, dtype=dtype)
        return self._random_dirs[layer_idx]

    def _get_rotated_dir(self, layer_idx: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if layer_idx not in self._rotated_dirs:
            d = self.corrective_directions[layer_idx].to(device=device, dtype=dtype)
            self._rotated_dirs[layer_idx] = _rotated_direction(d, self.random_seed, layer_idx)
        return self._rotated_dirs[layer_idx]

    def apply_in_trace(self, layer: Any, layer_idx: int, hooks: Any) -> None:
        if layer_idx not in self.layers or self.method == "none":
            return

        target = navigate_path(layer, hooks.mlp_output)

        if self.method == "directional_remove":
            # h' = h - (1-α) × proj(h, v̂)    — same as Exp5 directional_ablate_tensor
            target[:] = directional_ablate_tensor(
                target, self.corrective_directions[layer_idx], self.alpha
            )
            return

        if self.method == "directional_add":
            target[:] = directional_inject_tensor(
                target, self.corrective_directions[layer_idx], self.beta
            )
            return

        if self.method == "directional_random":
            d_model = self.corrective_directions[next(iter(self.corrective_directions))].shape[0]
            rdir = self._get_random_dir(layer_idx, d_model, target.device, target.dtype)
            target[:] = directional_inject_tensor(target, rdir, self.beta)
            return

        if self.method == "directional_rotated":
            rdir = self._get_rotated_dir(layer_idx, target.device, target.dtype)
            target[:] = directional_inject_tensor(target, rdir, self.beta)
            return

        if self.method == "content_direction":
            target[:] = directional_inject_tensor(
                target, self.content_directions[layer_idx], self.beta
            )
            return

        raise ValueError(f"Unknown A-experiment method: {self.method!r}")


    def register_hooks(self, model_raw: Any, cfg: Any) -> list:
        """Register forward hooks on raw HF model MLP modules (fast path).

        Returns a list of hook handles. Caller must call h.remove() for each.
        This replaces apply_in_trace() for A experiments, enabling use of
        model.generate() with KV cache instead of the slow token-by-token nnsight loop.
        """
        if self.method == "none" or not self.layers:
            return []

        handles = []
        for layer_idx in self.layers:
            mlp_mod = model_raw.language_model.layers[layer_idx].mlp

            def make_hook(li: int):
                def hook(mod: Any, inp: tuple, out: torch.Tensor) -> torch.Tensor:
                    with torch.no_grad():
                        if self.method == "directional_remove":
                            return directional_ablate_tensor(
                                out, self.corrective_directions[li], self.alpha
                            )
                        if self.method == "directional_add":
                            return directional_inject_tensor(
                                out, self.corrective_directions[li], self.beta
                            )
                        if self.method == "directional_random":
                            d_model = out.shape[-1]
                            rdir = self._get_random_dir(li, d_model, out.device, out.dtype)
                            return directional_inject_tensor(out, rdir, self.beta)
                        if self.method == "directional_rotated":
                            rdir = self._get_rotated_dir(li, out.device, out.dtype)
                            return directional_inject_tensor(out, rdir, self.beta)
                        if self.method == "content_direction":
                            return directional_inject_tensor(
                                out, self.content_directions[li], self.beta
                            )
                    return out
                return hook

            handles.append(mlp_mod.register_forward_hook(make_hook(layer_idx)))
        return handles


# ── Loader: build intervention from Exp6Config ────────────────────────────────

def load_directions_from_npz(
    npz_path: str, device: str = "cpu"
) -> dict[int, torch.Tensor]:
    """Load a corrective_directions.npz / content_directions.npz into dict[layer → tensor]."""
    out: dict[int, torch.Tensor] = {}
    if not npz_path or not Path(npz_path).exists():
        return out
    with np.load(npz_path) as data:
        for k in data.files:
            if k.startswith("layer_"):
                layer = int(k.split("_", 1)[1])
                out[layer] = torch.tensor(data[k], dtype=torch.float32, device=device)
            else:
                out[k] = torch.tensor(data[k], dtype=torch.float32, device=device)
    return out


def build_intervention(cfg) -> Exp6InterventionSpec:
    """Build the A-experiment intervention from an Exp6Config.

    Returns a no-op spec for B experiments (those use plain hooks in runtime.py).
    """
    if cfg.method in ("feature_clamp", "wdec_inject", "none"):
        return Exp6InterventionSpec(method="none")

    corrective_dirs = load_directions_from_npz(cfg.corrective_direction_path, cfg.device)
    content_dirs = load_directions_from_npz(cfg.content_direction_path, cfg.device)

    # Determine which layers this intervention targets.
    # cfg.ablation_layers overrides the default corrective_layers when explicitly set
    # (used by A1_early / A1_mid layer-specificity ablations).
    target_layers = cfg.ablation_layers if cfg.ablation_layers else cfg.corrective_layers

    # Aggregate content directions into a single direction when only one aggregate
    # direction file is present. Map it to every target layer.
    if content_dirs and "aggregate" in content_dirs:
        agg = content_dirs.pop("aggregate")
        for li in target_layers:
            content_dirs[li] = agg

    # For A1_early / A1_mid ablations: target layers (0–7 or 8–19) may not have
    # precomputed corrective directions (those were only extracted for layers 20–33).
    # Fill missing layers with the mean of available directions so we can test
    # whether applying the governance direction at a different depth has any effect.
    if cfg.method in ("directional_remove", "directional_add"):
        missing = [li for li in target_layers if li not in corrective_dirs]
        if missing:
            available = [v for k, v in corrective_dirs.items() if isinstance(k, int)]
            if available:
                mean_dir = torch.stack(available).mean(dim=0)
                mean_dir = mean_dir / (mean_dir.norm() + 1e-12)
                for li in missing:
                    corrective_dirs[li] = mean_dir

    return Exp6InterventionSpec(
        method=cfg.method,
        layers=target_layers,
        alpha=cfg.directional_alpha,
        beta=cfg.directional_beta,
        corrective_directions=corrective_dirs,
        content_directions=content_dirs,
        random_seed=cfg.random_direction_seed,
    )
