from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.poc.exp5.utils import navigate_path


@dataclass
class PrecomputedAblationStats:
    mean_mlp_outputs: dict[int, torch.Tensor]
    corrective_directions: dict[int, torch.Tensor]
    resample_bank: dict[int, torch.Tensor]

    @classmethod
    def empty(cls) -> "PrecomputedAblationStats":
        return cls(mean_mlp_outputs={}, corrective_directions={}, resample_bank={})

    @classmethod
    def load(
        cls,
        mean_acts_path: str = "",
        corrective_direction_path: str = "",
        resample_bank_path: str = "",
        device: str = "cpu",
    ) -> "PrecomputedAblationStats":
        stats = cls.empty()
        if mean_acts_path and Path(mean_acts_path).exists():
            with np.load(mean_acts_path) as data:
                for k in data.files:
                    if k.startswith("layer_"):
                        layer = int(k.split("_", 1)[1])
                        stats.mean_mlp_outputs[layer] = torch.tensor(
                            data[k], dtype=torch.float32, device=device
                        )
        if corrective_direction_path and Path(corrective_direction_path).exists():
            with np.load(corrective_direction_path) as data:
                for k in data.files:
                    if k.startswith("layer_"):
                        layer = int(k.split("_", 1)[1])
                        stats.corrective_directions[layer] = torch.tensor(
                            data[k], dtype=torch.float32, device=device
                        )
        if resample_bank_path and Path(resample_bank_path).exists():
            with np.load(resample_bank_path) as data:
                for k in data.files:
                    if k.startswith("layer_"):
                        layer = int(k.split("_", 1)[1])
                        stats.resample_bank[layer] = torch.tensor(
                            data[k], dtype=torch.float32, device=device
                        )
        return stats


def _broadcast_like(vec: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a [d_model] vector to the exact shape of an MLP output tensor."""
    v = vec.to(device=ref.device, dtype=ref.dtype)
    while v.dim() < ref.dim():
        v = v.unsqueeze(0)
    return v.expand_as(ref)


def directional_ablate_tensor(
    mlp_out: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Scale the corrective direction component in `mlp_out`.

    alpha is a "correction-strength" parameter per the exp5 design (Section C.2):

        h' = h - (1 - alpha) * proj(h, v̂)

    where v̂ = direction / ||direction||.  Interpretation:
      alpha = 1.0  → no change (baseline — full correction retained)
      alpha = 0.0  → corrective direction fully removed
      alpha = 0.5  → half-strength correction
      alpha = 2.0  → double correction (amplified)
      alpha = -1.0 → direction reversed (anti-correction)

    This matches the Arditi et al. 2024 abliteration formula
    (x' = x - (x·v̂)v̂) at alpha=0, and the ActAdd/RepE convention
    that alpha=0 is the maximally-steered state.
    """
    direction = direction.to(device=mlp_out.device, dtype=mlp_out.dtype)
    direction = direction / (direction.norm() + 1e-12)
    flat = mlp_out.reshape(-1, mlp_out.shape[-1])
    proj_coeff = flat @ direction
    proj = proj_coeff.unsqueeze(-1) * direction.unsqueeze(0)
    ablated = flat - (1.0 - alpha) * proj
    return ablated.reshape_as(mlp_out)


class InterventionSpec:
    """Runtime intervention object applied inside an nnsight trace."""

    def __init__(
        self,
        method: str = "none",
        layers: list[int] | None = None,
        alpha: float = 1.0,
        stats: PrecomputedAblationStats | None = None,
        proposal_boundary: int = 20,
    ) -> None:
        self.method = method
        self.layers = set(layers or [])
        self.alpha = alpha
        self.stats = stats or PrecomputedAblationStats.empty()
        self.proposal_boundary = proposal_boundary

    @property
    def active(self) -> bool:
        return self.method != "none" and bool(self.layers)

    def validate(self) -> None:
        if self.method == "mean":
            missing = sorted(i for i in self.layers if i not in self.stats.mean_mlp_outputs)
            if missing:
                raise ValueError(f"Missing mean MLP outputs for layers: {missing}")
        if self.method == "directional":
            bad = sorted(i for i in self.layers if i < self.proposal_boundary)
            if bad:
                raise ValueError(
                    f"Directional ablation is only supported for corrective layers >= "
                    f"{self.proposal_boundary}; got {bad}"
                )
            missing = sorted(i for i in self.layers if i not in self.stats.corrective_directions)
            if missing:
                raise ValueError(f"Missing corrective directions for layers: {missing}")
        if self.method == "resample":
            raise NotImplementedError(
                "Resample ablation is not yet implemented end-to-end: the current "
                "bank stores one static vector per layer, not a proper random draw. "
                "Use 'mean', 'skip', or 'directional' until a real bank is built."
            )

    def apply_in_trace(self, layer: Any, layer_idx: int, hooks: Any) -> None:
        if layer_idx not in self.layers or self.method == "none":
            return

        if self.method == "mean":
            target = navigate_path(layer, hooks.mlp_output)
            target[:] = _broadcast_like(self.stats.mean_mlp_outputs[layer_idx], target)
            return

        if self.method == "directional":
            target = navigate_path(layer, hooks.mlp_output)
            target[:] = directional_ablate_tensor(
                target, self.stats.corrective_directions[layer_idx], self.alpha
            )
            return

        if self.method == "resample":
            raise NotImplementedError(
                "Resample ablation is not yet implemented end-to-end. "
                "Use 'mean', 'skip', or 'directional' instead."
            )

        if self.method == "skip":
            # Zero both the attention hidden-state contribution and the MLP output so
            # the residual add for each is h + 0 = h, leaving the residual stream
            # unchanged — equivalent to a full layer skip per Lad et al. (2024).
            #
            # We write scalar 0 rather than using proxy-to-proxy assignment
            # (output[:] = input) because assignment between two nnsight proxies
            # can have ordering ambiguity when the proxies come from tuple outputs.
            # Writing a scalar 0 is unambiguous and matches the proven pattern
            # used by mean/directional/resample (proxy[:] = real value).
            #
            # Registration order within each layer:
            #   1. attn_hidden_output  (self_attn runs before MLP in forward pass)
            #   2. mlp_output          (MLP runs after attention)
            # This matches the Gemma 3 decoder forward-pass order, so no
            # nnsight OutOfOrderError is possible.
            attn_h = navigate_path(layer, hooks.attn_hidden_output)
            attn_h[:] = 0
            mlp_o = navigate_path(layer, hooks.mlp_output)
            mlp_o[:] = 0
            return

        raise ValueError(f"Unknown intervention method: {self.method}")

