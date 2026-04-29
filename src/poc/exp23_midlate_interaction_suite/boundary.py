"""Boundary-state capture and patch hooks for Exp23 Part B."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _as_hidden_tensor(output) -> torch.Tensor:
    """Return the residual hidden-state tensor from a transformer layer output."""

    if isinstance(output, tuple):
        return output[0]
    return output


@dataclass
class BoundaryStateCapture:
    """Capture the full hidden-state tensor entering one transformer layer."""

    layer: torch.nn.Module

    def __post_init__(self) -> None:
        self.state: torch.Tensor | None = None
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if not args:
            raise RuntimeError("Boundary pre-hook received no positional hidden state")
        hidden = args[0]
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(f"Expected [batch, seq, d_model] hidden state, got {type(hidden)}")
        self.state = hidden.detach().clone()

    def reset(self) -> None:
        self.state = None

    def snapshot(self) -> torch.Tensor:
        if self.state is None:
            raise RuntimeError("Boundary state was not captured")
        return self.state.detach().clone()

    def close(self) -> None:
        self.handle.remove()


@dataclass
class BoundaryStatePatch:
    """Replace the full hidden-state tensor entering one transformer layer.

    When ``noop_tol`` is provided, the hook asserts that the host's incoming
    hidden state matches the donor state to within ``noop_tol`` (per-coordinate
    max absolute deviation). This is the runtime identity check for the no-op
    factorial cells (donor = host), and guarantees that the diagonal cells of
    the 2x2 reproduce the unmodified-checkpoint forward pass to numerical
    tolerance. Off-diagonal (genuine cross-checkpoint) cells should be
    constructed with ``noop_tol=None``.
    """

    layer: torch.nn.Module
    donor_state: torch.Tensor
    noop_tol: float | None = None

    def __post_init__(self) -> None:
        self.n_patches = 0
        self.last_max_abs_input_delta: float | None = None
        self.handle = self.layer.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        if not args:
            raise RuntimeError("Boundary patch hook received no positional hidden state")
        hidden = args[0]
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(f"Expected [batch, seq, d_model] hidden state, got {type(hidden)}")
        if tuple(hidden.shape) != tuple(self.donor_state.shape):
            raise RuntimeError(
                "Boundary donor/host shape mismatch: "
                f"host={tuple(hidden.shape)} donor={tuple(self.donor_state.shape)}"
            )
        patched = self.donor_state.to(device=hidden.device, dtype=hidden.dtype)
        delta = float((patched.float() - hidden.float()).abs().max().item())
        self.last_max_abs_input_delta = delta
        if self.noop_tol is not None and delta > self.noop_tol:
            raise RuntimeError(
                "BoundaryStatePatch no-op identity check failed: "
                f"max-abs delta {delta:.3e} exceeds noop_tol {self.noop_tol:.3e}. "
                "This means the donor state does not match the host's incoming "
                "hidden state at the patched layer; no-op factorial cells are "
                "not reproducing the unmodified checkpoint to tolerance."
            )
        self.n_patches += 1
        return (patched, *args[1:])

    def close(self) -> None:
        self.handle.remove()


@dataclass
class LayerResidualCapture:
    """Capture last-token residual outputs at every layer in one forward."""

    layers: list[torch.nn.Module]
    adapter: object

    def __post_init__(self) -> None:
        self.residuals: list[torch.Tensor | None] = [None] * len(self.layers)
        self.handles = [
            layer.register_forward_hook(self._make_hook(layer_idx))
            for layer_idx, layer in enumerate(self.layers)
        ]

    def _make_hook(self, layer_idx: int):
        def hook(_module, _args, output):
            hidden = self.adapter.residual_from_output(output)
            self.residuals[layer_idx] = hidden[0, -1, :].detach().clone()

        return hook

    def reset(self) -> None:
        self.residuals = [None] * len(self.layers)

    def snapshot(self) -> list[torch.Tensor]:
        missing = [idx for idx, value in enumerate(self.residuals) if value is None]
        if missing:
            raise RuntimeError(f"Missing residual captures at layers {missing}")
        return [value.detach().clone() for value in self.residuals if value is not None]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

