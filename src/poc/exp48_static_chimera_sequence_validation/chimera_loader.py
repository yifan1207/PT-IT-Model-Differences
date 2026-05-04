"""Static module-swap helpers for Exp48 sequence chimeras."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Iterator

import torch


def layer_container(model: torch.nn.Module) -> torch.nn.ModuleList:
    try:
        return model.model.layers
    except AttributeError as exc:
        raise TypeError("Exp48 static chimeras require model.model.layers") from exc


def final_norm_parent(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    return model.model, "norm"


def lm_head_parent(model: torch.nn.Module) -> tuple[torch.nn.Module, str]:
    return model, "lm_head"


@dataclass
class _Swap:
    parent: torch.nn.Module
    name: str
    old_value: torch.nn.Module

    def restore(self) -> None:
        setattr(self.parent, self.name, self.old_value)


def _set(parent: torch.nn.Module, name: str, value: torch.nn.Module, swaps: list[_Swap]) -> None:
    swaps.append(_Swap(parent=parent, name=name, old_value=getattr(parent, name)))
    setattr(parent, name, value)


@contextlib.contextmanager
def static_chimera_context(
    *,
    host_model: torch.nn.Module,
    donor_model: torch.nn.Module,
    boundary: int,
    component: str,
    cell: str,
    permute_blocks: bool = False,
) -> Iterator[None]:
    """Temporarily replace late modules in ``host_model`` with donor modules.

    ``cell`` controls whether the context is active:
    - BB/FF are native cells and do not swap.
    - BF/FB swap the late side of the host with the donor.

    ``component`` controls exactly what late side means. For decomposition,
    ``mlp_only`` and ``attn_only`` replace only submodules in late blocks while
    keeping the donor readout, matching the planned "what component carries the
    static sequence behavior" diagnostic.
    """

    if cell in {"BB", "FF"}:
        yield
        return
    if component not in {
        "blocks_plus_head",
        "blocks_only",
        "head_only",
        "mlp_only",
        "attn_only",
    }:
        raise ValueError(f"Unknown static chimera component: {component}")

    host_layers = layer_container(host_model)
    donor_layers = layer_container(donor_model)
    n_layers = min(len(host_layers), len(donor_layers))
    if boundary < 0 or boundary >= n_layers:
        raise ValueError(f"Boundary {boundary} outside layer range 0..{n_layers - 1}")

    donor_indices = list(range(boundary, n_layers))
    if permute_blocks:
        donor_indices = list(reversed(donor_indices))
    swaps: list[_Swap] = []
    try:
        if component in {"blocks_plus_head", "blocks_only"}:
            for offset, layer_idx in enumerate(range(boundary, n_layers)):
                _set(host_layers, str(layer_idx), donor_layers[donor_indices[offset]], swaps)
        elif component == "mlp_only":
            for offset, layer_idx in enumerate(range(boundary, n_layers)):
                _set(host_layers[layer_idx], "mlp", donor_layers[donor_indices[offset]].mlp, swaps)
        elif component == "attn_only":
            for offset, layer_idx in enumerate(range(boundary, n_layers)):
                _set(host_layers[layer_idx], "self_attn", donor_layers[donor_indices[offset]].self_attn, swaps)

        if component in {"blocks_plus_head", "head_only", "mlp_only", "attn_only"}:
            norm_parent, norm_name = final_norm_parent(host_model)
            head_parent, head_name = lm_head_parent(host_model)
            donor_norm_parent, donor_norm_name = final_norm_parent(donor_model)
            donor_head_parent, donor_head_name = lm_head_parent(donor_model)
            _set(norm_parent, norm_name, getattr(donor_norm_parent, donor_norm_name), swaps)
            _set(head_parent, head_name, getattr(donor_head_parent, donor_head_name), swaps)

        yield
    finally:
        for swap in reversed(swaps):
            swap.restore()


@contextlib.contextmanager
def interpolated_late_context(
    *,
    host_model: torch.nn.Module,
    donor_model: torch.nn.Module,
    boundary: int,
    alpha: float,
    include_readout: bool = True,
) -> Iterator[None]:
    """In-place interpolate host late-stack parameters toward donor parameters.

    The context stores host parameters and restores them before exit. This is
    slower than module swaps but avoids writing merged checkpoints and keeps the
    interpolated-late control faithful to the pre-registered plan.
    """

    if alpha == 0.0:
        yield
        return
    host_layers = layer_container(host_model)
    donor_layers = layer_container(donor_model)
    n_layers = min(len(host_layers), len(donor_layers))
    modules: list[tuple[torch.nn.Module, torch.nn.Module]] = [
        (host_layers[i], donor_layers[i]) for i in range(boundary, n_layers)
    ]
    if include_readout:
        modules.extend(
            [
                (host_model.model.norm, donor_model.model.norm),
                (host_model.lm_head, donor_model.lm_head),
            ]
        )

    saved: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    try:
        with torch.no_grad():
            for host_module, donor_module in modules:
                for host_param, donor_param in zip(
                    host_module.parameters(recurse=True),
                    donor_module.parameters(recurse=True),
                    strict=False,
                ):
                    if host_param.shape != donor_param.shape:
                        raise RuntimeError(
                            "Cannot interpolate mismatched parameters: "
                            f"{tuple(host_param.shape)} vs {tuple(donor_param.shape)}"
                        )
                    saved.append((host_param, host_param.detach().clone()))
                    host_param.copy_(
                        host_param.mul(1.0 - float(alpha))
                        + donor_param.to(host_param.device, dtype=host_param.dtype).mul(float(alpha))
                    )
        yield
    finally:
        with torch.no_grad():
            for param, value in reversed(saved):
                param.copy_(value)
