"""Late-MLP hooks for Exp26 residual-opposition interventions."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import ArchitectureProbe
from src.poc.exp26_residual_opposition_mediation.variants import apply_variant, finite_mean, stable_seed


@dataclass
class LateMlpOppositionModifier:
    """Dynamically edit late MLP outputs at the current next-token position."""

    model: Any
    steering_adapter: Any
    late_layers: list[int]
    variant: str
    prompt_id: str
    cell_name: str
    seed: int | None = None
    alpha_by_layer: dict[int, float] | None = None

    def __post_init__(self) -> None:
        self.layers = self.steering_adapter.get_layers(self.model)
        self.arch_probe = ArchitectureProbe()
        self.handles: list[Any] = []
        self._pre_mlp_residual: dict[int, torch.Tensor] = {}
        self.layer_diagnostics: dict[int, list[dict[str, float | str | int | None]]] = defaultdict(list)
        self._register()

    def _register(self) -> None:
        for layer_idx in self.late_layers:
            layer = self.layers[layer_idx]
            capture_spec = self.arch_probe.get_capture_spec(layer)
            if capture_spec.mode == "norm":
                norm_module = capture_spec.module

                def norm_pre_hook(_module, args, li=layer_idx):
                    if not args or not torch.is_tensor(args[0]):
                        raise RuntimeError("Exp26 norm pre-hook expected hidden-state tensor")
                    self._pre_mlp_residual[li] = args[0][:, -1, :].detach()

                self.handles.append(norm_module.register_forward_pre_hook(norm_pre_hook))

            def mlp_hook(_module, args, output, li=layer_idx, mode=capture_spec.mode):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Exp26 expected tensor MLP output at layer {li}, got {type(output)}")
                if mode == "mlp_input_direct":
                    if not args or not torch.is_tensor(args[0]):
                        raise RuntimeError("Exp26 MLP direct mode expected hidden-state input")
                    residual = args[0][:, -1, :].detach()
                else:
                    residual = self._pre_mlp_residual.get(li)
                    if residual is None:
                        raise RuntimeError(f"Exp26 missing pre-MLP residual at layer {li}")
                update = output[:, -1, :]
                rand_seed = (
                    None
                    if self.seed is None
                    else stable_seed("exp26", self.seed, self.prompt_id, self.cell_name, li)
                )
                alpha = None if self.alpha_by_layer is None else self.alpha_by_layer.get(li)
                result = apply_variant(
                    update,
                    residual.to(device=update.device),
                    variant=self.variant,
                    rand_seed=rand_seed,
                    alpha=alpha,
                )
                new_output = output.clone()
                new_output[:, -1, :] = result.update
                diagnostics = dict(result.diagnostics)
                diagnostics["layer"] = int(li)
                diagnostics["seed"] = self.seed
                self.layer_diagnostics[li].append(diagnostics)
                return new_output

            self.handles.append(layer.mlp.register_forward_hook(mlp_hook))

    def summary(self) -> dict[str, Any]:
        rows = [row for layer_rows in self.layer_diagnostics.values() for row in layer_rows]
        return {
            "late_layers": [int(layer) for layer in self.late_layers],
            "n_layer_calls": int(len(rows)),
            "mean_opp_norm_frac": finite_mean([row.get("opp_norm_frac") for row in rows]),
            "mean_update_norm_ratio_after_hook": finite_mean(
                [row.get("update_norm_ratio_after_hook") for row in rows]
            ),
            "mean_postres_norm_ratio_after_hook": finite_mean(
                [row.get("postres_norm_ratio_after_hook") for row in rows]
            ),
            "by_layer": {
                str(layer): {
                    "n": len(layer_rows),
                    "mean_opp_norm_frac": finite_mean(
                        [row.get("opp_norm_frac") for row in layer_rows]
                    ),
                    "mean_update_norm_ratio_after_hook": finite_mean(
                        [row.get("update_norm_ratio_after_hook") for row in layer_rows]
                    ),
                    "mean_postres_norm_ratio_after_hook": finite_mean(
                        [row.get("postres_norm_ratio_after_hook") for row in layer_rows]
                    ),
                }
                for layer, layer_rows in sorted(self.layer_diagnostics.items())
            },
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

