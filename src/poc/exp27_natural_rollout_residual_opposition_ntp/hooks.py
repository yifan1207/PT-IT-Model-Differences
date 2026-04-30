"""Generated-span late-MLP hooks for Exp27."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch

from src.poc.exp11_matched_prefix_mlp_graft.mlp_graft import ArchitectureProbe
from src.poc.exp26_residual_opposition_mediation.variants import (
    decompose_negative_parallel,
    finite_mean,
    stable_seed,
)
from src.poc.exp27_natural_rollout_residual_opposition_ntp import RANDOM_VARIANTS
from src.poc.exp27_natural_rollout_residual_opposition_ntp.variants import apply_variant


@dataclass
class SpanMlpOppositionModifier:
    """Edit late MLP outputs only at selected source positions in a replay batch."""

    model: Any
    steering_adapter: Any
    late_layers: list[int]
    variant: str
    span_mask: torch.Tensor
    row_ids: list[str]
    batch_key: str
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.span_mask.ndim != 2:
            raise ValueError(f"span_mask must be [batch, seq], got {tuple(self.span_mask.shape)}")
        if self.span_mask.shape[0] != len(self.row_ids):
            raise ValueError("row_ids length must match span_mask batch dimension")
        self.layers = self.steering_adapter.get_layers(self.model)
        self.arch_probe = ArchitectureProbe()
        self.handles: list[Any] = []
        self._pre_mlp_residual: dict[int, torch.Tensor] = {}
        self._row_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._row_layer_calls: dict[str, int] = defaultdict(int)
        self._register()

    def _add_metric_values(
        self,
        *,
        row_indices: torch.Tensor,
        opp_norm_frac: torch.Tensor,
        update_norm_ratio: torch.Tensor,
        postres_norm_ratio: torch.Tensor,
    ) -> None:
        for row_idx in torch.unique(row_indices.detach().cpu()).tolist():
            row_id = self.row_ids[int(row_idx)]
            local = row_indices == int(row_idx)
            self._row_metrics[row_id]["opp_norm_frac"].extend(opp_norm_frac[local].detach().cpu().float().tolist())
            self._row_metrics[row_id]["update_norm_ratio_after_hook"].extend(
                update_norm_ratio[local].detach().cpu().float().tolist()
            )
            self._row_metrics[row_id]["postres_norm_ratio_after_hook"].extend(
                postres_norm_ratio[local].detach().cpu().float().tolist()
            )
            self._row_layer_calls[row_id] += 1

    def _register(self) -> None:
        for layer_idx in self.late_layers:
            layer = self.layers[layer_idx]
            capture_spec = self.arch_probe.get_capture_spec(layer)
            if capture_spec.mode == "norm":
                norm_module = capture_spec.module

                def norm_pre_hook(_module, args, li=layer_idx):
                    if not args or not torch.is_tensor(args[0]):
                        raise RuntimeError("Exp27 norm pre-hook expected hidden-state tensor")
                    self._pre_mlp_residual[li] = args[0].detach()

                self.handles.append(norm_module.register_forward_pre_hook(norm_pre_hook))

            def mlp_hook(_module, args, output, li=layer_idx, mode=capture_spec.mode):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Exp27 expected tensor MLP output at layer {li}, got {type(output)}")
                mask = self.span_mask.to(device=output.device)
                if tuple(mask.shape) != tuple(output.shape[:2]):
                    raise RuntimeError(
                        f"Exp27 mask/output shape mismatch at layer {li}: "
                        f"mask={tuple(mask.shape)} output={tuple(output.shape[:2])}"
                    )
                if not bool(mask.any().item()):
                    return output
                if mode == "mlp_input_direct":
                    if not args or not torch.is_tensor(args[0]):
                        raise RuntimeError("Exp27 MLP direct mode expected hidden-state input")
                    residual = args[0].detach()
                else:
                    residual = self._pre_mlp_residual.get(li)
                    if residual is None:
                        raise RuntimeError(f"Exp27 missing pre-MLP residual at layer {li}")
                update = output[mask]
                residual_selected = residual.to(device=output.device)[mask]
                if self.seed is not None and self.variant in RANDOM_VARIANTS:
                    # Seed random controls by semantic position, not by batch composition,
                    # so resumes or different batch sizes do not change the control vector.
                    selected_positions = mask.nonzero(as_tuple=False)
                    updates = []
                    for local_idx, (row_idx, seq_idx) in enumerate(selected_positions.tolist()):
                        rand_seed = stable_seed(
                            "exp27",
                            self.seed,
                            self.row_ids[int(row_idx)],
                            self.variant,
                            li,
                            int(seq_idx),
                        )
                        local_result = apply_variant(
                            update[local_idx : local_idx + 1],
                            residual_selected[local_idx : local_idx + 1],
                            variant=self.variant,
                            rand_seed=rand_seed,
                        )
                        updates.append(local_result.update)
                    transformed_update = torch.cat(updates, dim=0)
                else:
                    result = apply_variant(
                        update,
                        residual_selected,
                        variant=self.variant,
                        rand_seed=None,
                    )
                    transformed_update = result.update
                opposing, _, _ = decompose_negative_parallel(update.float(), residual_selected.float())
                update_norm = update.float().norm(dim=-1).clamp_min(1e-12)
                opp_norm_frac = opposing.norm(dim=-1) / update_norm
                update_norm_ratio = transformed_update.float().norm(dim=-1) / update_norm
                postres_norm_ratio = (residual_selected.float() + transformed_update.float()).norm(dim=-1) / (
                    residual_selected.float() + update.float()
                ).norm(dim=-1).clamp_min(1e-12)
                row_indices = mask.nonzero(as_tuple=False)[:, 0]
                self._add_metric_values(
                    row_indices=row_indices,
                    opp_norm_frac=opp_norm_frac,
                    update_norm_ratio=update_norm_ratio,
                    postres_norm_ratio=postres_norm_ratio,
                )
                new_output = output.clone()
                new_output[mask] = transformed_update
                return new_output

            self.handles.append(layer.mlp.register_forward_hook(mlp_hook))

    def summary_by_row(self) -> dict[str, dict[str, float | int | None]]:
        out: dict[str, dict[str, float | int | None]] = {}
        for row_id in self.row_ids:
            metrics = self._row_metrics.get(row_id, {})
            out[row_id] = {
                "n_layer_calls": int(self._row_layer_calls.get(row_id, 0)),
                "mean_opp_norm_frac": finite_mean(metrics.get("opp_norm_frac", [])),
                "mean_update_norm_ratio_after_hook": finite_mean(
                    metrics.get("update_norm_ratio_after_hook", [])
                ),
                "mean_postres_norm_ratio_after_hook": finite_mean(
                    metrics.get("postres_norm_ratio_after_hook", [])
                ),
            }
        return out

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
