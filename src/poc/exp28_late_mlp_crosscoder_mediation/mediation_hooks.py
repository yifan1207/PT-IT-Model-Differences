"""Online late-MLP crosscoder ablation hooks for Exp28 mediation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder


@dataclass
class FeatureSelection:
    name: str
    by_layer: dict[int, list[int]]
    mode: str = "ablate"  # ablate|full_reconstruct
    use_threshold: bool = True


class CrosscoderMlpModifier:
    """Modify the last-token MLP output in selected late layers."""

    def __init__(
        self,
        *,
        model: Any,
        steering_adapter: Any,
        run_root: Path,
        selection: FeatureSelection,
        device: torch.device,
        crosscoder_cache: dict[int, BatchTopKCrossCoder] | None = None,
        crosscoder_dtype: torch.dtype | None = None,
    ) -> None:
        self.model = model
        self.steering_adapter = steering_adapter
        self.run_root = run_root
        self.selection = selection
        self.device = device
        self.crosscoder_dtype = crosscoder_dtype
        self.layers = steering_adapter.get_layers(model)
        self.crosscoders = crosscoder_cache if crosscoder_cache is not None else {}
        self.latent_tensors: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.layer_diagnostics: dict[int, list[dict[str, float | int | str]]] = defaultdict(list)
        self._register()

    def _load_layer(self, layer_idx: int) -> BatchTopKCrossCoder:
        if layer_idx not in self.crosscoders:
            path = self.run_root / "dictionaries" / f"layer_{layer_idx}" / "crosscoder.pt"
            crosscoder = BatchTopKCrossCoder.load(path, device=self.device)
            if self.crosscoder_dtype is not None:
                crosscoder = crosscoder.to(dtype=self.crosscoder_dtype)
            self.crosscoders[layer_idx] = crosscoder
        return self.crosscoders[layer_idx]

    def _register(self) -> None:
        target_layers = sorted(self.selection.by_layer)
        if self.selection.mode == "full_reconstruct":
            target_layers = sorted(
                int(path.stem.split("_")[1])
                for path in (self.run_root / "cache").glob("layer_*.pt")
            )
        for layer_idx in target_layers:
            crosscoder = self._load_layer(layer_idx)
            latent_ids = self.selection.by_layer.get(layer_idx, [])
            self.latent_tensors[layer_idx] = torch.tensor(latent_ids, dtype=torch.long, device=self.device)

            def hook(_module, _args, output, li=layer_idx, cc=crosscoder):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Exp28 expected tensor MLP output at layer {li}, got {type(output)}")
                update = output[:, -1, :]
                if self.selection.mode == "full_reconstruct":
                    features = cc.encode_branch(update, branch=1, use_threshold=self.selection.use_threshold)
                    replacement = cc.decode_branch(features, branch=1).to(dtype=output.dtype)
                    delta = replacement.float() - update.float()
                    new_update = replacement
                    n_selected = int((features > 0).sum().item())
                else:
                    latent_tensor = self.latent_tensors[li]
                    contrib, features = cc.selected_branch_contribution(
                        update,
                        branch=1,
                        latent_ids=latent_tensor,
                        use_threshold=self.selection.use_threshold,
                    )
                    delta = -contrib.float()
                    new_update = (update.float() + delta).to(dtype=output.dtype)
                    n_selected = int(latent_tensor.numel())
                out = output.clone()
                out[:, -1, :] = new_update
                base_norm = update.float().norm(dim=-1).mean().clamp_min(1e-8)
                self.layer_diagnostics[li].append(
                    {
                        "layer": int(li),
                        "mode": self.selection.mode,
                        "n_selected": n_selected,
                        "feature_l0": float((features > 0).float().sum(dim=-1).mean().item()),
                        "delta_norm": float(delta.norm(dim=-1).mean().item()),
                        "delta_norm_frac": float((delta.norm(dim=-1).mean() / base_norm).item()),
                    }
                )
                return out

            self.handles.append(self.layers[layer_idx].mlp.register_forward_hook(hook))

    def summary(self) -> dict[str, Any]:
        rows = [row for layer_rows in self.layer_diagnostics.values() for row in layer_rows]
        return {
            "selection": self.selection.name,
            "mode": self.selection.mode,
            "n_layer_calls": len(rows),
            "mean_delta_norm_frac": _mean([row.get("delta_norm_frac") for row in rows]),
            "mean_feature_l0": _mean([row.get("feature_l0") for row in rows]),
            "by_layer": {
                str(layer): {
                    "n": len(layer_rows),
                    "mean_delta_norm_frac": _mean([row.get("delta_norm_frac") for row in layer_rows]),
                    "mean_feature_l0": _mean([row.get("feature_l0") for row in layer_rows]),
                    "n_selected": max(int(row.get("n_selected", 0)) for row in layer_rows) if layer_rows else 0,
                }
                for layer, layer_rows in sorted(self.layer_diagnostics.items())
            },
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _mean(values) -> float | None:
    finite = [float(v) for v in values if v is not None and torch.isfinite(torch.tensor(float(v)))]
    if not finite:
        return None
    return float(sum(finite) / len(finite))
