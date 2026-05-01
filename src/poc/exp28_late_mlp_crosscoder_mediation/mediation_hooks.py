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
    mode: str = "ablate"  # ablate|full_reconstruct|coverage_ablate
    use_threshold: bool = True
    coverage_fraction: float | None = None
    coverage_metric: str = "norm"


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
        coverage_margin_vector: torch.Tensor | None = None,
    ) -> None:
        self.model = model
        self.steering_adapter = steering_adapter
        self.run_root = run_root
        self.selection = selection
        self.device = device
        self.crosscoder_dtype = crosscoder_dtype
        self.coverage_margin_vector = coverage_margin_vector
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
        if self.selection.mode in {"full_reconstruct", "coverage_ablate"}:
            target_layers = self._available_dictionary_layers()
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
                    diag_extra = {}
                elif self.selection.mode == "coverage_ablate":
                    contrib, features, diag_extra = coverage_branch_contribution(
                        cc,
                        update,
                        branch=1,
                        coverage_fraction=float(self.selection.coverage_fraction or 0.0),
                        coverage_metric=self.selection.coverage_metric,
                        margin_vector=self.coverage_margin_vector,
                        use_threshold=self.selection.use_threshold,
                    )
                    delta = -contrib.float()
                    new_update = (update.float() + delta).to(dtype=output.dtype)
                    n_selected = int(diag_extra.get("n_selected", 0))
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
                    diag_extra = {}
                out = output.clone()
                out[:, -1, :] = new_update
                base_norm = update.float().norm(dim=-1).mean().clamp_min(1e-8)
                row = {
                    "layer": int(li),
                    "mode": self.selection.mode,
                    "coverage_metric": self.selection.coverage_metric,
                    "coverage_fraction": float(self.selection.coverage_fraction or 0.0),
                    "n_selected": n_selected,
                    "feature_l0": float((features > 0).float().sum(dim=-1).mean().item()),
                    "delta_norm": float(delta.norm(dim=-1).mean().item()),
                    "delta_norm_frac": float((delta.norm(dim=-1).mean() / base_norm).item()),
                }
                row.update(diag_extra)
                self.layer_diagnostics[li].append(row)
                return out

            self.handles.append(self.layers[layer_idx].mlp.register_forward_hook(hook))

    def _available_dictionary_layers(self) -> list[int]:
        """Return layers with trained crosscoder weights available.

        Exp28 coverage extensions are often run from compact result roots after
        activation caches have been uploaded to GCS and removed locally. The
        hooks only need dictionary weights at mediation time, so discover layers
        from `dictionaries/layer_*/crosscoder.pt` first and fall back to cache
        files for older in-progress run roots.
        """

        dict_layers = sorted(
            int(path.parent.name.split("_")[1])
            for path in (self.run_root / "dictionaries").glob("layer_*/crosscoder.pt")
        )
        if dict_layers:
            return dict_layers
        return sorted(
            int(path.stem.split("_")[1])
            for path in (self.run_root / "cache").glob("layer_*.pt")
        )

    def summary(self) -> dict[str, Any]:
        rows = [row for layer_rows in self.layer_diagnostics.values() for row in layer_rows]
        return {
            "selection": self.selection.name,
            "mode": self.selection.mode,
            "n_layer_calls": len(rows),
            "mean_delta_norm_frac": _mean([row.get("delta_norm_frac") for row in rows]),
            "mean_feature_l0": _mean([row.get("feature_l0") for row in rows]),
            "mean_active_l0": _mean([row.get("active_l0") for row in rows]),
            "mean_removed_mass_frac": _mean([row.get("removed_mass_frac") for row in rows]),
            "by_layer": {
                str(layer): {
                    "n": len(layer_rows),
                    "mean_delta_norm_frac": _mean([row.get("delta_norm_frac") for row in layer_rows]),
                    "mean_feature_l0": _mean([row.get("feature_l0") for row in layer_rows]),
                    "mean_active_l0": _mean([row.get("active_l0") for row in layer_rows]),
                    "mean_removed_mass_frac": _mean([row.get("removed_mass_frac") for row in layer_rows]),
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


@torch.no_grad()
def coverage_branch_contribution(
    crosscoder: BatchTopKCrossCoder,
    x: torch.Tensor,
    *,
    branch: int,
    coverage_fraction: float,
    coverage_metric: str,
    margin_vector: torch.Tensor | None = None,
    use_threshold: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    """Return the active-feature contribution needed to cover a mass fraction.

    The current Exp28 top-k mediation selects global features, but many are
    inactive on any given first-divergence prompt. Coverage ablation instead
    ranks only the active features in the current MLP output and removes enough
    of their contribution mass to cover a requested fraction.
    """

    if not 0.0 <= coverage_fraction <= 1.0:
        raise ValueError(f"coverage_fraction must be in [0,1], got {coverage_fraction}")
    if coverage_metric not in {"norm", "activation", "margin_pos", "margin_abs"}:
        raise ValueError(f"Unsupported coverage_metric={coverage_metric}")

    features = crosscoder.encode_branch(x, branch=branch, use_threshold=use_threshold)
    decoder = crosscoder.decoder[:, branch, :].to(device=x.device, dtype=torch.float32)
    out = torch.zeros_like(x.float())
    selected_counts: list[int] = []
    active_counts: list[int] = []
    removed_fracs: list[float] = []

    margin = None
    if coverage_metric in {"margin_pos", "margin_abs"}:
        if margin_vector is None:
            raise ValueError(f"coverage_metric={coverage_metric} requires margin_vector")
        margin = margin_vector.to(device=x.device, dtype=torch.float32)

    for batch_idx in range(features.shape[0]):
        active = torch.nonzero(features[batch_idx] > 0, as_tuple=False).flatten()
        active_counts.append(int(active.numel()))
        if active.numel() == 0 or coverage_fraction <= 0.0:
            selected_counts.append(0)
            removed_fracs.append(0.0)
            continue
        vals = features[batch_idx, active].float()
        dec = decoder[active]
        if coverage_metric == "norm":
            masses = vals * dec.norm(dim=-1)
        elif coverage_metric == "activation":
            masses = vals
        else:
            raw = vals * (dec @ margin)
            masses = raw.clamp_min(0.0) if coverage_metric == "margin_pos" else raw.abs()
        total = masses.sum()
        if not torch.isfinite(total) or float(total.item()) <= 1e-12:
            selected_counts.append(0)
            removed_fracs.append(0.0)
            continue
        order = torch.argsort(masses, descending=True)
        cumulative = torch.cumsum(masses[order], dim=0)
        threshold = coverage_fraction * total
        n_keep = int(torch.searchsorted(cumulative, threshold).item()) + 1
        n_keep = min(n_keep, int(active.numel()))
        chosen = active[order[:n_keep]]
        selected_counts.append(n_keep)
        removed_fracs.append(float((masses[order[:n_keep]].sum() / total).item()))
        out[batch_idx] = features[batch_idx, chosen].float() @ decoder[chosen]

    diag = {
        "coverage_metric": coverage_metric,
        "coverage_fraction": float(coverage_fraction),
        "n_selected": int(max(selected_counts) if selected_counts else 0),
        "active_l0": float(sum(active_counts) / max(len(active_counts), 1)),
        "selected_l0": float(sum(selected_counts) / max(len(selected_counts), 1)),
        "removed_mass_frac": float(sum(removed_fracs) / max(len(removed_fracs), 1)),
    }
    return out, features, diag
