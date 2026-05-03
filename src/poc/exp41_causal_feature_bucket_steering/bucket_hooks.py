"""Terminal MLP bucket-edit hooks for Exp41."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F

from src.poc.exp28_late_mlp_crosscoder_mediation.crosscoder import BatchTopKCrossCoder

ActivationMode = Literal["mediation_topk", "raw_relu", "threshold"]
ControlMode = Literal["feature", "same_delta_random"]


@dataclass
class LayerEditDiagnostics:
    layer: int
    n_latents: int
    activation_l0: float
    contribution_norm: float
    delta_norm: float
    mlp_norm: float
    delta_norm_frac: float


def stable_seed(*parts: Any) -> int:
    text = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) % (2**31 - 1)


def _load_crosscoder(path: Path, *, device: torch.device, dtype: torch.dtype | None) -> BatchTopKCrossCoder:
    crosscoder = BatchTopKCrossCoder.load(path, device=device)
    if dtype is not None:
        crosscoder = crosscoder.to(dtype=dtype)
    crosscoder.eval()
    return crosscoder


def selected_branch_contribution(
    crosscoder: BatchTopKCrossCoder,
    x: torch.Tensor,
    *,
    branch: int,
    latent_ids: torch.Tensor,
    activation_mode: ActivationMode,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected branch decoder contribution for one activation batch."""

    if latent_ids.numel() == 0:
        empty = torch.zeros((x.shape[0], 0), device=x.device, dtype=torch.float32)
        return torch.zeros_like(x.float()), empty

    selected = latent_ids.to(device=x.device, dtype=torch.long)
    if activation_mode == "mediation_topk":
        contrib, features = crosscoder.selected_branch_contribution(
            x,
            branch=branch,
            latent_ids=selected,
            use_threshold=False,
        )
        return contrib.float(), features.float()

    dtype = crosscoder.encoder.dtype
    mean = crosscoder.input_mean[branch].to(device=x.device, dtype=dtype)
    encoder = crosscoder.encoder[branch, :, selected].to(device=x.device, dtype=dtype)
    bias = crosscoder.encoder_bias[selected].to(device=x.device, dtype=dtype)
    features = F.relu((x.to(dtype=dtype) - mean) @ encoder + bias)
    if activation_mode == "threshold" and float(crosscoder.inference_threshold.item()) > 0:
        if crosscoder.config.scale_topk_by_decoder_norm:
            weights = crosscoder._score_weights(branch=branch).to(device=x.device, dtype=features.dtype)[selected]
            scores = features * weights
        else:
            scores = features
        threshold = crosscoder.inference_threshold.to(device=x.device, dtype=scores.dtype)
        features = features * (scores >= threshold).to(features.dtype)
    decoder = crosscoder.decoder[selected, branch, :].to(device=x.device, dtype=features.dtype)
    return (features @ decoder).float(), features.float()


def _orthogonal_random_unit(
    *,
    decoder_vectors: torch.Tensor,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    v = torch.randn(decoder_vectors.shape[-1], generator=generator, dtype=torch.float32).to(device)
    basis = decoder_vectors.detach().float().to(device)
    if basis.numel() > 0:
        q, _ = torch.linalg.qr(basis.T, mode="reduced")
        v = v - q @ (q.T @ v)
    norm = v.norm().clamp_min(1e-12)
    return (v / norm).to(dtype=dtype)


class BucketMlpEditor:
    """Context-managed terminal MLP editor for one model and bucket."""

    def __init__(
        self,
        *,
        model: Any,
        layers: list[Any],
        layer_latents: dict[int, list[int]],
        layer_roots: dict[int, Path],
        alpha: float,
        device: torch.device,
        activation_mode: ActivationMode = "mediation_topk",
        control_mode: ControlMode = "feature",
        seed: int = 0,
        crosscoder_dtype: torch.dtype | None = torch.bfloat16,
        target_positions: torch.Tensor | None = None,
        crosscoder_cache: dict[tuple[str, int], BatchTopKCrossCoder] | None = None,
    ) -> None:
        self.model = model
        self.layers = layers
        self.layer_latents = {int(k): [int(x) for x in v] for k, v in layer_latents.items() if v}
        self.layer_roots = {int(k): Path(v) for k, v in layer_roots.items()}
        self.alpha = float(alpha)
        self.device = device
        self.activation_mode = activation_mode
        self.control_mode = control_mode
        self.seed = int(seed)
        self.crosscoder_dtype = crosscoder_dtype
        self.target_positions = target_positions
        self.crosscoder_cache = crosscoder_cache if crosscoder_cache is not None else {}
        self.crosscoders: dict[int, BatchTopKCrossCoder] = {}
        self.latent_tensors: dict[int, torch.Tensor] = {}
        self.random_units: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.diagnostics: dict[int, list[LayerEditDiagnostics]] = defaultdict(list)

    def __enter__(self) -> "BucketMlpEditor":
        self._register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def summary(self) -> dict[str, float | int]:
        rows = [row for layer_rows in self.diagnostics.values() for row in layer_rows]
        if not rows:
            return {
                "n_hook_calls": 0,
                "mean_activation_l0": 0.0,
                "mean_contribution_norm": 0.0,
                "mean_delta_norm": 0.0,
                "mean_delta_norm_frac": 0.0,
                "max_delta_norm_frac": 0.0,
            }

        def mean(values: list[float]) -> float:
            return float(sum(values) / max(len(values), 1))

        return {
            "n_hook_calls": len(rows),
            "mean_activation_l0": mean([row.activation_l0 for row in rows]),
            "mean_contribution_norm": mean([row.contribution_norm for row in rows]),
            "mean_delta_norm": mean([row.delta_norm for row in rows]),
            "mean_delta_norm_frac": mean([row.delta_norm_frac for row in rows]),
            "max_delta_norm_frac": max(row.delta_norm_frac for row in rows),
        }

    def _load_layer(self, layer_idx: int) -> BatchTopKCrossCoder:
        if layer_idx not in self.crosscoders:
            root = self.layer_roots[layer_idx]
            cache_key = (str(root), int(layer_idx))
            if cache_key in self.crosscoder_cache:
                self.crosscoders[layer_idx] = self.crosscoder_cache[cache_key]
                return self.crosscoders[layer_idx]
            path = root / "dictionaries" / f"layer_{layer_idx}" / "crosscoder.pt"
            self.crosscoders[layer_idx] = _load_crosscoder(
                path,
                device=self.device,
                dtype=self.crosscoder_dtype,
            )
            self.crosscoder_cache[cache_key] = self.crosscoders[layer_idx]
        return self.crosscoders[layer_idx]

    def _random_unit(self, layer_idx: int, crosscoder: BatchTopKCrossCoder) -> torch.Tensor:
        if layer_idx not in self.random_units:
            latents = self.latent_tensors[layer_idx]
            decoder_vectors = crosscoder.decoder[latents, 1, :].detach()
            self.random_units[layer_idx] = _orthogonal_random_unit(
                decoder_vectors=decoder_vectors,
                seed=stable_seed("exp41", self.seed, layer_idx, tuple(int(x) for x in latents.detach().cpu().tolist())),
                device=self.device,
                dtype=decoder_vectors.dtype,
            )
        return self.random_units[layer_idx]

    def _register(self) -> None:
        for layer_idx in sorted(self.layer_latents):
            crosscoder = self._load_layer(layer_idx)
            latents = torch.tensor(self.layer_latents[layer_idx], dtype=torch.long, device=self.device)
            self.latent_tensors[layer_idx] = latents
            mlp = self.layers[layer_idx].mlp

            def hook(_module, _args, output, li=layer_idx, cc=crosscoder):
                if not torch.is_tensor(output):
                    raise RuntimeError(f"Expected tensor MLP output at layer {li}, got {type(output)}")
                if self.target_positions is None:
                    batch_idx = torch.arange(output.shape[0], device=output.device)
                    pos_idx = torch.full((output.shape[0],), output.shape[1] - 1, device=output.device, dtype=torch.long)
                else:
                    batch_idx = torch.arange(output.shape[0], device=output.device)
                    pos_idx = self.target_positions.to(device=output.device, dtype=torch.long)
                update = output[batch_idx, pos_idx, :]
                contrib, features = selected_branch_contribution(
                    cc,
                    update,
                    branch=1,
                    latent_ids=self.latent_tensors[li],
                    activation_mode=self.activation_mode,
                )
                if self.control_mode == "same_delta_random":
                    unit = self._random_unit(li, cc).to(device=update.device, dtype=contrib.dtype)
                    contrib = contrib.norm(dim=-1, keepdim=True) * unit[None, :]
                delta = -self.alpha * contrib.float()
                new_update = (update.float() + delta).to(dtype=output.dtype)
                out = output.clone()
                out[batch_idx, pos_idx, :] = new_update

                mlp_norm = update.float().norm(dim=-1).mean().clamp_min(1e-12)
                delta_norm = delta.norm(dim=-1).mean()
                self.diagnostics[li].append(
                    LayerEditDiagnostics(
                        layer=int(li),
                        n_latents=int(self.latent_tensors[li].numel()),
                        activation_l0=float((features > 0).float().sum(dim=-1).mean().item()),
                        contribution_norm=float(contrib.float().norm(dim=-1).mean().item()),
                        delta_norm=float(delta_norm.item()),
                        mlp_norm=float(mlp_norm.item()),
                        delta_norm_frac=float((delta_norm / mlp_norm).item()),
                    )
                )
                return out

            self.handles.append(mlp.register_forward_hook(hook))
