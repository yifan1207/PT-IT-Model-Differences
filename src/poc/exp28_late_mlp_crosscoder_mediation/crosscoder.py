"""BatchTopK PT/IT crosscoder used by Exp28.

The implementation is intentionally local and small: Exp28 needs audited
checkpoints that can be loaded by the mediation hooks without depending on an
external training package being present on every remote machine.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class CrosscoderConfig:
    activation_dim: int
    dict_size: int
    k: int
    n_branches: int = 2
    threshold_beta: float = 0.999
    threshold_start_step: int = 1000
    eps: float = 1e-8


def batch_topk_mask(values: torch.Tensor, k: int) -> torch.Tensor:
    """Return a mask that keeps approximately ``k`` latents per batch item."""

    if values.ndim != 2:
        raise ValueError(f"BatchTopK expects [batch, latents], got {tuple(values.shape)}")
    if k <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    budget = min(values.numel(), int(values.shape[0]) * int(k))
    if budget <= 0:
        return torch.zeros_like(values, dtype=torch.bool)
    flat = values.flatten()
    cutoff = torch.topk(flat, k=budget, largest=True).values[-1]
    return values >= cutoff


class BatchTopKCrossCoder(nn.Module):
    """Sparse shared-latent dictionary with PT/IT-specific encoder/decoder arms."""

    def __init__(
        self,
        config: CrosscoderConfig,
        *,
        input_mean: torch.Tensor | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        c = config
        init_std = 1.0 / math.sqrt(c.activation_dim)
        decoder = torch.randn(
            c.dict_size,
            c.n_branches,
            c.activation_dim,
            device=device,
            dtype=torch.float32,
        ) * init_std
        self.decoder = nn.Parameter(decoder)
        self.normalize_decoder_()

        # Initialize encoder close to the transpose of the normalized decoder.
        encoder = self.decoder.detach().permute(1, 2, 0).contiguous().clone()
        self.encoder = nn.Parameter(encoder)
        self.encoder_bias = nn.Parameter(torch.zeros(c.dict_size, device=device, dtype=torch.float32))
        if input_mean is None:
            input_mean = torch.zeros(
                c.n_branches,
                c.activation_dim,
                device=device,
                dtype=torch.float32,
            )
        else:
            input_mean = input_mean.detach().clone().float()
            if device is not None:
                input_mean = input_mean.to(device)
        self.register_buffer("input_mean", input_mean)
        self.decoder_bias = nn.Parameter(self.input_mean.detach().clone())
        self.register_buffer("inference_threshold", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("threshold_updates", torch.tensor(0, dtype=torch.long))

    def centered(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() - self.input_mean.to(device=x.device, dtype=torch.float32)

    def preacts(self, x: torch.Tensor) -> torch.Tensor:
        centered = self.centered(x)
        return torch.einsum("bmd,mdl->bl", centered, self.encoder) + self.encoder_bias

    def encode(self, x: torch.Tensor, *, batch_topk: bool = True) -> torch.Tensor:
        acts = F.relu(self.preacts(x))
        if not batch_topk:
            return acts
        return acts * batch_topk_mask(acts, self.config.k).to(dtype=acts.dtype)

    def encode_branch(
        self,
        x: torch.Tensor,
        *,
        branch: int,
        use_threshold: bool = True,
        topk: int | None = None,
    ) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Branch encoding expects [batch, d_model], got {tuple(x.shape)}")
        mean = self.input_mean[branch].to(device=x.device, dtype=torch.float32)
        pre = (x.float() - mean) @ self.encoder[branch] + self.encoder_bias
        acts = F.relu(pre)
        if use_threshold and float(self.inference_threshold.item()) > 0:
            return acts * (acts >= self.inference_threshold.to(device=acts.device)).to(acts.dtype)
        per_item_k = int(topk or self.config.k)
        if per_item_k <= 0:
            return torch.zeros_like(acts)
        keep = min(per_item_k, acts.shape[-1])
        cutoff = torch.topk(acts, k=keep, dim=-1).values[:, -1:]
        return acts * (acts >= cutoff).to(acts.dtype)

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        recon = torch.einsum("bl,lmd->bmd", features.float(), self.decoder)
        return recon + self.decoder_bias

    def decode_branch(self, features: torch.Tensor, *, branch: int) -> torch.Tensor:
        recon = features.float() @ self.decoder[:, branch, :]
        return recon + self.decoder_bias[branch]

    def selected_branch_contribution(
        self,
        x: torch.Tensor,
        *,
        branch: int,
        latent_ids: torch.Tensor,
        use_threshold: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_ids.numel() == 0:
            empty = torch.zeros((x.shape[0], 0), device=x.device, dtype=torch.float32)
            return torch.zeros_like(x.float()), empty
        selected = latent_ids.to(device=x.device, dtype=torch.long)
        if use_threshold and float(self.inference_threshold.item()) > 0:
            mean = self.input_mean[branch].to(device=x.device, dtype=torch.float32)
            encoder = self.encoder[branch, :, selected]
            bias = self.encoder_bias[selected]
            selected_features = F.relu((x.float() - mean) @ encoder + bias)
            threshold = self.inference_threshold.to(device=x.device, dtype=selected_features.dtype)
            selected_features = selected_features * (selected_features >= threshold).to(selected_features.dtype)
        else:
            features = self.encode_branch(x, branch=branch, use_threshold=use_threshold)
            selected_features = features[:, selected]
        contrib = selected_features @ self.decoder[selected, branch, :]
        return contrib, selected_features

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_features: bool = False,
        return_preacts: bool = False,
    ):
        pre = self.preacts(x)
        acts = F.relu(pre)
        features = acts * batch_topk_mask(acts, self.config.k).to(dtype=acts.dtype)
        recon = self.decode(features)
        outs: list[torch.Tensor] = [recon]
        if output_features:
            outs.append(features)
        if return_preacts:
            outs.append(pre)
        return tuple(outs) if len(outs) > 1 else recon

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        flat = self.decoder.data.flatten(start_dim=1)
        norms = flat.norm(dim=1, keepdim=True).clamp_min(self.config.eps)
        self.decoder.data.div_(norms.view(-1, 1, 1))

    @torch.no_grad()
    def update_threshold(self, preacts: torch.Tensor, *, step: int) -> None:
        if step < self.config.threshold_start_step:
            return
        acts = F.relu(preacts.detach())
        budget = min(acts.numel(), int(acts.shape[0]) * int(self.config.k))
        if budget <= 0:
            return
        batch_threshold = torch.topk(acts.flatten(), k=budget, largest=True).values[-1].float()
        if int(self.threshold_updates.item()) == 0:
            self.inference_threshold.copy_(batch_threshold.cpu())
        else:
            beta = float(self.config.threshold_beta)
            self.inference_threshold.mul_(beta).add_(batch_threshold.cpu() * (1.0 - beta))
        self.threshold_updates.add_(1)

    def save(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "extra": extra or {},
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, *, device: str | torch.device = "cpu") -> "BatchTopKCrossCoder":
        payload = torch.load(path, map_location=device, weights_only=False)
        config = CrosscoderConfig(**payload["config"])
        model = cls(config, device=device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def fvu(recon: torch.Tensor, target: torch.Tensor, *, eps: float = 1e-8) -> torch.Tensor:
    """Fraction of variance unexplained, averaged over branches."""

    err = (recon.float() - target.float()).pow(2).mean(dim=(0, 2))
    centered = target.float() - target.float().mean(dim=0, keepdim=True)
    var = centered.pow(2).mean(dim=(0, 2)).clamp_min(eps)
    return err / var
