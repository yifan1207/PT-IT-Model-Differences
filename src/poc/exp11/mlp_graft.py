from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass
class LayerStepMetrics:
    top1_token: list[int]
    top20_ids: list[list[int]]
    entropy: list[float]
    structural_mass_logit_tier1: list[float]
    structural_mass_prob_tier1: list[float]
    structural_mass_logit_tier12: list[float]
    structural_mass_prob_tier12: list[float]
    kl_to_own_final: list[float]
    top1_match_own_final: list[bool]
    next_token_prob: list[float]
    next_token_rank: list[int]
    delta_cosine: list[float]
    mlp_norm: list[float]
    diff_norm: list[float] | None = None
    pt_mlp_norm: list[float] | None = None
    relative_diff: list[float] | None = None
    cross_kl: list[float] | None = None
    residual_cosine: list[float] | None = None
    residual_divergence: list[float] | None = None


@dataclass
class PipelineStepRecord:
    token_id: int
    token_str: str
    metrics: LayerStepMetrics


@dataclass
class PipelineRun:
    generated_token_ids: list[int]
    generated_tokens: list[dict[str, Any]]
    generated_text: str
    step_records: list[PipelineStepRecord]
    baseline_cache: list[tuple[torch.Tensor, list[torch.Tensor]]]


@dataclass
class StepTensors:
    pre_mlp_residual: list[torch.Tensor]
    normed_mlp_input: list[torch.Tensor]
    mlp_output: list[torch.Tensor]
    residual_output: list[torch.Tensor]
    diff_norm: list[float | None]
    pt_mlp_norm: list[float | None]
    relative_diff: list[float | None]


class ArchitectureProbe:
    """Architecture-aware probe for locating the MLP-preceding norm module."""

    def get_mlp_input_norm(self, layer) -> torch.nn.Module:
        for attr in ("post_attention_layernorm", "pre_feedforward_layernorm", "ff_norm"):
            if hasattr(layer, attr):
                return getattr(layer, attr)
        raise ValueError(
            f"Unsupported layer structure for exp11: {layer.__class__.__name__} "
            "does not expose a known MLP-input norm module."
        )


class PipelineCapture:
    """Persistent forward-hook bundle for one model pipeline."""

    def __init__(
        self,
        *,
        model_raw,
        adapter,
        arch_probe: ArchitectureProbe,
        onset_layer: int,
        graft_it_model_raw=None,
    ) -> None:
        self.model_raw = model_raw
        self.adapter = adapter
        self.arch_probe = arch_probe
        self.onset_layer = onset_layer
        self.graft_it_model_raw = graft_it_model_raw
        self.layers = adapter.get_layers(model_raw)
        self.graft_it_layers = (
            adapter.get_layers(graft_it_model_raw) if graft_it_model_raw is not None else None
        )
        self._current = self._empty_store()
        self._handles = self._register()

    def _empty_store(self) -> dict[str, list[torch.Tensor | None]]:
        n_layers = len(self.layers)
        return {
            "pre_mlp_residual": [None] * n_layers,
            "normed_mlp_input": [None] * n_layers,
            "mlp_output": [None] * n_layers,
            "residual_output": [None] * n_layers,
            "diff_norm": [None] * n_layers,
            "pt_mlp_norm": [None] * n_layers,
            "relative_diff": [None] * n_layers,
        }

    def reset_step(self) -> None:
        self._current = self._empty_store()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def snapshot(self) -> StepTensors:
        def _require(key: str) -> list[torch.Tensor]:
            vals = self._current[key]
            if any(v is None for v in vals):
                missing = [i for i, v in enumerate(vals) if v is None]
                raise RuntimeError(f"Missing exp11 capture for {key} at layers {missing}")
            return [v for v in vals if v is not None]

        step = StepTensors(
            pre_mlp_residual=_require("pre_mlp_residual"),
            normed_mlp_input=_require("normed_mlp_input"),
            mlp_output=_require("mlp_output"),
            residual_output=_require("residual_output"),
            diff_norm=list(self._current["diff_norm"]),
            pt_mlp_norm=list(self._current["pt_mlp_norm"]),
            relative_diff=list(self._current["relative_diff"]),
        )
        return step

    def _register(self) -> list:
        handles: list = []
        for layer_idx, layer in enumerate(self.layers):
            norm_module = self.arch_probe.get_mlp_input_norm(layer)
            mlp_module = layer.mlp

            def norm_pre_hook(_module, args, li=layer_idx):
                tensor = args[0]
                self._current["pre_mlp_residual"][li] = tensor[:, -1, :].detach()

            def norm_hook(_module, _args, output, li=layer_idx):
                self._current["normed_mlp_input"][li] = output[:, -1, :].detach()

            def mlp_hook(_module, args, output, li=layer_idx):
                pt_out = output
                self._current["pt_mlp_norm"][li] = float(
                    pt_out[:, -1, :].float().norm(dim=-1).mean().item()
                )
                if self.graft_it_layers is not None and li >= self.onset_layer:
                    with torch.no_grad():
                        it_out = self.graft_it_layers[li].mlp(args[0])
                    diff = it_out[:, -1, :].float() - pt_out[:, -1, :].float()
                    diff_norm = float(diff.norm(dim=-1).mean().item())
                    pt_norm = max(self._current["pt_mlp_norm"][li], 1e-12)
                    self._current["diff_norm"][li] = diff_norm
                    self._current["relative_diff"][li] = float(diff_norm / pt_norm)
                    self._current["mlp_output"][li] = it_out[:, -1, :].detach()
                    return it_out

                self._current["diff_norm"][li] = 0.0
                self._current["relative_diff"][li] = 0.0
                self._current["mlp_output"][li] = pt_out[:, -1, :].detach()
                return output

            def layer_hook(_module, _args, output, li=layer_idx):
                resid = self.adapter.adapter.residual_from_output(output)
                self._current["residual_output"][li] = resid[:, -1, :].detach()

            handles.append(norm_module.register_forward_pre_hook(norm_pre_hook))
            handles.append(norm_module.register_forward_hook(norm_hook))
            handles.append(mlp_module.register_forward_hook(mlp_hook))
            handles.append(layer.register_forward_hook(layer_hook))
        return handles


def logits_from_residuals(final_norm, lm_head, residuals: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack([x.squeeze(0) for x in residuals], dim=0)
    normed = final_norm(stacked)
    return lm_head(normed).float()


def _entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)


def _kl_to_final(log_probs: torch.Tensor) -> torch.Tensor:
    log_p_final = log_probs[-1].unsqueeze(0).expand_as(log_probs)
    probs = log_probs.exp()
    return (probs * (log_probs - log_p_final)).sum(dim=-1)


def _rank_of_token(logits: torch.Tensor, token_id: int) -> int:
    target = logits[token_id]
    return int((logits > target).sum().item()) + 1


def compute_layer_metrics(
    *,
    pipeline_logits: torch.Tensor,
    step_tensors: StepTensors,
    chosen_token_id: int,
    tier1_mask: torch.Tensor,
    tier12_mask: torch.Tensor,
    baseline_logits: torch.Tensor | None = None,
    baseline_residuals: list[torch.Tensor] | None = None,
) -> LayerStepMetrics:
    probs = torch.softmax(pipeline_logits, dim=-1)
    log_probs = torch.log_softmax(pipeline_logits, dim=-1)
    kl_to_own_final = _kl_to_final(log_probs)
    top20 = torch.topk(pipeline_logits, k=min(20, pipeline_logits.shape[-1]), dim=-1).indices
    top1 = pipeline_logits.argmax(dim=-1)
    next_probs = probs[:, chosen_token_id]
    ent = _entropy_from_probs(probs)

    delta_cos = []
    mlp_norm = []
    for mlp_out, resid in zip(step_tensors.mlp_output, step_tensors.pre_mlp_residual, strict=False):
        mlp_vec = mlp_out.float()
        resid_vec = resid.float()
        delta_cos.append(float(F.cosine_similarity(mlp_vec, resid_vec, dim=-1).mean().item()))
        mlp_norm.append(float(mlp_vec.norm(dim=-1).mean().item()))

    metrics = LayerStepMetrics(
        top1_token=[int(x) for x in top1.tolist()],
        top20_ids=[[int(y) for y in row] for row in top20.tolist()],
        entropy=[float(x) for x in ent.tolist()],
        structural_mass_logit_tier1=[float(x) for x in pipeline_logits[:, tier1_mask].sum(dim=-1).tolist()],
        structural_mass_prob_tier1=[float(x) for x in probs[:, tier1_mask].sum(dim=-1).tolist()],
        structural_mass_logit_tier12=[float(x) for x in pipeline_logits[:, tier12_mask].sum(dim=-1).tolist()],
        structural_mass_prob_tier12=[float(x) for x in probs[:, tier12_mask].sum(dim=-1).tolist()],
        kl_to_own_final=[float(x) for x in kl_to_own_final.tolist()],
        top1_match_own_final=[bool(x) for x in (top1 == top1[-1]).tolist()],
        next_token_prob=[float(x) for x in next_probs.tolist()],
        next_token_rank=[_rank_of_token(pipeline_logits[i], chosen_token_id) for i in range(pipeline_logits.shape[0])],
        delta_cosine=delta_cos,
        mlp_norm=mlp_norm,
        diff_norm=[float(x) if x is not None else 0.0 for x in step_tensors.diff_norm],
        pt_mlp_norm=[float(x) if x is not None else 0.0 for x in step_tensors.pt_mlp_norm],
        relative_diff=[float(x) if x is not None else 0.0 for x in step_tensors.relative_diff],
    )

    if baseline_logits is not None and baseline_residuals is not None:
        base_log_probs = torch.log_softmax(baseline_logits, dim=-1)
        cross_kl = (log_probs.exp() * (log_probs - base_log_probs)).sum(dim=-1)
        residual_cos = []
        residual_div = []
        for grafted, base in zip(step_tensors.residual_output, baseline_residuals, strict=False):
            g = grafted.float()
            b = base.float()
            residual_cos.append(float(F.cosine_similarity(g, b, dim=-1).mean().item()))
            residual_div.append(float(((g - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-12)).mean().item()))
        metrics.cross_kl = [float(x) for x in cross_kl.tolist()]
        metrics.residual_cosine = residual_cos
        metrics.residual_divergence = residual_div

    return metrics


def first_stable_below_threshold(values: list[float], threshold: float) -> int | None:
    for i in range(len(values)):
        if values[i] < threshold and all(v < threshold for v in values[i:]):
            return i
    return None


def first_stable_true(values: list[bool]) -> int | None:
    for i in range(len(values)):
        if values[i] and all(values[i:]):
            return i
    return None
