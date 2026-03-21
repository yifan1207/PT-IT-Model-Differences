"""
Unified structural-semantic feature collection.

Runs generation mode (autoregressive, per-step per-layer metrics) and/or
encode mode (single-pass, last-token per-layer metrics) in one model load,
writing results incrementally to disk.

This script supersedes src/poc/exp3/run.py and src/poc/exp4/run.py.
Existing analysis scripts remain fully compatible — field names are unchanged.

Output (results/{run_name}/)
-----------------------------
  results.jsonl        one JSON record per line; streaming, resumable
  features.npz         generation-mode sparse features  {record_id: [n_steps, n_layers]}
  enc_features.npz     encode-mode sparse features      {record_id: [n_layers]}
  enc_residuals.npz    encode-mode dense residuals      {record_id: [n_layers, d_model]}
  gen_residuals.npz    generation-mode dense residuals  {record_id: [n_steps, n_layers, d_model]} float16
                         only saved when save_residual_layers is non-empty
  gen_mlp_inputs.npz   MLP pre-norm inputs at save_mlp_layers  {record_id: ...} float16
  gen_mlp_outputs.npz  MLP outputs at save_mlp_layers          {record_id: ...} float16
  run_config.json      full config snapshot

Record schema
-------------
Each JSONL line is a JSON object containing:

  -- Dataset provenance --
  record_id            str        unique id from dataset (e.g. "f_triviaqa_0001")
  prompt_id            str        alias for record_id (backward compat)
  split                str        F / R / OOD / GEN / A
  category             str        alias for split (backward compat)
  source               str        triviaqa / gsm8k / advbench / custom / …
  question             str        original question text
  question_type        str        factual / numerical / harmful / …
  domain               str        subject domain
  alignment_subcategory str|null  5a–5e for A-split records
  expected_behavior    str|null   refuse / comply / comply_safely
  ooc_type             str|null   novel_entity / post_training / counterfactual

  -- Collection metadata --
  prompt               str        exact prompt fed to the model
  model_id             str        HF model id
  variant              str        pt | it
  chat_template        bool       whether chat template was applied
  run_name             str        identifies this collection run
  force_decoded        bool       True when forced-token mode was used

  -- Generation metrics (present when mode includes 'generate') --
  generated_tokens     [{token_id, token_str}]  per step
  generated_text       str
  token_type           [str]      CONTENT/FUNCTION/DISCOURSE/STRUCTURAL/PUNCTUATION/OTHER
  final_layer_prob     [float]    probability of generated token at the final layer (layer 33)
  residual_norm        [[float]]  [n_steps][n_layers]
  layer_delta_norm     [[float?]] [n_steps][n_layers]  null at layer 0
  layer_delta_cosine   [[float?]] [n_steps][n_layers]  null at layer 0
  l0                   [[int]]    [n_steps][n_layers]
  output_entropy       [float]    [n_steps]
  logit_lens_entropy   [[float]]  [n_steps][n_layers]
  next_token_rank      [[int]]    [n_steps][n_layers]  if collect_emergence
  next_token_prob      [[float]]  [n_steps][n_layers]  if collect_emergence
  kl_to_final          [[float]]  [n_steps][n_layers]  if collect_emergence
  logit_delta_contrib  [[float?]] [n_steps][n_layers]  if collect_attribution
  transcoder_mse       [[float]]  [n_steps][n_layers]  if collect_transcoder_mse
  repulsion_contrib    [[float?]] [n_steps][n_layers]  if collect_repulsion
  forward_logit_contrib_k1 [[float?]] [n_steps][n_layers]  if collect_forward_contrib
  top1_token_per_layer  [[int]]   [n_steps][n_layers]  if collect_layer_extras
  kl_adjacent_layer     [[float]] [n_steps][n_layers]  if collect_layer_extras (nan at i=0)
  lens_entropy_delta    [[float]] [n_steps][n_layers]  if collect_layer_extras (nan at i=0)
  top5_token_ids_per_layer  [[[int]]]   [n_steps][n_layers][5]  if collect_top5_tokens
  top5_token_probs_per_layer [[[float]]] [n_steps][n_layers][5]  if collect_top5_tokens
  step_to_step_kl      [[float]]  [n_steps][n_layers]  if collect_step_kl (nan at step 0)
  transcoder_output_norm [[float]] [n_steps][n_layers]  if collect_transcoder_norms
  residual_cosine_to_final [[float]] [n_steps][n_layers]  if collect_residual_cosine
  mlp_contribution_norm  [[float]] [n_steps][n_layers]  if collect_mlp_norms
  active_features      → features.npz[record_id]  [n_steps, n_layers] object
  feature_values       → gen_feature_vals.npz[record_id]  [n_steps, n_layers] object (float32)
                           if collect_feature_values; parallel to active_features
  dense_residuals      → gen_residuals.npz[record_id]  if save_residual_layers set
  mlp_inputs           → gen_mlp_inputs.npz[record_id]  if save_mlp_layers set
  mlp_outputs          → gen_mlp_outputs.npz[record_id]  if save_mlp_layers set

  -- Encode metrics (present when mode includes 'encode') --
  enc_l0               [int]       [n_layers]
  enc_residual_norm    [float]     [n_layers]
  enc_output_entropy   float
  enc_logit_lens_entropy [float]   [n_layers]
  enc_kl_to_final      [float]     [n_layers]   if collect_emergence
  enc_attn_entropy     [[float?]]  [n_layers][n_heads]  if collect_attention
  enc_active_features  → enc_features.npz[record_id]  [n_layers] object
  enc_residuals        → enc_residuals.npz[record_id]  [n_layers, d_model]

Usage
-----
  uv run python -m src.poc.collect --variant it
  uv run python -m src.poc.collect --variant pt --mode generate
  uv run python -m src.poc.collect --variant it --mode generate,encode
  uv run python -m src.poc.collect --variant it --splits F,R
  uv run python -m src.poc.collect --variant it --ablation-layers 20-33
  uv run python -m src.poc.collect --variant it --max-gen-tokens 200 --gpus 4
  uv run python -m src.poc.collect --run-name debug_run --dataset data/exp3_dataset.jsonl
  uv run python -m src.poc.collect --variant it --save-residual-layers 10,11,12,20,25,30,33
  uv run python -m src.poc.collect --variant it --save-mlp-layers 20-33 --residual-strategy last
  uv run python -m src.poc.collect --variant it --force-decode-file results/pt_run/pt_tokens.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnsight import save as nnsave

from src.poc.shared.constants import N_LAYERS
from src.poc.shared.collect_config import CollectionConfig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entropy(logits: torch.Tensor,
             mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        logits = logits[mask]
    probs = torch.softmax(logits.float(), dim=-1)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


def _kl_div(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p ∥ q) in nats.  Both tensors must be probability distributions."""
    return float((p * torch.log(p / (q + 1e-12) + 1e-12)).sum().item())


def _sanitise(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for valid JSON output."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    return obj


def _navigate_path(obj: Any, path: str) -> Any:
    """Navigate a dotted attribute path, handling [idx] subscript notation.

    Works both inside and outside nnsight trace contexts.  Used to make hook
    paths in ModelHooks fully dynamic so collect.py is architecture-agnostic.

    Examples:
        _navigate_path(model, "language_model.layers")   → model.language_model.layers
        _navigate_path(layer, "output[0]")               → layer.output[0]
        _navigate_path(layer, "self_attn.output[1]")     → layer.self_attn.output[1]
        _navigate_path(layer, "pre_feedforward_layernorm.output")
    """
    for part in path.split("."):
        if "[" in part:
            attr_name, rest = part.split("[", 1)
            idx = int(rest.rstrip("]"))
            obj = getattr(obj, attr_name)[idx]
        else:
            obj = getattr(obj, part)
    return obj


# Generation steps to save when strategy = 'subsample' (1-indexed, i.e. step 1 = first token)
_SUBSAMPLE_STEP_INDICES: frozenset[int] = frozenset({0, 9, 49, 99, 199, 499})


def _should_save_step(step: int, strategy: str) -> bool:
    if strategy == "none":
        return False
    if strategy == "subsample":
        return step in _SUBSAMPLE_STEP_INDICES
    # "last" — caller decides at the end of the loop; here return True to collect all,
    # then trim post-loop.  We use a separate buffer for "last" to save memory.
    return False   # not used for "last"


def _tokens_to_float16(tensors: list[torch.Tensor]) -> np.ndarray:
    """Stack list of [d_model] tensors into [n, d_model] float16 ndarray."""
    return np.stack(
        [t.cpu().to(torch.float16).numpy() for t in tensors], axis=0
    )


def _empty_feature_summary(n_layers: int) -> dict[str, list[np.ndarray | None]]:
    return {
        "count": [None] * n_layers,
        "sum": [None] * n_layers,
    }


def _merge_feature_summary(dst: dict[str, list[np.ndarray | None]],
                           src: dict[str, list[np.ndarray | None]]) -> None:
    for layer_i in range(len(dst["count"])):
        src_count = src["count"][layer_i]
        src_sum = src["sum"][layer_i]
        if src_count is None or src_sum is None:
            continue
        if dst["count"][layer_i] is None:
            dst["count"][layer_i] = src_count
            dst["sum"][layer_i] = src_sum
        else:
            dst["count"][layer_i] += src_count
            dst["sum"][layer_i] += src_sum


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset_records(
    dataset_path: str,
    splits: list[str] | None = None,
    prompt_format: str = "B",
) -> list[dict]:
    """Load all records from a JSONL dataset, optionally filtered by split.

    Each returned dict contains the full record metadata plus 'prompt' (the
    formatted string that will be fed to the model).

    Args:
        dataset_path:  path to the JSONL dataset file
        splits:        list of split names to include; None = all
        prompt_format: 'A' or 'B' — which format to use from record['formats']

    Returns:
        list of dicts with all original record fields plus 'prompt'
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            f"Build it: uv run python -m src.poc.exp3.data.build_dataset"
        )
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if splits and rec.get("split") not in splits:
                continue
            rec["prompt"] = rec["formats"][prompt_format]
            records.append(rec)
    return records


def _load_force_decode_map(path: str) -> dict[str, list[int]]:
    """Load a JSON file mapping record_id → list[int] of token IDs."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"force_decode_file not found: {p}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return {k: [int(t) for t in v] for k, v in data.items()}


# ── Encode pass ──────────────────────────────────────────────────────────────

def _encode_pass(
    prompt_ids: torch.Tensor,  # [1, T] tokenized prompt
    loaded: Any,               # LoadedModel from shared/model.py
    cfg: CollectionConfig,
) -> dict:
    """Single forward pass on a prompt.  Returns encode-mode metrics.

    Captures last-token representations at every layer.  Uses cfg.hooks for
    architecture-agnostic hook paths so Gemma, Llama, and Qwen are supported.
    """
    device = loaded.W_U.device
    n_layers = cfg.n_layers
    hooks = cfg.hooks
    capture_attn = cfg.collect_attention or bool(cfg.attn_save_layers)

    with loaded.model.trace(prompt_ids, output_attentions=capture_attn):
        residual_saves  = []
        mlp_input_saves = []
        attn_saves      = []
        attn_raw_saves: dict[int, Any] = {}   # layer_idx → saved raw attn

        layers_root = _navigate_path(loaded.model, hooks.layers_root)

        for i in range(n_layers):
            layer = layers_root[i]

            residual_saves.append(nnsave(_navigate_path(layer, hooks.layer_residual)))
            mlp_input_saves.append(nnsave(_navigate_path(layer, hooks.mlp_input)))

            if capture_attn:
                attn_proxy = _navigate_path(layer, hooks.attn_weights)
                attn_saves.append(nnsave(attn_proxy))
                if i in cfg.attn_save_layers_set:
                    attn_raw_saves[i] = nnsave(attn_proxy)

        lm_head_mod = _navigate_path(loaded.model, hooks.lm_head)
        logits_save = nnsave(lm_head_mod.output)

    # Materialise: take last-token position
    residuals  = [r[0, -1, :].float() for r in residual_saves]
    mlp_inputs = [m[0, -1, :].float() for m in mlp_input_saves]
    logits     = logits_save[0, -1, :].float()

    attn_weights: list[torch.Tensor | None] = []
    if capture_attn:
        for a in attn_saves:
            try:
                attn_weights.append(a.float() if a is not None else None)
            except Exception:
                attn_weights.append(None)
    else:
        attn_weights = [None] * n_layers

    # Raw attention matrices for attn_save_layers (full [n_heads, T, T])
    attn_raw: dict[int, torch.Tensor] = {}
    for li, saved in attn_raw_saves.items():
        try:
            w = saved.float() if saved is not None else None
            if w is not None:
                # Remove batch dim if present → [n_heads, T, T]
                attn_raw[li] = w[0] if w.dim() == 4 else w
        except Exception:
            pass

    return {
        "residuals":    residuals,
        "mlp_inputs":   mlp_inputs,
        "logits":       logits,
        "attn_weights": attn_weights,
        "attn_raw":     attn_raw,
    }


def _compute_encode_metrics(
    raw: dict,
    loaded: Any,
    cfg: CollectionConfig,
) -> tuple[dict, list[list[int]], np.ndarray | None, np.ndarray | None]:
    """Compute encode-mode metrics from raw hook outputs.

    Returns:
        metrics:           dict of enc_* fields ready for the result record
        enc_features:      list[list[int]] active feature indices per layer
        enc_residuals:     ndarray [n_layers, d_model] float32, or None
        enc_attn_weights:  ndarray [n_save_layers] object (each [n_heads,T,T] float16), or None
    """
    residuals   = raw["residuals"]
    mlp_inputs  = raw["mlp_inputs"]
    logits      = raw["logits"]
    attn_weights = raw["attn_weights"]
    W_U         = loaded.W_U
    real_mask   = loaded.real_token_mask
    device      = W_U.device
    n_layers    = cfg.n_layers

    # ── L0 + active features ─────────────────────────────────────────────────
    enc_l0: list[int] = []
    enc_features: list[list[int]] = []
    with torch.inference_mode():
        for i in range(n_layers):
            tc = loaded.transcoder_list[i]
            x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device,
                                               dtype=tc.b_enc.dtype)
            acts = tc.encode(x)
            idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
            enc_l0.append(len(idxs))
            enc_features.append(idxs)

    # ── Residual norms ────────────────────────────────────────────────────────
    enc_residual_norm = [h.norm().item() for h in residuals]

    # ── Output entropy (final logits) ─────────────────────────────────────────
    enc_output_entropy = _entropy(logits, mask=real_mask)

    # ── Logit-lens + KL to final ─────────────────────────────────────────────
    enc_logit_lens_entropy: list[float] = []
    enc_kl_to_final: list[float] = []
    all_lens_logits: list[torch.Tensor] = []

    _final_norm = _navigate_path(loaded.model, cfg.hooks.final_norm)
    with torch.inference_mode():
        for i in range(n_layers):
            h = residuals[i].to(device)
            h_normed = _final_norm(h).float()
            ll = h_normed @ W_U                          # [vocab_size]
            all_lens_logits.append(ll)
            enc_logit_lens_entropy.append(_entropy(ll, mask=real_mask))

        if cfg.collect_emergence:
            final_real = all_lens_logits[-1][real_mask]
            p_final = torch.softmax(final_real, dim=-1)
            for i in range(n_layers):
                p_i = torch.softmax(all_lens_logits[i][real_mask], dim=-1)
                enc_kl_to_final.append(_kl_div(p_i, p_final))

    # ── Attention entropy per head ────────────────────────────────────────────
    enc_attn_entropy: list[list[float | None]] = []
    if cfg.collect_attention:
        T = attn_weights[0].shape[-1] if attn_weights[0] is not None else 0
        for i in range(n_layers):
            w = attn_weights[i]  # [B, n_heads, T, T] or [n_heads, T, T]
            if w is None:
                enc_attn_entropy.append([None] * cfg.n_heads)
                continue
            try:
                # Take last-token query row: [n_heads, T]
                if w.dim() == 4:
                    w = w[0]           # remove batch dim → [n_heads, T, T]
                attn_row = w[:, -1, :]  # [n_heads, T] — last token queries all keys
                head_ents = [
                    float(-(attn_row[h] * torch.log(attn_row[h] + 1e-12)).sum().item())
                    for h in range(attn_row.shape[0])
                ]
                enc_attn_entropy.append(head_ents)
            except Exception:
                enc_attn_entropy.append([None] * cfg.n_heads)

    # ── Dense residuals (for TwoNN / intrinsic dimension) ────────────────────
    enc_residuals_arr: np.ndarray | None = None
    if cfg.collect_residuals:
        enc_residuals_arr = np.stack(
            [h.cpu().to(torch.float32).numpy() for h in residuals],
            axis=0,
        )   # [n_layers, d_model]

    # ── Raw attention weight matrices (optional) ──────────────────────────────
    enc_attn_weights_arr: np.ndarray | None = None
    if cfg.attn_save_layers:
        attn_raw = raw.get("attn_raw", {})
        save_order = sorted(cfg.attn_save_layers)
        enc_attn_weights_arr = np.empty(len(save_order), dtype=object)
        for save_idx, layer_idx in enumerate(save_order):
            w = attn_raw.get(layer_idx)
            if w is not None:
                enc_attn_weights_arr[save_idx] = w.cpu().to(torch.float16).numpy()
            else:
                enc_attn_weights_arr[save_idx] = np.zeros((cfg.n_heads, 1, 1), dtype=np.float16)

    metrics = {
        "enc_l0":                  enc_l0,
        "enc_residual_norm":       enc_residual_norm,
        "enc_output_entropy":      enc_output_entropy,
        "enc_logit_lens_entropy":  enc_logit_lens_entropy,
        "enc_kl_to_final":         enc_kl_to_final if cfg.collect_emergence else [],
        "enc_attn_entropy":        enc_attn_entropy if cfg.collect_attention else [],
    }
    return metrics, enc_features, enc_residuals_arr, enc_attn_weights_arr


# ── Generate pass ─────────────────────────────────────────────────────────────

def _generate_pass(
    current_ids: torch.Tensor,
    loaded: Any,
    cfg: CollectionConfig,
) -> dict:
    """Single autoregressive step.  Returns raw hook outputs for one token.

    Uses cfg.hooks for architecture-agnostic hook paths.
    """
    device = loaded.W_U.device
    n_layers = cfg.n_layers
    hooks = cfg.hooks

    capture_mlp_output = (cfg.collect_transcoder_mse or bool(cfg.save_mlp_layers)
                          or cfg.collect_mlp_norms or cfg.collect_transcoder_norms)

    with loaded.model.trace(current_ids):
        residual_saves   = []
        mlp_input_saves  = []
        mlp_output_saves = []

        layers_root = _navigate_path(loaded.model, hooks.layers_root)

        for i in range(n_layers):
            layer = layers_root[i]

            mlp_input_saves.append(nnsave(_navigate_path(layer, hooks.mlp_input)))

            if capture_mlp_output:
                mlp_output_saves.append(nnsave(_navigate_path(layer, hooks.mlp_output)))

            if i in cfg.ablation_layers:
                # Zero MLP output in-place
                _navigate_path(layer, hooks.mlp_output)[:] = 0

            residual_saves.append(nnsave(_navigate_path(layer, hooks.layer_residual)))

        lm_head_mod = _navigate_path(loaded.model, hooks.lm_head)
        logits_save = nnsave(lm_head_mod.output)

    residuals   = [r[0, -1, :].float() for r in residual_saves]
    mlp_inputs  = [m[0, -1, :].float() for m in mlp_input_saves]
    logits      = logits_save[0, -1, :].float()
    mlp_outputs = (
        [m[0, -1, :].float() for m in mlp_output_saves]
        if mlp_output_saves else []
    )
    return {
        "residuals":   residuals,
        "mlp_inputs":  mlp_inputs,
        "logits":      logits,
        "mlp_outputs": mlp_outputs,
    }


def _compute_gen_step_metrics(
    raw: dict,
    loaded: Any,
    cfg: CollectionConfig,
    forced_token_id: int | None = None,
) -> tuple[dict, list[list[int]], int, list[torch.Tensor] | None]:
    """Compute all per-step metrics from one generation step's hook outputs.

    Args:
        raw:             hook outputs from _generate_pass
        loaded:          LoadedModel
        cfg:             CollectionConfig
        forced_token_id: when set, use this token for rank/prob/contrib rather
                         than argmax (force-decode mode for pt_token_rank_in_it)

    Returns:
        step_metrics:    dict of all per-step fields (includes repulsion_contrib if enabled)
        step_features:   list[list[int]] active features per layer
        next_token_id:   int — forced token id if forced, else greedy
        all_lens_logits: list[Tensor] [n_layers] vocab-sized logit tensors, or None.
                         Returned when collect_forward_contrib=True; caller buffers for
                         retroactive forward attribution of the next step's token.
    """
    residuals  = raw["residuals"]
    mlp_inputs = raw["mlp_inputs"]
    logits     = raw["logits"]
    mlp_outputs = raw["mlp_outputs"]
    W_U        = loaded.W_U
    real_mask  = loaded.real_token_mask
    device     = W_U.device
    n_layers   = cfg.n_layers

    # ── Residual norms ────────────────────────────────────────────────────────
    step_residual_norm = [h.norm().item() for h in residuals]

    # ── Layer delta norms + cosines ───────────────────────────────────────────
    step_delta_norm = [residuals[0].norm().item()]
    step_delta_cos  = [float("nan")]
    for i in range(1, n_layers):
        delta = residuals[i] - residuals[i - 1]
        step_delta_norm.append(delta.norm().item())
        denom = delta.norm() * residuals[i - 1].norm()
        cos = (torch.dot(delta, residuals[i - 1]) / denom).item() \
              if denom > 0 else float("nan")
        step_delta_cos.append(cos)

    # ── L0 + active features (+ optional activation values) ──────────────────
    step_l0: list[int] = []
    step_features: list[list[int]] = []
    step_feature_vals: list[list[float]] = []
    with torch.inference_mode():
        for i in range(n_layers):
            tc = loaded.transcoder_list[i]
            x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device,
                                               dtype=tc.b_enc.dtype)
            acts = tc.encode(x)
            idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
            step_l0.append(len(idxs))
            step_features.append(idxs)
            vals = acts[0][idxs].float().tolist() if idxs else []
            step_feature_vals.append(vals)

    # ── Output entropy ────────────────────────────────────────────────────────
    step_output_entropy = _entropy(logits, mask=real_mask)

    # ── Transcoder MSE + output norm (optional, shared forward pass) ──────────
    step_tc_mse: list[float] = []
    step_tc_output_norm: list[float] = []
    if (cfg.collect_transcoder_mse or cfg.collect_transcoder_norms) and mlp_outputs:
        with torch.inference_mode():
            for i in range(n_layers):
                tc = loaded.transcoder_list[i]
                x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device,
                                                   dtype=tc.b_enc.dtype)
                tc_out = tc.forward(x)[0].float()
                if cfg.collect_transcoder_norms:
                    step_tc_output_norm.append(tc_out.norm().item())
                if cfg.collect_transcoder_mse:
                    mse = ((tc_out - mlp_outputs[i]) ** 2).mean().item()
                    step_tc_mse.append(mse)

    # ── Logit-lens: compute for all layers in one pass ────────────────────────
    step_ll_ent: list[float] = []
    all_lens_logits: list[torch.Tensor] = []
    _final_norm = _navigate_path(loaded.model, cfg.hooks.final_norm)
    with torch.inference_mode():
        for i in range(n_layers):
            h = residuals[i].to(device)
            h_normed = _final_norm(h).float()
            ll = h_normed @ W_U
            all_lens_logits.append(ll)
            step_ll_ent.append(_entropy(ll, mask=real_mask))

    # ── Layer extras: top1 token, KL-adjacent, entropy-delta ─────────────────
    step_top1: list[int] = []
    step_kl_adj: list[float] = []
    step_ent_delta: list[float] = []
    if cfg.collect_layer_extras:
        with torch.inference_mode():
            prev_p: torch.Tensor | None = None
            for i in range(n_layers):
                # Top1: mask non-real tokens then argmax → vocab index
                masked_ll = all_lens_logits[i].clone()
                masked_ll[~real_mask] = float("-inf")
                step_top1.append(int(masked_ll.argmax().item()))

                # Entropy delta
                step_ent_delta.append(
                    float("nan") if i == 0 else step_ll_ent[i] - step_ll_ent[i - 1]
                )

                # KL(lens_i ∥ lens_{i-1})
                p_i = torch.softmax(all_lens_logits[i][real_mask], dim=-1)
                if i == 0 or prev_p is None:
                    step_kl_adj.append(float("nan"))
                else:
                    step_kl_adj.append(_kl_div(p_i, prev_p))
                prev_p = p_i

    # ── Top-5 tokens per layer ────────────────────────────────────────────────
    step_top5_ids: list[list[int]] = []
    step_top5_probs: list[list[float]] = []
    if cfg.collect_top5_tokens:
        with torch.inference_mode():
            for i in range(n_layers):
                masked_ll = all_lens_logits[i].clone()
                masked_ll[~real_mask] = float("-inf")
                probs = torch.softmax(masked_ll, dim=-1)
                top5 = torch.topk(probs, k=5)
                step_top5_ids.append(top5.indices.tolist())
                step_top5_probs.append([float(v) for v in top5.values.tolist()])

    # ── Residual cosine to final ──────────────────────────────────────────────
    step_res_cos: list[float] = []
    if cfg.collect_residual_cosine:
        h_final = residuals[-1].float()
        norm_final = h_final.norm()
        with torch.inference_mode():
            for i in range(n_layers):
                h_i = residuals[i].float()
                denom = h_i.norm() * norm_final
                cos = float((torch.dot(h_i, h_final) / denom).item()) \
                      if denom > 0 else float("nan")
                step_res_cos.append(cos)

    # ── MLP contribution norms ────────────────────────────────────────────────
    step_mlp_norm: list[float] = []
    if cfg.collect_mlp_norms and mlp_outputs and len(mlp_outputs) == n_layers:
        for i in range(n_layers):
            step_mlp_norm.append(float(mlp_outputs[i].norm().item()))

    # ── Greedy next token (or forced) ────────────────────────────────────────
    masked_logits = logits.clone()
    masked_logits[~real_mask] = float("-inf")
    greedy_token_id = int(masked_logits.argmax().item())
    next_token_id = forced_token_id if forced_token_id is not None else greedy_token_id

    # ── Emergence: rank / prob / KL (optional) ────────────────────────────────
    step_rank: list[int]   = []
    step_prob: list[float] = []
    step_kl:   list[float] = []
    if cfg.collect_emergence:
        with torch.inference_mode():
            final_real = all_lens_logits[-1][real_mask]
            p_final    = torch.softmax(final_real, dim=-1)
            real_idx_tensor = real_mask.nonzero(as_tuple=False).squeeze(1)

            for i in range(n_layers):
                lens_real = all_lens_logits[i][real_mask]
                p_i = torch.softmax(lens_real, dim=-1)

                if real_mask[next_token_id]:
                    match = (real_idx_tensor == next_token_id).nonzero(as_tuple=False)
                    if match.numel() > 0:
                        tok_idx   = match[0, 0].item()
                        tok_logit = lens_real[tok_idx].item()
                        rank = int((lens_real > tok_logit).sum().item()) + 1
                        prob = float(p_i[tok_idx].item())
                    else:
                        rank, prob = -1, float("nan")
                else:
                    rank, prob = -1, float("nan")

                step_rank.append(rank)
                step_prob.append(prob)
                step_kl.append(_kl_div(p_i, p_final))

    # ── Attribution: logit delta per layer (optional) ─────────────────────────
    step_contrib: list[float] = []
    if cfg.collect_attribution:
        step_contrib = [float("nan")]   # layer 0: no prior layer
        with torch.inference_mode():
            for i in range(1, n_layers):
                step_contrib.append(
                    all_lens_logits[i][next_token_id].item()
                    - all_lens_logits[i - 1][next_token_id].item()
                )

    # ── Repulsion: mean logit change of top-K wrong tokens (optional) ─────────
    step_repulsion: list[float] = []
    if cfg.collect_repulsion and len(all_lens_logits) > 1:
        K = cfg.repulsion_top_k
        step_repulsion.append(float("nan"))   # layer 0: no prior layer
        with torch.inference_mode():
            for i in range(1, n_layers):
                prev_ll = all_lens_logits[i - 1]
                curr_ll = all_lens_logits[i]
                # Top K+1 tokens by previous-layer logit, then exclude correct token
                top_indices = prev_ll.topk(K + 1).indices
                wrong_mask  = top_indices != next_token_id
                wrong_ids   = top_indices[wrong_mask][:K]
                if wrong_ids.numel() > 0:
                    delta = curr_ll[wrong_ids] - prev_ll[wrong_ids]
                    step_repulsion.append(float(delta.mean().item()))
                else:
                    step_repulsion.append(float("nan"))

    # Return all_lens_logits (CPU, to save GPU memory) when forward contrib or
    # step-to-step KL is needed — caller buffers for retroactive computation.
    lens_logits_out: list[torch.Tensor] | None = None
    if cfg.collect_forward_contrib or cfg.collect_step_kl:
        lens_logits_out = [ll.cpu() for ll in all_lens_logits]

    return {
        "residual_norm":             step_residual_norm,
        "layer_delta_norm":          step_delta_norm,
        "layer_delta_cosine":        step_delta_cos,
        "l0":                        step_l0,
        "output_entropy":            step_output_entropy,
        "logit_lens_entropy":        step_ll_ent,
        "next_token_rank":           step_rank,
        "next_token_prob":           step_prob,
        "kl_to_final":               step_kl,
        "logit_delta_contrib":       step_contrib,
        "transcoder_mse":            step_tc_mse,
        "repulsion_contrib":         step_repulsion,
        "top1_token_per_layer":      step_top1,
        "kl_adjacent_layer":         step_kl_adj,
        "lens_entropy_delta":        step_ent_delta,
        "top5_token_ids_per_layer":  step_top5_ids,
        "top5_token_probs_per_layer": step_top5_probs,
        "residual_cosine_to_final":  step_res_cos,
        "mlp_contribution_norm":     step_mlp_norm,
        "transcoder_output_norm":    step_tc_output_norm,
        "feature_vals":              step_feature_vals,
    }, step_features, next_token_id, lens_logits_out


# ── Per-record collection ─────────────────────────────────────────────────────

def collect_record(
    record: dict,
    loaded: Any,
    cfg: CollectionConfig,
    force_decode_map: dict[str, list[int]] | None = None,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray | None,
           np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None,
           dict[str, list[np.ndarray | None]]]:
    """Collect all metrics for one dataset record.

    Args:
        record:           full dataset record (from load_dataset_records)
        loaded:           LoadedModel (from shared/model.py)
        cfg:              CollectionConfig
        force_decode_map: optional {record_id: [token_ids]} for force-decode mode

    Returns (10-tuple):
        result_dict:       JSON-serializable result record
        gen_features:      ndarray [n_steps, n_layers] object dtype, or None
        enc_features:      ndarray [n_layers] object dtype, or None
        enc_residuals:     ndarray [n_layers, d_model] float32, or None
        gen_residuals:     ndarray [n_steps, n_layers, d_model] float16, or None
        gen_mlp_inputs:    ndarray [n_steps, n_layers, d_model] float16, or None
        gen_mlp_outputs:   ndarray [n_steps, n_layers, d_model] float16, or None
        enc_attn_weights:  ndarray [n_save_layers] object (each [n_heads,T,T] float16), or None
        gen_feature_vals:  ndarray [n_steps, n_layers] object (each float32 array), or None
        feature_summary:   {count, sum} per-layer sparse arrays → feature_importance_summary.npz
                           Always populated (cheap: sparse add.at over active features only)
    """
    tokenizer = loaded.tokenizer
    device    = loaded.W_U.device

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = record["prompt"]
    if cfg.is_instruction_tuned and cfg.apply_chat_template:
        input_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Complete the following: {prompt}"}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        chat_applied = True
    else:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        chat_applied = False

    # ── Identity metadata ─────────────────────────────────────────────────────
    record_id = record.get("id", record.get("record_id", "unknown"))
    meta = record.get("metadata", {})

    # Force-decode tokens for this record (if any)
    forced_tokens: list[int] | None = None
    if force_decode_map and record_id in force_decode_map:
        forced_tokens = force_decode_map[record_id]

    base = {
        # Dataset provenance
        "record_id":             record_id,
        "prompt_id":             record_id,   # backward compat alias
        "split":                 record.get("split", ""),
        "category":              record.get("split", ""),   # backward compat alias
        "source":                record.get("source", ""),
        "question":              record.get("question", ""),
        "question_type":         record.get("question_type", ""),
        "domain":                record.get("domain", ""),
        "alignment_subcategory": meta.get("alignment_subcategory"),
        "expected_behavior":     meta.get("expected_behavior"),
        "ooc_type":              record.get("ooc_type"),
        # Collection metadata
        "prompt":        prompt,
        "model_id":      cfg.model_id,
        "variant":       cfg.model_variant,
        "chat_template": chat_applied,
        "run_name":      cfg.run_name,
        "force_decoded": forced_tokens is not None,
    }

    gen_features_arr:    np.ndarray | None = None
    enc_features_arr:    np.ndarray | None = None
    enc_residuals_arr:   np.ndarray | None = None
    gen_residuals_arr:   np.ndarray | None = None
    gen_mlp_in_arr:      np.ndarray | None = None
    gen_mlp_out_arr:      np.ndarray | None = None
    enc_attn_weights_arr: np.ndarray | None = None
    gen_feature_vals_arr: np.ndarray | None = None
    feature_summary = _empty_feature_summary(cfg.n_layers)

    # ── Encode pass ───────────────────────────────────────────────────────────
    if "encode" in cfg.mode:
        enc_raw = _encode_pass(input_ids, loaded, cfg)
        enc_metrics, enc_feat_list, enc_residuals_arr, enc_attn_weights_arr = \
            _compute_encode_metrics(enc_raw, loaded, cfg)
        base.update(enc_metrics)

        # Pack encode features
        enc_features_arr = np.empty(cfg.n_layers, dtype=object)
        for i, idxs in enumerate(enc_feat_list):
            enc_features_arr[i] = np.array(idxs, dtype=np.int32)

    # ── Generate pass ─────────────────────────────────────────────────────────
    if "generate" in cfg.mode:
        eos_ids: set[int] = {tokenizer.eos_token_id}
        if cfg.is_instruction_tuned:
            eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            if eot and eot != tokenizer.unk_token_id:
                eos_ids.add(eot)

        # Per-step accumulators
        generated_tokens:    list[dict]         = []
        residual_norm:       list[list[float]]  = []
        layer_delta_norm:    list[list[float]]  = []
        layer_delta_cosine:  list[list[float]]  = []
        l0:                  list[list[int]]    = []
        output_entropy:      list[float]        = []
        logit_lens_entropy:  list[list[float]]  = []
        next_token_rank:     list[list[int]]    = []
        next_token_prob:     list[list[float]]  = []
        kl_to_final:         list[list[float]]  = []
        logit_delta_contrib: list[list[float]]  = []
        transcoder_mse:      list[list[float]]  = []
        repulsion_contrib:   list[list[float]]  = []
        all_features:        list[list[list[int]]] = []
        all_feature_vals:    list[list[list[float]]] = []   # parallel to all_features

        # New per-step accumulators
        top1_token_per_layer:       list[list[int]]   = []
        kl_adjacent_layer:          list[list[float]] = []
        lens_entropy_delta:         list[list[float]] = []
        top5_token_ids_per_layer:   list[list[list[int]]]   = []
        top5_token_probs_per_layer: list[list[list[float]]] = []
        step_to_step_kl:            list[list[float]] = []
        transcoder_output_norm:     list[list[float]] = []
        residual_cosine_to_final:   list[list[float]] = []
        mlp_contribution_norm:      list[list[float]] = []

        # Forward-looking attribution buffers
        prev_step_lens_logits: list[torch.Tensor] | None = None
        forward_logit_contrib_k1: list[list[float]] = []

        # Step-to-step KL buffer: prev step real-token softmax per layer
        prev_step_real_probs: list[torch.Tensor] | None = None

        # Dense residual/MLP accumulation
        save_res_layers = cfg.save_residual_layers
        save_mlp_layers = cfg.save_mlp_layers
        res_strategy    = cfg.save_residual_strategy

        # Buffer: list of [n_save_layers, d_model] float16 arrays per saved step
        res_steps_buf: list[np.ndarray]     = []  # for "subsample"
        res_last_buf:  np.ndarray | None    = None  # for "last"
        mlp_in_steps_buf:  list[np.ndarray] = []
        mlp_out_steps_buf: list[np.ndarray] = []
        mlp_in_last_buf:   np.ndarray | None = None
        mlp_out_last_buf:  np.ndarray | None = None

        n_forced = len(forced_tokens) if forced_tokens else 0
        max_steps = n_forced if forced_tokens else cfg.max_gen_tokens

        current_ids = input_ids.clone()
        for _step in range(max_steps):
            # Determine forced token for this step (if any)
            step_forced = forced_tokens[_step] if forced_tokens else None

            raw = _generate_pass(current_ids, loaded, cfg)
            step_metrics, step_feat, next_token_id, cur_lens_logits = \
                _compute_gen_step_metrics(raw, loaded, cfg, forced_token_id=step_forced)

            residual_norm.append(step_metrics["residual_norm"])
            layer_delta_norm.append(step_metrics["layer_delta_norm"])
            layer_delta_cosine.append(step_metrics["layer_delta_cosine"])
            l0.append(step_metrics["l0"])
            output_entropy.append(step_metrics["output_entropy"])
            logit_lens_entropy.append(step_metrics["logit_lens_entropy"])
            if step_metrics["next_token_rank"]:
                next_token_rank.append(step_metrics["next_token_rank"])
                next_token_prob.append(step_metrics["next_token_prob"])
                kl_to_final.append(step_metrics["kl_to_final"])
            if step_metrics["logit_delta_contrib"]:
                logit_delta_contrib.append(step_metrics["logit_delta_contrib"])
            if step_metrics["transcoder_mse"]:
                transcoder_mse.append(step_metrics["transcoder_mse"])
            if step_metrics["repulsion_contrib"]:
                repulsion_contrib.append(step_metrics["repulsion_contrib"])
            all_features.append(step_feat)
            if cfg.collect_feature_values:
                all_feature_vals.append(step_metrics["feature_vals"])
            for layer_i, idxs in enumerate(step_feat):
                if not idxs:
                    continue
                if feature_summary["count"][layer_i] is None:
                    feat_dim = len(loaded.transcoder_list[layer_i].b_enc)
                    feature_summary["count"][layer_i] = np.zeros(feat_dim, dtype=np.int64)
                    feature_summary["sum"][layer_i] = np.zeros(feat_dim, dtype=np.float32)
                vals = np.asarray(step_metrics["feature_vals"][layer_i], dtype=np.float32)
                np.add.at(feature_summary["count"][layer_i], idxs, 1)
                np.add.at(feature_summary["sum"][layer_i], idxs, vals)

            # ── New optional metrics ──────────────────────────────────────────
            if cfg.collect_layer_extras:
                top1_token_per_layer.append(step_metrics["top1_token_per_layer"])
                kl_adjacent_layer.append(step_metrics["kl_adjacent_layer"])
                lens_entropy_delta.append(step_metrics["lens_entropy_delta"])
            if cfg.collect_top5_tokens:
                top5_token_ids_per_layer.append(step_metrics["top5_token_ids_per_layer"])
                top5_token_probs_per_layer.append(step_metrics["top5_token_probs_per_layer"])
            if cfg.collect_transcoder_norms and step_metrics["transcoder_output_norm"]:
                transcoder_output_norm.append(step_metrics["transcoder_output_norm"])
            if cfg.collect_residual_cosine:
                residual_cosine_to_final.append(step_metrics["residual_cosine_to_final"])
            if cfg.collect_mlp_norms and step_metrics["mlp_contribution_norm"]:
                mlp_contribution_norm.append(step_metrics["mlp_contribution_norm"])

            # ── Step-to-step KL ───────────────────────────────────────────────
            if cfg.collect_step_kl and cur_lens_logits is not None:
                real_mask_skl = loaded.real_token_mask
                if prev_step_real_probs is None:
                    step_to_step_kl.append([float("nan")] * cfg.n_layers)
                else:
                    kls = []
                    with torch.inference_mode():
                        for li in range(cfg.n_layers):
                            p_curr = torch.softmax(
                                cur_lens_logits[li][real_mask_skl], dim=-1
                            )
                            kls.append(_kl_div(p_curr, prev_step_real_probs[li]))
                    step_to_step_kl.append(kls)
                # Buffer real-token softmax distributions for next step
                with torch.inference_mode():
                    prev_step_real_probs = [
                        torch.softmax(cur_lens_logits[li][real_mask_skl], dim=-1).cpu()
                        for li in range(cfg.n_layers)
                    ]

            # ── Forward-looking attribution: retroactively attribute prev step ──
            if cfg.collect_forward_contrib:
                if prev_step_lens_logits is not None:
                    # next_token_id is the +1 future token for the previous step
                    fwd = [float("nan")]  # layer 0 has no prior layer
                    real_mask = loaded.real_token_mask
                    for li in range(1, cfg.n_layers):
                        if real_mask[next_token_id]:
                            fwd.append(
                                prev_step_lens_logits[li][next_token_id].item()
                                - prev_step_lens_logits[li - 1][next_token_id].item()
                            )
                        else:
                            fwd.append(float("nan"))
                    forward_logit_contrib_k1.append(fwd)
                # Save current step's logit-lens for use in next iteration
                prev_step_lens_logits = cur_lens_logits

            tok_str = tokenizer.decode([next_token_id])
            generated_tokens.append({"token_id": next_token_id, "token_str": tok_str})

            # ── Save dense residuals for this step ────────────────────────────
            if save_res_layers:
                selected_res = [raw["residuals"][li] for li in save_res_layers
                                if li < len(raw["residuals"])]
                if selected_res:
                    arr_res = _tokens_to_float16(selected_res)  # [n_save_layers, d_model]
                    if res_strategy == "last":
                        res_last_buf = arr_res
                    elif res_strategy == "subsample" and _step in _SUBSAMPLE_STEP_INDICES:
                        res_steps_buf.append(arr_res)

            # ── Save MLP inputs/outputs for this step ─────────────────────────
            if save_mlp_layers and raw["mlp_outputs"]:
                sel_in  = [raw["mlp_inputs"][li]  for li in save_mlp_layers
                           if li < len(raw["mlp_inputs"])]
                sel_out = [raw["mlp_outputs"][li] for li in save_mlp_layers
                           if li < len(raw["mlp_outputs"])]
                if sel_in and sel_out:
                    arr_in  = _tokens_to_float16(sel_in)
                    arr_out = _tokens_to_float16(sel_out)
                    if res_strategy == "last":
                        mlp_in_last_buf  = arr_in
                        mlp_out_last_buf = arr_out
                    elif res_strategy == "subsample" and _step in _SUBSAMPLE_STEP_INDICES:
                        mlp_in_steps_buf.append(arr_in)
                        mlp_out_steps_buf.append(arr_out)

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token_id]], device=device)],
                dim=1,
            )
            if not forced_tokens and next_token_id in eos_ids:
                break

        n_steps = len(generated_tokens)
        generated_text = tokenizer.decode(
            [t["token_id"] for t in generated_tokens], skip_special_tokens=True
        )

        # ── Token type classification (post-generation, word-level) ───────────
        token_type: list[str] = []
        try:
            from src.poc.exp3.analysis.token_types import classify_generated_tokens
            token_type = classify_generated_tokens(generated_tokens)
        except Exception:
            token_type = ["OTHER"] * n_steps

        # ── final_layer_prob: prob of the generated token at the final layer ──
        final_layer_prob: list[float] = []
        if next_token_prob:
            final_layer_prob = [step_probs[-1] for step_probs in next_token_prob]

        # ── Finalise forward_logit_contrib_k1 (last step has no future token) ─
        if cfg.collect_forward_contrib:
            # Steps 0..n_steps-2 were computed retroactively; add nans for last
            forward_logit_contrib_k1.append([float("nan")] * cfg.n_layers)

        base.update({
            "generated_tokens":             generated_tokens,
            "generated_text":               generated_text,
            "token_type":                   token_type,
            "final_layer_prob":             final_layer_prob,
            "residual_norm":                residual_norm,
            "layer_delta_norm":             layer_delta_norm,
            "layer_delta_cosine":           layer_delta_cosine,
            "l0":                           l0,
            "output_entropy":               output_entropy,
            "logit_lens_entropy":           logit_lens_entropy,
            "next_token_rank":              next_token_rank,
            "next_token_prob":              next_token_prob,
            "kl_to_final":                  kl_to_final,
            "logit_delta_contrib":          logit_delta_contrib,
            "transcoder_mse":               transcoder_mse,
            "repulsion_contrib":            repulsion_contrib,
            "forward_logit_contrib_k1":     forward_logit_contrib_k1,
            "top1_token_per_layer":         top1_token_per_layer,
            "kl_adjacent_layer":            kl_adjacent_layer,
            "lens_entropy_delta":           lens_entropy_delta,
            "top5_token_ids_per_layer":     top5_token_ids_per_layer,
            "top5_token_probs_per_layer":   top5_token_probs_per_layer,
            "step_to_step_kl":              step_to_step_kl,
            "transcoder_output_norm":       transcoder_output_norm,
            "residual_cosine_to_final":     residual_cosine_to_final,
            "mlp_contribution_norm":        mlp_contribution_norm,
        })

        # Pack generation features
        gen_features_arr = np.empty((n_steps, cfg.n_layers), dtype=object)
        for s, step_feat in enumerate(all_features):
            for layer_i, idxs in enumerate(step_feat):
                gen_features_arr[s, layer_i] = np.array(idxs, dtype=np.int32)

        # Pack feature activation values (parallel to gen_features_arr)
        gen_feature_vals_arr: np.ndarray | None = None
        if cfg.collect_feature_values and all_feature_vals:
            gen_feature_vals_arr = np.empty((n_steps, cfg.n_layers), dtype=object)
            for s, step_vals in enumerate(all_feature_vals):
                for layer_i, vals in enumerate(step_vals):
                    gen_feature_vals_arr[s, layer_i] = np.array(vals, dtype=np.float32)

        # Pack dense residuals → [n_selected_steps, n_save_layers, d_model]
        if save_res_layers:
            if res_strategy == "last" and res_last_buf is not None:
                gen_residuals_arr = res_last_buf[np.newaxis]   # [1, n_layers, d_model]
            elif res_strategy == "subsample" and res_steps_buf:
                gen_residuals_arr = np.stack(res_steps_buf, axis=0)

        # Pack MLP IO → [n_selected_steps, n_save_mlp_layers, d_model]
        if save_mlp_layers:
            if res_strategy == "last":
                if mlp_in_last_buf is not None:
                    gen_mlp_in_arr  = mlp_in_last_buf[np.newaxis]
                    gen_mlp_out_arr = mlp_out_last_buf[np.newaxis]
            elif res_strategy == "subsample" and mlp_in_steps_buf:
                gen_mlp_in_arr  = np.stack(mlp_in_steps_buf,  axis=0)
                gen_mlp_out_arr = np.stack(mlp_out_steps_buf, axis=0)

    return (base, gen_features_arr, enc_features_arr, enc_residuals_arr,
            gen_residuals_arr, gen_mlp_in_arr, gen_mlp_out_arr, enc_attn_weights_arr,
            gen_feature_vals_arr, feature_summary)


# ── Streaming writer ──────────────────────────────────────────────────────────

class StreamingWriter:
    """Writes records to JSONL incrementally and accumulates NPZ data.

    Supports resuming interrupted runs: records already in results.jsonl are
    detected on __enter__ and skipped during collection.
    """

    def __init__(self, output_dir: str, cfg: CollectionConfig) -> None:
        self.out = Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        self._jsonl_path       = self.out / "results.jsonl"
        self._gen_npz_path     = self.out / "features.npz"
        self._enc_npz_path     = self.out / "enc_features.npz"
        self._res_npz_path     = self.out / "enc_residuals.npz"
        self._gres_npz_path    = self.out / "gen_residuals.npz"
        self._gmlp_in_path     = self.out / "gen_mlp_inputs.npz"
        self._gmlp_out_path    = self.out / "gen_mlp_outputs.npz"
        self._eattn_npz_path   = self.out / "enc_attn_weights.npz"
        self._gfeatvals_path   = self.out / "gen_feature_vals.npz"
        self._fimp_path        = self.out / "feature_importance_summary.npz"

        self._done_ids: set[str] = set()
        self._gen_data:       dict[str, np.ndarray] = {}
        self._enc_data:       dict[str, np.ndarray] = {}
        self._res_data:       dict[str, np.ndarray] = {}
        self._gres_data:      dict[str, np.ndarray] = {}
        self._gmlp_in_data:   dict[str, np.ndarray] = {}
        self._gmlp_out_data:  dict[str, np.ndarray] = {}
        self._eattn_data:     dict[str, np.ndarray] = {}
        self._gfeatvals_data: dict[str, np.ndarray] = {}
        self._fimp_summary = _empty_feature_summary(self.cfg.n_layers)
        self._n_written  = 0

    def __enter__(self) -> "StreamingWriter":
        # Load already-done IDs for resume support
        if self._jsonl_path.exists():
            with open(self._jsonl_path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._done_ids.add(rec.get("record_id", ""))
                    except json.JSONDecodeError:
                        pass
            print(f"  Resume: {len(self._done_ids)} records already done, skipping.")
            self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")
        else:
            self._jsonl_file = open(self._jsonl_path, "w", encoding="utf-8")
        return self

    def __exit__(self, *_) -> None:
        self._jsonl_file.close()
        self._flush_npz(final=True)

    def already_done(self, record_id: str) -> bool:
        return record_id in self._done_ids

    def write(
        self,
        result:            dict,
        gen_features:      np.ndarray | None,
        enc_features:      np.ndarray | None,
        enc_residuals:     np.ndarray | None,
        gen_residuals:     np.ndarray | None = None,
        gen_mlp_inputs:    np.ndarray | None = None,
        gen_mlp_outputs:   np.ndarray | None = None,
        enc_attn_weights:  np.ndarray | None = None,
        gen_feature_vals:  np.ndarray | None = None,
        feature_summary:   dict[str, list[np.ndarray | None]] | None = None,
    ) -> None:
        rid = result["record_id"]

        # Stream JSON record
        self._jsonl_file.write(json.dumps(_sanitise(result), ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

        # Accumulate NPZ data
        if gen_features is not None:
            self._gen_data[rid] = gen_features
        if enc_features is not None:
            self._enc_data[rid] = enc_features
        if enc_residuals is not None:
            self._res_data[rid] = enc_residuals
        if gen_residuals is not None:
            self._gres_data[rid] = gen_residuals
        if gen_mlp_inputs is not None:
            self._gmlp_in_data[rid]  = gen_mlp_inputs
        if gen_mlp_outputs is not None:
            self._gmlp_out_data[rid] = gen_mlp_outputs
        if enc_attn_weights is not None:
            self._eattn_data[rid] = enc_attn_weights
        if gen_feature_vals is not None:
            self._gfeatvals_data[rid] = gen_feature_vals
        if feature_summary is not None:
            _merge_feature_summary(self._fimp_summary, feature_summary)

        self._n_written += 1
        if self.cfg.checkpoint_every > 0 and \
                self._n_written % self.cfg.checkpoint_every == 0:
            self._flush_npz(final=False)

    def _flush_npz(self, final: bool) -> None:
        if self._gen_data:
            np.savez_compressed(str(self._gen_npz_path), **self._gen_data)
            if final:
                print(f"  Saved features          → {self._gen_npz_path}")
        if self._enc_data:
            np.savez_compressed(str(self._enc_npz_path), **self._enc_data)
            if final:
                print(f"  Saved enc_features      → {self._enc_npz_path}")
        if self._res_data:
            np.savez_compressed(str(self._res_npz_path), **self._res_data)
            if final:
                print(f"  Saved enc_residuals     → {self._res_npz_path}")
        if self._gres_data:
            np.savez_compressed(str(self._gres_npz_path), **self._gres_data)
            if final:
                print(f"  Saved gen_residuals     → {self._gres_npz_path}")
        if self._gmlp_in_data:
            np.savez_compressed(str(self._gmlp_in_path),  **self._gmlp_in_data)
            np.savez_compressed(str(self._gmlp_out_path), **self._gmlp_out_data)
            if final:
                print(f"  Saved gen_mlp_inputs    → {self._gmlp_in_path}")
                print(f"  Saved gen_mlp_outputs   → {self._gmlp_out_path}")
        if self._eattn_data:
            np.savez_compressed(str(self._eattn_npz_path), **self._eattn_data)
            if final:
                print(f"  Saved enc_attn_weights  → {self._eattn_npz_path}")
        if self._gfeatvals_data:
            np.savez_compressed(str(self._gfeatvals_path), **self._gfeatvals_data)
            if final:
                print(f"  Saved gen_feature_vals  → {self._gfeatvals_path}")
        fimp_payload = {}
        for layer_i in range(self.cfg.n_layers):
            counts = self._fimp_summary["count"][layer_i]
            sums = self._fimp_summary["sum"][layer_i]
            if counts is None or sums is None:
                continue
            fimp_payload[f"count_l{layer_i}"] = counts.astype(np.int64, copy=False)
            fimp_payload[f"sum_l{layer_i}"] = sums.astype(np.float32, copy=False)
        if fimp_payload:
            np.savez_compressed(str(self._fimp_path), **fimp_payload)
            if final:
                print(f"  Saved feature_importance → {self._fimp_path}")


# ── Multi-GPU worker ──────────────────────────────────────────────────────────

def _worker(gpu_id: int, records: list[dict], cfg: CollectionConfig,
            force_decode_map: dict[str, list[int]] | None) -> list[tuple]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg.device = "cuda"
    from src.poc.shared.model import load_model
    loaded = load_model(cfg)
    results = []
    for done, record in enumerate(records):
        rid = record.get("id", "unknown")
        print(f"  [GPU {gpu_id}] [{done + 1}/{len(records)}] {rid}", flush=True)
        result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp = collect_record(
            record, loaded, cfg, force_decode_map=force_decode_map
        )
        results.append((result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp))
    return results


# ── Main collection loop ──────────────────────────────────────────────────────

def run_collection(
    cfg: CollectionConfig,
    records: list[dict],
    loaded: Any | None = None,
    force_decode_map: dict[str, list[int]] | None = None,
) -> None:
    """Run collection for all records, writing results incrementally.

    Args:
        cfg:              CollectionConfig
        records:          list of dataset records (from load_dataset_records)
        loaded:           pre-loaded model, or None (loaded per worker for multi-GPU)
        force_decode_map: optional {record_id: [token_ids]} for force-decode mode
    """
    with StreamingWriter(cfg.output_dir, cfg) as writer:
        pending = [r for r in records
                   if not writer.already_done(r.get("id", ""))]
        total   = len(pending)
        print(f"  {total} records to collect → {cfg.output_dir}")

        if cfg.n_gpus <= 1:
            # Single-GPU or CPU: sequential
            if loaded is None:
                from src.poc.shared.model import load_model
                loaded = load_model(cfg)
            for done, record in enumerate(pending):
                rid = record.get("id", "unknown")
                print(f"  [{done + 1}/{total}] {rid}: '{record['prompt'][:55]}'")
                result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp = collect_record(
                    record, loaded, cfg, force_decode_map=force_decode_map
                )
                writer.write(result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp)
        else:
            # Multi-GPU: distribute across workers
            import multiprocessing as mp
            n_gpus = min(cfg.n_gpus, torch.cuda.device_count())
            if n_gpus == 0:
                raise RuntimeError("No CUDA GPUs available.")
            chunks = [pending[i::n_gpus] for i in range(n_gpus)]
            print(f"  Distributing {total} records across {n_gpus} GPUs")
            ctx = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as pool:
                futures = {
                    pool.submit(_worker, gid, chunks[gid], cfg, force_decode_map): gid
                    for gid in range(n_gpus)
                }
                for future in as_completed(futures):
                    for result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp in future.result():
                        writer.write(result, gf, ef, er, gr, gmi, gmo, eaw, gfv, fimp)

    print(f"\n  Collection complete → {cfg.output_dir}/results.jsonl")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_layer_list(s: str) -> list[int]:
    """Parse a layer specification string like '6-14' or '6,7,8' into a list of ints."""
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


def _parse_args() -> CollectionConfig:
    p = argparse.ArgumentParser(
        description="Unified structural-semantic feature collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    p.add_argument("--variant", choices=["pt", "it"], default="it",
                   help="Model variant: 'pt' (base) or 'it' (instruction-tuned).")
    p.add_argument("--model-id", default="",
                   help="Explicit HF model id. Auto-derived from --variant if empty.")
    p.add_argument("--transcoder-release", default="",
                   help="HF transcoder release. Auto-derived if empty.")
    p.add_argument("--transcoder-variant", default="width_16k_l0_big_affine")

    # Dataset
    p.add_argument("--dataset", default="data/exp3_dataset.jsonl",
                   help="Path to JSONL dataset.")
    p.add_argument("--splits", default=None,
                   help="Comma-separated splits to collect, e.g. 'F,R'. Default: all.")
    p.add_argument("--prompt-format", choices=["A", "B"], default="B",
                   help="Which format to use from each record's 'formats' dict.")
    p.add_argument("--chat-template", action="store_true",
                   help="Wrap IT prompts with the model's chat template.")

    # Collection mode
    p.add_argument("--mode", default="generate",
                   help="Comma-separated modes: 'generate', 'encode', or both.")

    # Generation
    p.add_argument("--max-gen-tokens", type=int, default=200)
    p.add_argument("--ablation-layers", default="",
                   help="Layers to ablate (zero MLP output). "
                        "Formats: '20,21,22' or '20-33' for a range.")

    # Force-decode
    p.add_argument("--force-decode-file", default="",
                   help="Path to JSON file mapping record_id → list[int] token IDs. "
                        "When set, generation uses forced tokens (pt_token_rank_in_it mode).")

    # Dense saving
    p.add_argument("--save-residual-layers", default="",
                   help="Layers to save full residuals for. Formats: '10,11,12' or '10-12'. "
                        "Suggested: '10,11,12,20,25,30,33'. Empty = disabled.")
    p.add_argument("--save-mlp-layers", default="",
                   help="Layers to save MLP inputs+outputs for. Format: '20-33'. "
                        "Intended for corrective stage transcoder adapter training.")
    p.add_argument("--residual-strategy", default="last",
                   choices=["last", "subsample", "none"],
                   help="Which steps to save dense residuals/MLP IO for. "
                        "'last'=final token only, 'subsample'=steps {1,10,50,100,200,500}.")

    # Metric flags
    p.add_argument("--no-emergence",   action="store_true",
                   help="Disable rank/prob/KL collection.")
    p.add_argument("--no-attribution", action="store_true",
                   help="Disable logit delta contribution.")
    p.add_argument("--transcoder-mse", action="store_true",
                   help="Collect transcoder MSE per layer per step.")
    p.add_argument("--no-attention",   action="store_true",
                   help="Disable attention entropy collection (encode mode).")
    p.add_argument("--collect-residuals", action="store_true",
                   help="Save dense residual vectors for TwoNN analysis (encode mode).")
    p.add_argument("--forward-contrib", action="store_true",
                   help="Collect forward-looking logit attribution (k=1 future token).")
    p.add_argument("--repulsion", action="store_true",
                   help="Collect repulsion: mean logit change of top-K wrong tokens per layer.")
    p.add_argument("--repulsion-top-k", type=int, default=10,
                   help="Number of wrong tokens to average for repulsion (default: 10).")
    p.add_argument("--no-attn-save", action="store_true",
                   help="Disable raw attention matrix saving (enc_attn_weights.npz).")
    p.add_argument("--attn-save-layers", default="",
                   help="Override attention save layers. Format: '6-14' or '6,7,8'. "
                        "Default: layers 6-14.")

    # New logit-lens extras
    p.add_argument("--layer-extras", action="store_true",
                   help="Collect top1_token_per_layer, kl_adjacent_layer, lens_entropy_delta "
                        "(all derived from existing logit-lens — near-zero overhead).")
    p.add_argument("--top5-tokens", action="store_true",
                   help="Collect top5_token_ids_per_layer + top5_token_probs_per_layer "
                        "from logit lens at each layer.")
    p.add_argument("--step-kl", action="store_true",
                   help="Collect step_to_step_kl: KL(lens_i at step t ∥ step t-1) per layer.")
    p.add_argument("--feature-values", action="store_true",
                   help="Collect feature activation magnitudes (parallel to active features). "
                        "Saved to gen_feature_vals.npz.")
    p.add_argument("--transcoder-norms", action="store_true",
                   help="Collect transcoder_output_norm: norm of tc.forward(x) per layer.")
    p.add_argument("--residual-cosine", action="store_true",
                   help="Collect residual_cosine_to_final: cos(h_i, h_33) per layer.")
    p.add_argument("--mlp-norms", action="store_true",
                   help="Collect mlp_contribution_norm: L2 norm of MLP output per layer. "
                        "Enables mlp_output hook for all layers.")

    # Hardware
    p.add_argument("--gpus", type=int, default=1)

    # Output
    p.add_argument("--run-name", default="",
                   help="Output subdirectory name. Auto-derived if empty.")
    p.add_argument("--output-base", default="results")
    p.add_argument("--checkpoint-every", type=int, default=100)

    args = p.parse_args()

    # Parse splits
    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None

    # Parse mode
    mode = {m.strip() for m in args.mode.split(",")}

    # Parse ablation layers
    ablation: list[int] = []
    if args.ablation_layers:
        for part in args.ablation_layers.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                ablation.extend(range(int(lo), int(hi) + 1))
            else:
                ablation.append(int(part))

    # Parse save-residual-layers
    save_res_layers: list[int] = []
    if args.save_residual_layers:
        for part in args.save_residual_layers.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                save_res_layers.extend(range(int(lo), int(hi) + 1))
            else:
                save_res_layers.append(int(part))

    # Parse save-mlp-layers
    save_mlp_layers: list[int] = []
    if args.save_mlp_layers:
        for part in args.save_mlp_layers.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-")
                save_mlp_layers.extend(range(int(lo), int(hi) + 1))
            else:
                save_mlp_layers.append(int(part))

    return CollectionConfig(
        run_name=args.run_name,
        model_variant=args.variant,
        model_id=args.model_id,
        transcoder_release=args.transcoder_release,
        transcoder_variant=args.transcoder_variant,
        dataset_path=args.dataset,
        splits=splits,
        prompt_format=args.prompt_format,
        apply_chat_template=args.chat_template,
        mode=mode,
        max_gen_tokens=args.max_gen_tokens,
        ablation_layers=ablation,
        force_decode_file=args.force_decode_file,
        save_residual_layers=save_res_layers,
        save_residual_strategy=args.residual_strategy,
        save_mlp_layers=save_mlp_layers,
        collect_emergence=not args.no_emergence,
        collect_attribution=not args.no_attribution,
        collect_transcoder_mse=args.transcoder_mse,
        collect_attention=not args.no_attention,
        collect_residuals=args.collect_residuals,
        collect_forward_contrib=args.forward_contrib,
        collect_repulsion=args.repulsion,
        repulsion_top_k=args.repulsion_top_k,
        attn_save_layers=_parse_layer_list(args.attn_save_layers)
                         if args.attn_save_layers else
                         ([] if args.no_attn_save else list(range(6, 15))),
        collect_layer_extras=True,
        collect_top5_tokens=args.top5_tokens,
        collect_step_kl=True,
        collect_feature_values=args.feature_values,
        collect_transcoder_norms=args.transcoder_norms,
        collect_residual_cosine=args.residual_cosine,
        collect_mlp_norms=args.mlp_norms,
        n_gpus=args.gpus,
        output_base=args.output_base,
        checkpoint_every=args.checkpoint_every,
    )


def main() -> None:
    cfg = _parse_args()

    # ── Print run summary ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  Structural-Semantic Feature Collection")
    print(f"  model              : {cfg.model_id}")
    print(f"  transcoder         : {cfg.transcoder_release} / {cfg.transcoder_variant}")
    print(f"  mode               : {sorted(cfg.mode)}")
    print(f"  prompt_format      : {cfg.prompt_format}")
    print(f"  chat_template      : {cfg.apply_chat_template}")
    print(f"  max_gen_tokens     : {cfg.max_gen_tokens}")
    print(f"  collect_emergence  : {cfg.collect_emergence}")
    print(f"  collect_attribution: {cfg.collect_attribution}")
    print(f"  collect_transcoder_mse: {cfg.collect_transcoder_mse}")
    print(f"  collect_attention  : {cfg.collect_attention}")
    print(f"  collect_residuals  : {cfg.collect_residuals}")
    if cfg.force_decode_file:
        print(f"  force_decode_file  : {cfg.force_decode_file}")
    if cfg.save_residual_layers:
        print(f"  save_residual_layers   : {cfg.save_residual_layers}")
        print(f"  save_residual_strategy : {cfg.save_residual_strategy}")
    if cfg.save_mlp_layers:
        print(f"  save_mlp_layers        : {cfg.save_mlp_layers}")
    if cfg.ablation_layers:
        print(f"  ablation_layers        : {cfg.ablation_layers}")
    if cfg.collect_forward_contrib:
        print(f"  collect_forward_contrib: True  (NH1 forward-looking attribution)")
    if cfg.collect_repulsion:
        print(f"  collect_repulsion      : True  (top-{cfg.repulsion_top_k} wrong tokens)")
    if cfg.attn_save_layers:
        print(f"  attn_save_layers       : {cfg.attn_save_layers}  → enc_attn_weights.npz")
    if cfg.collect_layer_extras:
        print(f"  collect_layer_extras   : True  (top1_token, kl_adj, entropy_delta per layer)")
    if cfg.collect_top5_tokens:
        print(f"  collect_top5_tokens    : True  (top5 ids+probs per layer)")
    if cfg.collect_step_kl:
        print(f"  collect_step_kl        : True  (step-to-step KL per layer)")
    if cfg.collect_feature_values:
        print(f"  collect_feature_values : True  (activation magnitudes → gen_feature_vals.npz)")
    if cfg.collect_transcoder_norms:
        print(f"  collect_transcoder_norms: True (tc.forward() norm per layer)")
    if cfg.collect_residual_cosine:
        print(f"  collect_residual_cosine: True  (cos(h_i, h_33) per layer)")
    if cfg.collect_mlp_norms:
        print(f"  collect_mlp_norms      : True  (MLP output norm per layer)")
    print(f"  n_gpus             : {cfg.n_gpus}")
    print(f"  dataset            : {cfg.dataset_path}")
    if cfg.splits:
        print(f"  splits             : {cfg.splits}")
    print(f"  output_dir         : {cfg.output_dir}")
    print("=" * 65)

    # ── Save config snapshot ──────────────────────────────────────────────────
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(cfg.output_dir) / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"\n  Config saved → {config_path}")

    # ── Load force-decode map (if any) ────────────────────────────────────────
    force_decode_map: dict[str, list[int]] | None = None
    if cfg.force_decode_file:
        print(f"\n  Loading force-decode tokens from {cfg.force_decode_file} …")
        force_decode_map = _load_force_decode_map(cfg.force_decode_file)
        print(f"  Loaded tokens for {len(force_decode_map)} records.")

    # ── Load dataset ──────────────────────────────────────────────────────────
    print("\n[1/3] Loading dataset …")
    records = load_dataset_records(
        cfg.dataset_path,
        splits=cfg.splits,
        prompt_format=cfg.prompt_format,
    )
    print(f"  {len(records)} records loaded")
    by_split: dict[str, int] = {}
    for r in records:
        s = r.get("split", "?")
        by_split[s] = by_split.get(s, 0) + 1
    for s, n in sorted(by_split.items()):
        print(f"    {s}: {n}")

    # ── Load model (single-GPU path only) ─────────────────────────────────────
    loaded = None
    if cfg.n_gpus <= 1:
        print("\n[2/3] Loading model + transcoders …")
        from src.poc.shared.model import load_model
        loaded = load_model(cfg)
    else:
        print("\n[2/3] Skipping main-process model load (workers load independently)")

    # ── Collect ───────────────────────────────────────────────────────────────
    print("\n[3/3] Collecting …")
    run_collection(cfg, records, loaded, force_decode_map=force_decode_map)

    print("\nDone.")


if __name__ == "__main__":
    main()
