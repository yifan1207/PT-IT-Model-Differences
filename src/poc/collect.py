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
  run_config.json      full config snapshot

Record schema
-------------
Each JSONL line is a JSON object containing:

  -- Dataset provenance --
  record_id            str        unique id from dataset (e.g. "f_triviaqa_0001")
  prompt_id            str        alias for record_id (backward compat)
  split                str        F / R / OOD / A
  category             str        alias for split (backward compat)
  source               str        triviaqa / gsm8k / advbench / custom / …
  question             str        original question text
  question_type        str        factual / numerical / harmful / …
  domain               str        subject domain
  alignment_subcategory str|null  4a–4e for A-split records
  expected_behavior    str|null   refuse / comply / comply_safely
  ooc_type             str|null   novel_entity / post_training / counterfactual

  -- Collection metadata --
  prompt               str        exact prompt fed to the model
  model_id             str        HF model id
  variant              str        pt | it
  chat_template        bool       whether chat template was applied
  run_name             str        identifies this collection run

  -- Generation metrics (present when mode includes 'generate') --
  generated_tokens     [{token_id, token_str}]  per step
  generated_text       str
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
  active_features      → features.npz[record_id]  [n_steps, n_layers] object

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


# ── Encode pass ──────────────────────────────────────────────────────────────

def _encode_pass(
    prompt_ids: torch.Tensor,  # [1, T] tokenized prompt
    loaded: Any,               # LoadedModel from shared/model.py
    cfg: CollectionConfig,
) -> dict:
    """Single forward pass on a prompt.  Returns encode-mode metrics.

    Captures last-token representations at every layer.

    Returns dict with keys:
      enc_residuals_raw: list[Tensor] [n_layers] each [d_model]  (if collect_residuals)
      enc_mlp_inputs:    list[Tensor] [n_layers] each [d_model]
      enc_logits:        Tensor [vocab_size]
      enc_attn_weights:  list[Tensor|None] [n_layers] each [n_heads, T, T]
    """
    device = loaded.W_U.device
    n_layers = cfg.n_layers
    hooks = cfg.hooks

    with loaded.model.trace(
        prompt_ids,
        output_attentions=cfg.collect_attention,
    ):
        residual_saves  = []
        mlp_input_saves = []
        attn_saves      = []

        for i in range(n_layers):
            layer = loaded.model.language_model.layers[i]

            # Residual stream — full tensor; extract last token post-trace
            residual_saves.append(nnsave(layer.output[0]))

            # MLP input (pre-feedforward layernorm output)
            mlp_input_saves.append(nnsave(layer.pre_feedforward_layernorm.output))

            # Attention weights (last-token row, all heads)
            if cfg.collect_attention:
                attn_saves.append(nnsave(layer.self_attn.output[1]))

        logits_save = nnsave(loaded.model.lm_head.output)

    # Materialise: take last-token position
    residuals  = [r[0, -1, :].float() for r in residual_saves]
    mlp_inputs = [m[0, -1, :].float() for m in mlp_input_saves]
    logits     = logits_save[0, -1, :].float()

    attn_weights: list[torch.Tensor | None] = []
    if cfg.collect_attention:
        for i, a in enumerate(attn_saves):
            try:
                attn_weights.append(a.float() if a is not None else None)
            except Exception:
                attn_weights.append(None)
    else:
        attn_weights = [None] * n_layers

    return {
        "residuals":   residuals,
        "mlp_inputs":  mlp_inputs,
        "logits":      logits,
        "attn_weights": attn_weights,
    }


def _compute_encode_metrics(
    raw: dict,
    loaded: Any,
    cfg: CollectionConfig,
) -> tuple[dict, list[list[int]], np.ndarray | None]:
    """Compute encode-mode metrics from raw hook outputs.

    Returns:
        metrics:       dict of enc_* fields ready for the result record
        enc_features:  list[list[int]] active feature indices per layer
        enc_residuals: ndarray [n_layers, d_model] or None
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

    with torch.inference_mode():
        for i in range(n_layers):
            h = residuals[i].to(device)
            h_normed = loaded.model.language_model.norm(h).float()
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

    metrics = {
        "enc_l0":                  enc_l0,
        "enc_residual_norm":       enc_residual_norm,
        "enc_output_entropy":      enc_output_entropy,
        "enc_logit_lens_entropy":  enc_logit_lens_entropy,
        "enc_kl_to_final":         enc_kl_to_final if cfg.collect_emergence else [],
        "enc_attn_entropy":        enc_attn_entropy if cfg.collect_attention else [],
    }
    return metrics, enc_features, enc_residuals_arr


# ── Generate pass ─────────────────────────────────────────────────────────────

def _generate_pass(
    current_ids: torch.Tensor,
    loaded: Any,
    cfg: CollectionConfig,
) -> dict:
    """Single autoregressive step.  Returns raw hook outputs for one token."""
    device = loaded.W_U.device
    n_layers = cfg.n_layers

    with loaded.model.trace(current_ids):
        residual_saves   = []
        mlp_input_saves  = []
        mlp_output_saves = []

        for i in range(n_layers):
            layer = loaded.model.language_model.layers[i]

            mlp_input_saves.append(nnsave(layer.pre_feedforward_layernorm.output))

            if cfg.collect_transcoder_mse:
                mlp_output_saves.append(nnsave(layer.mlp.output))

            if i in cfg.ablation_layers:
                layer.mlp.output[:] = 0

            residual_saves.append(nnsave(layer.output[0]))

        logits_save = nnsave(loaded.model.lm_head.output)

    residuals  = [r[0, -1, :].float() for r in residual_saves]
    mlp_inputs = [m[0, -1, :].float() for m in mlp_input_saves]
    logits     = logits_save[0, -1, :].float()
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
) -> tuple[dict, list[list[int]], int]:
    """Compute all per-step metrics from one generation step's hook outputs.

    Returns:
        step_metrics:    dict of all per-step fields
        step_features:   list[list[int]] active features per layer
        next_token_id:   int greedy next token
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

    # ── L0 + active features ─────────────────────────────────────────────────
    step_l0: list[int] = []
    step_features: list[list[int]] = []
    with torch.inference_mode():
        for i in range(n_layers):
            tc = loaded.transcoder_list[i]
            x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device,
                                               dtype=tc.b_enc.dtype)
            acts = tc.encode(x)
            idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
            step_l0.append(len(idxs))
            step_features.append(idxs)

    # ── Output entropy ────────────────────────────────────────────────────────
    step_output_entropy = _entropy(logits, mask=real_mask)

    # ── Transcoder MSE (optional) ─────────────────────────────────────────────
    step_tc_mse: list[float] = []
    if cfg.collect_transcoder_mse and mlp_outputs:
        with torch.inference_mode():
            for i in range(n_layers):
                tc = loaded.transcoder_list[i]
                x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device,
                                                   dtype=tc.b_enc.dtype)
                tc_out = tc.forward(x)[0].float()
                mse = ((tc_out - mlp_outputs[i]) ** 2).mean().item()
                step_tc_mse.append(mse)

    # ── Logit-lens: compute for all layers in one pass ────────────────────────
    step_ll_ent: list[float] = []
    all_lens_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(n_layers):
            h = residuals[i].to(device)
            h_normed = loaded.model.language_model.norm(h).float()
            ll = h_normed @ W_U
            all_lens_logits.append(ll)
            step_ll_ent.append(_entropy(ll, mask=real_mask))

    # ── Greedy next token ─────────────────────────────────────────────────────
    masked_logits = logits.clone()
    masked_logits[~real_mask] = float("-inf")
    next_token_id = int(masked_logits.argmax().item())

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
                        tok_idx  = match[0, 0].item()
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

    return {
        "residual_norm":      step_residual_norm,
        "layer_delta_norm":   step_delta_norm,
        "layer_delta_cosine": step_delta_cos,
        "l0":                 step_l0,
        "output_entropy":     step_output_entropy,
        "logit_lens_entropy": step_ll_ent,
        "next_token_rank":    step_rank,
        "next_token_prob":    step_prob,
        "kl_to_final":        step_kl,
        "logit_delta_contrib": step_contrib,
        "transcoder_mse":     step_tc_mse,
    }, step_features, next_token_id


# ── Per-record collection ─────────────────────────────────────────────────────

def collect_record(
    record: dict,
    loaded: Any,
    cfg: CollectionConfig,
) -> tuple[dict, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Collect all metrics for one dataset record.

    Args:
        record:  full dataset record (from load_dataset_records)
        loaded:  LoadedModel (from shared/model.py)
        cfg:     CollectionConfig

    Returns:
        result_dict:     JSON-serializable result record
        gen_features:    ndarray [n_steps, n_layers] object dtype, or None
        enc_features:    ndarray [n_layers] object dtype, or None
        enc_residuals:   ndarray [n_layers, d_model] float32, or None
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
    }

    gen_features_arr:  np.ndarray | None = None
    enc_features_arr:  np.ndarray | None = None
    enc_residuals_arr: np.ndarray | None = None

    # ── Encode pass ───────────────────────────────────────────────────────────
    if "encode" in cfg.mode:
        enc_raw = _encode_pass(input_ids, loaded, cfg)
        enc_metrics, enc_feat_list, enc_residuals_arr = _compute_encode_metrics(
            enc_raw, loaded, cfg
        )
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
        generated_tokens: list[dict]        = []
        residual_norm:     list[list[float]] = []
        layer_delta_norm:  list[list[float]] = []
        layer_delta_cosine:list[list[float]] = []
        l0:                list[list[int]]   = []
        output_entropy:    list[float]       = []
        logit_lens_entropy:list[list[float]] = []
        next_token_rank:   list[list[int]]   = []
        next_token_prob:   list[list[float]] = []
        kl_to_final:       list[list[float]] = []
        logit_delta_contrib: list[list[float]] = []
        transcoder_mse:    list[list[float]] = []
        all_features:      list[list[list[int]]] = []

        current_ids = input_ids.clone()
        for _step in range(cfg.max_gen_tokens):
            raw = _generate_pass(current_ids, loaded, cfg)
            step_metrics, step_feat, next_token_id = _compute_gen_step_metrics(
                raw, loaded, cfg
            )

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
            all_features.append(step_feat)

            tok_str = tokenizer.decode([next_token_id])
            generated_tokens.append({"token_id": next_token_id, "token_str": tok_str})

            current_ids = torch.cat(
                [current_ids, torch.tensor([[next_token_id]], device=device)],
                dim=1,
            )
            if next_token_id in eos_ids:
                break

        n_steps = len(generated_tokens)
        generated_text = tokenizer.decode(
            [t["token_id"] for t in generated_tokens], skip_special_tokens=True
        )

        base.update({
            "generated_tokens":    generated_tokens,
            "generated_text":      generated_text,
            "residual_norm":       residual_norm,
            "layer_delta_norm":    layer_delta_norm,
            "layer_delta_cosine":  layer_delta_cosine,
            "l0":                  l0,
            "output_entropy":      output_entropy,
            "logit_lens_entropy":  logit_lens_entropy,
            "next_token_rank":     next_token_rank,
            "next_token_prob":     next_token_prob,
            "kl_to_final":         kl_to_final,
            "logit_delta_contrib": logit_delta_contrib,
            "transcoder_mse":      transcoder_mse,
        })

        # Pack generation features
        gen_features_arr = np.empty((n_steps, cfg.n_layers), dtype=object)
        for s, step_feat in enumerate(all_features):
            for layer_i, idxs in enumerate(step_feat):
                gen_features_arr[s, layer_i] = np.array(idxs, dtype=np.int32)

    return base, gen_features_arr, enc_features_arr, enc_residuals_arr


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

        self._jsonl_path = self.out / "results.jsonl"
        self._gen_npz_path  = self.out / "features.npz"
        self._enc_npz_path  = self.out / "enc_features.npz"
        self._res_npz_path  = self.out / "enc_residuals.npz"

        self._done_ids: set[str] = set()
        self._gen_data:  dict[str, np.ndarray] = {}
        self._enc_data:  dict[str, np.ndarray] = {}
        self._res_data:  dict[str, np.ndarray] = {}
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
        result:          dict,
        gen_features:    np.ndarray | None,
        enc_features:    np.ndarray | None,
        enc_residuals:   np.ndarray | None,
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

        self._n_written += 1
        if self.cfg.checkpoint_every > 0 and \
                self._n_written % self.cfg.checkpoint_every == 0:
            self._flush_npz(final=False)

    def _flush_npz(self, final: bool) -> None:
        label = "final" if final else "checkpoint"
        if self._gen_data:
            np.savez_compressed(str(self._gen_npz_path), **self._gen_data)
            if final:
                print(f"  Saved features      → {self._gen_npz_path}")
        if self._enc_data:
            np.savez_compressed(str(self._enc_npz_path), **self._enc_data)
            if final:
                print(f"  Saved enc_features  → {self._enc_npz_path}")
        if self._res_data:
            np.savez_compressed(str(self._res_npz_path), **self._res_data)
            if final:
                print(f"  Saved enc_residuals → {self._res_npz_path}")


# ── Multi-GPU worker ──────────────────────────────────────────────────────────

def _worker(gpu_id: int, records: list[dict], cfg: CollectionConfig) -> list[tuple]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg.device = "cuda"
    from src.poc.shared.model import load_model
    loaded = load_model(cfg)
    results = []
    for done, record in enumerate(records):
        rid = record.get("id", "unknown")
        print(f"  [GPU {gpu_id}] [{done + 1}/{len(records)}] {rid}", flush=True)
        result, gen_feat, enc_feat, enc_res = collect_record(record, loaded, cfg)
        results.append((result, gen_feat, enc_feat, enc_res))
    return results


# ── Main collection loop ──────────────────────────────────────────────────────

def run_collection(
    cfg: CollectionConfig,
    records: list[dict],
    loaded: Any | None = None,
) -> None:
    """Run collection for all records, writing results incrementally.

    Args:
        cfg:     CollectionConfig
        records: list of dataset records (from load_dataset_records)
        loaded:  pre-loaded model, or None (loaded per worker for multi-GPU)
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
                result, gf, ef, er = collect_record(record, loaded, cfg)
                writer.write(result, gf, ef, er)
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
                futures = {pool.submit(_worker, gid, chunks[gid], cfg): gid
                           for gid in range(n_gpus)}
                for future in as_completed(futures):
                    for result, gf, ef, er in future.result():
                        writer.write(result, gf, ef, er)

    print(f"\n  Collection complete → {cfg.output_dir}/results.jsonl")


# ── CLI ───────────────────────────────────────────────────────────────────────

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
                   help="Comma-separated modes: 'generate', 'encode', or both. "
                        "Default: generate.")

    # Generation
    p.add_argument("--max-gen-tokens", type=int, default=200)
    p.add_argument("--ablation-layers", default="",
                   help="Layers to ablate (zero MLP output). "
                        "Formats: '20,21,22' or '20-33' for a range.")

    # Metric flags
    p.add_argument("--no-emergence",   action="store_true",
                   help="Disable rank/prob/KL collection.")
    p.add_argument("--no-attribution", action="store_true",
                   help="Disable logit delta contribution.")
    p.add_argument("--transcoder-mse", action="store_true",
                   help="Collect transcoder MSE (requires mlp.output hook).")
    p.add_argument("--no-attention",   action="store_true",
                   help="Disable attention entropy collection (encode mode).")
    p.add_argument("--collect-residuals", action="store_true",
                   help="Save dense residual vectors for TwoNN analysis (large).")

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
        collect_emergence=not args.no_emergence,
        collect_attribution=not args.no_attribution,
        collect_transcoder_mse=args.transcoder_mse,
        collect_attention=not args.no_attention,
        collect_residuals=args.collect_residuals,
        n_gpus=args.gpus,
        output_base=args.output_base,
        checkpoint_every=args.checkpoint_every,
    )


def main() -> None:
    cfg = _parse_args()

    # ── Print run summary ─────────────────────────────────────────────────────
    print("=" * 65)
    print("  Structural-Semantic Feature Collection")
    print(f"  model             : {cfg.model_id}")
    print(f"  transcoder        : {cfg.transcoder_release} / {cfg.transcoder_variant}")
    print(f"  mode              : {sorted(cfg.mode)}")
    print(f"  prompt_format     : {cfg.prompt_format}")
    print(f"  chat_template     : {cfg.apply_chat_template}")
    print(f"  max_gen_tokens    : {cfg.max_gen_tokens}")
    print(f"  collect_emergence : {cfg.collect_emergence}")
    print(f"  collect_attribution: {cfg.collect_attribution}")
    print(f"  collect_attention : {cfg.collect_attention}")
    print(f"  collect_residuals : {cfg.collect_residuals}")
    if cfg.ablation_layers:
        print(f"  ablation_layers   : {cfg.ablation_layers}")
    print(f"  n_gpus            : {cfg.n_gpus}")
    print(f"  dataset           : {cfg.dataset_path}")
    if cfg.splits:
        print(f"  splits            : {cfg.splits}")
    print(f"  output_dir        : {cfg.output_dir}")
    print("=" * 65)

    # ── Save config snapshot ──────────────────────────────────────────────────
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    config_path = Path(cfg.output_dir) / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    print(f"\n  Config saved → {config_path}")

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
    run_collection(cfg, records, loaded)

    print("\nDone.")


if __name__ == "__main__":
    main()
