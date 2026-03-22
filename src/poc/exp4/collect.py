"""
Exp4 data collection: single-pass forward pass for phase transition analysis.

Unlike exp3 (which runs autoregressive generation), this module performs ONE
forward pass per prompt.  We analyse how the model's internal representations
of the prompt evolve across layers — the mechanism responsible for the L0 dip.

Quantities collected per prompt
-------------------------------
  residuals  [n_layers, d_model] float32
      Last-token residual stream vector at each layer.
      Used by analysis/intrinsic_dim.py for TwoNN intrinsic dimension estimation.
      TwoNN pools across all prompts: shape [n_prompts, d_model] per layer.

  attn_entropy  [n_layers, n_heads] float32
      Per-head entropy of the last token's attention distribution at each layer.
      H(head) = -Σ_k attn[head, k] * log(attn[head, k] + ε)
      Summed over key positions.  For local-attention layers (window=1024),
      positions outside the window have weight 0 and contribute 0 to entropy.
      Requires cfg.collect_attention=True + model loaded with eager attention.

  attn_weights_at_save_layers  dict[int → [n_heads, T, T]] float32
      Full attention weight matrices at cfg.attn_save_layers (default 6–14).
      Used for head-level PT-IT KL divergence analysis.
      Only saved when cfg.collect_attention=True.

  active_features  [n_layers] list[int]
      Active transcoder feature indices at last-token position, each layer.
      Used by analysis/jaccard.py for adjacent-layer Jaccard curves.
      Requires cfg.collect_features=True.

  l0  [n_layers] int
      L0 sparsity (number of active features) per layer at last-token position.

Output files
------------
  {cfg.output_path}    : JSON — per-prompt metadata + l0 + attn_entropy
  {cfg.residuals_path} : .npz — {prompt_id: [n_layers, d_model]}
  {cfg.features_path}  : .npz — {prompt_id: object array [n_layers] of int32 arrays}
  {cfg.attn_path}      : .npz — {f"{prompt_id}_{layer}": [n_heads, T, T]}
                         (only for layers in cfg.attn_save_layers)

Attention implementation notes
------------------------------
To materialise the [B, H, T, T] attention weight matrix in Gemma 3:
  1. Model must be loaded with attn_implementation="eager"  (done in model.py).
  2. output_attentions=True must be passed to model.trace() at inference time.
  3. Attention output from self_attn is a tuple: (hidden_states, attn_weights, ...)
     where attn_weights has shape [B, n_heads, T, T].

If output_attentions=True is not supported by the circuit_tracer version,
the code falls back to entropy=NaN for all layers and logs a warning.
"""
import os
import json
import math
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from nnsight import save as nnsave

from src.poc.shared.model import LoadedModel
from src.poc.shared.constants import N_LAYERS
from src.poc.exp4.config import Exp4Config


# ── helpers ───────────────────────────────────────────────────────────────────

def _attn_entropy(attn_row: torch.Tensor) -> float:
    """Entropy of a single attention distribution over key positions.

    attn_row : [T_k] — already-normalised probabilities (post-softmax).
    Returns entropy in nats.  Zero entries contribute 0 (0 * log(0) := 0).
    """
    probs = attn_row.clamp(min=0.0)   # local-attention zeros can be negative fp noise
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


def _safe_float(x) -> float:
    """Convert to float, mapping NaN/inf to None for JSON serialisation."""
    if isinstance(x, float) and not math.isfinite(x):
        return None  # type: ignore[return-value]
    return float(x)


# ── single-prompt collection ──────────────────────────────────────────────────

def collect_prompt(
    prompt_id: str,
    category: str,
    prompt: str,
    loaded: LoadedModel,
    cfg: Exp4Config,
) -> dict:
    """Run a single forward pass for one prompt; return per-layer statistics.

    Returns
    -------
    result : dict with keys
        prompt_id, category, prompt,
        l0               [n_layers] int
        attn_entropy     [n_layers, n_heads] float | None (None if not collected)
    extras : dict with keys
        residuals        np.ndarray [n_layers, d_model]  | None
        active_features  list[list[int]] [n_layers]      | None
        attn_weights     dict[int, np.ndarray [n_heads, T, T]] | None
    """
    tokenizer = loaded.tokenizer
    device    = loaded.W_U.device

    # Tokenise — always Format B (no chat template) for confound control
    current_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    T = current_ids.shape[1]

    # ── nnsight single forward pass ───────────────────────────────────────────
    residual_saves  = []
    mlp_input_saves = []
    attn_saves      = {}   # layer_index → save proxy

    attn_save_set = set(cfg.attn_save_layers)

    # Attempt to pass output_attentions=True so self_attn.output is a tuple.
    # If the circuit_tracer version doesn't forward this kwarg, the trace will
    # succeed but self_attn.output will NOT be a tuple — we handle that below.
    trace_kwargs = {}
    if cfg.collect_attention:
        trace_kwargs["output_attentions"] = True

    try:
        with loaded.model.trace(current_ids, **trace_kwargs):
            for i in range(N_LAYERS):
                # Register saves in forward-pass order within each layer:
                # self_attn → pre_feedforward_layernorm → layer output.
                # nnsight requires registration order to match execution order.
                if cfg.collect_attention:
                    # self_attn runs first within each layer.
                    # self_attn.output is (hidden_states, attn_weights, [past_kv])
                    # when output_attentions=True with eager attention.
                    # Index [1] selects the attention weight matrix [B, H, T, T].
                    attn_saves[i] = nnsave(
                        loaded.model.language_model.layers[i].self_attn.output[1]
                    )
                if cfg.collect_features:
                    mlp_input_saves.append(
                        nnsave(loaded.model.language_model.layers[i].pre_feedforward_layernorm.output)
                    )
                if cfg.collect_residuals:
                    residual_saves.append(
                        nnsave(loaded.model.language_model.layers[i].output[0])
                    )
        attn_trace_ok = True
    except Exception as e:
        # Fallback: re-run without attention collection
        print(f"    [WARN] Attention collection failed for {prompt_id}: {e}")
        print(f"    [WARN] Retrying without output_attentions=True ...")
        attn_trace_ok = False
        attn_saves = {}
        # Reset lists — may have partial entries from the failed attempt above.
        residual_saves = []
        mlp_input_saves = []
        with loaded.model.trace(current_ids):
            for i in range(N_LAYERS):
                # Same forward-pass order: pre_feedforward_layernorm before layer output.
                if cfg.collect_features:
                    mlp_input_saves.append(
                        nnsave(loaded.model.language_model.layers[i].pre_feedforward_layernorm.output)
                    )
                if cfg.collect_residuals:
                    residual_saves.append(
                        nnsave(loaded.model.language_model.layers[i].output[0])
                    )

    # ── materialise residuals ─────────────────────────────────────────────────
    residuals_np: Optional[np.ndarray] = None
    if cfg.collect_residuals and residual_saves:
        # Each residual_saves[i] has shape [B, T, d_model] = [1, T, d_model]
        # We take the last token position (-1) for all layers.
        vecs = [r[0, -1, :].float().cpu().numpy() for r in residual_saves]
        residuals_np = np.array(vecs, dtype=np.float32)  # [n_layers, d_model]

    # ── materialise active features + L0 ─────────────────────────────────────
    active_features: Optional[list] = None
    l0_list: list[int] = []
    if cfg.collect_features and mlp_input_saves:
        mlp_inputs = [m[0, -1, :].float() for m in mlp_input_saves]
        active_features = []
        with torch.inference_mode():
            for i in range(N_LAYERS):
                tc = loaded.transcoder_list[i]
                x  = mlp_inputs[i].unsqueeze(0).to(
                    device=tc.b_enc.device, dtype=tc.b_enc.dtype
                )
                acts        = tc.encode(x)
                active_idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
                active_features.append(active_idxs)
                l0_list.append(len(active_idxs))
    else:
        l0_list = [0] * N_LAYERS

    # ── materialise attention ──────────────────────────────────────────────────
    attn_entropy_2d: Optional[list] = None   # [n_layers, n_heads]
    attn_weights_np: Optional[dict] = None   # layer → [n_heads, T, T]

    if cfg.collect_attention and attn_trace_ok and attn_saves:
        attn_entropy_2d = []
        attn_weights_np = {}

        for i in range(N_LAYERS):
            attn_proxy = attn_saves.get(i)
            if attn_proxy is None:
                attn_entropy_2d.append(None)
                continue

            try:
                # attn_proxy shape: [B, n_heads, T, T] = [1, 8, T, T]
                attn_mat = attn_proxy[0].float().cpu()  # [n_heads, T, T]
                n_heads  = attn_mat.shape[0]

                # Entropy of last token's attention row over all key positions
                last_row = attn_mat[:, -1, :]  # [n_heads, T]
                entropies = [_attn_entropy(last_row[h]) for h in range(n_heads)]
                attn_entropy_2d.append(entropies)   # [n_heads] floats

                # Save full matrix only for designated layers
                if i in attn_save_set:
                    attn_weights_np[i] = attn_mat.numpy()  # [n_heads, T, T]

            except Exception as e:
                print(f"    [WARN] Layer {i} attention materialisation failed: {e}")
                attn_entropy_2d.append(None)

    result = {
        "prompt_id": prompt_id,
        "category":  category,
        "prompt":    prompt,
        "seq_len":   T,
        "l0":        l0_list,
        "attn_entropy": attn_entropy_2d,   # None or [n_layers][n_heads]
    }
    extras = {
        "residuals":      residuals_np,
        "active_features": active_features,
        "attn_weights":    attn_weights_np,
    }
    return result, extras


# ── batch collection ──────────────────────────────────────────────────────────

def _build_flat_items(prompts_dict: dict) -> list:
    """Flatten nested {split: {source: [prompts]}} into [(id, cat, prompt)]."""
    items = []
    for category, subcats in prompts_dict.items():
        for subcat, prompt_list in subcats.items():
            for idx, prompt in enumerate(prompt_list):
                items.append((f"{subcat}_{idx}", category, prompt))
    return items


def _pack_active_features(active_features: list) -> np.ndarray:
    """Pack list[list[int]] into object-array [n_layers] of int32 arrays."""
    arr = np.empty(N_LAYERS, dtype=object)
    for layer_i, idxs in enumerate(active_features):
        arr[layer_i] = np.array(idxs, dtype=np.int32)
    return arr


def _collect_worker(gpu_id: int, prompt_items: list, cfg: "Exp4Config") -> tuple:
    """Worker: load model on one GPU, process a chunk of prompts."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + getattr(cfg, "gpu_offset", 0))
        cfg.device = "cuda"
        from src.poc.exp4.model import load_model
        loaded = load_model(cfg)

        results: list[dict] = []
        all_residuals: dict = {}
        all_features: dict = {}
        all_attn_weights: dict = {}

        for done, (prompt_id, category, prompt) in enumerate(prompt_items):
            print(f"  [GPU {gpu_id}] [{done + 1}/{len(prompt_items)}] {prompt_id}", flush=True)
            result, extras = collect_prompt(prompt_id, category, prompt, loaded, cfg)
            results.append(result)

            if extras["residuals"] is not None:
                all_residuals[prompt_id] = extras["residuals"]

            if extras["active_features"] is not None:
                all_features[prompt_id] = _pack_active_features(extras["active_features"])

            if extras["attn_weights"] is not None:
                for layer_i, mat in extras["attn_weights"].items():
                    all_attn_weights[f"{prompt_id}_{layer_i}"] = mat

        return results, all_residuals, all_features, all_attn_weights
    except Exception as e:
        # NNsightException and similar custom exceptions are not picklable,
        # which causes concurrent.futures to crash when sending the error back.
        # Re-raise as plain RuntimeError so multiprocessing can serialise it.
        raise RuntimeError(f"[GPU {gpu_id}] {type(e).__name__}: {e}") from None


def collect_all(
    loaded,   # LoadedModel | None — ignored when n_gpus > 1
    cfg: Exp4Config,
    prompts_dict: dict,
) -> tuple[list[dict], dict, dict, dict]:
    """Collect exp4 data for all prompts.

    Returns
    -------
    results         : list[dict]   — per-prompt metadata + l0 + attn_entropy
    all_residuals   : dict[str, np.ndarray [n_layers, d_model]]
    all_features    : dict[str, object-array [n_layers] of int32 arrays]
    all_attn_weights: dict[str, np.ndarray [n_heads, T, T]]  (keyed "pid_layer")
    """
    items = _build_flat_items(prompts_dict)
    total = len(items)

    # ── single-GPU path ───────────────────────────────────────────────────────
    if cfg.n_gpus <= 1:
        results: list[dict] = []
        all_residuals: dict = {}
        all_features: dict = {}
        all_attn_weights: dict = {}

        for done, (prompt_id, category, prompt) in enumerate(items):
            print(f"  [{done + 1}/{total}] {prompt_id}: '{prompt[:55]}'")
            result, extras = collect_prompt(prompt_id, category, prompt, loaded, cfg)
            results.append(result)

            if extras["residuals"] is not None:
                all_residuals[prompt_id] = extras["residuals"]

            if extras["active_features"] is not None:
                all_features[prompt_id] = _pack_active_features(extras["active_features"])

            if extras["attn_weights"] is not None:
                for layer_i, mat in extras["attn_weights"].items():
                    all_attn_weights[f"{prompt_id}_{layer_i}"] = mat

        return results, all_residuals, all_features, all_attn_weights

    # ── multi-GPU path ────────────────────────────────────────────────────────
    import multiprocessing as mp
    n_gpus = min(cfg.n_gpus, torch.cuda.device_count())
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    chunks = [items[i::n_gpus] for i in range(n_gpus)]
    print(f"  Distributing {total} prompts across {n_gpus} GPUs (offset={cfg.gpu_offset})")

    ctx = mp.get_context("spawn")
    all_results: list[dict] = []
    all_residuals_m: dict = {}
    all_features_m: dict = {}
    all_attn_weights_m: dict = {}

    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as pool:
        futures = {pool.submit(_collect_worker, gpu_id, chunks[gpu_id], cfg): gpu_id
                   for gpu_id in range(n_gpus)}
        for future in as_completed(futures):
            r, res, feat, attn = future.result()
            all_results.extend(r)
            all_residuals_m.update(res)
            all_features_m.update(feat)
            all_attn_weights_m.update(attn)

    return all_results, all_residuals_m, all_features_m, all_attn_weights_m


# ── serialisation ──────────────────────────────────────────────────────────────

def _sanitise_for_json(obj):
    """Recursively replace non-finite floats with None for valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    return obj


def save_results(
    results: list[dict],
    all_residuals: dict,
    all_features:  dict,
    all_attn_weights: dict,
    cfg: Exp4Config,
) -> None:
    """Write all four output files."""
    out_dir = Path(cfg.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON: metadata + l0 + attn_entropy
    with open(cfg.output_path, "w") as f:
        json.dump(_sanitise_for_json(results), f, indent=2)
    print(f"  Saved {len(results)} results → {cfg.output_path}")

    # Residuals .npz
    if all_residuals:
        np.savez_compressed(cfg.residuals_path, **all_residuals)
        print(f"  Saved residuals ({len(all_residuals)} prompts) → {cfg.residuals_path}")

    # Active features .npz
    if all_features:
        np.savez_compressed(cfg.features_path, **all_features)
        print(f"  Saved active features → {cfg.features_path}")

    # Attention weights .npz
    if all_attn_weights:
        np.savez_compressed(cfg.attn_path, **all_attn_weights)
        print(f"  Saved attention weights ({len(all_attn_weights)} entries) → {cfg.attn_path}")
    else:
        print(f"  No attention weights collected (collect_attention={cfg.collect_attention})")
