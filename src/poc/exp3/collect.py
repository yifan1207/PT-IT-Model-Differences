"""
Exp3 data collection: extended from exp2, adds corrective-stage evidence.

New quantities collected per prompt (in addition to all exp2 quantities):

  1. next_token_rank[step][layer]      (collect_emergence=True)
     Rank of the actually-generated token in the logit-lens distribution at
     layer i.  Rank 1 = already the top prediction at that layer.
     Formula: rank_i = #{k : lens_logits_i[k] > lens_logits_i[next_token_id]} + 1
     over real tokens only.

  2. next_token_prob[step][layer]      (collect_emergence=True)
     Probability of the generated token under the logit-lens softmax at layer i.
     Formula: softmax(lens_logits_i[real_mask])[index_of_next_token]

  3. kl_to_final[step][layer]          (collect_emergence=True)
     KL divergence from layer i's logit-lens distribution to the final layer's
     (layer 33) logit-lens distribution.
     Formula: KL( p_i ∥ p_33 ) = Σ p_i[k] * log(p_i[k] / p_33[k])
     over real tokens only.
     Interpretation: how far is each layer's "belief" from the final output?
     A sharp drop in KL at layer L = the model committed to its answer at L.

  4. logit_delta_contrib[step][layer]  (collect_attribution=True)
     How much layer i changed the prediction for the generated token.
     Formula: lens_logits_i[next_token_id] - lens_logits_{i-1}[next_token_id]
     Positive = layer i pushed toward the generated token (attraction).
     Negative = layer i pushed away (repulsion).
     Layer 0: NaN (no prior layer).
     NOTE: this uses the logit-lens difference (change in projected logit),
     not the raw residual delta.  The logit-lens projection is nonlinear
     (RMSNorm + W_U), so this is not simply delta_i @ W_U.

  5. transcoder_mse[step][layer]       (collect_transcoder_mse=True)
     Mean squared error of the transcoder's reconstruction of the MLP output.
     Formula: ||tc.forward(x_i) - mlp_actual_output_i||² / d_model
     x_i = pre_feedforward_layernorm.output  (same as exp2)
     mlp_actual_output_i = actual output of the MLP module at layer i
     REQUIRES a new nnsight hook — see TODO below.

Implementation notes
--------------------
Quantities 1–4 require NO new nnsight hooks.  They are computed in the
post-trace logit-lens loop from the residuals already captured in exp2.
Quantity 5 requires one new hook per layer for mlp.output.

The logit-lens loop is extended to compute all quantities in a single pass
over layers 0–33, keeping all lens_logits in a list before computing KL.
This adds one O(V_real) softmax and one O(V_real) sort per layer per step
when collect_emergence=True.

nnsight hook path for MLP output (TODO — verify on live model):
  loaded.model.language_model.layers[i].mlp.output
  Shape expected: [B, T, d_model] — the MLP's contribution before being
  added to the residual.  The exact submodule name depends on whether
  Gemma3ForConditionalGeneration uses `mlp` or a different attribute.
  Verify with: list(loaded.model.language_model.layers[0].named_children())
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
from src.poc.exp3.config import Exp3Config


# ── entropy helper (reused from exp2) ─────────────────────────────────────────

def _entropy_from_logits(logits: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> float:
    if mask is not None:
        logits = logits[mask]
    probs = torch.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum().item()


# ── KL divergence helper ───────────────────────────────────────────────────────

def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """KL(p ∥ q) for two probability distributions.  Both must sum to 1.
    Adds 1e-12 to q to avoid log(0).  Returns scalar in nats.
    """
    return (p * torch.log(p / (q + 1e-12) + 1e-12)).sum().item()


# ── per-prompt collection ──────────────────────────────────────────────────────

def collect_prompt(
    prompt_id: str,
    category: str,
    prompt: str,
    loaded: LoadedModel,
    cfg: Exp3Config,
    feature_summary: Optional[dict] = None,
) -> dict:
    """Autoregressively generate up to cfg.max_gen_tokens tokens for one prompt.

    Collects all exp2 quantities plus the new exp3 quantities controlled by
    the collect_* flags in cfg.

    Returns a result dict.  'active_features' is stripped before JSON
    serialisation (saved to .npz instead).
    """
    tokenizer = loaded.tokenizer
    W_U = loaded.W_U                    # [d_model, vocab_size]
    real_mask = loaded.real_token_mask  # [vocab_size] bool
    device = W_U.device

    # ── tokenise ──────────────────────────────────────────────────────────────
    # Primary analysis (apply_chat_template=False, default):
    #   Both PT and IT receive the prompt string verbatim (Format B:
    #   "Question: ...\nAnswer:").  Same input format → differences are due to
    #   weights only, not format.  This is the confound-controlled comparison.
    #
    # Secondary analysis (apply_chat_template=True, --chat-template flag):
    #   IT is wrapped with Gemma's chat template.  Use this to study IT-native
    #   behaviour or to compare chat-formatted vs neutral-formatted responses.
    if cfg.is_instruction_tuned and getattr(cfg, "apply_chat_template", False):
        user_message = f"Complete the following sentence: {prompt}"
        current_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
    else:
        current_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # ── stop tokens ───────────────────────────────────────────────────────────
    eos_token_id = tokenizer.eos_token_id
    stop_token_ids: set[int] = {eos_token_id}
    if cfg.is_instruction_tuned:
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id:
            stop_token_ids.add(end_of_turn_id)

    # ── result accumulators ───────────────────────────────────────────────────
    # exp2 quantities (always collected)
    generated_tokens: list[dict] = []
    residual_norm:      list[list[float]] = []
    layer_delta_norm:   list[list[float]] = []
    layer_delta_cosine: list[list[float]] = []
    l0:                 list[list[int]]   = []
    output_entropy:     list[float]       = []
    logit_lens_entropy: list[list[float]] = []
    active_features:    list[list[list[int]]] = []

    # exp3 quantities (conditional on flags)
    next_token_rank:      list[list[int]]   = []   # collect_emergence
    next_token_prob:      list[list[float]] = []   # collect_emergence
    kl_to_final:          list[list[float]] = []   # collect_emergence
    logit_delta_contrib:  list[list[float]] = []   # collect_attribution
    transcoder_mse:       list[list[float]] = []   # collect_transcoder_mse
    step_to_step_kl:      list[list[float]] = []
    top1_token_per_layer: list[list[int]] = []
    kl_adjacent_layer:    list[list[float]] = []
    top5_token_ids_per_layer: list[list[list[int]]] = []
    top5_token_probs_per_layer: list[list[list[float]]] = []
    prev_step_probs: Optional[list[torch.Tensor]] = None

    for step in range(cfg.max_gen_tokens):
        # ── forward pass (nnsight trace) ──────────────────────────────────────
        # Hook order within each Gemma3DecoderLayer (forward-pass order):
        #   pre_feedforward_layernorm  →  mlp  →  layer output
        # All new exp3 hooks must respect this order.
        with loaded.model.trace(current_ids):
            residual_saves  = []
            mlp_input_saves = []
            mlp_output_saves = []

            for i in range(N_LAYERS):
                mlp_input_saves.append(
                    nnsave(loaded.model.language_model.layers[i].pre_feedforward_layernorm.output)
                )
                
                if getattr(cfg, "collect_transcoder_mse", False):
                    mlp_output_saves.append(
                        nnsave(loaded.model.language_model.layers[i].mlp.output)
                    )
                
                if hasattr(cfg, "ablation_layers") and i in cfg.ablation_layers:
                    loaded.model.language_model.layers[i].mlp.output[:] = 0
                    
                residual_saves.append(
                    nnsave(loaded.model.language_model.layers[i].output[0])
                )
            logits_save = nnsave(loaded.model.lm_head.output)
            nnsave(residual_saves)
            nnsave(mlp_input_saves)
            if mlp_output_saves:
                nnsave(mlp_output_saves)

        # ── materialise ───────────────────────────────────────────────────────
        residuals  = [r[0, -1, :].float() for r in residual_saves]
        mlp_inputs = [m[0, -1, :].float() for m in mlp_input_saves]
        logits     = logits_save[0, -1, :].float()

        # ── exp2: residual norms ───────────────────────────────────────────────
        residual_norm.append([h.norm().item() for h in residuals])

        # ── exp2: layer delta norms + cosine ──────────────────────────────────
        step_delta = [residuals[0].norm().item()]
        step_cos   = [float("nan")]
        for i in range(1, N_LAYERS):
            delta = residuals[i] - residuals[i - 1]
            step_delta.append(delta.norm().item())
            denom = delta.norm() * residuals[i - 1].norm()
            cos = (torch.dot(delta, residuals[i - 1]) / denom).item() if denom > 0 else float("nan")
            step_cos.append(cos)
        layer_delta_norm.append(step_delta)
        layer_delta_cosine.append(step_cos)

        # ── exp2: L0 and active features ──────────────────────────────────────
        step_l0: list[int] = []
        step_active: list[list[int]] = []
        with torch.inference_mode():
            for i in range(N_LAYERS):
                tc = loaded.transcoder_list[i]
                x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device, dtype=tc.b_enc.dtype)
                acts = tc.encode(x)
                active_idxs_t = acts[0].nonzero(as_tuple=False).squeeze(1)
                active_idxs = active_idxs_t.tolist()
                step_l0.append(len(active_idxs))
                step_active.append(active_idxs)
                if feature_summary is not None and active_idxs:
                    if feature_summary["count"][i] is None:
                        d_feat = int(acts.shape[-1])
                        feature_summary["count"][i] = np.zeros(d_feat, dtype=np.int64)
                        feature_summary["sum"][i] = np.zeros(d_feat, dtype=np.float32)
                    active_vals = acts[0][active_idxs_t].float().cpu().numpy()
                    np.add.at(feature_summary["count"][i], active_idxs, 1)
                    np.add.at(feature_summary["sum"][i], active_idxs, active_vals)
        l0.append(step_l0)
        active_features.append(step_active)

        # ── exp2: output entropy ──────────────────────────────────────────────
        output_entropy.append(_entropy_from_logits(logits, mask=real_mask))

        # ── exp2 + exp3: logit-lens pass ──────────────────────────────────────
        # We compute lens_logits for all layers in a list so that:
        #   - exp2 logit_lens_entropy is computed as before
        #   - exp3 rank/prob/KL can be derived in a second pass over the same list
        step_ll_ent: list[float] = []
        all_lens_logits: list[torch.Tensor] = []  # kept for exp3 quantities

        with torch.inference_mode():
            step_top1: list[int] = []
            step_adj_kl: list[float] = [float("nan")]
            step_step_kl: list[float] = [float("nan")] * N_LAYERS if prev_step_probs is None else []
            step_top5_ids: list[list[int]] = []
            step_top5_probs: list[list[float]] = []
            cur_step_probs: list[torch.Tensor] = []
            for i in range(N_LAYERS):
                h = residuals[i].to(device=W_U.device)
                h_normed = loaded.model.language_model.norm(h).float()
                lens_logits = h_normed @ W_U                  # [vocab_size]
                all_lens_logits.append(lens_logits)
                step_ll_ent.append(_entropy_from_logits(lens_logits, mask=real_mask))
                masked_ll = lens_logits.clone()
                masked_ll[~real_mask] = float("-inf")
                step_top1.append(int(masked_ll.argmax().item()))
                top5 = torch.topk(torch.softmax(masked_ll.float(), dim=-1), k=5)
                step_top5_ids.append([int(v) for v in top5.indices.tolist()])
                step_top5_probs.append([float(v) for v in top5.values.tolist()])
                p_i = torch.softmax(lens_logits[real_mask].float(), dim=-1)
                cur_step_probs.append(p_i.cpu())
                if i > 0:
                    p_prev_layer = torch.softmax(all_lens_logits[i - 1][real_mask].float(), dim=-1)
                    step_adj_kl.append(_kl_divergence(p_i, p_prev_layer))
                    if prev_step_probs is not None:
                        step_step_kl.append(_kl_divergence(p_i, prev_step_probs[i].to(p_i.device)))
        logit_lens_entropy.append(step_ll_ent)
        top1_token_per_layer.append(step_top1)
        kl_adjacent_layer.append(step_adj_kl)
        step_to_step_kl.append(step_step_kl)
        top5_token_ids_per_layer.append(step_top5_ids)
        top5_token_probs_per_layer.append(step_top5_probs)
        prev_step_probs = cur_step_probs

        # ── exp3: transcoder MSE ──────────────────────────────────────────────
        if cfg.collect_transcoder_mse:
            step_mse = []
            with torch.inference_mode():
                for i in range(N_LAYERS):
                    tc = loaded.transcoder_list[i]
                    x  = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device, dtype=tc.b_enc.dtype)
                    tc_out      = tc.forward(x)[0].float()
                    # mlp.output is returned, we take the active token (-1)
                    mlp_actual = mlp_output_saves[i][0, -1, :].float()
                    mse = ((tc_out - mlp_actual) ** 2).mean().item()
                    step_mse.append(mse)
            transcoder_mse.append(step_mse)

        # ── greedy next token ─────────────────────────────────────────────────
        masked_logits = logits.clone()
        masked_logits[~real_mask] = float("-inf")
        next_token_id = int(masked_logits.argmax().item())

        is_stop_token = next_token_id in stop_token_ids

        # ── exp3: emergence / rank / prob / KL — computed AFTER we know next_token_id
        # All derived from all_lens_logits computed above.  No new nnsight hooks.
        if cfg.collect_emergence:
            with torch.inference_mode():
                # Softmax distribution of final layer (layer 33) over real tokens.
                final_logits_real = all_lens_logits[-1][real_mask]
                p_final = torch.softmax(final_logits_real.float(), dim=-1)

                step_rank: list[int]   = []
                step_prob: list[float] = []
                step_kl:   list[float] = []

                for i in range(N_LAYERS):
                    lens_real = all_lens_logits[i][real_mask]   # [V_real]
                    p_i = torch.softmax(lens_real.float(), dim=-1)

                    # next_token_id index within the real-token subset.
                    # If next_token_id is not a real token (shouldn't happen after masking),
                    # fall back gracefully.
                    if real_mask[next_token_id]:
                        real_indices = real_mask.nonzero(as_tuple=False).squeeze(1)
                        token_pos_in_real = (real_indices == next_token_id).nonzero(as_tuple=False)
                        if token_pos_in_real.numel() > 0:
                            tok_idx = token_pos_in_real[0, 0].item()
                            tok_logit = lens_real[tok_idx].item()
                            rank = int((lens_real > tok_logit).sum().item()) + 1
                            prob = float(p_i[tok_idx].item())
                        else:
                            rank, prob = -1, float("nan")
                    else:
                        rank, prob = -1, float("nan")

                    step_rank.append(rank)
                    step_prob.append(prob)
                    step_kl.append(_kl_divergence(p_i, p_final))

                next_token_rank.append(step_rank)
                next_token_prob.append(step_prob)
                kl_to_final.append(step_kl)

        # ── exp3: logit delta contribution ────────────────────────────────────
        if cfg.collect_attribution:
            with torch.inference_mode():
                step_contrib: list[float] = [float("nan")]  # layer 0: no prior
                for i in range(1, N_LAYERS):
                    logit_i   = all_lens_logits[i][next_token_id].item()
                    logit_im1 = all_lens_logits[i - 1][next_token_id].item()
                    step_contrib.append(logit_i - logit_im1)
                logit_delta_contrib.append(step_contrib)

        # Record token (we include special tokens now because Exp3 wants to study EOS)
        next_token_str = tokenizer.decode([next_token_id])
        generated_tokens.append({"token_id": next_token_id, "token_str": next_token_str})

        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_token_id]], device=current_ids.device)],
            dim=1,
        )

        if is_stop_token:
            break

    result = {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "generated_tokens": generated_tokens,
        # exp2 quantities
        "residual_norm":      residual_norm,
        "layer_delta_norm":   layer_delta_norm,
        "layer_delta_cosine": layer_delta_cosine,
        "l0":                 l0,
        "output_entropy":     output_entropy,
        "logit_lens_entropy": logit_lens_entropy,
        "active_features":    active_features,   # stripped before JSON
        # exp3 quantities (empty lists if flag was False)
        "next_token_rank":     next_token_rank,
        "next_token_prob":     next_token_prob,
        "kl_to_final":         kl_to_final,
        "logit_delta_contrib": logit_delta_contrib,
        "transcoder_mse":      transcoder_mse,
        "step_to_step_kl":     step_to_step_kl,
        "top1_token_per_layer": top1_token_per_layer,
        "kl_adjacent_layer":   kl_adjacent_layer,
        "top5_token_ids_per_layer": top5_token_ids_per_layer,
        "top5_token_probs_per_layer": top5_token_probs_per_layer,
    }
    return result


# ── batch collection (mirrors exp2 pattern) ───────────────────────────────────

def _pack_active_features(af: list, n_steps: int) -> np.ndarray:
    arr = np.empty((n_steps, N_LAYERS), dtype=object)
    for s in range(n_steps):
        for layer in range(N_LAYERS):
            arr[s, layer] = np.array(af[s][layer], dtype=np.int32)
    return arr


def _collect_worker(gpu_id: int, prompt_items: list, cfg: "Exp3Config") -> tuple:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg.device = "cuda"
    from src.poc.shared.model import load_model
    loaded = load_model(cfg)
    results, npz_data = [], {}
    feature_summary = {
        "count": [None] * N_LAYERS,
        "sum": [None] * N_LAYERS,
    }
    for done, (prompt_id, category, prompt) in enumerate(prompt_items):
        print(f"  [GPU {gpu_id}] [{done + 1}/{len(prompt_items)}] {prompt_id}", flush=True)
        result = collect_prompt(prompt_id, category, prompt, loaded, cfg, feature_summary=feature_summary)
        af = result.pop("active_features")
        npz_data[prompt_id] = _pack_active_features(af, len(af))
        results.append(result)
    return results, npz_data, feature_summary


def _build_flat_items(prompts_dict: dict) -> list:
    items = []
    for category, subcats in prompts_dict.items():
        for subcat, prompt_list in subcats.items():
            for idx, prompt in enumerate(prompt_list):
                items.append((f"{subcat}_{idx}", category, prompt))
    return items


def collect_all(loaded, cfg: "Exp3Config", prompts_dict: dict) -> tuple:
    items = _build_flat_items(prompts_dict)
    total = len(items)
    feature_summary = {
        "count": [None] * N_LAYERS,
        "sum": [None] * N_LAYERS,
    }

    if cfg.n_gpus <= 1:
        results, npz_data = [], {}
        for done, (prompt_id, category, prompt) in enumerate(items):
            print(f"  [{done + 1}/{total}] {prompt_id}: '{prompt[:60]}'")
            result = collect_prompt(prompt_id, category, prompt, loaded, cfg, feature_summary=feature_summary)
            af = result.pop("active_features")
            npz_data[prompt_id] = _pack_active_features(af, len(af))
            results.append(result)
        return results, npz_data, feature_summary

    import multiprocessing as mp
    n_gpus = min(cfg.n_gpus, torch.cuda.device_count())
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available.")
    chunks = [items[i::n_gpus] for i in range(n_gpus)]
    print(f"  Distributing {total} prompts across {n_gpus} GPUs")

    ctx = mp.get_context("spawn")
    all_results, all_npz = [], {}
    def _merge_feature_summary(dst: dict, src: dict) -> None:
        for layer_i in range(N_LAYERS):
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
    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as pool:
        futures = {pool.submit(_collect_worker, gpu_id, chunks[gpu_id], cfg): gpu_id
                   for gpu_id in range(n_gpus)}
        for future in as_completed(futures):
            r, npz, fs = future.result()
            all_results.extend(r)
            all_npz.update(npz)
            _merge_feature_summary(feature_summary, fs)
    return all_results, all_npz, feature_summary


def _sanitise_for_json(obj):
    """Recursively replace float nan/inf with None so output is valid JSON.

    Python's json.dump writes float('nan') as the bare token NaN, which is
    not valid JSON.  layer_delta_cosine and logit_delta_contrib deliberately
    store NaN at layer 0; all others are incidental.  None round-trips as
    JSON null and downstream code should already guard for None/nan.
    """
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise_for_json(v) for v in obj]
    return obj


def save_results(results: list[dict], npz_data: dict, feature_summary: dict, cfg: Exp3Config) -> None:
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(_sanitise_for_json(results), f, indent=2)
    print(f"  Saved {len(results)} results → {out_path}")
    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(str(npz_path), **npz_data)
    print(f"  Saved active_features → {npz_path}")
    summary_payload = {}
    for layer_i in range(N_LAYERS):
        counts = feature_summary["count"][layer_i]
        sums = feature_summary["sum"][layer_i]
        if counts is None or sums is None:
            continue
        summary_payload[f"count_l{layer_i}"] = counts.astype(np.int64, copy=False)
        summary_payload[f"sum_l{layer_i}"] = sums.astype(np.float32, copy=False)
    if summary_payload:
        np.savez_compressed(cfg.feature_importance_path, **summary_payload)
        print(f"  Saved feature importance summary → {cfg.feature_importance_path}")
