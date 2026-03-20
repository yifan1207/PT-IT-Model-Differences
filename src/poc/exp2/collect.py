"""
Exp2 data collection: IC vs OOC vs R mechanistic comparison.

For each prompt, generates up to max_gen_tokens tokens via greedy decoding and
captures per-layer, per-generation-step statistics:

  residual_norm[step][layer]       : ||h_i||₂ — residual stream norm after full layer i
  layer_delta_norm[step][layer]    : ||h_i - h_{i-1}||₂ — each layer's update magnitude
  l0[step][layer]                  : number of active transcoder features
  output_entropy[step]             : H(softmax(logits_real)) — generation uncertainty
  logit_lens_entropy[step][layer]  : H(softmax(h_i @ W_U)_real) — early-exit entropy
  active_features[step][layer]     : active feature indices (saved to .npz, not JSON)

Implementation notes on nnsight hook paths (Gemma3ForConditionalGeneration):
  - google/gemma-3-4b-pt loads as Gemma3ForConditionalGeneration, NOT Gemma3ForCausalLM.
    The nnsight path to transformer layers is therefore:
      model.language_model.layers[i]          (NOT model.model.layers[i])
    Confirmed by circuit_tracer/utils/tl_nnsight_mapping.py: gemma_3_conditional_mapping.

  - Gemma3DecoderLayer.forward() returns a tuple (hidden_states,) or
    (hidden_states, self_attn_weights) when output_attentions=True.
    In nnsight 0.6, .output IS the real tuple (not a proxy) — tuples cannot have
    .save() patched onto them. The correct pattern is .output[0].save() to unpack
    to the hidden_states tensor first, then save. After the trace, .value is [B,T,D]
    and [0, -1, :] gives batch 0, last token, all d_model.

  - pre_feedforward_layernorm.forward() returns a plain tensor [B, T, D].
    Save with .output.save(); after trace: .value[0, -1, :] indexes directly.
    This is the MLP input (= 'mlp.hook_in' in TransformerLens naming) AND the point
    used by circuit-tracer for feature activation extraction.

  - lm_head is an nn.Linear at the top level of Gemma3ForConditionalGeneration.
    .output is a plain tensor [B, T, vocab_size]; [0, -1, :] = last token logits.

  - Greedy decoding MUST mask <unusedXXXX> tokens before argmax.
    Gemma 3's 262k vocab has many unused slots with high-norm W_U columns from
    random initialization; without masking, argmax frequently lands on these
    placeholder tokens and produces nonsensical output.

  - L0 post-trace: call transcoder_list[i].encode(x) where x = [1, d_model].
    encode() returns [1, d_transcoder]; nonzero count = L0.

Speed notes:
  - torch.inference_mode() is NOT used to wrap the trace (it caused nnsight proxy
    issues). Instead, inference_mode is applied only for post-trace computations
    (L0 encoding, logit-lens entropy) where no proxy objects are involved.
    It is strictly faster than torch.no_grad() because it also disables view
    tracking. Since exp2 never backpropagates, inference_mode is always safe for
    those post-trace steps.
  - scan=False / validate=False are silently no-ops in nnsight 0.6.2 — those
    kwargs pass through to HuggingFace's **kwargs and are ignored. Removed to
    avoid misleading documentation.
"""
import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from nnsight import save as nnsave

from src.poc.shared.model import LoadedModel
from src.poc.shared.constants import N_LAYERS
from src.poc.exp2.config import Exp2Config


# ─── entropy helper ───────────────────────────────────────────────────────────

def _entropy_from_logits(logits: torch.Tensor,
                         mask: Optional[torch.Tensor] = None) -> float:
    """Shannon entropy (nats) of softmax(logits[mask])."""
    if mask is not None:
        logits = logits[mask]
    probs = torch.softmax(logits.float(), dim=-1)
    return -(probs * torch.log(probs + 1e-12)).sum().item()


# ─── per-prompt collection ─────────────────────────────────────────────────────

def collect_prompt(
    prompt_id: str,
    category: str,
    prompt: str,
    loaded: LoadedModel,
    cfg: Exp2Config,
) -> dict:
    """Autoregressively generate up to cfg.max_gen_tokens tokens for one prompt.

    Returns a result dict. The key 'active_features' contains ragged lists
    (list[step][layer] of int lists) that should be saved to .npz separately
    and stripped before JSON serialization.
    """
    tokenizer = loaded.tokenizer
    W_U = loaded.W_U                    # [d_model, vocab_size] float32
    real_mask = loaded.real_token_mask  # [vocab_size] bool
    device = W_U.device

    # ── tokenise prompt ───────────────────────────────────────────────────────
    if cfg.is_instruction_tuned:
        # IT model requires the Gemma chat template.
        # Raw completion-style prompts ("The capital of France is") don't work well
        # as bare user messages for an instruction-tuned model. We wrap them with a
        # brief instruction so the model treats it as a sentence-completion task.
        user_message = f"Complete the following sentence: {prompt}"
        # apply_chat_template(tokenize=True) handles BOS and all special tokens
        # correctly in one shot — avoids double-BOS that tokenizer.encode() would add.
        # Produces: <bos><start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n
        current_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_message}],
            return_tensors="pt",
            add_generation_prompt=True,   # appends <start_of_turn>model\n
        ).to(device)
    else:
        # PT model: plain text, tokenizer adds BOS automatically.
        current_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # ── stop tokens ───────────────────────────────────────────────────────────
    # PT: stop only at <eos>.
    # IT: stop at <eos> OR <end_of_turn> (token 107 in the Gemma tokenizer).
    #     The model emits <end_of_turn> to signal end of its assistant turn;
    #     this is the natural stopping point for IT generation.
    eos_token_id = tokenizer.eos_token_id
    stop_token_ids: set[int] = {eos_token_id}
    if cfg.is_instruction_tuned:
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if end_of_turn_id is not None and end_of_turn_id != tokenizer.unk_token_id:
            stop_token_ids.add(end_of_turn_id)

    generated_tokens: list[dict] = []
    residual_norm: list[list[float]] = []        # [step][layer]
    layer_delta_norm: list[list[float]] = []     # [step][layer]
    layer_delta_cosine: list[list[float]] = []   # [step][layer]  cos(delta_i, h_{i-1})
    l0: list[list[int]] = []                     # [step][layer]
    output_entropy: list[float] = []             # [step]
    logit_lens_entropy: list[list[float]] = []   # [step][layer]
    active_features: list[list[list[int]]] = []  # [step][layer]

    for step in range(cfg.max_gen_tokens):
        # ── forward pass ──────────────────────────────────────────────────────
        # The trace runs without inference_mode: wrapping the trace in
        # inference_mode caused nnsight proxy issues (proxies need grad tracking
        # to propagate correctly). inference_mode is used only for post-trace
        # computations (L0 encoding, logit-lens entropy) below.
        with loaded.model.trace(current_ids):
            # nnsight 0.6 scoping rule: only variables explicitly passed to nnsave()
            # are pushed back to the caller's frame after the trace exits (see base.py
            # push(): filtered to id(v) in Globals.saves when stack==1).
            # So we must nnsave() the list containers themselves, not just their elements.
            #
            # Forward-pass order within each Gemma3DecoderLayer:
            #   pre_feedforward_layernorm  →  MLP/transcoder  →  layer output
            # So MLP input must be registered before layer output per layer.
            residual_saves = []
            mlp_input_saves = []
            for i in range(N_LAYERS):
                mlp_input_saves.append(
                    nnsave(loaded.model.language_model.layers[i].pre_feedforward_layernorm.output)
                )
                residual_saves.append(
                    nnsave(loaded.model.language_model.layers[i].output[0])
                )
            logits_save = nnsave(loaded.model.lm_head.output)
            # Register the list containers so nnsight pushes them back to outer scope.
            nnsave(residual_saves)
            nnsave(mlp_input_saves)

        # ── materialise ───────────────────────────────────────────────────────
        # nnsave() returns the tensor directly — no .value, just index.
        # residual_saves[i] = [B, T, D];  mlp_input_saves[i] = [B, T, D];  logits_save = [B, T, V]
        residuals = [r[0, -1, :].float() for r in residual_saves]
        mlp_inputs = [m[0, -1, :].float() for m in mlp_input_saves]
        logits = logits_save[0, -1, :].float()

        # ── residual stream norms ─────────────────────────────────────────────
        step_res_norms = [h.norm().item() for h in residuals]
        residual_norm.append(step_res_norms)

        # ── layer delta norms + cosine(delta_i, h_{i-1}) ─────────────────────
        # For each layer i:
        #   delta_i = h_i - h_{i-1}   (what this layer added to the stream)
        #   cos_i   = dot(delta_i, h_{i-1}) / (||delta_i|| * ||h_{i-1}||)
        # cos_i > 0 → layer reinforces the existing stream direction
        # cos_i ≈ 0 → layer adds orthogonal / new information
        # cos_i < 0 → layer suppresses / contradicts the existing stream
        # Layer 0: no h_{i-1} available (would need embedding layer), store nan.
        step_delta = [residuals[0].norm().item()]
        step_cos = [float("nan")]
        for i in range(1, N_LAYERS):
            delta = residuals[i] - residuals[i - 1]
            step_delta.append(delta.norm().item())
            denom = delta.norm() * residuals[i - 1].norm()
            cos = (torch.dot(delta, residuals[i - 1]) / denom).item() if denom > 0 else float("nan")
            step_cos.append(cos)
        layer_delta_norm.append(step_delta)
        layer_delta_cosine.append(step_cos)

        # ── L0 and active feature indices ─────────────────────────────────────
        step_l0: list[int] = []
        step_active: list[list[int]] = []
        with torch.inference_mode():
            for i in range(N_LAYERS):
                tc = loaded.transcoder_list[i]
                # encode() expects [..., d_model] and returns [..., d_transcoder]
                x = mlp_inputs[i].unsqueeze(0).to(device=tc.b_enc.device, dtype=tc.b_enc.dtype)  # [1, d_model]
                acts = tc.encode(x)                                  # [1, d_transcoder]
                active_idxs = acts[0].nonzero(as_tuple=False).squeeze(1).tolist()
                step_l0.append(len(active_idxs))
                step_active.append(active_idxs)
        l0.append(step_l0)
        active_features.append(step_active)

        # ── output entropy ────────────────────────────────────────────────────
        output_entropy.append(_entropy_from_logits(logits, mask=real_mask))

        # ── logit-lens entropy per layer ──────────────────────────────────────
        step_ll_ent: list[float] = []
        with torch.inference_mode():
            for i in range(N_LAYERS):
                h = residuals[i].to(device=W_U.device)
                h_normed = loaded.model.language_model.norm(h).float()
                lens_logits = h_normed @ W_U   # [vocab_size]
                step_ll_ent.append(_entropy_from_logits(lens_logits, mask=real_mask))
        logit_lens_entropy.append(step_ll_ent)

        # ── greedy next token ─────────────────────────────────────────────────
        # MUST mask <unusedXXXX> tokens before argmax: Gemma 3's 262k vocab has
        # many placeholder slots with high logits from random init. Without masking,
        # argmax frequently lands on these and the generation is nonsensical.
        masked_logits = logits.clone()
        masked_logits[~real_mask] = float("-inf")
        next_token_id = int(masked_logits.argmax().item())

        # Stop tokens: break before recording — never appear in generated_tokens.
        if next_token_id in stop_token_ids:
            break

        # Skip ALL other special tokens (BOS, PAD, etc.) from the recorded token list.
        # They should never appear mid-generation but if they do we don't want to
        # analyse them. We still append to current_ids so the context stays correct.
        if next_token_id not in tokenizer.all_special_ids:
            next_token_str = tokenizer.decode([next_token_id])
            generated_tokens.append({"token_id": next_token_id, "token_str": next_token_str})

        current_ids = torch.cat(
            [current_ids, torch.tensor([[next_token_id]], device=current_ids.device)],
            dim=1,
        )

    return {
        "prompt_id": prompt_id,
        "category": category,
        "prompt": prompt,
        "generated_tokens": generated_tokens,
        "residual_norm": residual_norm,
        "layer_delta_norm": layer_delta_norm,
        "layer_delta_cosine": layer_delta_cosine,
        "l0": l0,
        "output_entropy": output_entropy,
        "logit_lens_entropy": logit_lens_entropy,
        "active_features": active_features,  # stripped before JSON — save to .npz
    }


# ─── batch collection ──────────────────────────────────────────────────────────

def _pack_active_features(af: list, n_steps: int) -> np.ndarray:
    """Convert active_features list[step][layer] → object ndarray [n_steps, N_LAYERS]."""
    arr = np.empty((n_steps, N_LAYERS), dtype=object)
    for s in range(n_steps):
        for layer in range(N_LAYERS):
            arr[s, layer] = np.array(af[s][layer], dtype=np.int32)
    return arr


def _collect_worker(
    gpu_id: int,
    prompt_items: list,   # list of (prompt_id, category, prompt)
    cfg: "Exp2Config",
) -> tuple:
    """Worker function for one GPU. Runs in a spawned subprocess.

    Loads the model independently on cuda:{gpu_id}, processes its chunk of
    prompts, and returns (results, npz_data) — both picklable.
    """
    # Bind this process to exactly one GPU so circuit-tracer / nnsight see cuda:0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cfg.device = "cuda"   # within this process, 'cuda' == physical GPU gpu_id

    from src.poc.shared.model import load_model   # local import — spawned process
    loaded = load_model(cfg)

    results: list[dict] = []
    npz_data: dict = {}
    total = len(prompt_items)

    for done, (prompt_id, category, prompt) in enumerate(prompt_items):
        print(f"  [GPU {gpu_id}] [{done + 1}/{total}] {prompt_id}: '{prompt[:50]}'",
              flush=True)
        result = collect_prompt(
            prompt_id=prompt_id,
            category=category,
            prompt=prompt,
            loaded=loaded,
            cfg=cfg,
        )
        af = result.pop("active_features")
        npz_data[prompt_id] = _pack_active_features(af, len(af))
        results.append(result)

    return results, npz_data


def _build_flat_items(prompts_dict: dict) -> list:
    """Flatten prompts_dict → list of (prompt_id, category, prompt)."""
    items = []
    for category, subcats in prompts_dict.items():
        for subcat, prompt_list in subcats.items():
            for idx, prompt in enumerate(prompt_list):
                items.append((f"{subcat}_{idx}", category, prompt))
    return items


def collect_all(loaded: "LoadedModel | None", cfg: "Exp2Config", prompts_dict: dict) -> tuple:
    """Collect data for all prompts, optionally across multiple GPUs.

    When cfg.n_gpus > 1: spawns one worker process per GPU, each loads the
    model independently. `loaded` is unused and may be None.
    When cfg.n_gpus == 1: uses the already-loaded `loaded` model in-process.

    Returns (results, npz_data) — results is a list of dicts without
    active_features; npz_data is a dict keyed by prompt_id.
    """
    items = _build_flat_items(prompts_dict)
    total = len(items)

    if cfg.n_gpus <= 1:
        # ── single-GPU in-process path ────────────────────────────────
        results: list[dict] = []
        npz_data: dict = {}
        for done, (prompt_id, category, prompt) in enumerate(items):
            print(f"  [{done + 1}/{total}] {prompt_id}: '{prompt[:60]}'")
            result = collect_prompt(
                prompt_id=prompt_id,
                category=category,
                prompt=prompt,
                loaded=loaded,
                cfg=cfg,
            )
            af = result.pop("active_features")
            npz_data[prompt_id] = _pack_active_features(af, len(af))
            results.append(result)
        return results, npz_data

    # ── multi-GPU path ────────────────────────────────────────────────
    import multiprocessing as mp

    n_gpus = cfg.n_gpus
    available = torch.cuda.device_count()
    if n_gpus > available:
        print(f"  Warning: n_gpus={n_gpus} but only {available} GPU(s) available — clamping to {available}")
        n_gpus = available
    if n_gpus == 0:
        raise RuntimeError("No CUDA GPUs available. Set cfg.n_gpus=1 to run on CPU or check your CUDA setup.")
    # Round-robin split: GPU i gets items[i], items[i+n], items[i+2n], ...
    chunks = [items[i::n_gpus] for i in range(n_gpus)]
    sizes = [len(c) for c in chunks]
    print(f"  Distributing {total} prompts across {n_gpus} GPUs: {sizes} each")

    ctx = mp.get_context("spawn")
    all_results: list[dict] = []
    all_npz: dict = {}

    with ProcessPoolExecutor(max_workers=n_gpus, mp_context=ctx) as pool:
        futures = {
            pool.submit(_collect_worker, gpu_id, chunks[gpu_id], cfg): gpu_id
            for gpu_id in range(n_gpus)
        }
        for future in as_completed(futures):
            gpu_id = futures[future]
            r, npz = future.result()   # raises if worker crashed
            print(f"  GPU {gpu_id} finished: {len(r)} prompts", flush=True)
            all_results.extend(r)
            all_npz.update(npz)

    return all_results, all_npz


def save_results(results: list[dict], npz_data: dict, cfg: Exp2Config) -> None:
    """Persist results JSON and active_features .npz to the run directory."""
    out_path = Path(cfg.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results → {out_path}")

    npz_path = out_path.with_suffix(".npz")
    np.savez_compressed(str(npz_path), **npz_data)
    print(f"  Saved active_features → {npz_path}")
