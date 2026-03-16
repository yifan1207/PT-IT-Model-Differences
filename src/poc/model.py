"""
Model loading for Gemma 3 4B PT + Gemma Scope 2 transcoders.

Key facts baked into this file:
  - google/gemma-3-4b-pt loads as Gemma3ForCausalLM (text-only causal LM).
    Do NOT use the instruct variant (-it) or the multimodal class
    (Gemma3ForConditionalGeneration) — we want raw next-token prediction.
  - google/gemma-scope-2-4b-pt has NO top-level config.yaml that circuit-tracer
    can auto-parse. We build the TranscoderSet manually.
  - 34 transformer layers (0–33). Each layer has transcoders at:
      transcoder_all/layer_{i}_{variant}/params.safetensors
  - Hook names (from gemma_3_mapping in tl_nnsight_mapping.py):
      feature_input_hook  = "mlp.hook_in"    → pre_feedforward_layernorm.output
      feature_output_hook = "hook_mlp_out"   → post_feedforward_layernorm.output
  - unembed_weight = lm_head.weight  shape [vocab_size, d_model]
    → W_U = unembed_weight.T          shape [d_model, vocab_size]
  - BOS token added automatically by tokenizer. Prompts are raw text.
    Next token has a leading space (SentencePiece convention, e.g. " Paris").
"""
import torch
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download
from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder.single_layer_transcoder import (
    load_gemma_scope_2_transcoder,
    TranscoderSet,
)

# Hook names for Gemma 3 — must match gemma_3_mapping in tl_nnsight_mapping.py
_FEATURE_INPUT_HOOK = "mlp.hook_in"    # pre_feedforward_layernorm output
_FEATURE_OUTPUT_HOOK = "hook_mlp_out"  # post_feedforward_layernorm output
_N_LAYERS = 34


@dataclass
class LoadedModel:
    model: ReplacementModel
    W_U: torch.Tensor  # [d_model, vocab_size] float32, on model device
    tokenizer: object  # HuggingFace tokenizer


def _load_transcoder_set(cfg) -> TranscoderSet:
    """Download and assemble all 34 Gemma Scope 2 transcoders into a TranscoderSet.

    Downloads only the `params.safetensors` files for the chosen variant
    (e.g. width_16k_l0_small_affine) across all 34 layers using a single
    snapshot_download call with a glob filter.
    """
    repo_id = cfg.transcoder_set  # "google/gemma-scope-2-4b-pt"
    variant = cfg.transcoder_variant  # e.g. "width_16k_l0_small_affine"
    dtype = torch.float32 if cfg.dtype_str == "float32" else torch.bfloat16
    device = torch.device(cfg.device)

    pattern = f"transcoder_all/layer_*_{variant}/params.safetensors"
    print(f"  Downloading transcoders ({variant}, {_N_LAYERS} layers) from {repo_id} ...")
    local_dir = snapshot_download(repo_id, allow_patterns=[pattern])

    transcoders = {}
    for layer in range(_N_LAYERS):
        path = Path(local_dir) / "transcoder_all" / f"layer_{layer}_{variant}" / "params.safetensors"
        if not path.exists():
            raise FileNotFoundError(
                f"Transcoder not found: {path}\n"
                f"Check that '{variant}' exists for all 34 layers in {repo_id}/transcoder_all/"
            )
        transcoders[layer] = load_gemma_scope_2_transcoder(
            str(path), layer=layer, device=device, dtype=dtype,
            lazy_encoder=False, lazy_decoder=False,
        )

    scan = f"{repo_id}//transcoder_all/{variant}"
    return TranscoderSet(
        transcoders,
        feature_input_hook=_FEATURE_INPUT_HOOK,
        feature_output_hook=_FEATURE_OUTPUT_HOOK,
        scan=scan,
    )


def load_model(cfg) -> "LoadedModel":
    dtype = torch.float32 if cfg.dtype_str == "float32" else torch.bfloat16
    device = torch.device(cfg.device)

    transcoder_set = _load_transcoder_set(cfg)

    print(f"  Loading {cfg.model_name} via {cfg.backend} backend ...")
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=cfg.model_name,
        transcoders=transcoder_set,
        backend=cfg.backend,
        device=device,
        dtype=dtype,
    )
    model.eval()

    # W_U: nnsight exposes lm_head.weight as [vocab_size, d_model] → transpose to [d_model, vocab_size]
    W_U = model.unembed_weight.detach().float().T.contiguous().to(device)

    # Tokenizer: nnsight LanguageModel exposes .tokenizer directly
    tokenizer = model.tokenizer

    return LoadedModel(model=model, W_U=W_U, tokenizer=tokenizer)


def get_token_id(loaded: LoadedModel, token_str: str) -> int:
    """Get single token id for a target string.

    Uses add_special_tokens=False so we get only the token(s) for token_str itself,
    without BOS. Raises AssertionError if token_str is multi-token.
    """
    ids = loaded.tokenizer.encode(token_str, add_special_tokens=False)
    assert len(ids) == 1, f"'{token_str}' tokenizes to {len(ids)} tokens {ids} — fix in config"
    return ids[0]
