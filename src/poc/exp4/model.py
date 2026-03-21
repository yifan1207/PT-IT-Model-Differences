"""
Model loading for Experiment 4 — extends shared/model.py with eager attention.

Exp4 needs attention weights for the attention-entropy analysis (E0a).
PyTorch's SDPA / FlashAttention backend does NOT materialise the [B, H, T, T]
attention weight matrix.  To capture attention weights we must:

  1. Load the model with attn_implementation="eager" (forces the non-flash path)
  2. Pass output_attentions=True at trace time (tells HF to return attn weights)

This file wraps the shared load_model logic and injects attn_implementation
when cfg.collect_attention is True.  The shared/model.py is unchanged so that
exp1, exp2, exp3 continue to work without eager attention overhead.

Important notes
---------------
- Eager attention adds ~30% memory overhead vs SDPA.  If you only need
  entropy (not full matrix divergence), consider computing entropy on the
  fly and skipping attn_weights.npz to save memory.
- The attn_implementation kwarg is forwarded to
  ReplacementModel.from_pretrained_and_transcoders → AutoModelForCausalLM.from_pretrained
  via **kwargs.  Verify this works with the pinned circuit-tracer commit
  (rev 26a976e) before a large run.
- Global attention layers in Gemma 3 4B: 5, 11, 17, 23, 29 (i % 6 == 5).
  All other layers use sliding-window local attention (window = 1024 tokens).
  Attention entropy for local layers is only meaningful over the attended
  positions (clamped to 1024); global layers have full-sequence entropy.
"""
import re
import torch
from pathlib import Path

from huggingface_hub import snapshot_download
from circuit_tracer import ReplacementModel
from circuit_tracer.transcoder.single_layer_transcoder import (
    load_gemma_scope_2_transcoder,
    TranscoderSet,
)

from src.poc.shared.model import LoadedModel
from src.poc.exp4.config import Exp4Config

_FEATURE_INPUT_HOOK  = "mlp.hook_in"
_FEATURE_OUTPUT_HOOK = "hook_mlp_out"
_N_LAYERS            = 34

# Gemma 3 4B: layers where i % 6 == 5 use global (full-sequence) attention.
# All others use sliding-window local attention (window = 1024 tokens).
GLOBAL_ATTN_LAYERS = frozenset(i for i in range(_N_LAYERS) if i % 6 == 5)


def _load_transcoder_set(cfg: Exp4Config) -> TranscoderSet:
    """Download and assemble all 34 Gemma Scope 2 transcoders.

    Identical to shared/model.py._load_transcoder_set but takes Exp4Config.
    """
    repo_id = cfg.transcoder_set
    variant  = cfg.transcoder_variant
    dtype    = torch.float32 if cfg.dtype_str == "float32" else torch.bfloat16
    device   = torch.device(cfg.device)

    pattern   = f"transcoder_all/layer_*_{variant}/params.safetensors"
    print(f"  Downloading transcoders ({variant}, {_N_LAYERS} layers) from {repo_id} ...")
    local_dir = snapshot_download(repo_id, allow_patterns=[pattern])

    transcoders = {}
    for layer in range(_N_LAYERS):
        path = (Path(local_dir) / "transcoder_all"
                / f"layer_{layer}_{variant}" / "params.safetensors")
        if not path.exists():
            raise FileNotFoundError(
                f"Transcoder not found: {path}\n"
                f"Check that '{variant}' exists for all 34 layers in {repo_id}/transcoder_all/"
            )
        transcoders[layer] = load_gemma_scope_2_transcoder(
            str(path), layer=layer, device=device, dtype=dtype,
            lazy_encoder=False, lazy_decoder=False,
        )

    scan = f"{repo_id}/transcoder_all/{variant}"
    return TranscoderSet(
        transcoders,
        feature_input_hook=_FEATURE_INPUT_HOOK,
        feature_output_hook=_FEATURE_OUTPUT_HOOK,
        scan=scan,
    )


def load_model(cfg: Exp4Config) -> LoadedModel:
    """Load the model for exp4.

    When cfg.collect_attention is True, loads with attn_implementation="eager"
    so that attention weight matrices are materialised (required for E0a).

    Returns a LoadedModel identical in structure to what shared/model.py returns,
    so all existing exp3 analysis helpers work unchanged.
    """
    dtype  = torch.float32 if cfg.dtype_str == "float32" else torch.bfloat16
    device = torch.device(cfg.device)

    transcoder_set  = _load_transcoder_set(cfg)
    transcoder_list = [transcoder_set[i] for i in range(len(transcoder_set))]

    extra_kwargs: dict = {}
    if cfg.collect_attention:
        extra_kwargs["attn_implementation"] = "eager"
        print("  Loading with attn_implementation='eager' (required for attention weights)")

    print(f"  Loading {cfg.model_name} via {cfg.backend} backend ...")
    model = ReplacementModel.from_pretrained_and_transcoders(
        model_name=cfg.model_name,
        transcoders=transcoder_set,
        backend=cfg.backend,
        device=device,
        dtype=dtype,
        **extra_kwargs,
    )
    model.eval()

    # W_U: [d_model, vocab_size] — same extraction as shared/model.py
    W_U = model.unembed_weight.detach().float().T.contiguous().to(device)

    tokenizer = model.tokenizer

    # Real-token mask: filter out <unusedXXXX> placeholders
    vocab_size     = W_U.shape[1]
    all_token_strs = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    _unused_re     = re.compile(r"^<unused\d+>$")
    real_token_mask = torch.tensor(
        [not _unused_re.match(t or "") for t in all_token_strs],
        dtype=torch.bool,
        device=device,
    )
    n_real = int(real_token_mask.sum().item())
    print(f"  Vocabulary: {vocab_size} total, {n_real} real tokens, "
          f"{vocab_size - n_real} <unusedXXXX> filtered")

    return LoadedModel(
        model=model,
        W_U=W_U,
        tokenizer=tokenizer,
        transcoder_list=transcoder_list,
        real_token_mask=real_token_mask,
    )
