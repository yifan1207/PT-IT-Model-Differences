"""
L9: Attention entropy divergence (IT − PT per layer).

Single forward pass per prompt with output_attentions=True and
attn_implementation="eager" (required to materialize [B, H, T, T] matrices).

For each layer ℓ and head h, computes:
  H(ℓ, h) = -Σ_k attn[h, -1, k] * log(attn[h, -1, k] + ε)

Expected result: max(IT − PT entropy) at ~1/3 depth (phase boundary).

Reuses: _attn_entropy() logic from src/poc/exp4/collect.py

Notes on sliding window attention (Gemma 3 local layers, Mistral):
  - Entropy is bounded by log(window_size) instead of log(T)
  - Normalize by log(effective_context) for fair cross-model comparison

Output: results/cross_model/{model}/{variant}/L9_attn_entropy.json
  {"model": str, "variant": str, "n_prompts": int,
   "mean_entropy": [[float]*n_heads]*n_layers,  # mean over prompts
   "sem_entropy":  [[float]*n_heads]*n_layers}
"""
from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.poc.cross_model.config import get_spec, MODEL_REGISTRY, model_id_for_variant
from src.poc.cross_model.adapters import get_adapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _attn_entropy(attn_row: torch.Tensor) -> float:
    """Entropy of one attention distribution over key positions.
    attn_row: [T_k] — already post-softmax probabilities.
    Returns entropy in nats.
    """
    probs = attn_row.clamp(min=0.0)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


# ── collection ────────────────────────────────────────────────────────────────

def collect_attn_entropy(
    records: list[dict],
    model,
    tokenizer,
    adapter,
    spec,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward pass with output_attentions=True; compute per-head entropy at last token.

    Returns:
      mean_entropy: [n_layers, n_heads] float32
      sem_entropy:  [n_layers, n_heads] float32
    """
    n_layers = spec.n_layers
    n_heads = spec.n_heads

    # accum[n_prompts, n_layers, n_heads]
    all_entropy: list[np.ndarray] = []

    for i, rec in enumerate(records):
        formats = rec.get("formats", {})
        prompt_text = formats.get("B") or formats.get("A") or rec.get("prompt", "")
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                out = model(input_ids, output_attentions=True)
        except Exception as e:
            log.warning("Forward pass failed for prompt %d: %s", i, e)
            continue

        if out.attentions is None:
            log.warning("output_attentions returned None — model may not support eager attn.")
            break

        entropy_mat = np.full((n_layers, n_heads), float("nan"), dtype=np.float32)
        for ℓ, attn in enumerate(out.attentions):
            if attn is None:
                continue
            # attn: [1, n_heads_q, T, T] — use last-token query row
            n_h = attn.shape[1]
            attn_row = attn[0, :, -1, :]  # [n_heads_q, T_k]
            for h in range(min(n_h, n_heads)):
                entropy_mat[ℓ, h] = _attn_entropy(attn_row[h])

        all_entropy.append(entropy_mat)

        if (i + 1) % 100 == 0:
            log.info("  %d/%d prompts", i + 1, len(records))

    if not all_entropy:
        log.error("No entropy data collected.")
        mean_entropy = np.full((n_layers, n_heads), float("nan"), dtype=np.float32)
        sem_entropy  = np.full((n_layers, n_heads), float("nan"), dtype=np.float32)
        return mean_entropy, sem_entropy

    stack = np.stack(all_entropy, axis=0)  # [N, n_layers, n_heads]
    mean_entropy = np.nanmean(stack, axis=0)
    sem_entropy  = np.nanstd(stack, axis=0) / np.sqrt(np.sum(~np.isnan(stack), axis=0).clip(min=1))
    return mean_entropy, sem_entropy


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="L9: Attention entropy divergence.")
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant", required=True, choices=["pt", "it"])
    parser.add_argument("--dataset", default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    spec = get_spec(args.model)
    adapter = get_adapter(args.model)
    model_id = model_id_for_variant(spec, args.variant)

    out_dir = Path(args.out_dir) if args.out_dir else spec.result_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "L9_attn_entropy.json"

    if out_path.exists():
        log.info("Output already exists: %s — skipping.", out_path)
        return

    log.info("=== L9 attention entropy: %s %s ===", spec.name, args.variant)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    device = torch.device(args.device)
    # MUST use eager attention to materialize attention weight matrices
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    model.eval()

    with open(args.dataset) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.n_eval_examples:
        records = records[:args.n_eval_examples]

    log.info("Collecting attention entropy for %d prompts...", len(records))
    mean_entropy, sem_entropy = collect_attn_entropy(records, model, tokenizer, adapter, spec, device)

    result = {
        "model": spec.name,
        "variant": args.variant,
        "n_prompts": len(records),
        "n_layers": spec.n_layers,
        "n_heads": spec.n_heads,
        "global_attn_layers": sorted(spec.global_attn_layers),
        "is_sliding_window": spec.is_sliding_window,
        "sliding_window_size": spec.sliding_window_size,
        "mean_entropy": mean_entropy.tolist(),
        "sem_entropy": sem_entropy.tolist(),
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved → %s", out_path)


if __name__ == "__main__":
    main()
