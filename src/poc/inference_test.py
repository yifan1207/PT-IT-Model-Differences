"""
Inference smoke test: verify Gemma-3-4B-PT next-token predictions match our prompts.

Loads ONLY the base model (no transcoders, no circuit-tracer) to check:
  1. Each prompt's expected token is top-1 or near top-1.
  2. Every target token is a single token (multi-token = will be skipped in attribution).
  3. Prompt format is correct: raw sentence completion, no chat template.

Why google/gemma-3-4b-pt (base, pretrained)?
  We use the *pretrained* base model, not the instruction-tuned (-it) variant.
  Base models do raw next-token prediction: "The capital of France is" → " Paris".
  No system prompt, no chat template, no <start_of_turn> tokens.
  The tokenizer adds BOS automatically. Tokens in the middle of a sentence have
  a leading space (SentencePiece convention): " Paris", " 6", " cold", etc.

Usage:
    uv run python -m src.poc.inference_test               # CPU (slow)
    uv run python -m src.poc.inference_test --device mps
    uv run python -m src.poc.inference_test --device cuda
    uv run python -m src.poc.inference_test --topk 10     # show top-10 instead of top-5
"""
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.poc.exp1.config import PocConfig


def run_test(device: str, topk: int = 5) -> None:
    cfg = PocConfig()

    print(f"Model : {cfg.model_name}")
    print(f"Device: {device}")
    print("Format: raw sentence-completion (base model, NO chat template)\n")

    print("Loading tokenizer ...")
    # Use AutoTokenizer — adds BOS automatically, no chat template
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print("Loading model (this downloads ~8 GB on first run) ...")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, torch_dtype=dtype, device_map=device,
    )
    model.eval()
    print()

    # Sanity-check tokenization of a known example
    sample = " Paris"
    sample_ids = tokenizer.encode(sample, add_special_tokens=False)
    print(f"Tokenization check: {repr(sample)} → {sample_ids}  "
          f"({'single token ✓' if len(sample_ids) == 1 else f'MULTI-TOKEN {len(sample_ids)} ✗'})\n")

    # Flatten all prompts
    all_prompts = [
        (group, prompt, target)
        for group, items in cfg.prompts.items()
        for prompt, target in items
    ]

    col = f"{'Grp':<4} {'Expected':>14}  {'Rank':>5}  Top-{topk} predictions"
    print(col)
    print("-" * (len(col) + 20))

    n_top1 = n_top5 = n_multitoken = n_total = 0

    for group, prompt, target in all_prompts:
        n_total += 1
        grp = f"[{group[0].upper()}]"

        # --- check tokenization ---
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if len(target_ids) != 1:
            print(f"  {grp:<4} {repr(target):>14}  MULTI-TOKEN({len(target_ids)})  {target_ids}  ← fix config")
            n_multitoken += 1
            continue

        target_id = target_ids[0]

        # --- forward pass: get next-token logits ---
        # Tokenizer adds BOS automatically (add_special_tokens=True default)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]  # [vocab_size], last token position

        probs = torch.softmax(logits, dim=-1)
        top_result = torch.topk(probs, topk)
        top_tokens = [tokenizer.decode([idx]) for idx in top_result.indices.tolist()]
        top_probs = top_result.values.tolist()

        rank = int((logits > logits[target_id]).sum().item()) + 1

        if rank == 1:
            n_top1 += 1; n_top5 += 1; rank_tag = "  1 ✓"
        elif rank <= topk:
            n_top5 += 1; rank_tag = f"  {rank} ~"
        else:
            rank_tag = f"{rank:>3} ✗"

        top_str = "  ".join(f"{repr(t)}({p:.2f})" for t, p in zip(top_tokens, top_probs))
        print(f"  {grp:<4} {repr(target):>14} {rank_tag}  {top_str}")

    print("-" * (len(col) + 20))
    print(f"\nSummary: {n_top1}/{n_total} top-1  |  {n_top5}/{n_total} top-{topk}"
          f"  |  {n_multitoken} multi-token skipped")

    if n_multitoken:
        print("\n  → Multi-token targets will be skipped during attribution.")
        print("    Fix them in config.py: use the first token only (e.g. ' 1' for 11, ' 5' for 56).")
    if n_top1 < n_total * 0.3:
        print("\n  ⚠  Low top-1 rate. Consider:")
        print("     - Checking if prompts have trailing spaces that shift tokenization")
        print("     - Verifying expected tokens have the right leading space")
        print("     - Running with --topk 20 to see where expected tokens rank")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    run_test(args.device, args.topk)
