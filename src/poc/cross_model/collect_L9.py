"""
L9: Attention entropy divergence (IT − PT per layer).

Single forward pass per prompt with output_attentions=True and
attn_implementation="eager" (plan §4.L9 Step 1).

For each layer ℓ and head h, computes:
  H(ℓ, h) = -Σ_k attn[h, last_token, k] * log(attn[h, last_token, k] + ε)

Expected finding: max(IT − PT mean entropy) at ~1/3 depth (phase boundary).

═══ Key design decisions ═════════════════════════════════════════════════
  • NO CHAT TEMPLATE — same raw text for PT and IT (plan §4.L9 Step 1).
  • attn_implementation="eager" is mandatory — Flash/SDPA do not return
    attention weight matrices.
  • 8-GPU parallel workers + merge, same pattern as L1L2 / L8.
  • Output is per-prompt JSONL (plan §4.L9 Step 7), not aggregated JSON.
    Aggregation (mean/SEM) is computed at merge time.
  • Sliding-window normalisation flag stored in output so downstream
    plots can normalise by log(effective_context_length) for fair
    cross-model comparison (plan §4.L9 note on Gemma/Mistral).

═══ Output ═══════════════════════════════════════════════════════════════
  results/cross_model/{model}/{variant}/
    L9_w{worker}.jsonl      per-worker, one JSON line per prompt
    L9_results.jsonl        merged per-prompt results
    L9_summary.json         aggregated mean/SEM across prompts

  Per-prompt JSON line (plan §4.L9 Step 7):
    {
      "prompt_id": str,
      "model":     str,
      "variant":   str,
      "n_layers":  int,
      "n_heads":   int,
      "seq_len":   int,
      "attn_entropy": [[float * n_heads] * n_layers]
    }

  Summary JSON:
    {
      "model": str, "variant": str, "n_prompts": int,
      "n_layers": int, "n_heads": int,
      "global_attn_layers":   [int],
      "is_sliding_window":    bool,
      "sliding_window_size":  int | null,
      "mean_entropy":  [[float * n_heads] * n_layers],
      "sem_entropy":   [[float * n_heads] * n_layers]
    }
"""
from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import torch
import numpy as np

from src.poc.cross_model.config import get_spec, MODEL_REGISTRY, model_id_for_variant
from src.poc.cross_model.adapters import get_adapter
from src.poc.cross_model.utils import (
    load_model_and_tokenizer,
    load_dataset,
    get_raw_prompt,
    get_prompt_for_variant,
    read_done_ids,
    merge_worker_jsonls,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── entropy helper (matches exp4/collect.py) ──────────────────────────────────

def _attn_entropy(attn_row: torch.Tensor) -> float:
    """Shannon entropy (nats) of one post-softmax attention distribution.
    attn_row: [T_k] — clamped to ≥ 0 to handle fp noise in local-attn layers.
    """
    probs = attn_row.clamp(min=0.0)
    return float(-(probs * torch.log(probs + 1e-12)).sum().item())


# ── per-prompt collection ─────────────────────────────────────────────────────

def collect_prompt_L9(
    raw_prompt: str,
    prompt_id: str,
    model,
    tokenizer,
    spec,
    device: torch.device,
) -> dict | None:
    """Single forward pass with output_attentions=True.

    Returns dict for JSON serialisation, or None on failure.
    """
    input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(device)
    seq_len   = int(input_ids.shape[1])

    try:
        with torch.no_grad():
            out = model(input_ids, output_attentions=True)
    except Exception as e:
        log.warning("Forward pass failed for %s: %s", prompt_id, e)
        return None

    if out.attentions is None:
        log.warning("output_attentions=None for %s — model may not support eager attn.", prompt_id)
        return None

    n_layers = spec.n_layers
    n_heads  = spec.n_heads

    entropy_mat: list[list[float]] = []
    for ℓ, attn in enumerate(out.attentions):
        if attn is None or ℓ >= n_layers:
            entropy_mat.append([float("nan")] * n_heads)
            continue
        # attn: [1, n_heads_q, T, T] — query rows for last token
        attn_row = attn[0, :, -1, :]       # [n_heads_q, T_k]
        n_h_actual = attn_row.shape[0]
        row: list[float] = []
        for h in range(n_heads):
            if h < n_h_actual:
                row.append(_attn_entropy(attn_row[h]))
            else:
                row.append(float("nan"))
        entropy_mat.append(row)

    return {
        "prompt_id":    prompt_id,
        "model":        spec.name,
        "variant":      "?",         # filled in by worker
        "n_layers":     n_layers,
        "n_heads":      n_heads,
        "seq_len":      seq_len,
        "attn_entropy": entropy_mat,
    }


# ── worker ────────────────────────────────────────────────────────────────────

def run_worker(
    model_name: str,
    variant: str,
    dataset_path: str,
    out_dir: str,
    device_str: str,
    worker_index: int,
    n_workers: int,
    n_eval_examples: int | None,
    apply_chat_template: bool = False,
) -> None:
    """Single-GPU worker: collect attention entropy for this worker's slice."""
    spec    = get_spec(model_name)
    device  = torch.device(device_str)

    model_id = model_id_for_variant(spec, variant)
    # MUST use eager attention to materialise [B, H, T, T] matrices
    model, tokenizer = load_model_and_tokenizer(model_id, device_str, eager_attn=True)

    records = load_dataset(
        dataset_path,
        worker_index=worker_index,
        n_workers=n_workers,
        n_examples=n_eval_examples,
    )

    out_path = Path(out_dir) / f"L9_w{worker_index}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(out_path)

    if done_ids:
        log.info("[w%d] Resuming: %d/%d already done", worker_index, len(done_ids), len(records))

    with open(out_path, "a") as fout:
        for i, rec in enumerate(records):
            pid = rec.get("id", f"rec_{worker_index}_{i}")
            if pid in done_ids:
                continue

            result = collect_prompt_L9(
                raw_prompt=get_prompt_for_variant(
                    rec, variant=variant, tokenizer=tokenizer,
                    apply_chat_template=apply_chat_template,
                ),
                prompt_id=pid,
                model=model,
                tokenizer=tokenizer,
                spec=spec,
                device=device,
            )
            if result is None:
                continue

            result["variant"] = variant
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            if (i + 1) % 100 == 0:
                log.info("[w%d] %d/%d prompts done", worker_index, i + 1, len(records))

    log.info("[w%d] Done → %s", worker_index, out_path)


# ── merge + aggregate ─────────────────────────────────────────────────────────

def merge_and_aggregate(
    out_dir: Path,
    model_name: str,
    variant: str,
    n_workers: int,
) -> None:
    """Merge per-worker JSONL files and compute mean/SEM entropy summary."""
    spec = get_spec(model_name)

    all_records = merge_worker_jsonls(
        out_dir, n_workers,
        worker_prefix="L9",
        merged_name="L9_results.jsonl",
    )
    if not all_records:
        log.warning("No records found — skipping summary.")
        return

    n_layers = spec.n_layers
    n_heads  = spec.n_heads

    # Stack entropy matrices: shape [N, n_layers, n_heads]
    stack: list[np.ndarray] = []
    for rec in all_records:
        mat = rec.get("attn_entropy", [])
        if len(mat) == n_layers:
            stack.append(np.array(mat, dtype=np.float32))

    if not stack:
        log.warning("No valid entropy matrices found.")
        return

    arr = np.stack(stack, axis=0)          # [N, n_layers, n_heads]
    mean_ent = np.nanmean(arr, axis=0)     # [n_layers, n_heads]
    n_valid  = np.sum(~np.isnan(arr), axis=0).clip(min=1)
    sem_ent  = np.nanstd(arr, axis=0) / np.sqrt(n_valid)

    summary = {
        "model":               model_name,
        "variant":             variant,
        "n_prompts":           len(stack),
        "n_layers":            n_layers,
        "n_heads":             n_heads,
        "global_attn_layers":  sorted(spec.global_attn_layers),
        "is_sliding_window":   spec.is_sliding_window,
        "sliding_window_size": spec.sliding_window_size,
        "mean_entropy":        mean_ent.tolist(),
        "sem_entropy":         sem_ent.tolist(),
    }

    summary_path = out_dir / "L9_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("L9 summary (%d prompts) saved → %s", len(stack), summary_path)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="L9: Attention entropy divergence.")
    parser.add_argument("--model",           required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant",         required=True, choices=["pt", "it"])
    parser.add_argument("--dataset",         default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--worker-index",    type=int, default=0)
    parser.add_argument("--n-workers",       type=int, default=1)
    parser.add_argument("--out-dir",         default=None)
    parser.add_argument("--apply-chat-template", action="store_true", default=False,
                        help="Apply native chat template for IT variants")
    parser.add_argument("--no-chat-template", action="store_true", default=False,
                        help="Explicitly disable chat template (ablation mode)")
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip collection; merge JSONL files and compute summary.",
    )
    args = parser.parse_args()

    spec    = get_spec(args.model)
    out_dir = Path(args.out_dir) if args.out_dir else spec.result_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_and_aggregate(out_dir, args.model, args.variant, args.n_workers)
        return

    use_chat_template = args.apply_chat_template and not args.no_chat_template
    if args.variant == "it" and not args.no_chat_template:
        use_chat_template = True
    log.info("Chat template: %s (variant=%s)", use_chat_template, args.variant)

    run_worker(
        model_name=args.model,
        variant=args.variant,
        dataset_path=args.dataset,
        out_dir=str(out_dir),
        device_str=args.device,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_eval_examples=args.n_eval_examples,
        apply_chat_template=use_chat_template,
    )


if __name__ == "__main__":
    main()
