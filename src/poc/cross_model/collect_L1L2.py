"""
L1 + L2: δ-cosine heatmap and commitment delay.

Collected in a single autoregressive generation pass per prompt.

═══ L1 — δ-cosine heatmap ═══════════════════════════════════════════════
  At each generation step t and layer ℓ:
    δ_ℓ^(t)   = h_ℓ^(t) − h_{ℓ-1}^(t)
    cos_ℓ^(t) = dot(δ_ℓ, h_{ℓ-1}) / (‖δ_ℓ‖ ‖h_{ℓ-1}‖)
  Negative values at corrective layers in IT → opposition signal.

═══ L2 — commitment delay ════════════════════════════════════════════════
  Logit-lens at every layer of every generation step:
    probs_ℓ = softmax(final_norm(h_ℓ) @ W_U.T)
    top1_ℓ  = argmax(probs_ℓ)
    ent_ℓ   = -Σ_v probs_ℓ[v] log probs_ℓ[v]

  Commitment layer for step t = earliest ℓ such that:
    top1_ℓ == top1_{n-1}  AND  top1_{ℓ'} == top1_{n-1} for all ℓ' > ℓ
  (forward scan; "no flip-back" requirement from plan §4.L2)

═══ Key design decisions ═════════════════════════════════════════════════
  • NO CHAT TEMPLATE for either PT or IT.  Plan §4.L1: "No templates for
    PT or IT. Raw text, identical tokenisation."  This isolates the
    difference to model weights only.
  • Greedy decoding (do_sample=False, temperature not applied) for
    deterministic, reproducible results — same as exp2 setup.
  • Logit lens computed INLINE on GPU immediately after every layer's
    hook fires on the last layer (not post-hoc on CPU), avoiding a
    second GPU round-trip per step.
  • Residual tensors live on GPU only during the single step they are
    needed; scalars (cosines, entropies, top-1 tokens) go to Python
    lists immediately.

═══ Output ═══════════════════════════════════════════════════════════════
  results/cross_model/{model}/{variant}/
    L1L2_w{worker}.jsonl    per-worker JSONL (one line per prompt)
    L1L2_results.jsonl      merged results
    L1L2_heatmap_pt.npy     mean δ-cosine [max_steps, n_layers] for PT
    L1L2_heatmap_it.npy     same for IT

  Per-prompt JSON line (matches plan §4.L1 Step 5 + §4.L2 Step 5):
    {
      "prompt_id":          str,
      "model":              str,
      "variant":            "pt" | "it",
      "n_layers":           int,
      "n_steps":            int,
      "generated_text":     str,
      "delta_cosine":       [[float * n_layers] * n_steps],
      "logit_lens_entropy": [[float * n_layers] * n_steps],
      "logit_lens_top1":    [[int   * n_layers] * n_steps],
      "commitment_layer":   [int * n_steps]
    }
"""
from __future__ import annotations

import json
import math
import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from src.poc.cross_model.config import get_spec, MODEL_REGISTRY, model_id_for_variant
from src.poc.cross_model.adapters import get_adapter
from src.poc.cross_model.utils import (
    load_model_and_tokenizer,
    load_dataset,
    get_raw_prompt,
    read_done_ids,
    merge_worker_jsonls,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── per-prompt collection ─────────────────────────────────────────────────────

def collect_prompt_L1L2(
    raw_prompt: str,
    prompt_id: str,
    model,
    tokenizer,
    adapter,
    spec,
    device: torch.device,
    max_new_tokens: int = 512,
) -> dict:
    """
    Generate up to max_new_tokens tokens (greedy, no template).
    Inline-collect δ-cosine + logit-lens for every (step, layer) pair.

    Returns a dict ready for JSON serialisation.
    """
    n_layers = spec.n_layers

    # W_U: [d_model, vocab_size] float32 — extracted once.
    # For multi-GPU, lm_head and final_norm may be on different devices from the
    # step_buf tensors (which are on CPU). Move both to CPU so logit-lens works.
    final_norm_mod = adapter.final_norm(model)
    W_U = model.lm_head.weight.detach().float().T  # [d_model, vocab_size]
    if spec.multi_gpu:
        final_norm_mod = final_norm_mod.cpu()
        W_U = W_U.cpu()

    # ── per-step accumulators (Python lists of scalars — small) ──────────────
    all_delta_cosine:       list[list[float]] = []
    all_logit_lens_entropy: list[list[float]] = []
    all_logit_lens_top1:    list[list[int]]   = []
    all_commitment_layer:   list[int]         = []

    # Step buffer: holds residuals for the current generation step.
    # With multi_gpu=True, tensors land on different CUDA devices, so we move
    # them to CPU immediately in the hook to avoid cross-device operations.
    multi_gpu = spec.multi_gpu
    step_buf: list[torch.Tensor | None] = [None] * n_layers

    def _process_step() -> None:
        """Called at layer n_layers-1: all residuals for this step are ready on GPU."""
        # ── δ-cosine ─────────────────────────────────────────────────────────
        row_cos: list[float] = [float("nan")]  # layer 0: no h_{ℓ-1}
        for ℓ in range(1, n_layers):
            h_prev = step_buf[ℓ - 1]
            h_cur  = step_buf[ℓ]
            if h_prev is None or h_cur is None:
                row_cos.append(float("nan"))
                continue
            h_prev_f = h_prev.float()
            h_cur_f  = h_cur.float()
            delta = h_cur_f - h_prev_f
            denom = delta.norm() * h_prev_f.norm()
            cos = (torch.dot(delta, h_prev_f) / denom).item() if denom > 1e-8 else float("nan")
            row_cos.append(cos)
        all_delta_cosine.append(row_cos)

        # ── logit lens: apply final_norm + W_U at every layer ────────────────
        row_ent:  list[float] = []
        row_top1: list[int]   = []
        with torch.no_grad():
            for ℓ in range(n_layers):
                h = step_buf[ℓ]
                if h is None:
                    row_ent.append(float("nan"))
                    row_top1.append(-1)
                    continue
                # final_norm expects [..., d_model]; unsqueeze gives [1, 1, d_model]
                normed = final_norm_mod(h.float().unsqueeze(0).unsqueeze(0)).squeeze()
                logits = normed @ W_U  # [vocab_size]
                top1   = int(logits.argmax().item())
                probs  = torch.softmax(logits, dim=-1)
                ent    = float(-(probs * torch.log(probs + 1e-12)).sum().item())
                row_ent.append(ent)
                row_top1.append(top1)
        all_logit_lens_entropy.append(row_ent)
        all_logit_lens_top1.append(row_top1)

        # ── commitment layer — forward scan, "no flip-back" ──────────────────
        # Plan §4.L2: "earliest ℓ where top1_ℓ == top1_final AND for all ℓ' > ℓ:
        # top1_{ℓ'} == top1_final"
        final_top1 = row_top1[n_layers - 1]
        commit = n_layers - 1  # default: only commits at very last layer
        for ℓ in range(n_layers):
            if row_top1[ℓ] == final_top1:
                # Check stability: all subsequent layers must also match
                if all(row_top1[ℓ2] == final_top1 for ℓ2 in range(ℓ + 1, n_layers)):
                    commit = ℓ
                    break
        all_commitment_layer.append(commit)

    # ── forward hooks ─────────────────────────────────────────────────────────
    layer_modules = adapter.layers(model)

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            # Prefill: h.shape = [1, T_prompt, d] — skip.
            # Generation step: h.shape = [1, 1, d] (KV-cache enabled by default).
            if h.shape[1] != 1:
                return
            vec = h[0, 0, :]  # [d_model]
            # For multi-GPU models, each layer lives on a different device.
            # Move to CPU immediately so _process_step() works on uniform tensors.
            step_buf[layer_idx] = vec.cpu() if multi_gpu else vec

            # On the last layer, all residuals for this step are ready → process.
            if layer_idx == n_layers - 1:
                _process_step()
        return hook

    handles = [layer_modules[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

    # ── greedy generation — no template, deterministic ────────────────────────
    # Both PT and IT use identical raw tokenisation (no chat template).
    # Plan §4.L1: "No templates for PT or IT. Raw text, identical tokenisation."
    # For multi-GPU (device_map), use model.device (first shard); HF handles the rest.
    gen_device = model.device if multi_gpu else device
    input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(gen_device)
    stop_ids  = list(adapter.stop_token_ids(tokenizer))

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,              # greedy — deterministic
            temperature=1.0,              # no-op when do_sample=False, but explicit
            eos_token_id=stop_ids or None,
        )

    for h in handles:
        h.remove()

    generated_ids   = out_ids[0, input_ids.shape[1]:]
    generated_text  = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return {
        "prompt_id":          prompt_id,
        "model":              spec.name,
        "variant":            "?",        # filled in by worker
        "n_layers":           n_layers,
        "n_steps":            len(all_delta_cosine),
        "generated_text":     generated_text,
        "delta_cosine":       all_delta_cosine,
        "logit_lens_entropy": all_logit_lens_entropy,
        "logit_lens_top1":    all_logit_lens_top1,
        "commitment_layer":   all_commitment_layer,
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
    max_new_tokens: int,
    n_eval_examples: int | None,
) -> None:
    """Single-GPU worker: processes one slice of the dataset."""
    spec    = get_spec(model_name)
    adapter = get_adapter(model_name)
    device  = torch.device(device_str)

    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(model_id, device_str, multi_gpu=spec.multi_gpu)

    records = load_dataset(
        dataset_path,
        worker_index=worker_index,
        n_workers=n_workers,
        n_examples=n_eval_examples,
    )

    out_path = Path(out_dir) / f"L1L2_w{worker_index}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = read_done_ids(out_path)

    if done_ids:
        log.info("[w%d] Resuming: %d/%d already done", worker_index, len(done_ids), len(records))

    with open(out_path, "a") as fout:
        for i, rec in enumerate(records):
            pid = rec.get("id", f"rec_{worker_index}_{i}")
            if pid in done_ids:
                continue

            raw_prompt = get_raw_prompt(rec)
            try:
                result = collect_prompt_L1L2(
                    raw_prompt=raw_prompt,
                    prompt_id=pid,
                    model=model,
                    tokenizer=tokenizer,
                    adapter=adapter,
                    spec=spec,
                    device=device,
                    max_new_tokens=max_new_tokens,
                )
                result["variant"] = variant
                fout.write(json.dumps(result) + "\n")
                fout.flush()
            except Exception as e:
                log.warning("[w%d] prompt %s failed: %s", worker_index, pid, e)

            if (i + 1) % 50 == 0:
                log.info("[w%d] %d/%d prompts done", worker_index, i + 1, len(records))

    log.info("[w%d] Done → %s", worker_index, out_path)


# ── merge ─────────────────────────────────────────────────────────────────────

def merge_and_aggregate(out_dir: Path, n_workers: int, n_layers: int) -> None:
    """Merge per-worker JSONL files and compute mean δ-cosine heatmap.

    The heatmap is 2D [max_steps, n_layers] (plan §4.L1 Step 4/5),
    averaged across all prompts while keeping the step dimension intact.
    Steps beyond the shortest prompt are NaN-filled.
    """
    all_records = merge_worker_jsonls(
        out_dir, n_workers,
        worker_prefix="L1L2",
        merged_name="L1L2_results.jsonl",
    )
    if not all_records:
        log.warning("No records to aggregate — no heatmap saved.")
        return

    # Determine max step count across all prompts
    max_steps = max(r.get("n_steps", 0) for r in all_records)
    if max_steps == 0:
        return

    accum = np.zeros((max_steps, n_layers), dtype=np.float64)
    count = np.zeros((max_steps, n_layers), dtype=np.int64)

    for rec in all_records:
        dc = rec.get("delta_cosine", [])
        for t, row in enumerate(dc):
            if t >= max_steps:
                break
            for ℓ, v in enumerate(row):
                if ℓ < n_layers and v == v and not math.isnan(v):
                    accum[t, ℓ] += v
                    count[t, ℓ] += 1

    mean_heatmap = np.where(count > 0, accum / count, np.nan)

    # Determine variant from first record
    variant = all_records[0].get("variant", "unknown") if all_records else "unknown"
    np.save(out_dir / f"L1L2_heatmap_{variant}.npy", mean_heatmap)
    log.info("δ-cosine heatmap [%d × %d] saved → %s",
             max_steps, n_layers, out_dir / f"L1L2_heatmap_{variant}.npy")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="L1+L2: δ-cosine heatmap + commitment delay collection."
    )
    parser.add_argument("--model",           required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant",         required=True, choices=["pt", "it"])
    parser.add_argument("--dataset",         default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--worker-index",    type=int, default=0)
    parser.add_argument("--n-workers",       type=int, default=1)
    parser.add_argument("--max-new-tokens",  type=int, default=512)
    parser.add_argument("--out-dir",         default=None)
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip collection; only merge existing worker JSONL files.",
    )
    args = parser.parse_args()

    spec    = get_spec(args.model)
    out_dir = Path(args.out_dir) if args.out_dir else spec.result_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_and_aggregate(out_dir, args.n_workers, spec.n_layers)
        return

    run_worker(
        model_name=args.model,
        variant=args.variant,
        dataset_path=args.dataset,
        out_dir=str(out_dir),
        device_str=args.device,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        max_new_tokens=args.max_new_tokens,
        n_eval_examples=args.n_eval_examples,
    )


if __name__ == "__main__":
    main()
