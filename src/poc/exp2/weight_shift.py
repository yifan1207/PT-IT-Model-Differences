"""
Weight shift analysis: Gemma 3 4B PT → IT.

For every parameter shared between the pretrained (PT) and instruction-tuned (IT)
checkpoints, compute:

  1. Relative Frobenius shift  Δ_rel = ||W_it − W_pt||_F / ||W_pt||_F
  2. Cosine distance           d_cos = 1 − cos(vec(W_pt), vec(W_it))

See src/poc/shared/metrics.py for the metric definitions.

GPU strategy
------------
Both models are loaded to CPU in bfloat16 (~8 GB RAM each).  Metric computation
is parallelised across n_gpus CUDA devices using a ThreadPoolExecutor — one
thread per GPU, no subprocess overhead, no tensor pickling.  Each thread moves
its chunk of parameter pairs to float32 on its assigned GPU and returns results.
Round-robin splitting distributes large and small matrices evenly.

Results are written to <output_dir>/weight_shift.json, one dict per parameter,
sorted by descending Δ_rel.  Visualisation is handled by
src/poc/exp2/plots/plot_weight_shift.py.
"""
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from src.poc.shared.constants import PT_MODEL_ID, IT_MODEL_ID
from src.poc.shared.metrics import frob_shift, cosine_distance


# ── GPU chunk worker (runs in a thread) ───────────────────────────────────────

def _process_chunk(
    chunk: list[tuple[str, torch.Tensor, torch.Tensor]],
    gpu_id: int,
) -> list[dict]:
    """Compute both metrics for a list of (name, w_pt_cpu, w_it_cpu) on one GPU.

    Tensors arrive as bfloat16 on CPU; they are promoted to float32 on the
    target device for numerical precision.  Called from a ThreadPoolExecutor
    thread — tensors are shared in-process, no pickling.
    """
    device = f"cuda:{gpu_id}"
    results = []
    for name, w_pt_cpu, w_it_cpu in chunk:
        w_pt = w_pt_cpu.to(device=device, dtype=torch.float32)
        w_it = w_it_cpu.to(device=device, dtype=torch.float32)
        results.append({
            "name": name,
            "shape": list(w_pt_cpu.shape),
            "n_params": w_pt_cpu.numel(),
            "frob_shift": frob_shift(w_pt, w_it),
            "cos_dist": cosine_distance(w_pt, w_it),
        })
    return results


# ── main entry point ───────────────────────────────────────────────────────────

def compute_weight_shift(
    n_gpus: int = 8,
    output_dir: str = "results/weight_shift",
) -> list[dict]:
    """Load PT and IT checkpoints; compute shift metrics for all shared parameters.

    Parameters
    ----------
    n_gpus : int
        CUDA devices to use.  Clamped to torch.cuda.device_count().
    output_dir : str
        Directory where weight_shift.json is written.

    Returns
    -------
    list[dict]
        One dict per shared parameter, sorted by frob_shift descending.
        Keys: name, shape, n_params, frob_shift, cos_dist.
    """
    from transformers import AutoModelForCausalLM

    available = torch.cuda.device_count()
    if available == 0:
        raise RuntimeError("No CUDA GPUs available.")
    n_gpus = min(n_gpus, available)
    print(f"  Using {n_gpus} / {available} GPU(s)")

    print(f"[1/4] Loading PT model ({PT_MODEL_ID}) to CPU ...")
    pt_model = AutoModelForCausalLM.from_pretrained(
        PT_MODEL_ID, device_map="cpu", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"[2/4] Loading IT model ({IT_MODEL_ID}) to CPU ...")
    it_model = AutoModelForCausalLM.from_pretrained(
        IT_MODEL_ID, device_map="cpu", torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    pt_params = dict(pt_model.named_parameters())
    it_params = dict(it_model.named_parameters())

    shared_keys = sorted(set(pt_params.keys()) & set(it_params.keys()))
    pt_only = set(pt_params.keys()) - set(it_params.keys())
    it_only = set(it_params.keys()) - set(pt_params.keys())

    print(f"  Shared parameters : {len(shared_keys)}")
    if pt_only:
        print(f"  PT-only (skipped) : {len(pt_only)} — e.g. {sorted(pt_only)[:3]}")
    if it_only:
        print(f"  IT-only (skipped) : {len(it_only)} — e.g. {sorted(it_only)[:3]}")

    # Detach + clone before freeing model objects so tensors remain alive in RAM.
    pairs: list[tuple[str, torch.Tensor, torch.Tensor]] = [
        (k, pt_params[k].detach().clone(), it_params[k].detach().clone())
        for k in shared_keys
    ]
    del pt_model, it_model, pt_params, it_params

    print(f"[3/4] Computing metrics across {n_gpus} GPU(s) ...")
    chunks = [pairs[i::n_gpus] for i in range(n_gpus)]  # round-robin split

    all_results: list[dict] = []
    with ThreadPoolExecutor(max_workers=n_gpus) as exe:
        futures = {
            exe.submit(_process_chunk, chunk, gpu_id): gpu_id
            for gpu_id, chunk in enumerate(chunks)
            if chunk
        }
        for fut in as_completed(futures):
            gpu_id = futures[fut]
            chunk_results = fut.result()
            print(f"  GPU {gpu_id} done: {len(chunk_results)} parameters", flush=True)
            all_results.extend(chunk_results)

    all_results.sort(
        key=lambda r: r["frob_shift"] if not math.isnan(r["frob_shift"]) else -1.0,
        reverse=True,
    )

    print(f"[4/4] Saving results ...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_json = Path(output_dir) / "weight_shift.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {len(all_results)} parameter results → {out_json}")

    return all_results


# ── console summary ────────────────────────────────────────────────────────────

def print_summary(results: list[dict], top_n: int = 20) -> None:
    """Print aggregate statistics and a ranked top-N table."""
    import numpy as np

    valid = [r for r in results if not math.isnan(r["frob_shift"])]
    if not valid:
        print("No valid results to summarise.")
        return

    frob_arr = [r["frob_shift"] for r in valid]
    cos_arr  = [r["cos_dist"]  for r in valid if not math.isnan(r["cos_dist"])]

    print("\n" + "=" * 72)
    print("Weight shift summary: PT → IT  (google/gemma-3-4b)")
    print("=" * 72)
    print(f"  Parameters analysed : {len(results)}")
    print(f"  Δ_rel Frobenius — mean {np.mean(frob_arr):.4f} | "
          f"median {np.median(frob_arr):.4f} | max {np.max(frob_arr):.4f}")
    if cos_arr:
        print(f"  d_cos cosine    — mean {np.mean(cos_arr):.6f} | "
              f"median {np.median(cos_arr):.6f} | max {np.max(cos_arr):.6f}")

    print(f"\n  Top {top_n} most-shifted parameters:")
    print(f"  {'Name':<55} {'Δ_rel':>8}  {'d_cos':>10}  Shape")
    print("  " + "─" * 88)
    for r in results[:top_n]:
        frob_s = f"{r['frob_shift']:.4f}" if not math.isnan(r["frob_shift"]) else "     nan"
        cos_s  = f"{r['cos_dist']:.6f}"   if not math.isnan(r["cos_dist"])   else "       nan"
        shape  = "×".join(str(s) for s in r["shape"])
        print(f"  {r['name']:<55} {frob_s:>8}  {cos_s:>10}  {shape}")
