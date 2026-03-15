"""
POC: Hierarchical Distributional Narrowing in Next-Token Prediction.

Multi-GPU mode: distributes prompts across all available GPUs in parallel.
Each GPU loads its own model copy and processes its assigned prompts.
Results are merged, then regression + scatter plot are produced.

Usage:
    uv run python -m src.poc.run_poc                    # all available GPUs
    uv run python -m src.poc.run_poc --gpus 0 1 2 3    # specific GPUs
    uv run python -m src.poc.run_poc --device mps       # Apple Silicon
    uv run python -m src.poc.run_poc --device cpu       # single CPU (slow)
"""
import argparse
import dataclasses
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from src.poc.analyze import build_result, run_regression, save_results, save_scatter_plot
from src.poc.attribution import run_attribution
from src.poc.config import PocConfig
from src.poc.model import get_token_id, load_model

PROMPT_IDS = [
    "A1", "A2", "A3", "A4", "A5",
    "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "C4", "C5",
    "D1", "D2", "D3", "D4", "D5",
]


def _gpu_worker(gpu_idx: int, prompt_items: list, cfg: PocConfig) -> list[dict]:
    """Spawned subprocess: loads model on cuda:{gpu_idx}, processes assigned prompts.

    All imports are local because spawn creates a fresh Python interpreter.
    cfg is received fully configured — only device is overridden for this GPU.
    """
    import dataclasses
    import time
    from src.poc.attribution import run_attribution
    from src.poc.analyze import build_result
    from src.poc.model import load_model, get_token_id

    cfg = dataclasses.replace(cfg, device=f"cuda:{gpu_idx}")
    loaded = load_model(cfg)

    results = []
    for prompt, tok_str, prompt_id in prompt_items:
        try:
            correct_id = get_token_id(loaded, tok_str)
        except AssertionError as e:
            print(f"[GPU{gpu_idx}] SKIP [{prompt_id}]: {e}", flush=True)
            continue

        t0 = time.time()
        _, records = run_attribution(prompt, correct_id, prompt_id, loaded, cfg)
        elapsed = time.time() - t0

        result = build_result(prompt, prompt_id, tok_str, records, elapsed)
        results.append(result)
        print(f"[GPU{gpu_idx}] [{prompt_id}] done — {len(records)} features  {elapsed:.1f}s", flush=True)

    return results


def _run_sequential(cfg: PocConfig) -> list[dict]:
    """Single device (CPU or MPS) — runs prompts sequentially in the main process."""
    loaded = load_model(cfg)
    results = []

    for idx, (prompt, tok_str) in enumerate(cfg.prompts):
        prompt_id = PROMPT_IDS[idx]
        print(f"[{prompt_id}] '{prompt}'  →  '{tok_str}'")
        try:
            correct_id = get_token_id(loaded, tok_str)
        except AssertionError as e:
            print(f"  SKIP: {e}")
            continue

        t0 = time.time()
        _, records = run_attribution(prompt, correct_id, prompt_id, loaded, cfg)
        elapsed = time.time() - t0

        results.append(build_result(prompt, prompt_id, tok_str, records, elapsed))
        print(f"  {len(records)} features  {elapsed:.1f}s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", choices=["cuda", "mps", "cpu"])
    parser.add_argument("--gpus", nargs="+", type=int, default=None,
                        help="Specific GPU indices (default: all available)")
    args = parser.parse_args()

    # Build config with device-appropriate defaults
    if args.device == "cpu":
        cfg = PocConfig(device="cpu", dtype_str="float32", batch_size=32)
    elif args.device == "mps":
        cfg = PocConfig(device="mps", dtype_str="bfloat16", batch_size=64)
    else:
        cfg = PocConfig(device="cuda", dtype_str="bfloat16", batch_size=512)

    # --- Single-device path ---
    if args.device != "cuda":
        print(f"Running on {args.device} (sequential)...")
        all_results = _run_sequential(cfg)

    # --- Multi-GPU path ---
    else:
        n_available = torch.cuda.device_count()
        gpu_indices = args.gpus if args.gpus else list(range(n_available))
        print(f"Found {n_available} CUDA GPUs. Using: {gpu_indices}")
        for i in gpu_indices:
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.0f}GB)")

        # Round-robin distribute prompts across GPUs
        tagged = [(p, t, PROMPT_IDS[i]) for i, (p, t) in enumerate(cfg.prompts)]
        gpu_batches: dict[int, list] = {g: [] for g in gpu_indices}
        for i, item in enumerate(tagged):
            gpu_batches[gpu_indices[i % len(gpu_indices)]].append(item)

        print(f"\nDispatching {len(tagged)} prompts across {len(gpu_indices)} GPUs...")
        t_start = time.time()

        all_results = []
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=len(gpu_indices), mp_context=ctx) as pool:
            futures = {
                pool.submit(_gpu_worker, gpu_idx, items, cfg): gpu_idx
                for gpu_idx, items in gpu_batches.items()
                if items
            }
            for future in as_completed(futures):
                gpu_idx = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"GPU {gpu_idx} finished — {len(results)} prompts")
                except Exception as e:
                    print(f"GPU {gpu_idx} failed: {e}")

        order = {pid: i for i, pid in enumerate(PROMPT_IDS)}
        all_results.sort(key=lambda r: order.get(r["prompt_id"], 99))
        print(f"\nAll GPUs done in {time.time() - t_start:.1f}s total")

    if not all_results:
        print("No results — all prompts skipped.")
        return

    stats, xs, ys, group_labels = run_regression(all_results)
    save_scatter_plot(xs, ys, group_labels, stats, cfg.plot_path)
    save_results(all_results, stats, cfg.output_path)


if __name__ == "__main__":
    main()
