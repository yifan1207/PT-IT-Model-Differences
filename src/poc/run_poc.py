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
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from src.poc.analyze import build_result, run_regression, save_results, save_scatter_plot
from src.poc.attribution import run_attribution
from src.poc.config import PocConfig
from src.poc.model import get_token_id, load_model


def _tagged_prompts(cfg: PocConfig) -> list[tuple[str, str, str]]:
    """Flatten cfg.prompts dict → [(prompt, tok_str, prompt_id), ...].

    prompt_id format: first letter of group (uppercase) + 1-based index.
    "memorization" → M1, M2, ...   "reasoning" → R1, R2, ...
    """
    tagged = []
    for group_name, group_prompts in cfg.prompts.items():
        prefix = group_name[0].upper()
        for i, (prompt, tok_str) in enumerate(group_prompts, 1):
            tagged.append((prompt, tok_str, f"{prefix}{i}"))
    return tagged


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
        try:
            _, records = run_attribution(prompt, correct_id, prompt_id, loaded, cfg)
        except Exception as e:
            print(f"[GPU{gpu_idx}] ERROR [{prompt_id}]: attribution failed: {e}", flush=True)
            continue
        elapsed = time.time() - t0

        result = build_result(prompt, prompt_id, tok_str, records, elapsed)
        results.append(result)
        print(f"[GPU{gpu_idx}] [{prompt_id}] done — {len(records)} features  {elapsed:.1f}s", flush=True)

    return results


def _run_sequential(cfg: PocConfig) -> list[dict]:
    """Single device (CPU or MPS) — runs prompts sequentially in the main process."""
    loaded = load_model(cfg)
    results = []

    for prompt, tok_str, prompt_id in _tagged_prompts(cfg):
        print(f"[{prompt_id}] '{prompt}'  →  '{tok_str}'")
        try:
            correct_id = get_token_id(loaded, tok_str)
        except AssertionError as e:
            print(f"  SKIP: {e}")
            continue

        t0 = time.time()
        try:
            _, records = run_attribution(prompt, correct_id, prompt_id, loaded, cfg)
        except Exception as e:
            print(f"  ERROR: attribution failed: {e}")
            continue
        elapsed = time.time() - t0

        results.append(build_result(prompt, prompt_id, tok_str, records, elapsed))
        print(f"  {len(records)} features  {elapsed:.1f}s")

    return results


def _default_device() -> str:
    """Pick the best available device automatically."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=_default_device(), choices=["cuda", "mps", "cpu"])
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

    print(f"Run dir  : {cfg.run_dir}/")
    print(f"  variant: {cfg.transcoder_variant}")
    print(f"  features: {cfg.max_feature_nodes}   logits: {cfg.max_n_logits}")
    print(f"  results → {cfg.output_path}")
    print(f"  scatter → {cfg.plot_path}")
    print()

    # --- Single-device path ---
    if args.device != "cuda":
        print(f"Running on {args.device} (sequential)...")
        all_results = _run_sequential(cfg)

    # --- Multi-GPU path ---
    else:
        n_available = torch.cuda.device_count()
        if n_available == 0:
            raise RuntimeError("--device cuda specified but no CUDA GPUs found. Use --device cpu or --device mps.")
        gpu_indices = args.gpus if args.gpus else list(range(n_available))
        print(f"Found {n_available} CUDA GPUs. Using: {gpu_indices}")
        for i in gpu_indices:
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.0f}GB)")

        tagged = _tagged_prompts(cfg)
        order = {pid: i for i, (_, _, pid) in enumerate(tagged)}

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
