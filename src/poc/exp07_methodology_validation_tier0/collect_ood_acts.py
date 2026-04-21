"""Collect MLP activations from out-of-distribution prompts (Exp7 0A OOD test).

Pulls 600 prompts from TriviaQA and ARC (not in eval_dataset_v2.jsonl) and
collects per-record MLP activations using the same hook methodology as
collect_unified_acts.py. This tests whether the corrective direction
generalises beyond the calibration prompt distribution.

Usage:
  # Collect on 8 GPUs
  for i in {0..7}; do
      uv run python -m src.poc.exp07_methodology_validation_tier0.collect_ood_acts \\
          --worker-index $i --n-workers 8 --device cuda:$i &
  done; wait

  # Merge
  uv run python -m src.poc.exp07_methodology_validation_tier0.collect_ood_acts --merge-only --n-workers 8

  # Quick test
  uv run python -m src.poc.exp07_methodology_validation_tier0.collect_ood_acts \\
      --worker-index 0 --n-workers 1 --device cuda:0 --n-records 20
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.poc.exp05_corrective_direction_ablation_cartography.config import Exp5Config
from src.poc.shared.model import load_model

ALL_LAYERS = list(range(1, 34))
D_MODEL = 2560
MAX_GEN = 80
N_OOD = 600  # target number of OOD prompts

OUTPUT_DIR = Path("results/exp7/0A/ood_acts")


def _get_raw(loaded):
    return loaded.model._model


def _load_existing_prompts() -> set[str]:
    """Load all prompts from eval_dataset_v2.jsonl to exclude from OOD set."""
    existing = set()
    eval_path = Path("data/eval_dataset_v2.jsonl")
    if eval_path.exists():
        with open(eval_path) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    # Hash the prompt text for comparison
                    existing.add(hashlib.md5(r.get("prompt", "").encode()).hexdigest())
    return existing


def _build_ood_prompts(n: int = N_OOD) -> list[dict]:
    """Build OOD prompt set from TriviaQA and ARC (HuggingFace datasets).

    Returns list of dicts with 'prompt' and 'record_id' fields.
    """
    try:
        import datasets
    except ImportError:
        print("[OOD] ERROR: 'datasets' package required. Install: uv add datasets", flush=True)
        sys.exit(1)

    existing_hashes = _load_existing_prompts()
    ood_prompts: list[dict] = []
    target_per_source = n // 2

    # TriviaQA — use question text as prompt
    print(f"[OOD] Loading TriviaQA (target={target_per_source})...", flush=True)
    try:
        trivia = datasets.load_dataset("trivia_qa", "unfiltered.nocontext", split="validation")
        count = 0
        for item in trivia:
            q = item.get("question", "")
            if not q:
                continue
            h = hashlib.md5(q.encode()).hexdigest()
            if h in existing_hashes:
                continue
            prompt = f"Question: {q}\nAnswer:"
            ood_prompts.append({"prompt": prompt, "record_id": f"ood_trivia_{count}"})
            existing_hashes.add(h)
            count += 1
            if count >= target_per_source:
                break
        print(f"[OOD] Got {count} TriviaQA prompts", flush=True)
    except Exception as e:
        print(f"[OOD] WARNING: TriviaQA load failed: {e}", flush=True)

    # ARC — use question text with choices
    remaining = n - len(ood_prompts)
    print(f"[OOD] Loading ARC-Challenge (target={remaining})...", flush=True)
    try:
        arc = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        count = 0
        for item in arc:
            q = item.get("question", "")
            choices = item.get("choices", {})
            if not q:
                continue
            h = hashlib.md5(q.encode()).hexdigest()
            if h in existing_hashes:
                continue
            # Format as multiple choice
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            options = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
            prompt = f"Question: {q}\n{options}\nAnswer:"
            ood_prompts.append({"prompt": prompt, "record_id": f"ood_arc_{count}"})
            existing_hashes.add(h)
            count += 1
            if count >= remaining:
                break
        print(f"[OOD] Got {count} ARC prompts", flush=True)
    except Exception as e:
        print(f"[OOD] WARNING: ARC load failed: {e}", flush=True)

    print(f"[OOD] Total OOD prompts: {len(ood_prompts)}", flush=True)

    # Save prompts for reproducibility
    prompts_path = OUTPUT_DIR / "ood_prompts.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(json.dumps(
        [{"record_id": p["record_id"], "prompt": p["prompt"][:200]} for p in ood_prompts],
        indent=2,
    ))
    return ood_prompts


def _collect_per_record_means(
    model_raw,
    tokenizer,
    prompts: list[dict],
    device: str,
    tag: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Collect per-record mean MLP output activations (same as collect_unified_acts)."""
    n = len(prompts)
    means: dict[int, np.ndarray] = {li: np.zeros((n, D_MODEL), dtype=np.float64) for li in ALL_LAYERS}
    valid: dict[int, np.ndarray] = {li: np.zeros(n, dtype=bool) for li in ALL_LAYERS}

    for idx, row in enumerate(prompts):
        gen_acts: dict[int, list] = {li: [] for li in ALL_LAYERS}

        def make_hook(li: int):
            def hook(mod, inp, out):
                if out.shape[1] == 1 and len(gen_acts[li]) < MAX_GEN:
                    gen_acts[li].append(out[0, 0, :].float().cpu())
            return hook

        handles = [
            model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
            for li in ALL_LAYERS
        ]
        try:
            input_ids = tokenizer.encode(row["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                model_raw.generate(
                    input_ids, max_new_tokens=MAX_GEN, do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        except Exception as e:
            if idx < 5:
                print(f"  Warning: gen failed for {row['record_id']}: {e}", flush=True)
        finally:
            for h in handles:
                h.remove()

        for li in ALL_LAYERS:
            if gen_acts[li]:
                stacked = torch.stack(gen_acts[li])
                means[li][idx] = stacked.mean(dim=0).numpy().astype(np.float64)
                valid[li][idx] = True

        if (idx + 1) % 20 == 0:
            print(f"  [{tag}] {idx+1}/{n} records", flush=True)

    return means, valid


def run_worker(
    worker_index: int,
    n_workers: int,
    device: str,
    n_records: int | None = None,
) -> None:
    """Collect IT and PT MLP activations for OOD prompts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"w{worker_index}.npz"

    if out_path.exists():
        print(f"[OOD w{worker_index}] already exists, skipping.", flush=True)
        return

    # Build or load OOD prompts — worker 0 builds, others wait to avoid race condition
    prompts_path = OUTPUT_DIR / "ood_prompts_full.json"
    lock_path = OUTPUT_DIR / ".ood_build.lock"

    if worker_index == 0:
        # Worker 0 builds prompts and saves full version for other workers
        all_prompts = _build_ood_prompts()
        prompts_path.write_text(json.dumps(
            [{"record_id": p["record_id"], "prompt": p["prompt"]} for p in all_prompts],
        ))
        # Signal that prompts are ready
        lock_path.write_text("done")
    else:
        # Non-zero workers wait for worker 0 to finish building
        import time
        for _ in range(300):  # wait up to 5 minutes
            if lock_path.exists():
                break
            time.sleep(1)
        if not prompts_path.exists():
            print(f"[OOD w{worker_index}] ERROR: prompts not built by worker 0", flush=True)
            return
        all_prompts = json.loads(prompts_path.read_text())

    # Worker's slice
    my_prompts = all_prompts[worker_index::n_workers]
    if n_records is not None:
        my_prompts = my_prompts[:n_records]
    my_ids = [p["record_id"] for p in my_prompts]

    print(f"[OOD w{worker_index}/{n_workers}] {len(my_prompts)} prompts on {device}", flush=True)

    # IT activations
    it_cfg = Exp5Config(
        experiment="baseline", model_variant="it", model_id="",
        run_name=f"exp7_ood_it_w{worker_index}", device=device, skip_transcoders=True,
    )
    it_loaded = load_model(it_cfg)
    it_means, it_valid = _collect_per_record_means(
        _get_raw(it_loaded), it_loaded.tokenizer, my_prompts, device, f"IT w{worker_index}"
    )
    del it_loaded
    torch.cuda.empty_cache()

    # PT activations
    pt_cfg = Exp5Config(
        experiment="baseline", model_variant="pt", model_id="",
        run_name=f"exp7_ood_pt_w{worker_index}", device=device, skip_transcoders=True,
    )
    pt_loaded = load_model(pt_cfg)
    pt_means, pt_valid = _collect_per_record_means(
        _get_raw(pt_loaded), pt_loaded.tokenizer, my_prompts, device, f"PT w{worker_index}"
    )
    del pt_loaded
    torch.cuda.empty_cache()

    payload: dict = {"record_ids": np.array(my_ids, dtype=object)}
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = it_means[li].astype(np.float32)
        payload[f"pt_acts_{li}"] = pt_means[li].astype(np.float32)

    np.savez_compressed(str(out_path), **payload)
    print(f"[OOD w{worker_index}] saved → {out_path}", flush=True)


def merge_workers(n_workers: int) -> None:
    """Merge per-worker NPZ files into merged.npz."""
    merged_path = OUTPUT_DIR / "merged.npz"
    if merged_path.exists():
        print("[OOD] merged.npz already exists, skipping.", flush=True)
        return

    all_ids: list[str] = []
    layer_it: dict[int, list] = {li: [] for li in ALL_LAYERS}
    layer_pt: dict[int, list] = {li: [] for li in ALL_LAYERS}

    for w in range(n_workers):
        wp = OUTPUT_DIR / f"w{w}.npz"
        if not wp.exists():
            print(f"[OOD] WARNING: {wp} not found", flush=True)
            continue
        with np.load(wp, allow_pickle=True) as d:
            all_ids.extend(list(d["record_ids"]))
            for li in ALL_LAYERS:
                layer_it[li].append(d[f"it_acts_{li}"])
                layer_pt[li].append(d[f"pt_acts_{li}"])

    payload: dict = {"record_ids": np.array(all_ids, dtype=object)}
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.concatenate(layer_it[li], axis=0)
        payload[f"pt_acts_{li}"] = np.concatenate(layer_pt[li], axis=0)

    np.savez_compressed(str(merged_path), **payload)
    print(f"[OOD] merged {len(all_ids)} records → {merged_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="OOD activation collector (Exp7 0A)")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-records", type=int, default=None)
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    if args.merge_only:
        merge_workers(args.n_workers)
    else:
        run_worker(args.worker_index, args.n_workers, args.device, args.n_records)


if __name__ == "__main__":
    main()
