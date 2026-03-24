#!/usr/bin/env python3
"""Precompute IT-PT corrective directions for early (0-7) and mid (8-19) layers.

Two outputs:
  1. results/exp6/precompute/ablation_directions_early.npz  — layer_0 … layer_7
  2. results/exp6/precompute/ablation_directions_mid.npz    — layer_8 … layer_19

Strategy: compute the TRUE IT-PT MLP-output difference at each target layer
(same methodology as precompute_extra_directions.py).  Then save per-layer
unit-normalised directions.

Usage:
    uv run python scripts/precompute_ablation_directions.py --device cuda:0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.poc.collect import load_dataset_records


def _get_mlp(model, layer_idx: int):
    return model.language_model.layers[layer_idx].mlp


def _load_hf_model(model_id: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dt = torch.bfloat16 if dtype == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dt, device_map=device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, tokenizer


@torch.no_grad()
def _run_and_capture(model, tokenizer, records: list, layer_ids: list[int], device: str, prompt_format: str):
    """Run model on records and capture MLP outputs at given layers."""
    cache: dict[int, list[torch.Tensor]] = {l: [] for l in layer_ids}
    hooks = []

    for lid in layer_ids:
        def hook_fn(module, inp, out, _l=lid):
            cache[_l].append(out[0][:, -1, :].float().cpu() if isinstance(out, tuple) else out[:, -1, :].float().cpu())
        hooks.append(_get_mlp(model, lid).register_forward_hook(hook_fn))

    for rec in records:
        prompt = rec.get("formats", {}).get(prompt_format) or rec.get("prompt", "")
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        model(ids)

    for h in hooks:
        h.remove()

    # Mean last-token MLP output per layer
    return {l: torch.cat(cache[l], dim=0).mean(dim=0).numpy() for l in layer_ids}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dataset", default="data/exp6_dataset.jsonl")
    p.add_argument("--prompt-format", default="B")
    p.add_argument("--n-records", type=int, default=500,
                   help="Number of records to use for direction estimation")
    p.add_argument("--layers", default="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19",
                   help="Comma-separated layer indices to compute")
    p.add_argument("--out-dir", default="results/exp6/precompute")
    args = p.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    records = load_dataset_records(args.dataset, prompt_format=args.prompt_format)
    import random; random.seed(42)
    sampled = random.sample(records, min(args.n_records, len(records)))
    print(f"Using {len(sampled)} records for direction computation", flush=True)
    print(f"Target layers: {layer_ids}", flush=True)

    # Pass 1: PT model
    pt_id = "google/gemma-3-4b-pt"
    print(f"Loading PT model: {pt_id}", flush=True)
    pt_model, pt_tok = _load_hf_model(pt_id, args.device, args.dtype)
    print("Computing PT MLP outputs...", flush=True)
    pt_means = _run_and_capture(pt_model, pt_tok, sampled, layer_ids, args.device, args.prompt_format)
    del pt_model, pt_tok
    torch.cuda.empty_cache()
    print("PT done.", flush=True)

    # Pass 2: IT model
    it_id = "google/gemma-3-4b-it"
    print(f"Loading IT model: {it_id}", flush=True)
    it_model, it_tok = _load_hf_model(it_id, args.device, args.dtype)
    print("Computing IT MLP outputs...", flush=True)
    it_means = _run_and_capture(it_model, it_tok, sampled, layer_ids, args.device, args.prompt_format)
    del it_model, it_tok
    torch.cuda.empty_cache()
    print("IT done.", flush=True)

    # Compute directions (IT - PT, unit-normalised)
    directions: dict[str, np.ndarray] = {}
    for l in layer_ids:
        diff = it_means[l] - pt_means[l]
        norm = np.linalg.norm(diff)
        if norm < 1e-8:
            print(f"  WARNING: layer {l} direction is near-zero (norm={norm:.2e})")
        directions[f"layer_{l}"] = (diff / (norm + 1e-12)).astype(np.float32)
        print(f"  layer {l}: IT-PT diff norm={norm:.4f}")

    # Save: split into early (0-7) and mid (8-19)
    early_layers = [l for l in layer_ids if l < 8]
    mid_layers = [l for l in layer_ids if 8 <= l < 20]

    if early_layers:
        early_path = out_dir / "ablation_directions_early.npz"
        np.savez(str(early_path), **{k: v for k, v in directions.items()
                                     if int(k.split("_")[1]) in early_layers})
        print(f"Saved early directions: {early_path}")

    if mid_layers:
        mid_path = out_dir / "ablation_directions_mid.npz"
        np.savez(str(mid_path), **{k: v for k, v in directions.items()
                                   if int(k.split("_")[1]) in mid_layers})
        print(f"Saved mid directions: {mid_path}")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
