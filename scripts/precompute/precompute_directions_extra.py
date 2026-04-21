#!/usr/bin/env python3
"""Compute IT-PT MLP-output directions for arbitrary layers using plain PyTorch hooks.

Unlike precompute.py (which uses nnsight and materialises the full computation
graph), this script registers lightweight register_forward_hook calls that
capture only the requested layers' MLP outputs.  Peak memory ≈ model weights
(~8 GB bfloat16) + hook buffers — comfortably within a single 80 GB GPU.

Usage:
    uv run python scripts/precompute_extra_directions.py \
        --layers 5,15 \
        --out-npz results/exp05_corrective_direction_ablation_cartography/precompute_it/precompute/extra_directions.npz \
        --merge-into results/exp05_corrective_direction_ablation_cartography/precompute_it/precompute/corrective_directions.npz

The --merge-into flag updates the target npz in-place (adds/overwrites the
layer keys produced here without touching other keys).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ── Model loading (mirrors load_model but without nnsight / transcoder) ───────

def _load_hf_model(model_id: str, device: str, dtype_str: str = "bfloat16"):
    """Load a HuggingFace causal LM + tokenizer without nnsight wrapper."""
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_str]
    print(f"  loading {model_id} on {device} as {dtype_str} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    print(f"  loaded {model_id}", flush=True)
    return model, tokenizer


def _get_mlp_module(model: Any, layer_idx: int):
    """Return model.language_model.layers[layer_idx].mlp (Gemma3 layout)."""
    return model.language_model.layers[layer_idx].mlp


# ── Input preparation ─────────────────────────────────────────────────────────

def _prepare_input_ids(rec: dict, tokenizer: Any, device: str, prompt_format: str) -> torch.Tensor:
    prompt = rec["formats"][prompt_format]
    return tokenizer.encode(prompt, return_tensors="pt").to(device)


def _sample_records(records: list[dict], n: int) -> list[dict]:
    if len(records) <= n:
        return records
    return random.Random(42).sample(records, n)


# ── Hook-based activation capture ────────────────────────────────────────────

def _run_and_capture(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    layer_ids: list[int],
    device: str,
    prompt_format: str,
    position: str = "last",
) -> dict[int, list[torch.Tensor]]:
    """Run the model on records and capture MLP outputs at given layers.

    Returns dict[layer_id -> list of float32 CPU tensors, one per record].
    """
    cache: dict[int, list[torch.Tensor]] = {li: [] for li in layer_ids}
    hooks: list = []

    def _make_hook(li: int):
        def hook(module, input, output):
            # output shape: (B, T, d_model)
            if position == "last":
                vec = output[0, -1, :].float().cpu()
            else:
                vec = output[0, :, :].float().cpu()
            cache[li].append(vec)
        return hook

    for li in layer_ids:
        mlp = _get_mlp_module(model, li)
        hooks.append(mlp.register_forward_hook(_make_hook(li)))

    try:
        for idx, rec in enumerate(records):
            if (idx + 1) % 50 == 0:
                print(f"    {idx + 1}/{len(records)}", flush=True)
            ids = _prepare_input_ids(rec, tokenizer, device, prompt_format)
            with torch.no_grad():
                model(ids)
    finally:
        for h in hooks:
            h.remove()

    return cache


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layers", required=True,
                   help="Comma-separated layer indices, e.g. '5,15'")
    p.add_argument("--dataset", default="data/exp3_dataset.jsonl")
    p.add_argument("--prompt-format", choices=["A", "B"], default="B")
    p.add_argument("--n-records", type=int, default=500)
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out-npz", required=True,
                   help="Output npz path for the new directions.")
    p.add_argument("--merge-into",
                   help="Existing corrective_directions.npz to merge the new layers into.")
    args = p.parse_args()

    layer_ids = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    print(f"Computing IT-PT directions for layers {layer_ids} on {args.device}", flush=True)

    # Load dataset
    from src.poc.collect import load_dataset_records
    records = load_dataset_records(args.dataset, prompt_format=args.prompt_format)
    sampled = _sample_records(records, args.n_records)
    print(f"Using {len(sampled)} records from {args.dataset}", flush=True)

    # Pass 1: PT
    pt_model_id = "google/gemma-3-4b-pt"
    pt_model, pt_tokenizer = _load_hf_model(pt_model_id, args.device, args.dtype)
    print(f"Pass 1 (PT): capturing MLP outputs for layers {layer_ids} ...", flush=True)
    pt_cache = _run_and_capture(pt_model, pt_tokenizer, sampled, layer_ids, args.device, args.prompt_format)
    del pt_model, pt_tokenizer
    torch.cuda.empty_cache()
    print("PT done, model unloaded.", flush=True)

    # Pass 2: IT
    it_model_id = "google/gemma-3-4b-it"
    it_model, it_tokenizer = _load_hf_model(it_model_id, args.device, args.dtype)
    print(f"Pass 2 (IT): capturing MLP outputs for layers {layer_ids} ...", flush=True)
    it_cache = _run_and_capture(it_model, it_tokenizer, sampled, layer_ids, args.device, args.prompt_format)
    del it_model, it_tokenizer
    torch.cuda.empty_cache()
    print("IT done, model unloaded.", flush=True)

    # Compute mean(IT - PT), normalize
    d_model = pt_cache[layer_ids[0]][0].shape[0]
    payload: dict[str, np.ndarray] = {}
    for li in layer_ids:
        diffs = [it_cache[li][i] - pt_cache[li][i] for i in range(len(sampled))]
        mean_diff = torch.stack(diffs).mean(dim=0).numpy().astype(np.float64)
        norm = float(np.linalg.norm(mean_diff))
        unit_vec = (mean_diff / (norm + 1e-12)).astype(np.float32)
        payload[f"layer_{li}"] = unit_vec
        print(f"  layer {li}: mean_diff norm={norm:.4f}", flush=True)

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, **payload)

    meta = {
        "pt_model_id": pt_model_id,
        "it_model_id": it_model_id,
        "dataset_path": args.dataset,
        "prompt_format": args.prompt_format,
        "n_records": len(sampled),
        "layers": layer_ids,
        "token_position": "last",
        "activation_space": "mlp_output",
        "method": "plain_pytorch_hooks",
    }
    meta_path = out_npz.with_suffix("").with_suffix("") .parent / (out_npz.stem + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved → {out_npz}", flush=True)

    # Merge into existing corrective_directions.npz if requested
    if args.merge_into:
        target = Path(args.merge_into)
        if target.exists():
            with np.load(target) as d:
                existing = dict(d)
        else:
            existing = {}
        existing.update(payload)
        np.savez_compressed(target, **existing)
        print(f"Merged layers {layer_ids} into {target}", flush=True)
        print(f"  all keys now: {sorted(existing.keys())}", flush=True)


if __name__ == "__main__":
    main()
