#!/usr/bin/env python3
"""Direction stability bootstrap for cross-model corrective directions.

For each model, loads the selected high-contrast calibration records, generates
with both IT and PT models to collect per-record mean MLP activations at
corrective layers, then performs N bootstrap resamples (with replacement) to
test direction stability.

For each resample:
  - Sample records with replacement
  - Compute mean IT acts - mean PT acts per layer -> normalize -> bootstrap direction
  - Compare to full-data direction (cosine similarity)
  - Also compute pairwise cosine similarities between all bootstrap directions

Efficiency: all per-record activations are collected in a single forward pass
per model (200 records x IT + PT). Bootstrap resamples then just re-index the
numpy arrays -- no re-generation needed.

Output per model:
  results/cross_model/{model}/directions/bootstrap_stability.json

Usage:
  python scripts/phase0_direction_bootstrap.py --model-name llama31_8b --device cuda:2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
from src.poc.exp6.model_adapter import get_steering_adapter


def _load_model(model_id: str, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


def collect_per_record_acts(
    model_raw: Any,
    tokenizer: Any,
    adapter: Any,
    records: list[dict],
    corrective_layers: list[int],
    device: str,
    max_gen: int,
) -> dict[int, np.ndarray]:
    """Generate each record, hook corrective layer MLPs, return per-record mean acts.

    Returns:
        dict: layer_idx -> np.ndarray [n_records, d_model] of per-record mean MLP acts
    """
    n_records = len(records)
    d_model = adapter.d_model

    record_acts: dict[int, list[np.ndarray]] = {li: [] for li in corrective_layers}

    layers_list = adapter.get_layers(model_raw)
    eos_ids = adapter.eos_token_ids(tokenizer)

    for idx, record in enumerate(records):
        prompt = record["prompt"]

        gen_acts: dict[int, list[np.ndarray]] = {li: [] for li in corrective_layers}
        handles = []

        for li in corrective_layers:
            def make_hook(layer_idx: int):
                def hook(mod, inp, out):
                    if out.shape[1] == 1:  # generated token only
                        gen_acts[layer_idx].append(
                            out[0, 0, :].float().detach().cpu().numpy()
                        )
                return hook
            handles.append(layers_list[li].mlp.register_forward_hook(make_hook(li)))

        try:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model_raw.generate(
                    input_ids,
                    max_new_tokens=max_gen,
                    do_sample=False,
                    eos_token_id=eos_ids,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        finally:
            for h in handles:
                h.remove()

        for li in corrective_layers:
            if gen_acts[li]:
                mean_act = np.mean(gen_acts[li], axis=0)  # [d_model]
                record_acts[li].append(mean_act)
            else:
                record_acts[li].append(np.zeros(d_model, dtype=np.float32))

        if (idx + 1) % 50 == 0:
            print(f"    {idx+1}/{n_records} records", flush=True)

    return {li: np.stack(vecs) for li, vecs in record_acts.items()}


def bootstrap_directions(
    it_acts: dict[int, np.ndarray],
    pt_acts: dict[int, np.ndarray],
    full_directions: dict[int, np.ndarray],
    corrective_layers: list[int],
    n_resamples: int,
    seed: int = 42,
) -> dict:
    """Perform bootstrap resampling on pre-collected activations.

    Args:
        it_acts: layer_idx -> [n_records, d_model] IT MLP acts
        pt_acts: layer_idx -> [n_records, d_model] PT MLP acts
        full_directions: layer_idx -> [d_model] normalized full-data direction
        corrective_layers: list of layer indices
        n_resamples: number of bootstrap resamples
        seed: random seed

    Returns:
        dict with per-layer bootstrap statistics
    """
    rng = np.random.default_rng(seed)
    n_records = it_acts[corrective_layers[0]].shape[0]

    per_layer: dict[str, dict] = {}

    for li in corrective_layers:
        it_mat = it_acts[li]     # [n_records, d_model]
        pt_mat = pt_acts[li]     # [n_records, d_model]
        full_dir = full_directions[li]  # [d_model], normalized

        # Collect bootstrap directions
        boot_directions = []
        for _ in range(n_resamples):
            indices = rng.choice(n_records, size=n_records, replace=True)
            boot_it_mean = it_mat[indices].mean(axis=0)
            boot_pt_mean = pt_mat[indices].mean(axis=0)
            boot_diff = boot_it_mean - boot_pt_mean
            norm = np.linalg.norm(boot_diff)
            if norm > 0:
                boot_diff = boot_diff / norm
            boot_directions.append(boot_diff)

        boot_directions = np.stack(boot_directions)  # [n_resamples, d_model]

        # Bootstrap vs full-data cosines
        cosines_vs_full = boot_directions @ full_dir  # [n_resamples]

        # Pairwise cosines between all bootstrap directions
        # boot_directions is already normalized, so pairwise cosine = dot product
        pairwise_mat = boot_directions @ boot_directions.T  # [n_resamples, n_resamples]
        # Extract upper triangle (excluding diagonal)
        triu_indices = np.triu_indices(n_resamples, k=1)
        pairwise_cosines = pairwise_mat[triu_indices]

        per_layer[str(li)] = {
            "bootstrap_vs_full_cosines": {
                "mean": float(np.mean(cosines_vs_full)),
                "std": float(np.std(cosines_vs_full)),
                "min": float(np.min(cosines_vs_full)),
                "p5": float(np.percentile(cosines_vs_full, 5)),
                "p95": float(np.percentile(cosines_vs_full, 95)),
            },
            "pairwise_cosines": {
                "mean": float(np.mean(pairwise_cosines)),
                "std": float(np.std(pairwise_cosines)),
                "min": float(np.min(pairwise_cosines)),
                "p5": float(np.percentile(pairwise_cosines, 5)),
                "p95": float(np.percentile(pairwise_cosines, 95)),
            },
        }

    return per_layer


def run_model(model_name: str, device: str, max_records: int = 200,
              n_resamples: int = 100) -> dict:
    spec = get_spec(model_name)
    adapter = get_steering_adapter(model_name)
    max_gen = adapter.max_gen_tokens
    corrective_layers = list(range(spec.corrective_onset, spec.n_layers))

    # Load selected records from precompute work dir
    work_dir = Path(f"results/cross_model/{model_name}/directions_work")
    selected_path = work_dir / "selected.json"
    gen_merged_path = work_dir / "gen_merged.jsonl"

    if not selected_path.exists():
        print(f"[bootstrap] SKIP {model_name}: no selected.json at {selected_path}")
        return {}

    selected_ids = set(json.loads(selected_path.read_text()))
    records_by_id: dict[str, dict] = {}
    with open(gen_merged_path) as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in selected_ids:
                records_by_id[r["record_id"]] = r

    records = [records_by_id[rid] for rid in sorted(records_by_id.keys())]
    if max_records and len(records) > max_records:
        records = records[:max_records]
    print(f"[bootstrap] {model_name}: {len(records)} records (of {len(records_by_id)} selected), "
          f"corrective layers {corrective_layers[0]}-{corrective_layers[-1]}", flush=True)

    # Load full-data directions
    directions_path = Path(f"results/cross_model/{model_name}/directions/corrective_directions.npz")
    if not directions_path.exists():
        print(f"[bootstrap] SKIP {model_name}: no corrective_directions.npz at {directions_path}")
        return {}

    npz = np.load(directions_path)
    full_directions: dict[int, np.ndarray] = {}
    for li in corrective_layers:
        key = f"layer_{li}"
        if key in npz:
            d = npz[key].astype(np.float32)
            norm = np.linalg.norm(d)
            if norm > 0:
                d = d / norm
            full_directions[li] = d
        else:
            print(f"[bootstrap] WARNING: layer {li} not found in {directions_path}, skipping")

    # Filter corrective_layers to those present in full directions
    corrective_layers = [li for li in corrective_layers if li in full_directions]
    if not corrective_layers:
        print(f"[bootstrap] SKIP {model_name}: no corrective layers found in directions file")
        return {}

    # Collect IT activations
    print(f"[bootstrap] {model_name}: collecting IT MLP activations...", flush=True)
    it_model_id = model_id_for_variant(spec, "it")
    it_raw, tokenizer = _load_model(it_model_id, device)
    it_acts = collect_per_record_acts(
        it_raw, tokenizer, adapter, records, corrective_layers, device, max_gen
    )
    del it_raw
    torch.cuda.empty_cache()

    # Collect PT activations
    print(f"[bootstrap] {model_name}: collecting PT MLP activations...", flush=True)
    pt_model_id = model_id_for_variant(spec, "pt")
    pt_raw, tokenizer = _load_model(pt_model_id, device)
    pt_acts = collect_per_record_acts(
        pt_raw, tokenizer, adapter, records, corrective_layers, device, max_gen
    )
    del pt_raw
    torch.cuda.empty_cache()

    # Bootstrap resampling (pure numpy, no GPU needed)
    print(f"[bootstrap] {model_name}: running {n_resamples} bootstrap resamples...", flush=True)
    per_layer = bootstrap_directions(
        it_acts, pt_acts, full_directions, corrective_layers, n_resamples
    )

    # Summary statistics across all corrective layers
    boot_vs_full_means = [
        per_layer[str(li)]["bootstrap_vs_full_cosines"]["mean"]
        for li in corrective_layers
    ]
    pairwise_means = [
        per_layer[str(li)]["pairwise_cosines"]["mean"]
        for li in corrective_layers
    ]
    pairwise_mins = [
        per_layer[str(li)]["pairwise_cosines"]["min"]
        for li in corrective_layers
    ]

    mean_boot_vs_full = float(np.mean(boot_vs_full_means))
    mean_pairwise = float(np.mean(pairwise_means))
    min_pairwise = float(np.min(pairwise_mins))

    results = {
        "model": model_name,
        "n_records": len(records),
        "n_resamples": n_resamples,
        "corrective_layers": corrective_layers,
        "per_layer": per_layer,
        "summary": {
            "mean_bootstrap_vs_full": mean_boot_vs_full,
            "mean_pairwise": mean_pairwise,
            "min_pairwise": min_pairwise,
            "stable": bool(mean_pairwise > 0.90),
        },
    }

    print(f"[bootstrap] {model_name}: mean boot-vs-full cosine = {mean_boot_vs_full:.4f}, "
          f"mean pairwise = {mean_pairwise:.4f}, min pairwise = {min_pairwise:.4f}, "
          f"{'STABLE' if mean_pairwise > 0.90 else 'UNSTABLE'}", flush=True)

    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Direction stability bootstrap for cross-model corrective directions"
    )
    p.add_argument("--model-name", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-records", type=int, default=200,
                   help="Max records to use (sample from these for bootstrap)")
    p.add_argument("--n-resamples", type=int, default=100,
                   help="Number of bootstrap resamples (default 100)")
    args = p.parse_args()

    results = run_model(
        args.model_name, args.device,
        max_records=args.max_records,
        n_resamples=args.n_resamples,
    )

    if results:
        out_dir = Path(f"results/cross_model/{args.model_name}/directions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "bootstrap_stability.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bootstrap] {args.model_name} -> {out_path}")


if __name__ == "__main__":
    main()
