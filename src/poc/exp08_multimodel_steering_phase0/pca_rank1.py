#!/usr/bin/env python3
"""1A: PCA of IT-PT direction — is it rank-1?

For each model, loads the 600 selected high-contrast records from precompute,
generates with IT and PT models, hooks MLP outputs at corrective layers,
and computes per-record mean MLP activations.

Then computes PCA on the per-record IT-PT differences at each corrective layer:
  diff_i = mean_IT_acts_i - mean_PT_acts_i   (for record i)
  PCA on [n_records, d_model] matrix of diff_i

Key question: Is PC1 > 60% of variance? If yes → rank-1 direction justified.
If distributed → corrective stage decomposes into sub-directions.

Output per model:
  results/cross_model/{model}/directions/pca_scree.json

Usage:
  uv run python -m src.poc.exp08_multimodel_steering_phase0.pca_rank1 --model-name llama31_8b --device cuda:2
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
from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter


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
    is_it: bool = False,
) -> dict[int, np.ndarray]:
    """Generate each record, hook corrective layer MLPs, return per-record mean acts.

    Returns:
        dict: layer_idx -> np.ndarray [n_records, d_model] of per-record mean MLP acts
    """
    n_records = len(records)
    d_model = adapter.d_model

    # Accumulators: per-record mean MLP activations
    record_acts: dict[int, list[np.ndarray]] = {li: [] for li in corrective_layers}

    layers_list = adapter.get_layers(model_raw)
    eos_ids = adapter.eos_token_ids(tokenizer)

    for idx, record in enumerate(records):
        prompt = record["prompt"]

        # Per-record token accumulator
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
            if is_it:
                templated = adapter.adapter.apply_template(tokenizer, prompt, is_it=True)
                input_ids = tokenizer.encode(templated, return_tensors="pt").to(device)
            else:
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

        # Compute mean MLP activation per record per corrective layer
        for li in corrective_layers:
            if gen_acts[li]:
                mean_act = np.mean(gen_acts[li], axis=0)  # [d_model]
                record_acts[li].append(mean_act)
            else:
                record_acts[li].append(np.zeros(d_model, dtype=np.float32))

        if (idx + 1) % 100 == 0:
            print(f"    {idx+1}/{n_records} records", flush=True)

    # Stack into [n_records, d_model] per layer
    return {li: np.stack(vecs) for li, vecs in record_acts.items()}


def run_pca(diff_matrix: np.ndarray, n_components: int = 10) -> dict:
    """Run PCA on [n_records, d_model] difference matrix.

    Returns dict with variance explained, cumulative variance, top singular values.
    """
    # Center
    diff_centered = diff_matrix - diff_matrix.mean(axis=0, keepdims=True)

    # SVD (more numerically stable than covariance eigendecomposition for wide matrices)
    # n_records << d_model, so this is efficient
    U, S, Vt = np.linalg.svd(diff_centered, full_matrices=False)

    # Variance explained
    total_var = np.sum(S ** 2)
    var_explained = (S ** 2) / total_var if total_var > 0 else S * 0
    cumulative_var = np.cumsum(var_explained)

    n_report = min(n_components, len(S))
    return {
        "variance_explained": var_explained[:n_report].tolist(),
        "cumulative_variance": cumulative_var[:n_report].tolist(),
        "singular_values": S[:n_report].tolist(),
        "total_variance": float(total_var),
        "pc1_variance_ratio": float(var_explained[0]) if len(var_explained) > 0 else 0.0,
        "n_records": diff_matrix.shape[0],
        "d_model": diff_matrix.shape[1],
        "rank_1_justified": bool(var_explained[0] > 0.6) if len(var_explained) > 0 else False,
    }


def run_model(model_name: str, device: str, max_records: int = 200) -> dict:
    spec = get_spec(model_name)
    adapter = get_steering_adapter(model_name)
    max_gen = adapter.max_gen_tokens
    corrective_layers = list(range(spec.corrective_onset, spec.n_layers))

    # Load selected records from precompute work dir
    work_dir = Path(f"results/cross_model/{model_name}/directions_work")
    selected_path = work_dir / "selected.json"
    gen_merged_path = work_dir / "gen_merged.jsonl"

    if not selected_path.exists():
        print(f"[1A] SKIP {model_name}: no selected.json at {selected_path}")
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
    print(f"[1A] {model_name}: {len(records)} records (of {len(records_by_id)} selected), "
          f"corrective layers {corrective_layers[0]}-{corrective_layers[-1]}", flush=True)

    # Collect IT activations
    print(f"[1A] {model_name}: collecting IT MLP activations...", flush=True)
    it_model_id = model_id_for_variant(spec, "it")
    it_raw, tokenizer = _load_model(it_model_id, device)
    it_acts = collect_per_record_acts(
        it_raw, tokenizer, adapter, records, corrective_layers, device, max_gen,
        is_it=True,
    )
    del it_raw; torch.cuda.empty_cache()

    # Collect PT activations
    print(f"[1A] {model_name}: collecting PT MLP activations...", flush=True)
    pt_model_id = model_id_for_variant(spec, "pt")
    pt_raw, tokenizer = _load_model(pt_model_id, device)
    pt_acts = collect_per_record_acts(
        pt_raw, tokenizer, adapter, records, corrective_layers, device, max_gen
    )
    del pt_raw; torch.cuda.empty_cache()

    # PCA per corrective layer
    print(f"[1A] {model_name}: running PCA on IT-PT differences...", flush=True)
    layer_results: dict[str, dict] = {}
    for li in corrective_layers:
        diff = it_acts[li] - pt_acts[li]  # [n_records, d_model]
        pca_result = run_pca(diff)
        layer_results[str(li)] = pca_result

    # Aggregate: mean PC1 variance ratio across corrective layers
    pc1_ratios = [layer_results[str(li)]["pc1_variance_ratio"] for li in corrective_layers]
    mean_pc1 = float(np.mean(pc1_ratios))
    min_pc1 = float(np.min(pc1_ratios))
    max_pc1 = float(np.max(pc1_ratios))

    results = {
        "model": model_name,
        "corrective_layers": corrective_layers,
        "n_records": len(records),
        "per_layer_pca": layer_results,
        "summary": {
            "mean_pc1_variance_ratio": mean_pc1,
            "min_pc1_variance_ratio": min_pc1,
            "max_pc1_variance_ratio": max_pc1,
            "rank_1_justified": bool(mean_pc1 > 0.6),
        },
    }

    print(f"[1A] {model_name}: mean PC1 = {mean_pc1:.3f} "
          f"(range {min_pc1:.3f}-{max_pc1:.3f}), "
          f"rank-1 {'JUSTIFIED' if mean_pc1 > 0.6 else 'NOT justified'}", flush=True)

    return results


def main() -> None:
    p = argparse.ArgumentParser(description="1A: PCA of IT-PT corrective direction")
    p.add_argument("--model-name", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-records", type=int, default=200,
                   help="Max records to use (200 is plenty for PCA stability)")
    args = p.parse_args()

    results = run_model(args.model_name, args.device, max_records=args.max_records)

    if results:
        out_dir = Path(f"results/cross_model/{args.model_name}/directions")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pca_scree.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[1A] {args.model_name} -> {out_path}")


if __name__ == "__main__":
    main()
