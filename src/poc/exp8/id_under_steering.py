#!/usr/bin/env python3
"""1C: Intrinsic dimensionality under steering (TwoNN).

At 3 key α values (1.0 baseline, 0.0 full removal, -1.0 amplification),
generate ~200 prompts with hooks on ALL decoder layers capturing the residual
stream. Then compute TwoNN ID per layer.

Tests whether removing the corrective direction reduces late-layer ID.
Completes the triad: direction <-> commitment <-> ID.

Output per model:
  results/cross_model/{model}/exp6/id_under_steering.json

Usage:
  uv run python -m src.poc.exp8.id_under_steering --model-name llama31_8b --device cuda:2
  uv run python -m src.poc.exp8.id_under_steering --model-name gemma3_4b --device cuda:3
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
from src.poc.exp6.interventions import Exp6InterventionSpec, load_directions_from_npz
from src.poc.exp4.analysis.intrinsic_dim import estimate_id_twonn
from src.poc.collect import load_dataset_records


# ── Config ────────────────────────────────────────────────────────────────────
ALPHA_VALUES = [1.0, 0.0, -1.0]
N_PROMPTS = 200
MAX_TOKENS_PER_LAYER = 5000  # subsample for TwoNN efficiency


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


def collect_residuals_under_steering(
    model_raw: Any,
    tokenizer: Any,
    adapter: Any,
    intervention: Exp6InterventionSpec,
    records: list[dict],
    device: str,
    max_gen: int,
    n_layers: int,
) -> dict[int, np.ndarray]:
    """Generate with intervention active, collect residual stream at each layer.

    Returns:
        dict mapping layer_idx -> np.ndarray [n_tokens, d_model] (float32, CPU)
    """
    # Register intervention hooks
    handles = intervention.register_hooks(model_raw, None, adapter=adapter)

    # Residual collection hooks
    residuals: dict[int, list[np.ndarray]] = {li: [] for li in range(n_layers)}
    res_handles = []

    layers_list = adapter.get_layers(model_raw)

    for li in range(n_layers):
        def make_hook(layer_idx: int):
            def hook(mod, inp, out):
                hidden = out[0] if isinstance(out, tuple) else out
                # Only collect generated tokens (shape[1]==1 with KV cache)
                if hidden.shape[1] == 1:
                    # Take the single token, detach to CPU
                    residuals[layer_idx].append(
                        hidden[0, 0, :].float().detach().cpu().numpy()
                    )
            return hook
        res_handles.append(layers_list[li].register_forward_hook(make_hook(li)))

    eos_ids = adapter.eos_token_ids(tokenizer)

    try:
        for idx, record in enumerate(records):
            prompt = record["formats"]["B"]
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

            if (idx + 1) % 50 == 0:
                n_tok = len(residuals[0])
                print(f"    {idx+1}/{len(records)} prompts, ~{n_tok} tokens collected", flush=True)

    finally:
        for h in res_handles:
            h.remove()
        for h in handles:
            h.remove()

    # Stack and subsample
    result: dict[int, np.ndarray] = {}
    rng = np.random.default_rng(42)
    for li in range(n_layers):
        if residuals[li]:
            arr = np.stack(residuals[li])  # [n_tokens, d_model]
            if arr.shape[0] > MAX_TOKENS_PER_LAYER:
                idx = rng.choice(arr.shape[0], size=MAX_TOKENS_PER_LAYER, replace=False)
                arr = arr[idx]
            result[li] = arr
        else:
            result[li] = np.empty((0, 0))

    return result


def compute_id_profile(residuals: dict[int, np.ndarray], n_layers: int) -> list[float]:
    """Compute TwoNN ID at each layer from collected residuals."""
    profile = []
    for li in range(n_layers):
        if li in residuals and residuals[li].shape[0] >= 10:
            id_val = estimate_id_twonn(residuals[li])
        else:
            id_val = float("nan")
        profile.append(id_val)
    return profile


def run_model(model_name: str, device: str) -> dict:
    spec = get_spec(model_name)
    adapter = get_steering_adapter(model_name)
    n_layers = spec.n_layers
    max_gen = adapter.max_gen_tokens

    # Load corrective directions
    dir_path = f"results/cross_model/{model_name}/directions/corrective_directions.npz"
    if not Path(dir_path).exists():
        print(f"[1C] SKIP {model_name}: no directions at {dir_path}")
        return {}

    corrective_dirs = load_directions_from_npz(dir_path, device=device)
    corrective_layers = list(range(spec.corrective_onset, spec.n_layers))

    # Load dataset
    records = load_dataset_records("data/eval_dataset_v2.jsonl", prompt_format="B")
    records = records[:N_PROMPTS]

    # Load model
    it_model_id = model_id_for_variant(spec, "it")
    model_raw, tokenizer = _load_model(it_model_id, device)

    results: dict[str, Any] = {"model": model_name, "n_prompts": N_PROMPTS, "profiles": {}}

    for alpha in ALPHA_VALUES:
        print(f"[1C] {model_name} alpha={alpha}: generating {N_PROMPTS} prompts...", flush=True)

        intervention = Exp6InterventionSpec(
            method="directional_remove",
            layers=corrective_layers,
            alpha=alpha,
            corrective_directions=corrective_dirs,
        )

        residuals = collect_residuals_under_steering(
            model_raw, tokenizer, adapter, intervention,
            records, device, max_gen, n_layers,
        )

        profile = compute_id_profile(residuals, n_layers)
        n_tokens = residuals[0].shape[0] if 0 in residuals and residuals[0].ndim == 2 else 0

        results["profiles"][str(alpha)] = {
            "alpha": alpha,
            "id_twonn": profile,
            "n_tokens": n_tokens,
            "n_layers": n_layers,
        }

        print(f"[1C] {model_name} alpha={alpha}: ID profile computed "
              f"(mean ID={np.nanmean(profile):.1f}, n_tokens={n_tokens})", flush=True)

    del model_raw
    torch.cuda.empty_cache()

    return results


def main() -> None:
    global N_PROMPTS

    p = argparse.ArgumentParser(description="1C: ID under steering (TwoNN)")
    p.add_argument("--model-name", required=True, choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    args = p.parse_args()

    N_PROMPTS = args.n_prompts

    results = run_model(args.model_name, args.device)

    if results:
        out_dir = Path(f"results/cross_model/{args.model_name}/exp6")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "id_under_steering.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[1C] {args.model_name} -> {out_path}")


if __name__ == "__main__":
    main()
