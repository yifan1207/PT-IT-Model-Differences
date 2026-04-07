"""
Validate eval_commitment_fast vs eval_commitment on a real model.
Runs both on the same 5 prompts and compares output bit-for-bit.

    modal run scripts/modal_validate_fast_eval.py
"""
from __future__ import annotations
import modal

app = modal.App("validate-fast-eval")

probes_vol = modal.Volume.from_name("0g-probes-v3", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7.0", "transformers==4.57.3", "accelerate",
        "numpy", "scipy", "sentencepiece", "protobuf",
        "huggingface_hub", "safetensors", "einops",
    )
    .add_local_python_source("src")
    .add_local_file("data/exp3_dataset.jsonl", remote_path="/root/data/exp3_dataset.jsonl")
)


@app.function(
    gpu="B200", timeout=3600, image=image,
    volumes={"/probes": probes_vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536, cpu=8.0,
)
def validate(model_name: str = "llama31_8b", variant: str = "it") -> dict:
    import json, logging, numpy as np
    from pathlib import Path
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("validate")

    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer, load_dataset
    from src.poc.cross_model.tuned_lens import (
        _load_probes, eval_commitment, eval_commitment_fast,
    )

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    probes_vol.reload()
    probe_dir = Path(f"/probes/{model_name}/{variant}")
    probes = _load_probes(probe_dir, spec.d_model, device)
    log.info("Loaded %d probes", len(probes))

    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(model_id, device, eager_attn=spec.is_moe)
    model.requires_grad_(False)

    records = load_dataset("/root/data/exp3_dataset.jsonl", n_examples=5)
    log.info("Testing on %d prompts", len(records))

    use_template = variant == "it"

    # Run ORIGINAL eval
    log.info("=== Running eval_commitment (original) ===")
    out_orig = Path("/tmp/test_orig.jsonl")
    arr_orig = Path("/tmp/arrays_orig")
    eval_commitment(
        model, tokenizer, adapter, spec, probes, records, device,
        output_path=out_orig, variant=variant, max_new_tokens=50,
        collect_full=True, arrays_dir=arr_orig,
        apply_chat_template=use_template,
    )

    # Run FAST eval
    log.info("=== Running eval_commitment_fast (new) ===")
    out_fast = Path("/tmp/test_fast.jsonl")
    arr_fast = Path("/tmp/arrays_fast")
    eval_commitment_fast(
        model, tokenizer, adapter, spec, probes, records, device,
        output_path=out_fast, variant=variant, max_new_tokens=50,
        collect_full=True, arrays_dir=arr_fast,
        apply_chat_template=use_template,
    )

    # Compare JSONL
    log.info("=== Comparing results ===")
    with open(out_orig) as f:
        orig_lines = [json.loads(l) for l in f]
    with open(out_fast) as f:
        fast_lines = [json.loads(l) for l in f]

    results = {"model": model_name, "variant": variant, "n_prompts": len(records)}

    # Compare prompt-by-prompt
    mismatches = []
    for i, (o, f_) in enumerate(zip(orig_lines, fast_lines)):
        for key in o:
            if key in ("prompt_id", "model", "variant"):
                continue
            if o[key] != f_.get(key):
                mismatches.append(f"prompt {i}, key={key}: orig={str(o[key])[:50]} vs fast={str(f_.get(key))[:50]}")

    results["jsonl_mismatches"] = len(mismatches)
    if mismatches:
        results["first_mismatches"] = mismatches[:10]
        log.warning("JSONL MISMATCHES: %d", len(mismatches))
        for m in mismatches[:5]:
            log.warning("  %s", m)
    else:
        log.info("JSONL: EXACT MATCH")

    # Compare NPY arrays
    for arr_name in ["raw_kl_final.npy", "tuned_kl_final.npy", "raw_top1.npy", "tuned_top1.npy"]:
        p_orig = arr_orig / arr_name
        p_fast = arr_fast / arr_name
        if p_orig.exists() and p_fast.exists():
            a = np.load(p_orig)
            b = np.load(p_fast)
            if a.shape != b.shape:
                log.warning("%s: SHAPE MISMATCH %s vs %s", arr_name, a.shape, b.shape)
                results[f"array_{arr_name}_match"] = False
            else:
                max_diff = np.abs(a.astype(np.float32) - b.astype(np.float32)).max()
                results[f"array_{arr_name}_max_diff"] = float(max_diff)
                if max_diff < 1e-3:
                    log.info("%s: MATCH (max_diff=%.2e)", arr_name, max_diff)
                else:
                    log.warning("%s: DIFF %.2e", arr_name, max_diff)

    results["pass"] = len(mismatches) == 0
    log.info("=== RESULT: %s ===", "PASS" if results["pass"] else "FAIL")
    return results


@app.local_entrypoint()
def main():
    # Test on llama IT (most common case) and deepseek PT (MoE edge case)
    for model, variant in [("llama31_8b", "it"), ("deepseek_v2_lite", "pt")]:
        print(f"\n{'='*60}")
        print(f"Validating {model}/{variant}")
        print(f"{'='*60}")
        result = validate.remote(model, variant)
        print(json.dumps(result, indent=2))
        if not result.get("pass"):
            print("FAILED — do not use eval_commitment_fast")
            return
    print("\nALL VALIDATIONS PASSED")

import json
