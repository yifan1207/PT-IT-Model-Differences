"""
TEST version of Modal eval — runs 1 model (deepseek_v2_lite) with 5 prompts.
Validates: image build, probe loading, model loading, eval pipeline, Volume I/O.

Usage:
    modal run scripts/modal_eval_0G_test.py
"""
from __future__ import annotations

import modal

app = modal.App("0g-eval-test")

probes_vol = modal.Volume.from_name("0g-probes", create_if_missing=True)
results_vol = modal.Volume.from_name("0g-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0,<2.7.0",  # pin to CUDA 12.x compatible range
        "transformers==4.57.3",
        "accelerate",
        "numpy",
        "scipy",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
    )
    .add_local_python_source("src")
    .add_local_file(
        "data/exp3_dataset.jsonl",
        remote_path="/root/data/exp3_dataset.jsonl",
    )
)


@app.function(
    gpu=["H100", "A100-80GB"],
    timeout=3600,  # 1 hour for test
    image=image,
    volumes={
        "/probes": probes_vol,
        "/results": results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
    cpu=8.0,
)
def test_eval():
    """Smoke test: load deepseek_v2_lite/pt, eval 5 prompts, verify output."""
    import json
    import logging
    import time
    from pathlib import Path

    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("test_eval")

    model_name = "deepseek_v2_lite"
    variant = "pt"

    log.info("=== TEST: %s/%s on %s ===", model_name, variant, torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    log.info("VRAM: %.1f GB", vram / 1e9)

    # --- Step 1: Check probes ---
    log.info("Step 1: Checking probes...")
    probes_vol.reload()
    probe_dir = Path(f"/probes/{model_name}/{variant}")
    probe_files = list(probe_dir.glob("probe_layer_*.pt"))
    log.info("  Found %d probe files in %s", len(probe_files), probe_dir)
    assert len(probe_files) == 27, f"Expected 27 probes, got {len(probe_files)}"
    log.info("  PASS: probe count correct")

    # --- Step 2: Import project code ---
    log.info("Step 2: Importing project code...")
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer, load_dataset
    from src.poc.cross_model.tuned_lens import _load_probes, eval_commitment
    log.info("  PASS: all imports successful")

    # --- Step 3: Load probes ---
    log.info("Step 3: Loading probes...")
    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")
    probes = _load_probes(probe_dir, spec.d_model, device)
    log.info("  PASS: loaded %d/%d probes", len(probes), spec.n_layers)

    # --- Step 4: Load model ---
    log.info("Step 4: Loading model...")
    t0 = time.time()
    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(
        model_id, device, eager_attn=True, multi_gpu=False,
    )
    model.requires_grad_(False)
    log.info("  PASS: model loaded in %.1fs", time.time() - t0)
    log.info("  GPU memory: %.1f GB used", torch.cuda.memory_allocated() / 1e9)

    # Commit HF cache for future runs
    try:
        hf_cache_vol.commit()
    except Exception:
        pass

    # --- Step 5: Load dataset (first 5 only) ---
    log.info("Step 5: Loading dataset...")
    records = load_dataset("/root/data/exp3_dataset.jsonl", n_examples=5)
    log.info("  PASS: loaded %d records", len(records))
    log.info("  First record ID: %s", records[0].get("id", "?"))

    # --- Step 6: Run eval ---
    log.info("Step 6: Running eval (5 prompts, max_new_tokens=64, collect_full=True)...")
    results_dir = Path("/results/_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "test_commitment.jsonl"
    arrays_dir = results_dir / "arrays"
    # Remove old test results
    if out_path.exists():
        out_path.unlink()

    t0 = time.time()
    summary = eval_commitment(
        model, tokenizer, adapter, spec, probes, records, device,
        output_path=out_path,
        variant=variant,
        max_new_tokens=64,  # small for speed
        collect_full=True,
        collect_top5=True,
        top5_max_prompts=5,
        arrays_dir=arrays_dir,
    )
    elapsed = time.time() - t0
    log.info("  PASS: eval completed in %.1fs", elapsed)
    log.info("  Summary: %s", json.dumps(summary, indent=2))

    # --- Step 7: Verify output ---
    log.info("Step 7: Verifying output...")
    assert out_path.exists(), "Output JSONL not created"
    with open(out_path) as f:
        lines = [json.loads(l) for l in f if l.strip()]
    log.info("  Output records: %d", len(lines))
    assert len(lines) > 0, "No output records"

    # Check first record has all expected fields
    r0 = lines[0]
    expected_fields = [
        "prompt_id", "model", "variant", "n_steps",
        "commitment_layer_raw",
        "commitment_layer_raw_kl_0.1",
        "commitment_layer_tuned_0.1",
        "commitment_layer_top1_tuned",
        "commitment_layer_cosine_0.95",
        "commitment_layer_entropy_0.1",
        "commitment_layer_raw_top1_qual_top3",
        "commitment_layer_tuned_top1_qual_top5",
        "commitment_layer_raw_kl_qual_0.1_3x",
        "commitment_layer_tuned_kl_qual_0.1_5x",
    ]
    missing = [f for f in expected_fields if f not in r0]
    if missing:
        log.error("  FAIL: missing fields: %s", missing)
        log.error("  Available fields: %s", sorted(r0.keys()))
        return {"status": "FAIL", "missing_fields": missing}
    log.info("  PASS: all %d expected fields present", len(expected_fields))
    log.info("  Total fields per record: %d", len(r0))

    # --- Step 8: Verify NPY arrays (Groups B-G) ---
    log.info("Step 8: Verifying NPY arrays...")
    import numpy as np
    expected_npy = [
        "raw_top1.npy", "tuned_top1.npy", "generated_ids.npy",
        "raw_kl_final.npy", "tuned_kl_final.npy",
        "raw_kl_adj.npy", "tuned_kl_adj.npy",
        "raw_ntprob.npy", "tuned_ntprob.npy",
        "raw_ntrank.npy", "tuned_ntrank.npy",
        "raw_entropy.npy", "tuned_entropy.npy",
        "delta_cosine.npy", "cosine_h_to_final.npy",
        "raw_top5_ids.npy", "raw_top5_probs.npy",
        "tuned_top5_ids.npy", "tuned_top5_probs.npy",
    ]
    missing_npy = [f for f in expected_npy if not (arrays_dir / f).exists()]
    if missing_npy:
        log.error("  FAIL: missing NPY files: %s", missing_npy)
        return {"status": "FAIL", "missing_npy": missing_npy}
    log.info("  PASS: all %d NPY files exist", len(expected_npy))

    # Check shapes
    with open(arrays_dir / "step_index.jsonl") as f:
        step_recs = [json.loads(l) for l in f if l.strip()]
    total_steps = sum(r["n_steps"] for r in step_recs)
    n_layers = spec.n_layers
    raw_top1 = np.load(arrays_dir / "raw_top1.npy")
    assert raw_top1.shape == (total_steps, n_layers), \
        f"raw_top1 shape {raw_top1.shape} != ({total_steps}, {n_layers})"
    log.info("  PASS: raw_top1 shape (%d, %d) correct", *raw_top1.shape)

    # Semantic check: generated_ids match raw_top1 final layer
    gen_ids = np.load(arrays_dir / "generated_ids.npy")
    assert np.array_equal(gen_ids, raw_top1[:, -1]), \
        "generated_ids != raw_top1 at final layer"
    log.info("  PASS: generated_ids == raw_top1[:, -1]")

    # Check KL at final layer ~0
    kl = np.load(arrays_dir / "raw_kl_final.npy").astype(np.float32)
    assert np.allclose(kl[:, -1], 0.0, atol=0.01), \
        f"raw_kl_final at final layer not ~0: {kl[:, -1][:5]}"
    log.info("  PASS: raw_kl_final at final layer ~0")

    # Check delta_cosine layer 0 is NaN
    dc = np.load(arrays_dir / "delta_cosine.npy")
    assert np.all(np.isnan(dc[:, 0])), "delta_cosine layer 0 should be NaN"
    log.info("  PASS: delta_cosine layer 0 is NaN")

    # Check adjacent KL layer 0 is NaN
    adj_kl = np.load(arrays_dir / "raw_kl_adj.npy")
    assert np.all(np.isnan(adj_kl[:, 0])), "raw_kl_adj layer 0 should be NaN"
    log.info("  PASS: raw_kl_adj layer 0 is NaN")

    # Storage estimate
    total_bytes = sum((arrays_dir / f).stat().st_size for f in expected_npy)
    log.info("  Total NPY storage: %.1f MB", total_bytes / 1e6)
    per_step_bytes = total_bytes / max(total_steps, 1)
    est_full_mb = per_step_bytes * (total_steps / max(len(lines), 1)) * 2936 / 1e6
    log.info("  Estimated full dataset: %.0f MB", est_full_mb)

    # --- Step 9: Volume commit ---
    log.info("Step 9: Committing results to Volume...")
    results_vol.commit()
    log.info("  PASS: Volume committed")

    # --- Step 10: Resume test ---
    log.info("Step 10: Testing resume logic...")
    from src.poc.cross_model.utils import read_done_ids
    done_ids = read_done_ids(out_path)
    log.info("  done_ids: %d", len(done_ids))
    assert len(done_ids) == len(lines), f"Expected {len(lines)} done_ids, got {len(done_ids)}"
    log.info("  PASS: resume logic works")

    log.info("")
    log.info("=" * 60)
    log.info("ALL TESTS PASSED (with collect_full + collect_top5)")
    log.info("=" * 60)
    log.info("  Probes: %d loaded", len(probes))
    log.info("  Model: %s", model_id)
    log.info("  Prompts evaluated: %d", len(lines))
    log.info("  JSONL fields per record: %d", len(r0))
    log.info("  NPY files: %d", len(expected_npy))
    log.info("  Total steps: %d", total_steps)
    log.info("  Time per prompt: %.1fs", elapsed / len(lines))
    log.info("  ETA for 2,936 prompts: %.1f hours", (elapsed / len(lines) * 2936) / 3600)
    log.info("=" * 60)

    return {
        "status": "PASS",
        "n_prompts": len(lines),
        "n_fields": len(r0),
        "n_npy_files": len(expected_npy),
        "total_steps": total_steps,
        "npy_storage_mb": total_bytes / 1e6,
        "time_per_prompt_s": elapsed / len(lines),
        "eta_full_hours": (elapsed / len(lines) * 2936) / 3600,
    }


@app.local_entrypoint()
def main():
    result = test_eval.remote()
    print(f"\nTest result: {result}")
    if result.get("status") == "PASS":
        print("\nAll checks passed! Safe to run full eval:")
        print("  modal run --detach scripts/modal_eval_0G.py")
    else:
        print("\nTest FAILED — fix issues before running full eval")
