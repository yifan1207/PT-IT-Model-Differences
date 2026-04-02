"""
Prototype test for Phase 0 Modal script.

Runs a minimal end-to-end test of the precompute pipeline on deepseek_v2_lite
(smallest model, fastest to load, max_new_tokens=64).

Tests: precompute_gen (1 worker, 20 records) → precompute_score → precompute_acts → merge

Usage:
    modal run scripts/modal_phase0_test.py
"""
from __future__ import annotations

import modal

app = modal.App("phase0-test")

results_vol = modal.Volume.from_name("phase0-test-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("phase0-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.5.0,<2.7.0",
        "transformers==4.57.3",
        "accelerate",
        "numpy",
        "scipy",
        "scikit-dimension>=0.3.4",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "tqdm",
        "matplotlib",
        "openai",
        "httpx",
        "requests",
        "nnsight>=0.6.0",
        "transformer-lens>=2.17.0",
    )
    .pip_install(
        "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git@26a976e",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file(
        "data/eval_dataset_v2.jsonl",
        remote_path="/root/data/eval_dataset_v2.jsonl",
    )
)

VOLUME_MOUNTS = {
    "/root/results": results_vol,
    "/root/.cache/huggingface": hf_cache_vol,
}


def _setup():
    import os, sys
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/hub")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


def _run(cmd, timeout=None):
    import os, subprocess as sp
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    result = sp.run(cmd, capture_output=True, text=True, cwd="/root", env=env, timeout=timeout)
    if result.stdout:
        print(result.stdout[-5000:])
    if result.returncode != 0:
        err = result.stderr[-3000:] if result.stderr else "(no stderr)"
        print(f"STDERR:\n{err}")
        raise RuntimeError(f"Command failed: {' '.join(cmd[:6])}...\n{err[-1000:]}")
    return result.stdout


@app.function(
    gpu="A100-80GB:1",
    timeout=3600,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def test_precompute_gen() -> str:
    """Test precompute gen on deepseek with only 20 records."""
    _setup()
    import torch
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    # Patch the dataset to only use 20 records by creating a small subset
    import json
    from pathlib import Path
    full_ds = Path("/root/data/eval_dataset_v2.jsonl")
    mini_ds = Path("/root/data/eval_mini.jsonl")
    with open(full_ds) as f:
        lines = [next(f) for _ in range(20)]
    with open(mini_ds, "w") as f:
        f.writelines(lines)

    # Monkey-patch the DATASET constant in the precompute script
    # Run gen phase with the mini dataset
    # We'll use subprocess but override the dataset by modifying the script's constant
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "precompute", "/root/scripts/precompute_directions_multimodel.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Override DATASET before exec
    import sys
    sys.modules["precompute"] = mod
    spec.loader.exec_module(mod)
    mod.DATASET = "data/eval_mini.jsonl"

    print("Running phase_gen (deepseek_v2_lite, 1 worker, 20 records)...")
    mod.phase_gen("deepseek_v2_lite", worker_index=0, n_workers=1, device="cuda:0")

    results_vol.commit()

    # Verify output
    out_path = Path("/root/results/cross_model/deepseek_v2_lite/directions_work/gen/w0.jsonl")
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        first = json.loads(open(out_path).readline())
        return (
            f"SUCCESS: {n_lines} records saved\n"
            f"Keys: {list(first.keys())}\n"
            f"IT text preview: {first['it_text'][:100]}\n"
            f"PT text preview: {first['pt_text'][:100]}\n"
            f"STR: it={first['it_str']:.3f} pt={first['pt_str']:.3f}\n"
            f"PT NLL: {first['pt_nll']:.3f}"
        )
    else:
        return f"FAIL: output not found at {out_path}"


@app.function(
    gpu="A100-80GB:1",
    timeout=3600,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def test_steer_import() -> str:
    """Test that steering pipeline imports work (nnsight, circuit_tracer, etc)."""
    _setup()

    checks = []

    # Test full import chain for steering
    try:
        from src.poc.exp6.config import Exp6Config
        checks.append("Exp6Config: OK")
    except Exception as e:
        checks.append(f"FAIL Exp6Config: {e}")

    try:
        from src.poc.exp6.interventions import build_intervention
        checks.append("interventions: OK")
    except Exception as e:
        checks.append(f"FAIL interventions: {e}")

    try:
        from src.poc.exp6.runtime import generate_records_A_batch
        checks.append("runtime: OK")
    except Exception as e:
        checks.append(f"FAIL runtime: {e}")

    try:
        from src.poc.shared.model import load_model, LoadedModel
        checks.append("shared.model: OK")
    except Exception as e:
        checks.append(f"FAIL shared.model: {e}")

    try:
        from src.poc.exp6.model_adapter import get_steering_adapter
        adapter = get_steering_adapter("deepseek_v2_lite")
        checks.append(f"adapter (deepseek): max_gen={adapter.max_gen_tokens}")
    except Exception as e:
        checks.append(f"FAIL adapter: {e}")

    # Test constructing an Exp6Config for multi-model
    try:
        cfg = Exp6Config(
            experiment="A1",
            model_variant="it",
            model_family="deepseek_v2_lite",
            skip_transcoders=True,
            device="cuda:0",
            apply_chat_template=False,
            n_eval_examples=5,
        )
        checks.append(
            f"Exp6Config(deepseek): model_id={cfg.model_id}, "
            f"n_layers={cfg.n_layers}, d_model={cfg.d_model}, "
            f"pb={cfg.proposal_boundary}"
        )
    except Exception as e:
        checks.append(f"FAIL Exp6Config construction: {e}")

    result = "\n".join(checks)
    print(result)
    return result


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("PHASE 0 PROTOTYPE TEST")
    print("=" * 60)

    # Test 1: Import chain
    print("\n--- Test 1: Steering imports ---")
    import_result = test_steer_import.remote()
    print(import_result)
    if "FAIL" in import_result:
        print("*** Import test has failures ***")

    # Test 2: Precompute gen (actual GPU work)
    print("\n--- Test 2: Precompute gen (deepseek, 20 records) ---")
    gen_result = test_precompute_gen.remote()
    print(gen_result)

    print("\n" + "=" * 60)
    print("PROTOTYPE TEST COMPLETE")
    print("=" * 60)
