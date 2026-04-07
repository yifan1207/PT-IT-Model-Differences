"""
Modal script for Exp10: Contrastive Activation Patching.

Runs the full Exp10 pipeline on Modal cloud B200 GPUs:
  Phase 1: Forced-decoding paired data collection (6 models in parallel)
            Regression target: per-layer KL excess (Δkl_ℓ), winsorized.
  Phase 2: Ridge regression probes + direction comparison (CPU — no GPU needed)
            Outputs: convergence_directions.npz (+ backward-compat commitment_directions.npz)
  Phase 3: Causal activation patching (6 models in parallel)
            Metric: ΔKL at downstream layers (not scalar commitment)
  Phase 4B: Steering with convergence-gap directions (conditional on go/no-go)

Usage:
    # Smoke test (1 model, 5 prompts — validates full pipeline end-to-end)
    modal run src/poc/exp10/modal_exp10.py::run_smoke

    # Prototype (2 models in parallel, 10 prompts — real validation)
    modal run src/poc/exp10/modal_exp10.py::run_prototype

    # Full run (6 models, 600 prompts)
    modal run --detach src/poc/exp10/modal_exp10.py

Monitor:
    modal app logs exp10-contrastive-patching

Download results:
    modal volume get exp10-results exp10 results/exp10
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import threading

import modal

# ── Modal app & infrastructure ────────────────────────────────────────────────

app = modal.App("exp10-contrastive-patching")

results_vol = modal.Volume.from_name("exp10-results", create_if_missing=True)
tuned_lens_vol = modal.Volume.from_name("0g-probes-v3")  # read-only, tuned lens probes (v3: IT trained with chat template)
phase0_vol = modal.Volume.from_name("phase0-results")  # read-only, corrective_directions.npz

# All model IDs needed (PT + IT for forced decoding)
_HF_MODELS_TO_BAKE = [
    # PT variants
    "google/gemma-3-4b-pt",
    "meta-llama/Llama-3.1-8B",
    "Qwen/Qwen3-4B-Base",
    "mistralai/Mistral-7B-v0.3",
    "deepseek-ai/DeepSeek-V2-Lite",
    "allenai/OLMo-2-1124-7B",
    # IT variants
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "allenai/OLMo-2-1124-7B-Instruct",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch>=2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==4.57.3",
        "accelerate",
        "numpy",
        "scipy",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "tqdm",
    )
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_commands(
        *[f"huggingface-cli download {mid}" for mid in _HF_MODELS_TO_BAKE],
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_file(
        "data/eval_dataset_v2.jsonl",
        remote_path="/root/data/eval_dataset_v2.jsonl",
    )
)

# ── Constants ─────────────────────────────────────────────────────────────────

MODELS = [
    "gemma3_4b", "llama31_8b", "qwen3_4b",
    "mistral_7b", "deepseek_v2_lite", "olmo2_7b",
]

# Per-model info for max_gen_tokens (MoE models capped at 64)
MODEL_MAX_GEN = {
    "gemma3_4b": 128, "llama31_8b": 128, "qwen3_4b": 128,
    "mistral_7b": 128, "deepseek_v2_lite": 64, "olmo2_7b": 128,
}

VOLUME_MOUNTS = {
    "/root/results": results_vol,
    "/root/tuned_lens_probes": tuned_lens_vol,
    "/root/phase0_results": phase0_vol,  # corrective_directions.npz live here
}

GPU_RETRIES = modal.Retries(max_retries=3, initial_delay=15.0, backoff_coefficient=2.0)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _setup():
    """Common container setup."""
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


def _commit_volumes():
    try:
        results_vol.commit()
    except Exception as e:
        print(f"[volume] commit warn: {e}")


def _setup_sigterm_handler():
    import atexit
    import signal

    def _shutdown(*_args):
        print("[shutdown] Committing volumes...")
        _commit_volumes()
        if _args:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)


def _start_periodic_commit(interval: int = 120) -> tuple[threading.Event, threading.Thread]:
    """Start a daemon thread that commits the results volume periodically.

    Returns (stop_event, thread). Call stop_event.set() then thread.join()
    when the long-running function completes.
    """
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            stop.wait(interval)
            if not stop.is_set():
                try:
                    results_vol.commit()
                    print(f"[volume] periodic commit OK")
                except Exception as e:
                    print(f"[volume] periodic commit warn: {e}")

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop, t


def _find_mean_dir(model_name: str) -> str | None:
    """Locate corrective_directions.npz for a model, checking multiple paths."""
    candidates = [
        f"/root/phase0_results/cross_model/{model_name}/directions/corrective_directions.npz",
        f"/root/tuned_lens_probes/{model_name}/directions/corrective_directions.npz",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


# ── Phase 1: Forced-decoding data collection ─────────────────────────────────

@app.function(
    gpu="B200",
    timeout=3600,  # 60 min
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def collect_one(model_name: str, n_prompts: int = 600) -> str:
    """Phase 1: Collect forced-decoding paired data for one model."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== Phase 1 collect: {model_name} (n_prompts={n_prompts}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    results_vol.reload()

    from src.poc.exp10.collect_paired import collect_paired_data

    output_dir = f"/root/results/exp10/{model_name}/paired_data"

    # Tuned lens probes (v3, IT trained with chat template) on 0g-probes-v3 volume
    # {model}/{variant}/probe_layer_*.pt
    tuned_lens_dir = "/root/tuned_lens_probes"

    max_gen = MODEL_MAX_GEN.get(model_name, 128)

    # Periodic volume commits (every 2 min) to survive preemption
    stop_commit, commit_thread = _start_periodic_commit(interval=120)
    try:
        metadata = collect_paired_data(
            model_name=model_name,
            device="cuda:0",
            output_dir=output_dir,
            n_prompts=n_prompts,
            max_gen_tokens=max_gen,
            dataset_path="/root/data/eval_dataset_v2.jsonl",
            tuned_lens_dir=tuned_lens_dir,
            compute_kl_gradient=True,
        )
    finally:
        stop_commit.set()
        commit_thread.join(timeout=5)

    _commit_volumes()
    return f"collect {model_name}: {metadata['n_prompts_processed']} prompts, {metadata['total_tokens']} tokens"


# ── Phase 2+2.5: Probe training + direction comparison ───────────────────────

@app.function(
    # CPU only — ridge solve is O(d³) on d_model≤4096, PCA SVD on ≤10k tokens.
    # No GPU needed. Saves cost and avoids GPU queue contention.
    timeout=1800,  # 30 min
    retries=modal.Retries(max_retries=1),
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def train_probes_one(model_name: str) -> str:
    """Phase 2+2.5: Train ridge probes and compare to existing directions."""
    _setup()
    _setup_sigterm_handler()

    print(f"=== Phase 2+2.5 probes: {model_name} ===")
    results_vol.reload()

    from src.poc.exp10.train_probes import train_probes

    mean_dir_path = _find_mean_dir(model_name)
    if mean_dir_path is None:
        print(f"[WARN] No corrective_directions.npz for {model_name} — cosine comparison will be 0")
        mean_dir_path = "/dev/null"  # train_probes handles missing file gracefully

    summary = train_probes(
        model_name=model_name,
        paired_data_dir=f"/root/results/exp10/{model_name}/paired_data",
        mean_dir_path=mean_dir_path,
        output_dir=f"/root/results/exp10/{model_name}/probes",
    )

    _commit_volumes()
    go_nogo = summary.get("go_nogo", "unknown")
    mean_cos = summary.get("mean_corrective_cosine_abs", 0)
    return f"probes {model_name}: go_nogo={go_nogo}, mean_cos={mean_cos:.3f}"


# ── Phase 3: Causal patching ─────────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=5400,  # 90 min
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def patch_one(model_name: str, n_test_prompts: int = 120,
              max_tokens_per_prompt: int = 5) -> str:
    """Phase 3: Causal activation patching for one model."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== Phase 3 patching: {model_name} (n_test={n_test_prompts}) ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    results_vol.reload()

    from src.poc.exp10.patching import validate_patching

    tuned_lens_dir = "/root/tuned_lens_probes"
    mean_dir_path = _find_mean_dir(model_name)

    # Periodic volume commits (every 2 min) to survive preemption
    stop_commit, commit_thread = _start_periodic_commit(interval=120)
    try:
        validate_patching(
            model_name=model_name,
            device="cuda:0",
            probes_dir=f"/root/results/exp10/{model_name}/probes",
            paired_data_dir=f"/root/results/exp10/{model_name}/paired_data",
            output_dir=f"/root/results/exp10/{model_name}/patching",
            mean_dir_path=mean_dir_path,
            n_test_prompts=n_test_prompts,
            max_tokens_per_prompt=max_tokens_per_prompt,
            top_k_layers=10,
            dataset_path="/root/data/eval_dataset_v2.jsonl",
            tuned_lens_dir=tuned_lens_dir,
        )
    finally:
        stop_commit.set()
        commit_thread.join(timeout=5)

    _commit_volumes()
    return f"patching {model_name} done"


# ── Phase 4B: Steering with commitment directions ────────────────────────────

@app.function(
    gpu="B200",
    timeout=10800,  # 3h
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def steer_one(model_name: str) -> str:
    """Phase 4B: A1 α-sweep with BOTH d_conv and d_mean directions.

    Runs two steering sweeps under identical conditions for fair comparison:
      1. d_conv (convergence-gap direction from exp10 ridge probes)
      2. d_mean (mean IT-PT direction from Phase 0 precompute)

    Both use chat template (IT default), batch=16 on B200, same 1,400 prompts.
    """
    _setup()
    _setup_sigterm_handler()

    import json
    import torch
    print(f"=== Phase 4B steering: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    # Check go/no-go
    summary_path = f"/root/results/exp10/{model_name}/probes/probe_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    go_nogo = summary.get("go_nogo", "unknown")
    if go_nogo == "redundant":
        print(f"[Phase 4B] {model_name}: SKIP — go_nogo=redundant (d_commit ≈ d_mean)")
        return f"steer {model_name}: SKIPPED (redundant)"

    import subprocess as sp
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    output_base = f"/root/results/exp10/{model_name}/steering"
    results = []

    # ── Run 1: Steer with d_conv (convergence-gap direction from exp10) ──────
    d_conv_path = f"/root/results/exp10/{model_name}/probes/commitment_directions.npz"
    cmd_conv = [
        "python", "-m", "src.poc.exp6.run",
        "--experiment", "A1",
        "--variant", "it",
        "--device", "cuda:0",
        "--model-name", model_name,
        "--corrective-direction-path", d_conv_path,
        "--run-name", f"A1_d_conv_{model_name}",
        "--output-base", output_base,
        "--batch-size", "16",
    ]
    print(f"\n[Phase 4B] Steering with d_conv...")
    r1 = sp.run(cmd_conv, capture_output=True, text=True, cwd="/root", env=env, timeout=10000)
    if r1.stdout:
        print(r1.stdout[-2000:])
    if r1.returncode != 0:
        err = r1.stderr[-1000:] if r1.stderr else "(no stderr)"
        print(f"[d_conv] FAILED:\n{err}")
        results.append(f"d_conv=FAIL")
    else:
        results.append(f"d_conv=OK")

    _commit_volumes()

    # ── Run 2: Steer with d_mean (Phase 0 precomputed IT-PT mean direction) ──
    d_mean_path = _find_mean_dir(model_name)
    if d_mean_path is None:
        print(f"[Phase 4B] No d_mean directions for {model_name} — skipping d_mean steering")
        results.append("d_mean=SKIP(no_dirs)")
    else:
        cmd_mean = [
            "python", "-m", "src.poc.exp6.run",
            "--experiment", "A1",
            "--variant", "it",
            "--device", "cuda:0",
            "--model-name", model_name,
            "--corrective-direction-path", d_mean_path,
            "--run-name", f"A1_d_mean_{model_name}",
            "--output-base", output_base,
            "--batch-size", "16",
        ]
        print(f"\n[Phase 4B] Steering with d_mean...")
        r2 = sp.run(cmd_mean, capture_output=True, text=True, cwd="/root", env=env, timeout=10000)
        if r2.stdout:
            print(r2.stdout[-2000:])
        if r2.returncode != 0:
            err = r2.stderr[-1000:] if r2.stderr else "(no stderr)"
            print(f"[d_mean] FAILED:\n{err}")
            results.append("d_mean=FAIL")
        else:
            results.append("d_mean=OK")

    _commit_volumes()
    return f"steer {model_name}: {', '.join(results)}"


# ── Smoke test: end-to-end validation with 1 model, 5 prompts ───────────────

@app.function(
    gpu="B200",
    timeout=1800,  # 30 min
    retries=modal.Retries(max_retries=1),
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def smoke_test_gpu(model_name: str) -> str:
    """End-to-end smoke test: Phases 1→2→3 on one GPU, 5 prompts.

    Validates:
      1. Volume mounts & tuned lens probes exist
      2. Chat template works for this model
      3. Phase 1: collect 5 prompts → accumulators, delta_kl.npy, JSONL
      4. Phase 2: ridge probes → convergence_directions.npz, probe_summary.json
      5. Phase 3: patching 3 prompts × 2 tokens → patching_summary.json
      6. Output shapes & values are sane
    """
    _setup()
    _setup_sigterm_handler()

    import json
    import time
    import numpy as np
    import torch

    t0 = time.time()
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"=== SMOKE TEST: {model_name} on {gpu_name} ({vram_gb:.0f} GB) ===\n")

    checks = []
    errors = []

    def check(name, condition, detail=""):
        status = "PASS" if condition else "FAIL"
        msg = f"  [{status}] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        checks.append((name, condition, detail))
        if not condition:
            errors.append(name)

    # ── 0. Imports ────────────────────────────────────────────────────────────
    print("[Step 0] Imports & volumes")
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.exp10.collect_paired import collect_paired_data
    from src.poc.exp10.train_probes import train_probes
    from src.poc.exp10.patching import validate_patching
    check("core imports", True)

    spec = get_spec(model_name)
    check("model spec", spec is not None, f"n_layers={spec.n_layers}, d_model={spec.d_model}")

    # ── 1. Volume checks ─────────────────────────────────────────────────────
    print("\n[Step 1] Volume & probe checks")
    ds_path = Path("/root/data/eval_dataset_v2.jsonl")
    check("dataset exists", ds_path.exists())
    if ds_path.exists():
        n_records = sum(1 for _ in open(ds_path))
        check("dataset non-empty", n_records > 10, f"{n_records} records")

    # Tuned lens probes
    it_probe_dir = Path(f"/root/tuned_lens_probes/{model_name}/it")
    pt_probe_dir = Path(f"/root/tuned_lens_probes/{model_name}/pt")
    # Fallback to v2 layout
    if not it_probe_dir.exists():
        it_probe_dir = Path(f"/root/tuned_lens_probes/{model_name}/tuned_lens/it")
        pt_probe_dir = Path(f"/root/tuned_lens_probes/{model_name}/tuned_lens/pt")
    it_probes_exist = it_probe_dir.exists()
    pt_probes_exist = pt_probe_dir.exists()
    check("tuned lens IT probes", it_probes_exist, str(it_probe_dir))
    check("tuned lens PT probes", pt_probes_exist, str(pt_probe_dir))
    if it_probes_exist:
        n_it = len(list(it_probe_dir.glob("probe_layer_*.pt")))
        check("IT probe count", n_it == spec.n_layers, f"{n_it}/{spec.n_layers} layers")
    if pt_probes_exist:
        n_pt = len(list(pt_probe_dir.glob("probe_layer_*.pt")))
        check("PT probe count", n_pt == spec.n_layers, f"{n_pt}/{spec.n_layers} layers")

    # Mean directions (optional — not all models may have Phase 0 done)
    mean_dir = _find_mean_dir(model_name)
    check("mean directions", mean_dir is not None,
          mean_dir if mean_dir else "not found (cosine comparison will be 0)")

    # ── 2. Chat template ─────────────────────────────────────────────────────
    print("\n[Step 2] Chat template")
    from transformers import AutoTokenizer
    adapter = get_adapter(model_name)
    tok = AutoTokenizer.from_pretrained(spec.it_id)
    raw = "Write a haiku about neural networks."
    templated = adapter.apply_template(tok, raw, is_it=True)
    check("chat template applied", len(templated) > len(raw),
          f"raw={len(raw)} chars → templated={len(templated)} chars")

    # ── 3. Phase 1: Collect 5 prompts ─────────────────────────────────────────
    print("\n[Step 3] Phase 1: Collect forced-decoding data (5 prompts)")
    output_dir = f"/root/results/exp10/_smoke_{model_name}/paired_data"
    max_gen = MODEL_MAX_GEN.get(model_name, 128)

    # Clean old smoke data to avoid resume-from-stale-data
    import shutil
    smoke_root = Path(f"/root/results/exp10/_smoke_{model_name}")
    if smoke_root.exists():
        shutil.rmtree(smoke_root)
        print("  Cleaned old smoke data")

    t1 = time.time()
    metadata = collect_paired_data(
        model_name=model_name,
        device="cuda:0",
        output_dir=output_dir,
        n_prompts=5,
        max_gen_tokens=max_gen,
        dataset_path="/root/data/eval_dataset_v2.jsonl",
        tuned_lens_dir="/root/tuned_lens_probes",
        compute_kl_gradient=True,
    )
    t1_elapsed = time.time() - t1
    print(f"  Phase 1 took {t1_elapsed:.1f}s")

    check("phase1 completed", metadata is not None)
    check("phase1 prompts", metadata["n_prompts_processed"] == 5,
          f"{metadata['n_prompts_processed']} prompts")
    check("phase1 tokens", metadata["total_tokens"] > 0,
          f"{metadata['total_tokens']} tokens")
    check("phase1 target", metadata.get("regression_target") == "delta_kl",
          f"target={metadata.get('regression_target')}")
    check("phase1 tuned_lens", metadata.get("use_tuned_lens", False),
          f"use_tuned_lens={metadata.get('use_tuned_lens')}")

    # Validate accumulator files
    acc_dir = Path(output_dir) / "accumulators"
    check("accumulators dir", acc_dir.exists())
    sentinel = acc_dir / "target_version.txt"
    check("version sentinel", sentinel.exists() and sentinel.read_text().strip() == "delta_kl")

    # Check accumulator count matches n_layers
    n_acc = len(list(acc_dir.glob("layer_*_XtX.npy")))
    check("accumulator count", n_acc == spec.n_layers,
          f"{n_acc}/{spec.n_layers} layers")

    # Validate delta_kl.npy shape
    pca_dir = Path(output_dir) / "pca_subsample"
    dkl_path = pca_dir / "delta_kl.npy"
    check("delta_kl.npy exists", dkl_path.exists())
    if dkl_path.exists():
        dkl = np.load(dkl_path)
        check("delta_kl shape", dkl.ndim == 2 and dkl.shape[1] == spec.n_layers,
              f"shape={dkl.shape}, expected (*, {spec.n_layers})")
        # Sanity: values differ across layers (not the old shared Δc)
        if dkl.shape[0] > 1:
            col_means = dkl.mean(axis=0)
            all_same = np.allclose(col_means, col_means[0], atol=1e-6)
            check("delta_kl varies by layer", not all_same,
                  "layer means differ (good)" if not all_same else "WARNING: all layer means identical")

    # Sanity: accumulators at different layers have different ysum
    # (proves per-layer target, not shared Δc)
    if n_acc >= 2:
        from src.poc.exp10.collect_paired import LayerAccumulator
        acc_0 = LayerAccumulator.load(acc_dir, 0, spec.d_model)
        acc_mid = LayerAccumulator.load(acc_dir, spec.n_layers // 2, spec.d_model)
        check("accum ysum differs", abs(acc_0.ysum - acc_mid.ysum) > 1e-6,
              f"layer0={acc_0.ysum:.4f}, layer{spec.n_layers//2}={acc_mid.ysum:.4f}")

    # Check JSONL
    jsonl_files = list(Path(output_dir).glob("forced_decoding_*.jsonl"))
    check("JSONL output", len(jsonl_files) > 0)
    if jsonl_files:
        with open(jsonl_files[0]) as f:
            first_record = json.loads(f.readline())
        check("JSONL has target field", first_record.get("target") == "delta_kl")
        check("JSONL has convergence_gap", "convergence_gap" in first_record)

    # GPU memory after Phase 1 cleanup
    torch.cuda.empty_cache()
    mem_used = torch.cuda.memory_allocated() / 1e9
    print(f"  GPU memory after Phase 1 cleanup: {mem_used:.2f} GB")

    # ── 4. Phase 2: Ridge probes ──────────────────────────────────────────────
    print("\n[Step 4] Phase 2: Ridge probes")
    probes_dir = f"/root/results/exp10/_smoke_{model_name}/probes"
    mean_dir_for_train = mean_dir if mean_dir else "/dev/null"

    t2 = time.time()
    summary = train_probes(
        model_name=model_name,
        paired_data_dir=output_dir,
        mean_dir_path=mean_dir_for_train,
        output_dir=probes_dir,
    )
    t2_elapsed = time.time() - t2
    print(f"  Phase 2 took {t2_elapsed:.1f}s")

    check("phase2 completed", summary is not None)
    check("phase2 target", summary.get("regression_target") == "delta_kl")
    check("phase2 go_nogo", summary.get("go_nogo") in ("proceed", "redundant"),
          f"go_nogo={summary.get('go_nogo')}")

    # Check output files
    conv_dir = Path(probes_dir) / "convergence_directions.npz"
    compat_dir = Path(probes_dir) / "commitment_directions.npz"
    probe_json = Path(probes_dir) / "probe_summary.json"
    check("convergence_directions.npz", conv_dir.exists())
    check("commitment_directions.npz (compat)", compat_dir.exists())
    check("probe_summary.json", probe_json.exists())

    if conv_dir.exists():
        dirs = np.load(conv_dir)
        n_dir_layers = len([k for k in dirs.files if k.startswith("layer_")])
        check("direction count", n_dir_layers == spec.n_layers,
              f"{n_dir_layers}/{spec.n_layers}")
        # Check a direction has unit norm (or zero for skipped layers)
        d_mid = dirs[f"layer_{spec.n_layers // 2}"]
        norm = np.linalg.norm(d_mid)
        check("direction norm", abs(norm - 1.0) < 0.01 or norm < 1e-6,
              f"layer {spec.n_layers//2} norm={norm:.4f}")

    # Per-layer R² sanity
    per_layer = summary.get("per_layer", [])
    if per_layer:
        r2_values = [r["r2_test"] for r in per_layer if r["r2_test"] > 0]
        peak_r2 = max(r["r2_test"] for r in per_layer)
        peak_layer = max(per_layer, key=lambda r: r["r2_test"])["layer"]
        check("R² non-trivial", peak_r2 > 0.0,
              f"peak R²={peak_r2:.4f} at layer {peak_layer}, {len(r2_values)} layers with R²>0")

    # ── 5. Phase 3: Patching (3 prompts, 2 tokens) ───────────────────────────
    print("\n[Step 5] Phase 3: Causal patching (3 prompts, 2 tokens)")
    patching_dir = f"/root/results/exp10/_smoke_{model_name}/patching"

    t3 = time.time()
    validate_patching(
        model_name=model_name,
        device="cuda:0",
        probes_dir=probes_dir,
        paired_data_dir=output_dir,
        output_dir=patching_dir,
        mean_dir_path=mean_dir,
        n_test_prompts=3,
        max_tokens_per_prompt=2,
        top_k_layers=3,  # only 3 layers to keep it fast
        dataset_path="/root/data/eval_dataset_v2.jsonl",
        tuned_lens_dir="/root/tuned_lens_probes",
    )
    t3_elapsed = time.time() - t3
    print(f"  Phase 3 took {t3_elapsed:.1f}s")

    patch_summary = Path(patching_dir) / "patching_summary.json"
    check("patching_summary.json", patch_summary.exists())
    if patch_summary.exists():
        with open(patch_summary) as f:
            pdata = json.load(f)
        # Check we have results for multiple conditions
        all_conditions = set()
        for layer_key, layer_data in pdata.items():
            if isinstance(layer_data, dict):
                all_conditions.update(layer_data.keys())
        expected_conds = {"commit", "full", "random"}
        check("patching conditions", expected_conds.issubset(all_conditions),
              f"found: {all_conditions}")
        # Check mean_delta_kl_downstream exists in at least one condition
        has_dkl = False
        for layer_key, layer_data in pdata.items():
            if isinstance(layer_data, dict):
                for cond, cond_data in layer_data.items():
                    if isinstance(cond_data, dict) and "mean_delta_kl_downstream" in cond_data:
                        has_dkl = True
                        break
        check("ΔKL downstream metric", has_dkl)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_time = time.time() - t0
    n_pass = sum(1 for _, ok, _ in checks if ok)
    n_fail = sum(1 for _, ok, _ in checks if not ok)

    _commit_volumes()

    print(f"\n{'='*60}")
    print(f"SMOKE TEST: {n_pass} passed, {n_fail} failed ({total_time:.0f}s total)")
    print(f"  Phase 1 (collect):  {t1_elapsed:.0f}s")
    print(f"  Phase 2 (probes):   {t2_elapsed:.0f}s")
    print(f"  Phase 3 (patching): {t3_elapsed:.0f}s")
    if errors:
        print(f"\nFAILED CHECKS:")
        for e in errors:
            print(f"  - {e}")
    print(f"{'='*60}")

    status = "PASS" if n_fail == 0 else f"FAIL ({n_fail} failures)"
    return f"smoke {model_name}: {status} — {n_pass}/{n_pass+n_fail} checks, {total_time:.0f}s"


# ── GCS sync helper ──────────────────────────────────────────────────────────

@app.function(
    timeout=600,
    image=image,
    volumes=VOLUME_MOUNTS,
)
def sync_to_gcs() -> str:
    """Sync results to GCS bucket."""
    _setup()
    results_vol.reload()

    import subprocess as sp
    try:
        sp.run([
            "gsutil", "-m", "rsync", "-r",
            "/root/results/exp10/",
            "gs://pt-vs-it-results/exp10/",
        ], check=True, timeout=300)
        return "GCS sync done"
    except Exception as e:
        return f"GCS sync failed: {e}"


# ── Local entrypoints ────────────────────────────────────────────────────────

@app.local_entrypoint()
def run_smoke():
    """Smoke test: 1 model (deepseek_v2_lite — smallest), 5 prompts, all phases.

    Usage: modal run src/poc/exp10/modal_exp10.py::run_smoke
    """
    # deepseek_v2_lite: 27 layers, d=2048, MoE — smallest model, fastest to load
    print("=" * 60)
    print("EXP10 SMOKE TEST — deepseek_v2_lite, 5 prompts")
    print("=" * 60)
    result = smoke_test_gpu.remote("deepseek_v2_lite")
    print(f"\n{result}")


@app.local_entrypoint()
def run_prototype():
    """Prototype: 2 models in parallel, 10 prompts, Phases 1→2→3.

    Uses 2 B200 GPUs simultaneously. Tests parallelism + cross-model correctness.
    Models: deepseek_v2_lite (MoE, 27 layers) + gemma3_4b (dense, 34 layers).

    Usage: modal run --detach src/poc/exp10/modal_exp10.py::run_prototype
    """
    proto_models = ["deepseek_v2_lite", "gemma3_4b"]

    print("=" * 60)
    print(f"EXP10 PROTOTYPE — {proto_models}, 10 prompts each")
    print("=" * 60)

    # Phase 1: 2 GPU jobs in parallel
    print("\n[Phase 1] Collecting (2 models × 10 prompts, parallel)...")
    results_1 = list(collect_one.map(proto_models, [10, 10]))
    for r in results_1:
        print(f"  {r}")

    # Phase 2: 2 CPU jobs in parallel (no GPU contention)
    print("\n[Phase 2] Ridge probes (parallel, CPU)...")
    results_2 = list(train_probes_one.map(proto_models))
    for r in results_2:
        print(f"  {r}")

    # Phase 3: 2 GPU jobs in parallel
    print("\n[Phase 3] Patching (2 models × 5 prompts × 3 tokens, parallel)...")
    results_3 = list(patch_one.map(
        proto_models,
        [5, 5],           # n_test_prompts
        [3, 3],           # max_tokens_per_prompt
    ))
    for r in results_3:
        print(f"  {r}")

    # Quick validation of output structure
    print("\n[Validate] Checking outputs on volume...")
    for m in proto_models:
        try:
            subprocess.run([
                "modal", "volume", "ls", "exp10-results",
                f"exp10/{m}/probes/",
            ], check=True, timeout=30)
        except Exception as e:
            print(f"  {m}: ls failed — {e}")

    print("\n" + "=" * 60)
    print("PROTOTYPE COMPLETE")
    print("=" * 60)
    print("\nDownload results:")
    print("  modal volume get exp10-results exp10/_smoke_deepseek_v2_lite results/exp10/_smoke_deepseek_v2_lite")
    print("  modal volume get exp10-results exp10/_smoke_gemma3_4b results/exp10/_smoke_gemma3_4b")


@app.local_entrypoint()
def main():
    """Full run: 6 models, 600 prompts, all 4 phases.

    Usage: modal run --detach src/poc/exp10/modal_exp10.py
    """
    print("=" * 70)
    print("EXP10: CONTRASTIVE ACTIVATION PATCHING (FULL RUN)")
    print("=" * 70)

    # ── Phase 1: 6 parallel collection jobs ───────────────────────────────────
    print("\n[Phase 1] Collecting forced-decoding paired data (6 models)...")
    results_1 = list(collect_one.map(MODELS))
    for r in results_1:
        print(f"  {r}")

    # ── Phase 2+2.5: 6 parallel probe training (CPU — no GPU queue) ──────────
    print("\n[Phase 2+2.5] Training ridge probes + direction comparison...")
    results_2 = list(train_probes_one.map(MODELS))
    for r in results_2:
        print(f"  {r}")

    # ── Phase 3: 6 parallel patching ─────────────────────────────────────────
    print("\n[Phase 3] Causal activation patching (6 models)...")
    results_3 = list(patch_one.map(MODELS))
    for r in results_3:
        print(f"  {r}")

    # ── Phase 4B: Conditional steering ────────────────────────────────────────
    print("\n[Phase 4B] Steering with commitment directions (conditional)...")
    results_4 = list(steer_one.map(MODELS))
    for r in results_4:
        print(f"  {r}")

    # ── Pull results locally ──────────────────────────────────────────────────
    print("\n[Download] Pulling results from Modal volume...")
    try:
        subprocess.run([
            "modal", "volume", "get",
            "exp10-results", "exp10",
            "results/exp10",
        ], check=True, timeout=300)
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Run manually: modal volume get exp10-results exp10 results/exp10")

    print("\n" + "=" * 70)
    print("EXP10 COMPLETE")
    print("=" * 70)
    print("\nGenerate plots:")
    print("  uv run python scripts/plot_exp10.py")
    print("\nPush to GCS:")
    print("  gsutil -m rsync -r results/exp10/ gs://pt-vs-it-results/exp10/")
