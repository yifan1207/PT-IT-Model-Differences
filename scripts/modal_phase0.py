"""
Modal script for Phase 0: Multi-model steering experiments.

Runs the full Phase 0 pipeline on Modal cloud GPUs:
  1. Precompute corrective directions (6 models × 4 phases)
  2. A1 α-sweep steering (6 models × 2 workers, 18 conditions each)
  3. Merge workers + LLM judge (CPU/API-bound)
  4. PCA rank-1 check (1A), ID under steering (1C)
  5. Commitment delay post-processing (1B, CPU)

Cost estimate (B200 @ ~$6.25/GPU-hr):
  Precompute gen:      10 jobs × 1.5h  = 15 GPU-hr  ~$94
  Precompute acts:     10 jobs × 1.7h  = 17 GPU-hr  ~$106
  A1 steering:         12 jobs × 4h    = 48 GPU-hr  ~$300
  PCA + ID-steering:   12 jobs × 0.2h  =  2 GPU-hr  ~$13
  ─────────────────────────────────────────────────────────
  Total GPU:            82 GPU-hr                    ~$513
  LLM judge (Gemini):  ~44,000 calls               ~$4
  GRAND TOTAL:                                       ~$517
  Note: B200 ~2-3× faster than A100, model weights baked into image.

Wall time with 12 containers: ~12-14h

Setup (one-time):
    # 1. Modal auth (already done if you see this)
    modal token set ...

    # 2. Secrets (already exist):
    #    huggingface-token   → HF_TOKEN
    #    OPENROUTER_API_KEY  → OPENROUTER_API_KEY

    # 3. Run:
    modal run --detach scripts/modal_phase0.py

    # 4. Monitor:
    modal app logs phase0-multimodel

    # 5. Download results:
    modal volume get phase0-results cross_model results/cross_model

    # 6. Sync to GCS:
    gsutil -m rsync -r results/cross_model/ gs://pt-vs-it-results/cross_model/
"""
from __future__ import annotations

import itertools
import os
import subprocess
import sys
from pathlib import Path

import modal

# ── Modal app & infrastructure ────────────────────────────────────────────────

app = modal.App("phase0-multimodel")

results_vol = modal.Volume.from_name("phase0-results", create_if_missing=True)
# HF cache volume no longer needed — model weights are baked into the image.
# This avoids the "cannot mount volume on non-empty path" conflict.

# HuggingFace model IDs to pre-download into the image (IT variants only for steering)
_HF_MODELS_TO_BAKE = [
    "google/gemma-3-4b-it",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B",              # IT = base for Qwen3
    "mistralai/Mistral-7B-Instruct-v0.3",
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    "allenai/OLMo-2-1124-7B-Instruct",
]

# Container image with all dependencies + pre-downloaded model weights
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")  # for circuit-tracer install + git rev-parse
    .pip_install(
        "torch>=2.7.0",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "transformers==4.57.3",       # pinned — DeepSeek breaks on >=5.0
        "accelerate",
        "numpy",
        "scipy",
        "scikit-dimension>=0.3.4",    # TwoNN intrinsic dimensionality
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "tqdm",
        "matplotlib",
        "openai",                     # OpenRouter LLM judge client
        "httpx",
        "requests",
        "nnsight>=0.6.0",            # model wrapping for steering
        "transformer-lens>=2.17.0",   # needed by circuit-tracer
    )
    .pip_install(
        "circuit-tracer @ git+https://github.com/safety-research/circuit-tracer.git@26a976e",
    )
    # Pre-download all IT model weights into the image — eliminates re-download
    # on every preemption restart. Modal caches images, so this is a one-time cost.
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_commands(
        *[
            f"huggingface-cli download {model_id}"
            for model_id in _HF_MODELS_TO_BAKE
        ],
        secrets=[modal.Secret.from_name("huggingface-token")],
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file(
        "data/eval_dataset_v2.jsonl",
        remote_path="/root/data/eval_dataset_v2.jsonl",
    )
    .add_local_file(
        "data/exp3_dataset.jsonl",
        remote_path="/root/data/exp3_dataset.jsonl",
    )
)

# ── Model registry ────────────────────────────────────────────────────────────

MODELS = [
    "gemma3_4b", "llama31_8b", "qwen3_4b",
    "mistral_7b", "deepseek_v2_lite", "olmo2_7b",
]

MODEL_INFO = {
    "gemma3_4b":        {"n_layers": 34, "d_model": 2560, "pb": 20, "max_gen": 200},
    "llama31_8b":       {"n_layers": 32, "d_model": 4096, "pb": 19, "max_gen": 200},
    "qwen3_4b":         {"n_layers": 36, "d_model": 2560, "pb": 22, "max_gen": 200},
    "mistral_7b":       {"n_layers": 32, "d_model": 4096, "pb": 19, "max_gen": 200},
    "deepseek_v2_lite": {"n_layers": 27, "d_model": 2048, "pb": 16, "max_gen": 64},
    "olmo2_7b":         {"n_layers": 32, "d_model": 4096, "pb": 19, "max_gen": 200},
}

N_WORKERS = 2  # workers per model for gen/acts/steer

# ── Helpers ───────────────────────────────────────────────────────────────────

VOLUME_MOUNTS = {
    "/root/results": results_vol,
}

GPU_RETRIES = modal.Retries(max_retries=10, initial_delay=10.0, backoff_coefficient=1.0)


def _setup():
    """Common setup: working directory, Python path, env vars."""
    import os
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/hub")
    import sys as _sys
    if "/root" not in _sys.path:
        _sys.path.insert(0, "/root")


def _run(cmd: list[str], timeout: int | None = None) -> str:
    """Run subprocess, print output, raise on failure."""
    import os
    import subprocess as sp
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    result = sp.run(
        cmd, capture_output=True, text=True,
        cwd="/root", env=env, timeout=timeout,
    )
    if result.stdout:
        print(result.stdout[-5000:])  # last 5K chars
    if result.returncode != 0:
        err = result.stderr[-3000:] if result.stderr else "(no stderr)"
        print(f"STDERR:\n{err}")
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd[:6])}...\n{err[-1000:]}"
        )
    return result.stdout


def _commit_volumes():
    """Commit results volume."""
    try:
        results_vol.commit()
    except Exception as e:
        print(f"[volume] results commit warn: {e}")


def _setup_sigterm_handler():
    """Graceful shutdown: commit volumes on SIGTERM/exit."""
    import atexit
    import signal

    def _shutdown(*_args):
        print("[shutdown] Committing volumes...")
        _commit_volumes()
        if _args:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)


# ── Phase 1: Precompute directions ────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=10800,  # 3h
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def precompute_gen(model_name: str, worker_index: int, n_workers: int) -> str:
    """Phase 1: Generate IT/PT texts + PT NLL for one model shard."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== precompute_gen: {model_name} w{worker_index}/{n_workers} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    _run([
        "python", "/root/scripts/precompute_directions_multimodel.py",
        "--model-name", model_name,
        "--phase", "gen",
        "--worker-index", str(worker_index),
        "--n-workers", str(n_workers),
        "--device", "cuda:0",
    ], timeout=10000)

    _commit_volumes()
    return f"gen {model_name} w{worker_index} done"


@app.function(
    timeout=7200,  # 2h (LLM judge is API-bound)
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0),
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("OPENROUTER_API_KEY"),
    ],
    memory=16384,
)
def precompute_score(model_name: str) -> str:
    """Phase 2: Score & select top-600 high-contrast records. CPU + LLM judge."""
    _setup()
    _setup_sigterm_handler()

    print(f"=== precompute_score: {model_name} ===")
    results_vol.reload()

    _run([
        "python", "/root/scripts/precompute_directions_multimodel.py",
        "--model-name", model_name,
        "--phase", "score",
    ], timeout=6000)

    _commit_volumes()
    return f"score {model_name} done"


@app.function(
    gpu="B200",
    timeout=10800,  # 3h
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def precompute_acts(model_name: str, worker_index: int, n_workers: int) -> str:
    """Phase 3: Collect MLP activations at all layers for selected records."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== precompute_acts: {model_name} w{worker_index}/{n_workers} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    _run([
        "python", "/root/scripts/precompute_directions_multimodel.py",
        "--model-name", model_name,
        "--phase", "acts",
        "--worker-index", str(worker_index),
        "--n-workers", str(n_workers),
        "--device", "cuda:0",
    ], timeout=10000)

    _commit_volumes()
    return f"acts {model_name} w{worker_index} done"


@app.function(
    timeout=1800,  # 30 min
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=16384,
)
def precompute_merge(model_name: str) -> str:
    """Phase 4: Merge worker activations into corrective_directions.npz."""
    _setup()

    print(f"=== precompute_merge: {model_name} ===")
    results_vol.reload()

    _run([
        "python", "/root/scripts/precompute_directions_multimodel.py",
        "--model-name", model_name,
        "--phase", "merge",
    ], timeout=1200)

    _commit_volumes()
    return f"merge {model_name} done"


# ── Phase 2: A1 α-sweep steering ─────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=86400,  # 24h — steering is the longest step
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
    max_containers=12,
)
def steer_model(model_name: str, worker_index: int, n_workers: int) -> str:
    """A1 α-sweep with logit-lens collection. One worker shard."""
    _setup()
    _setup_sigterm_handler()

    import threading
    import torch
    info = MODEL_INFO[model_name]
    print(f"=== steer: {model_name} w{worker_index}/{n_workers} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: n_layers={info['n_layers']} pb={info['pb']} max_gen={info['max_gen']}")

    results_vol.reload()

    # Periodic Volume commit (every 60s) — steering runs for hours.
    # If preempted, at most 60s of condition-level progress is lost.
    # exp6/run.py has condition-level resume via scores.jsonl, so a retry
    # skips all conditions already written.
    stop_event = threading.Event()

    def _periodic_commit():
        while not stop_event.is_set():
            stop_event.wait(timeout=60)
            if stop_event.is_set():
                break
            try:
                results_vol.commit()
            except Exception:
                pass

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    run_name = f"A1_{model_name}_it_v1"
    corr_dir = f"results/cross_model/{model_name}/directions/corrective_directions.npz"
    output_base = f"results/cross_model/{model_name}/exp6"

    try:
        _run([
            "python", "-m", "src.poc.exp6.run",
            "--experiment", "A1",
            "--model-name", model_name,
            "--variant", "it",
            "--dataset", "data/eval_dataset_v2.jsonl",
            "--n-eval-examples", "1400",
            "--device", "cuda:0",
            "--worker-index", str(worker_index),
            "--n-workers", str(n_workers),
            "--run-name", run_name,
            "--corrective-direction-path", corr_dir,
            "--output-base", output_base,
            "--proposal-boundary", str(info["pb"]),
            "--n-layers", str(info["n_layers"]),
            "--max-gen-tokens", str(info["max_gen"]),
            "--collect-logit-lens",
        ], timeout=82800)
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    _commit_volumes()
    return f"steer {model_name} w{worker_index} done"


# ── Phase 3: Merge & judge ────────────────────────────────────────────────────

@app.function(
    timeout=3600,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=16384,
)
def merge_workers(model_name: str) -> str:
    """Merge per-worker steering results into a single merged directory."""
    _setup()

    print(f"=== merge: {model_name} ===")
    results_vol.reload()

    run_name = f"A1_{model_name}_it_v1"
    output_base = f"results/cross_model/{model_name}/exp6"
    merged_name = f"merged_{run_name}"

    # Build source dirs
    src_dirs = [
        f"{output_base}/{run_name}_w{wi}"
        for wi in range(N_WORKERS)
    ]

    _run([
        "python", "/root/scripts/merge_steering_workers.py",
        "--experiment", "A1",
        "--variant", "it",
        "--n-workers", str(N_WORKERS),
        "--output-base", output_base,
        "--merged-name", merged_name,
        "--source-dirs", *src_dirs,
    ], timeout=3000)

    _commit_volumes()
    return f"merge {model_name} done"


@app.function(
    timeout=43200,  # 12h — orchestrates entire steer+merge
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=4096,
    nonpreemptible=True,  # CPU orchestrator must not be preempted — prevents cascade restarts
)
def orchestrate_steer(models: list[str], n_workers: int = 2) -> str:
    """Remote orchestrator for steer+merge — runs on Modal without local client.

    Spawns all steer workers in parallel, waits for completion, then merges.
    Use with `--detach` so it survives local client disconnect.
    """
    import time

    print(f"=== Orchestrating steer for {len(models)} models × {n_workers} workers ===")

    # Launch all steer jobs in parallel using spawn
    handles = []
    for m in models:
        for wi in range(n_workers):
            h = steer_model.spawn(m, wi, n_workers)
            handles.append((m, wi, h))
            print(f"  Spawned steer {m} w{wi}/{n_workers}")

    # Wait for all to complete
    results = []
    for m, wi, h in handles:
        try:
            result = h.get()
            print(f"  ✓ {result}")
            results.append(result)
        except Exception as e:
            msg = f"  ✗ steer {m} w{wi} FAILED: {e}"
            print(msg)
            results.append(msg)

    # Merge workers
    print(f"\nMerging {len(models)} models...")
    merge_handles = []
    for m in models:
        h = merge_workers.spawn(m)
        merge_handles.append((m, h))

    for m, h in merge_handles:
        try:
            result = h.get()
            print(f"  ✓ {result}")
            results.append(result)
        except Exception as e:
            msg = f"  ✗ merge {m} FAILED: {e}"
            print(msg)
            results.append(msg)

    # LLM judge (runs after merge, API-bound not GPU-bound)
    print(f"\nLaunching LLM judge for {len(models)} models...")
    judge_handles = []
    for m in models:
        h = judge_model.spawn(m)
        judge_handles.append((m, h))
        print(f"  Spawned judge {m}")

    for m, h in judge_handles:
        try:
            result = h.get()
            print(f"  ✓ {result}")
            results.append(result)
        except Exception as e:
            msg = f"  ✗ judge {m} FAILED: {e}"
            print(msg)
            results.append(msg)

    summary = "\n".join(results)
    print(f"\n=== STEER+MERGE+JUDGE COMPLETE ===\n{summary}")
    return summary


@app.function(
    timeout=14400,  # 4h — API-bound
    retries=modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=1.0),
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("OPENROUTER_API_KEY")],
    memory=8192,
)
def judge_model(model_name: str) -> str:
    """Post-hoc LLM judge (G1, G2, S1, S2) using Gemini 2.5 Flash."""
    _setup()
    _setup_sigterm_handler()

    print(f"=== judge: {model_name} ===")
    results_vol.reload()

    run_name = f"A1_{model_name}_it_v1"
    merged_dir = f"results/cross_model/{model_name}/exp6/merged_{run_name}"

    _run([
        "python", "/root/scripts/llm_judge.py",
        "--merged-dir", merged_dir,
        "--model", "google/gemini-2.5-flash",
        "--workers", "16",
        "--tasks", "g1", "g2", "s1", "s2",
    ], timeout=13000)

    _commit_volumes()
    return f"judge {model_name} done"


# ── Phase 4: Piggybacked experiments (1A, 1B, 1C) ────────────────────────────

@app.function(
    gpu="B200",
    timeout=7200,
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def pca_model(model_name: str) -> str:
    """Experiment 1A: PCA rank-1 check of corrective direction."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== pca (1A): {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    _run([
        "python", "-m", "src.poc.exp8.pca_rank1",
        "--model-name", model_name,
        "--device", "cuda:0",
        "--max-records", "200",
    ], timeout=6000)

    _commit_volumes()
    return f"pca {model_name} done"


@app.function(
    gpu="B200",
    timeout=7200,
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def id_steering_model(model_name: str) -> str:
    """Experiment 1C: TwoNN ID profiles at α=1.0, 0.0, -1.0."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== id_steering (1C): {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    _run([
        "python", "-m", "src.poc.exp8.id_under_steering",
        "--model-name", model_name,
        "--device", "cuda:0",
    ], timeout=6000)

    _commit_volumes()
    return f"id_steering {model_name} done"


@app.function(
    timeout=1800,
    image=image,
    volumes=VOLUME_MOUNTS,
    memory=8192,
)
def commitment_model(model_name: str) -> str:
    """Experiment 1B: Commitment delay vs α (CPU post-processing)."""
    _setup()

    print(f"=== commitment (1B): {model_name} ===")
    results_vol.reload()

    _run([
        "python", "-m", "src.poc.exp8.commitment_vs_alpha",
        "--model-name", model_name,
    ], timeout=1200)

    _commit_volumes()
    return f"commitment {model_name} done"


@app.function(
    gpu="B200",
    timeout=7200,
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def bootstrap_model(model_name: str) -> str:
    """Direction stability bootstrap (§4.4): 100 resamples, 200 records."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== bootstrap (§4.4): {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    _run([
        "python", "-m", "src.poc.exp8.direction_bootstrap",
        "--model-name", model_name,
        "--device", "cuda:0",
        "--max-records", "200",
        "--n-resamples", "100",
    ], timeout=6000)

    _commit_volumes()
    return f"bootstrap {model_name} done"


# ── Smoke test function ──────────────────────────────────────────────────────

@app.function(
    gpu="B200",
    timeout=600,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
)
def smoke_test() -> str:
    """Quick sanity check: verify image, imports, GPU, volume, dataset."""
    _setup()
    import torch
    import json
    from pathlib import Path

    checks = []

    # 1. GPU
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    checks.append(f"GPU: {gpu_name} ({vram:.0f} GB)")

    # 2. Core imports
    try:
        from src.poc.cross_model.config import MODEL_REGISTRY, get_spec
        checks.append(f"MODEL_REGISTRY: {len(MODEL_REGISTRY)} models")
        for m in MODELS:
            spec = get_spec(m)
            checks.append(f"  {m}: {spec.n_layers}L, d={spec.d_model}")
    except Exception as e:
        checks.append(f"FAIL cross_model imports: {e}")

    # 3. Steering adapter
    try:
        from src.poc.exp6.model_adapter import get_steering_adapter
        adapter = get_steering_adapter("llama31_8b")
        checks.append(f"SteeringAdapter: {adapter.model_name}, max_gen={adapter.max_gen_tokens}")
    except Exception as e:
        checks.append(f"FAIL model_adapter: {e}")

    # 4. Exp6 imports (needs nnsight + circuit_tracer)
    try:
        from src.poc.shared.model import LoadedModel
        checks.append("shared.model: OK (nnsight + circuit_tracer)")
    except Exception as e:
        checks.append(f"FAIL shared.model: {e}")

    # 5. Dataset
    dataset_path = Path("/root/data/eval_dataset_v2.jsonl")
    if dataset_path.exists():
        n_records = sum(1 for _ in open(dataset_path))
        checks.append(f"Dataset: {n_records} records")
    else:
        checks.append("FAIL: eval_dataset_v2.jsonl not found")

    # 6. Volume
    results_dir = Path("/root/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    test_file = results_dir / "_smoke_test.txt"
    test_file.write_text("ok")
    results_vol.commit()
    checks.append("Volume write+commit: OK")

    # 7. Scripts accessible
    scripts = [
        "/root/scripts/precompute_directions_multimodel.py",
        "/root/scripts/llm_judge.py",
        "/root/src/poc/exp8/pca_rank1.py",
        "/root/src/poc/exp8/id_under_steering.py",
        "/root/src/poc/exp8/commitment_vs_alpha.py",
        "/root/scripts/merge_steering_workers.py",
    ]
    for s in scripts:
        status = "OK" if Path(s).exists() else "MISSING"
        checks.append(f"  {Path(s).name}: {status}")

    # 8. Quick model load test (small model check)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite")
        checks.append(f"Tokenizer load (deepseek): vocab_size={tok.vocab_size}")
    except Exception as e:
        checks.append(f"Tokenizer load: {e}")

    result = "\n".join(checks)
    print(result)
    return result


# ── Local entrypoint ──────────────────────────────────────────────────────────

def _local_bootstrap_from_gcs():
    """Pull saved results from GCS and upload to Modal volume for resume."""
    import shutil
    import tempfile

    GCS_PATH = "gs://pt-vs-it-results/cross_model_phase0"

    print("=" * 70)
    print("BOOTSTRAP: Pulling saved results from GCS into Modal volume")
    print("=" * 70)

    # Step 1: Pull from GCS to local temp dir
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = os.path.join(tmpdir, "cross_model")
        os.makedirs(local_dir)
        print(f"\n[1/3] Downloading from {GCS_PATH} ...")
        subprocess.run(
            ["gsutil", "-m", "cp", "-r", f"{GCS_PATH}/*", local_dir + "/"],
            check=True, timeout=1800,
        )

        # Step 2: Upload to Modal volume
        print("\n[2/3] Uploading to Modal volume 'phase0-results' ...")
        # modal volume put <vol> <local_path> <remote_path>
        subprocess.run(
            ["modal", "volume", "put", "phase0-results", local_dir, "cross_model"],
            check=True, timeout=1800,
        )

    # Step 3: Verify
    print("\n[3/3] Verifying volume contents ...")
    for m in MODELS:
        result = subprocess.run(
            ["modal", "volume", "ls", "phase0-results", f"cross_model/{m}/directions/"],
            capture_output=True, text=True,
        )
        has_dirs = "npz" in result.stdout
        result2 = subprocess.run(
            ["modal", "volume", "ls", "phase0-results", f"cross_model/{m}/exp6/"],
            capture_output=True, text=True,
        )
        has_exp6 = "A1_" in result2.stdout
        print(f"  {m}: directions={'OK' if has_dirs else 'MISSING'}, exp6={'OK' if has_exp6 else 'none'}")

    print("\n=== BOOTSTRAP COMPLETE ===")
    print("You can now run: modal run scripts/modal_phase0.py --step steer --skip-precompute")


def _local_sync_gcs(desc: str):
    """Best-effort download from Volume + sync to GCS. Runs locally."""
    import shutil
    print(f"\n[GCS] Syncing intermediate results: {desc}")
    try:
        subprocess.run(
            ["modal", "volume", "get", "phase0-results", "cross_model",
             "results/cross_model"],
            check=True, capture_output=True, timeout=600,
        )
        print("[GCS] Downloaded from Volume")
    except Exception as e:
        print(f"[GCS] Volume download failed: {e}")
        return

    try:
        subprocess.run(
            ["gsutil", "-m", "rsync", "-r",
             "results/cross_model/", "gs://pt-vs-it-results/cross_model/"],
            check=True, capture_output=True, timeout=600,
        )
        print(f"[GCS] Synced to gs://pt-vs-it-results/cross_model/ ({desc})")
    except FileNotFoundError:
        print("[GCS] gsutil not found — skip GCS sync (download from Volume manually)")
    except Exception as e:
        print(f"[GCS] GCS sync failed: {e}")


@app.local_entrypoint()
def main(
    step: str = "all",
    model: str = "",
    skip_precompute: bool = False,
    dry_run: bool = False,
    sync_gcs: bool = True,
):
    """Orchestrate Phase 0 multi-model steering on Modal.

    Args:
        step: Which step to run. Options:
            "test"       — smoke test only
            "precompute" — direction extraction (phases 1-4)
            "steer"      — A1 α-sweep (+ merge)
            "judge"      — post-hoc LLM judge
            "analysis"   — PCA (1A), ID-steering (1C), commitment (1B)
            "bootstrap"  — pull saved results from GCS into volume for resume
            "all"        — full pipeline
        model: Run for a single model only (e.g., "llama31_8b")
        skip_precompute: Skip direction precompute (use existing on volume)
        dry_run: Print plan without launching jobs
        sync_gcs: Download from Volume + sync to GCS between phases (default True)
    """
    models = [model] if model else MODELS

    # ── Bootstrap from GCS ───────────────────────────────────────────────
    if step == "bootstrap":
        _local_bootstrap_from_gcs()
        return

    # ── Cost estimates ────────────────────────────────────────────────────
    print("=" * 70)
    print("PHASE 0: MULTI-MODEL STEERING — MODAL EXECUTION PLAN")
    print("=" * 70)
    n_models = len(models)
    n_precompute = n_models  # all models get precomputed

    est_gpu_hrs = {
        "precompute_gen":  n_precompute * N_WORKERS * 1.5,
        "precompute_acts": n_precompute * N_WORKERS * 1.7,
        "steer":           n_models * N_WORKERS * 8.0,
        "pca":             n_models * 0.3,
        "id_steering":     n_models * 0.3,
    }
    total_gpu_hrs = sum(est_gpu_hrs.values())
    cost_a100 = total_gpu_hrs * 2.78
    cost_h100 = total_gpu_hrs * 3.95

    print(f"\nModels: {', '.join(models)}")
    print(f"Step: {step}")
    print(f"\nEstimated GPU-hours:")
    for name, hrs in est_gpu_hrs.items():
        print(f"  {name:20s}: {hrs:6.1f}h")
    print(f"  {'TOTAL':20s}: {total_gpu_hrs:6.1f}h")
    print(f"\nEstimated cost:")
    print(f"  A100-80GB: ${cost_a100:,.0f}")
    print(f"  H100:      ${cost_h100:,.0f}")
    print(f"  LLM judge: ~$4 (Gemini 2.5 Flash via OpenRouter)")
    print(f"  TOTAL:     ${cost_a100 + 4:,.0f} - ${cost_h100 + 4:,.0f}")
    print()

    if dry_run:
        print("[dry-run] Would launch the above jobs. Exiting.")
        return

    # ── Smoke test ────────────────────────────────────────────────────────
    if step in ("test", "all"):
        print("─" * 70)
        print("SMOKE TEST")
        print("─" * 70)
        result = smoke_test.remote()
        print(result)
        if "FAIL" in result:
            print("\n*** SMOKE TEST HAS FAILURES — fix before proceeding ***")
            if step == "test":
                return
            # Continue for "all" — some failures may be non-blocking
        if step == "test":
            return

    # ── Precompute directions ─────────────────────────────────────────────
    if step in ("precompute", "all") and not skip_precompute:
        print("\n" + "─" * 70)
        print("PRECOMPUTE: Direction extraction (Phases 1-4)")
        print("─" * 70)

        # Phase 1: gen (GPU, parallel)
        print(f"\n[Phase 1] gen — {n_precompute} models × {N_WORKERS} workers")
        gen_args = [
            (m, wi, N_WORKERS)
            for m in models
            for wi in range(N_WORKERS)
        ]
        gen_results = list(precompute_gen.starmap(gen_args))
        for r in gen_results:
            print(f"  ✓ {r}")

        # Phase 2: score (CPU + LLM judge, sequential per model)
        print(f"\n[Phase 2] score — {n_precompute} models (CPU + LLM judge)")
        score_results = list(precompute_score.map(models))
        for r in score_results:
            print(f"  ✓ {r}")

        # Phase 3: acts (GPU, parallel)
        print(f"\n[Phase 3] acts — {n_precompute} models × {N_WORKERS} workers")
        acts_args = [
            (m, wi, N_WORKERS)
            for m in models
            for wi in range(N_WORKERS)
        ]
        acts_results = list(precompute_acts.starmap(acts_args))
        for r in acts_results:
            print(f"  ✓ {r}")

        # Phase 4: merge (CPU)
        print(f"\n[Phase 4] merge — {n_precompute} models (CPU)")
        merge_results = list(precompute_merge.map(models))
        for r in merge_results:
            print(f"  ✓ {r}")

        print("\n=== PRECOMPUTE COMPLETE ===")
        if sync_gcs:
            _local_sync_gcs("precompute directions")

    # ── A1 α-sweep steering ───────────────────────────────────────────────
    if step in ("steer", "all"):
        print("\n" + "─" * 70)
        print("STEER: A1 α-sweep (18 conditions × 1400 records, logit-lens on)")
        print("─" * 70)

        # Use remote orchestrator so it survives local client disconnect.
        # Run with `modal run --detach` for fire-and-forget.
        print(f"Launching remote orchestrator for {n_models} models × {N_WORKERS} workers...")
        print("(The orchestrator runs entirely on Modal — safe to disconnect.)")
        result = orchestrate_steer.remote(models, N_WORKERS)
        print(result)

        # Merge already done inside orchestrator
        merge_results = []
        for r in merge_results:
            print(f"  ✓ {r}")

        print("\n=== STEERING COMPLETE ===")
        if sync_gcs:
            _local_sync_gcs("steering results")

    # ── LLM judge ─────────────────────────────────────────────────────────
    if step in ("judge", "all"):
        print("\n" + "─" * 70)
        print("JUDGE: Post-hoc LLM evaluation (G1, G2, S1, S2)")
        print("─" * 70)

        judge_results = list(judge_model.map(models))
        for r in judge_results:
            print(f"  ✓ {r}")

        print("\n=== JUDGE COMPLETE ===")
        if sync_gcs:
            _local_sync_gcs("LLM judge scores")

    # ── Analysis experiments (1A, 1B, 1C) ─────────────────────────────────
    if step in ("analysis", "all"):
        print("\n" + "─" * 70)
        print("ANALYSIS: 1A (PCA), 1C (ID-steering), 1B (commitment)")
        print("─" * 70)

        # PCA and ID-steering run in parallel (both GPU)
        print(f"\n[1A] PCA + [1C] ID-steering — {n_models} models each (GPU)")
        pca_args = [(m,) for m in models]
        id_args = [(m,) for m in models]

        # Launch PCA and ID-steering together
        pca_futures = list(pca_model.starmap(pca_args))
        for r in pca_futures:
            print(f"  ✓ {r}")

        id_futures = list(id_steering_model.starmap(id_args))
        for r in id_futures:
            print(f"  ✓ {r}")

        # Commitment: CPU post-processing after steering logit-lens data
        print(f"\n[1B] Commitment vs α — {n_models} models (CPU)")
        commit_results = list(commitment_model.map(models))
        for r in commit_results:
            print(f"  ✓ {r}")

        print("\n=== ANALYSIS COMPLETE ===")
        if sync_gcs:
            _local_sync_gcs("analysis (PCA + ID + commitment)")

    # ── Direction bootstrap (§4.4) ────────────────────────────────────────
    if step in ("dir-bootstrap", "all"):
        print("\n" + "─" * 70)
        print("DIRECTION BOOTSTRAP: 100 resamples × 200 records per model (GPU)")
        print("─" * 70)

        bs_args = [(m,) for m in models]
        bs_results = list(bootstrap_model.starmap(bs_args))
        for r in bs_results:
            print(f"  ✓ {r}")

        print("\n=== DIRECTION BOOTSTRAP COMPLETE ===")
        if sync_gcs:
            _local_sync_gcs("direction bootstrap")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PHASE 0 COMPLETE")
    print("=" * 70)
    print("\nDownload all results:")
    print("  modal volume get phase0-results cross_model results/cross_model")
    print("\nSync to GCS:")
    print("  gsutil -m rsync -r results/cross_model/ gs://pt-vs-it-results/cross_model/")
    print("\nGenerate plots locally:")
    print("  uv run python scripts/plot_phase0_multimodel_dose_response.py")
    print("  uv run python scripts/plot_validation_tier0.py")
