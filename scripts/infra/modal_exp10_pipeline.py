"""
Unified Modal pipeline: Recompute directions + dual steering + exp10.

Runs the full pipeline on Modal B200 GPUs with chat template for IT models:
  Phase A: Precompute exp8 corrective directions (IT with chat template)
  Phase B: Exp10 paired data collection + ridge probes
  Phase C: A1 alpha-sweep with BOTH direction types x 6 models
  Phase D: Merge workers + LLM judge
  Phase E: Piggybacked analyses (PCA, ID-under-steering, commitment, patching)

Volumes:
  phase0-results-v2   — directions + steering results (RW)
  exp10-results        — exp10 paired data, probes, patching (RW)
  0g-probes-v2         — tuned lens probe weights (RO)

Cost estimate (B200 @ ~$6.25/GPU-hr):
  Phase A (precompute):     38 GPU-hr   ~$238
  Phase B (exp10):           2 GPU-hr    ~$13
  Phase C (dual steering):  96 GPU-hr   ~$600
  Phase E (piggybacked):     6 GPU-hr    ~$38
  LLM judge (Gemini):                     ~$7
  Grand total:             142 GPU-hr   ~$896

Usage:
    # Smoke test:
    modal run scripts/modal_unified_pipeline.py::smoke_test

    # Full run:
    modal run --detach scripts/modal_unified_pipeline.py

    # Individual phases (for debugging):
    modal run scripts/modal_unified_pipeline.py::run_phase_a
    modal run scripts/modal_unified_pipeline.py::run_phase_b

    # Download results:
    modal volume get phase0-results-v2 cross_model results/cross_model
    modal volume get exp10-results exp10 results/exp10_contrastive_activation_patching
"""
from __future__ import annotations

import os
import subprocess as sp
import sys
import threading
from pathlib import Path

import modal

# ── Modal app & infrastructure ────────────────────────────────────────────────

app = modal.App("unified-pipeline-v2")

results_vol = modal.Volume.from_name("phase0-results-v2", create_if_missing=True)
exp10_vol = modal.Volume.from_name("exp10-results", create_if_missing=True)
probes_vol = modal.Volume.from_name("0g-probes-v3")  # read-only, tuned lens probes (v3: IT trained with chat template)

# All 12 model weights (PT + IT) baked into image
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
    .env({"HF_HOME": "/root/.cache/huggingface"})
    .run_commands(
        *[f"huggingface-cli download {mid}" for mid in _HF_MODELS_TO_BAKE],
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

# ── Constants ────────────────────────────────────────────────────────────────

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

N_WORKERS = 2

VOLUME_MOUNTS = {
    "/root/results": results_vol,
    "/root/exp10_results": exp10_vol,
    "/root/probes": probes_vol,
}

GPU_RETRIES = modal.Retries(max_retries=10, initial_delay=10.0, backoff_coefficient=1.0)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _setup():
    os.chdir("/root")
    os.environ["PYTHONPATH"] = "/root"
    os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/root/.cache/huggingface/datasets")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/root/.cache/huggingface/hub")
    if "/root" not in sys.path:
        sys.path.insert(0, "/root")


def _run(cmd: list[str], timeout: int | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    result = sp.run(cmd, capture_output=True, text=True, cwd="/root", env=env, timeout=timeout)
    if result.stdout:
        print(result.stdout[-5000:])
    if result.returncode != 0:
        err = result.stderr[-3000:] if result.stderr else "(no stderr)"
        print(f"STDERR:\n{err}")
        raise RuntimeError(
            f"Command failed (exit {result.returncode}): {' '.join(cmd[:6])}...\n{err[-1000:]}"
        )
    return result.stdout


def _commit_all():
    for name, vol in [("results", results_vol), ("exp10", exp10_vol)]:
        try:
            vol.commit()
        except Exception as e:
            print(f"[volume] {name} commit warn: {e}")


def _setup_sigterm():
    import atexit, signal
    def _shutdown(*_args):
        print("[shutdown] Committing volumes...")
        _commit_all()
        if _args:
            raise SystemExit(1)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)


def _periodic_commit_thread():
    """Returns (thread, stop_event) — call stop_event.set() to stop."""
    stop = threading.Event()
    def _loop():
        while not stop.is_set():
            stop.wait(timeout=60)
            if not stop.is_set():
                _commit_all()
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t, stop


# ══════════════════════════════════════════════════════════════════════════════
# Phase A: Precompute exp8 corrective directions (WITH chat template for IT)
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="B200", timeout=10800, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def precompute_gen(model_name: str, worker_index: int, n_workers: int) -> str:
    """Phase A.1: Generate IT (with template) + PT (raw) texts."""
    _setup(); _setup_sigterm()
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
    _commit_all()
    return f"gen {model_name} w{worker_index} done"


@app.function(
    timeout=7200, image=image, volumes=VOLUME_MOUNTS,
    retries=modal.Retries(max_retries=3, initial_delay=10.0, backoff_coefficient=1.0),
    secrets=[
        modal.Secret.from_name("huggingface-token"),
        modal.Secret.from_name("OPENROUTER_API_KEY"),
    ],
    memory=16384,
)
def precompute_score(model_name: str) -> str:
    """Phase A.2: Score + select top-600 records (LLM judge, CPU)."""
    _setup(); _setup_sigterm()
    print(f"=== precompute_score: {model_name} ===")
    results_vol.reload()
    _run([
        "python", "/root/scripts/precompute_directions_multimodel.py",
        "--model-name", model_name,
        "--phase", "score",
    ], timeout=6000)
    _commit_all()
    return f"score {model_name} done"


@app.function(
    gpu="B200", timeout=10800, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def precompute_acts(model_name: str, worker_index: int, n_workers: int) -> str:
    """Phase A.3: Collect MLP activations at all layers (IT with template, PT raw)."""
    _setup(); _setup_sigterm()
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
    _commit_all()
    return f"acts {model_name} w{worker_index} done"


@app.function(
    timeout=1800, image=image, volumes=VOLUME_MOUNTS, memory=16384,
)
def precompute_merge(model_name: str) -> str:
    """Phase A.4: Merge worker activations -> corrective_directions.npz."""
    _setup()
    print(f"=== precompute_merge: {model_name} ===")
    results_vol.reload()
    _run([
        "python", "/root/scripts/precompute_directions_multimodel.py",
        "--model-name", model_name,
        "--phase", "merge",
    ], timeout=1200)
    _commit_all()
    return f"merge {model_name} done"


# ══════════════════════════════════════════════════════════════════════════════
# Phase B: Exp10 paired data + ridge probes
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="B200", timeout=3600, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def exp10_collect(model_name: str) -> str:
    """Phase B.1: Forced-decoding paired data collection (IT with template)."""
    _setup(); _setup_sigterm()
    import torch
    print(f"=== exp10_collect: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results_vol.reload()
    exp10_vol.reload()

    output_dir = f"/root/exp10_results/{model_name}/paired_data"
    # Tuned lens probes at /root/probes/{model}/{variant}/probe_layer_*.pt
    tuned_lens_dir = "/root/probes"

    _run([
        "python", "-m", "src.poc.exp10_contrastive_activation_patching.collect_paired",
        "--model", model_name,
        "--device", "cuda:0",
        "--output-dir", output_dir,
        "--n-prompts", "600",
        "--max-gen-tokens", str(MODEL_INFO[model_name]["max_gen"]),
        "--tuned-lens-dir", tuned_lens_dir,
    ], timeout=3000)

    _commit_all()
    return f"exp10_collect {model_name} done"


@app.function(
    timeout=1800, image=image, volumes=VOLUME_MOUNTS, memory=32768,
)
def exp10_probes(model_name: str) -> str:
    """Phase B.2: Ridge probes + direction comparison (CPU-heavy)."""
    _setup()
    print(f"=== exp10_probes: {model_name} ===")
    results_vol.reload()
    exp10_vol.reload()

    paired_dir = f"/root/exp10_results/{model_name}/paired_data"
    output_dir = f"/root/exp10_results/{model_name}/probes"
    # Mean IT-PT direction from Phase A (for cosine comparison)
    mean_dir = f"/root/results/cross_model/{model_name}/directions/corrective_directions.npz"

    _run([
        "python", "-m", "src.poc.exp10_contrastive_activation_patching.train_probes",
        "--model", model_name,
        "--paired-data-dir", paired_dir,
        "--mean-dir-path", mean_dir,
        "--output-dir", output_dir,
    ], timeout=1500)

    _commit_all()
    return f"exp10_probes {model_name} done"


# ══════════════════════════════════════════════════════════════════════════════
# Phase C: Unified A1 alpha-sweep with BOTH direction types
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="B200", timeout=86400, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536, max_containers=24,
)
def steer_one(model_name: str, dir_type: str, worker_index: int, n_workers: int) -> str:
    """A1 alpha-sweep with one direction type. IT uses chat template (default).

    dir_type: "exp8" uses corrective_directions.npz, "exp10" uses commitment_directions.npz.
    """
    _setup(); _setup_sigterm()
    import torch
    info = MODEL_INFO[model_name]
    print(f"=== steer: {model_name} {dir_type} w{worker_index}/{n_workers} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results_vol.reload()
    exp10_vol.reload()

    thread, stop = _periodic_commit_thread()

    if dir_type == "exp8":
        corr_dir = f"/root/results/cross_model/{model_name}/directions/corrective_directions.npz"
        run_name = f"A1_{model_name}_it_exp8_v2"
        output_base = f"/root/results/cross_model/{model_name}/exp6"
    elif dir_type == "exp10":
        corr_dir = f"/root/exp10_results/{model_name}/probes/commitment_directions.npz"
        run_name = f"A1_{model_name}_it_exp10_v1"
        output_base = f"/root/results/cross_model/{model_name}/exp6"
    else:
        raise ValueError(f"Unknown dir_type: {dir_type}")

    try:
        _run([
            "python", "-m", "src.poc.exp06_corrective_direction_steering.run",
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
        stop.set()
        thread.join(timeout=5)

    _commit_all()
    return f"steer {model_name} {dir_type} w{worker_index} done"


# ══════════════════════════════════════════════════════════════════════════════
# Phase D: Merge workers + LLM judge
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    timeout=3600, image=image, volumes=VOLUME_MOUNTS, memory=16384,
)
def merge_workers(model_name: str, dir_type: str) -> str:
    """Merge per-worker steering results into merged directory."""
    _setup()
    print(f"=== merge: {model_name} {dir_type} ===")
    results_vol.reload()

    if dir_type == "exp8":
        run_name = f"A1_{model_name}_it_exp8_v2"
    else:
        run_name = f"A1_{model_name}_it_exp10_v1"

    output_base = f"/root/results/cross_model/{model_name}/exp6"
    merged_name = f"merged_{run_name}"
    src_dirs = [f"{output_base}/{run_name}_w{wi}" for wi in range(N_WORKERS)]

    _run([
        "python", "/root/scripts/merge_steering_workers.py",
        "--experiment", "A1",
        "--variant", "it",
        "--n-workers", str(N_WORKERS),
        "--output-base", output_base,
        "--merged-name", merged_name,
        "--source-dirs", *src_dirs,
    ], timeout=3000)

    _commit_all()
    return f"merge {model_name} {dir_type} done"


@app.function(
    timeout=14400, image=image, volumes=VOLUME_MOUNTS,
    retries=modal.Retries(max_retries=3, initial_delay=30.0, backoff_coefficient=1.0),
    secrets=[modal.Secret.from_name("OPENROUTER_API_KEY")],
    memory=8192,
)
def judge_one(model_name: str, dir_type: str) -> str:
    """LLM judge (G1, G2, S1, S2) on merged steering results."""
    _setup(); _setup_sigterm()
    print(f"=== judge: {model_name} {dir_type} ===")
    results_vol.reload()

    if dir_type == "exp8":
        run_name = f"A1_{model_name}_it_exp8_v2"
    else:
        run_name = f"A1_{model_name}_it_exp10_v1"

    merged_dir = f"/root/results/cross_model/{model_name}/exp6/merged_{run_name}"

    _run([
        "python", "/root/scripts/llm_judge.py",
        "--merged-dir", merged_dir,
        "--model", "google/gemini-2.5-flash",
        "--workers", "16",
        "--tasks", "g1", "g2", "s1", "s2",
    ], timeout=13000)

    _commit_all()
    return f"judge {model_name} {dir_type} done"


# ══════════════════════════════════════════════════════════════════════════════
# Phase E: Piggybacked analyses
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="B200", timeout=7200, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def pca_one(model_name: str) -> str:
    """1A: PCA rank-1 check of corrective direction."""
    _setup(); _setup_sigterm()
    import torch
    print(f"=== pca (1A): {model_name} | GPU: {torch.cuda.get_device_name(0)} ===")
    results_vol.reload()
    _run([
        "python", "-m", "src.poc.exp08_multimodel_steering_phase0.pca_rank1",
        "--model-name", model_name,
        "--device", "cuda:0",
        "--max-records", "200",
    ], timeout=6000)
    _commit_all()
    return f"pca {model_name} done"


@app.function(
    gpu="B200", timeout=7200, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def id_steering_one(model_name: str) -> str:
    """1C: TwoNN ID profiles at alpha=1.0, 0.0, -1.0."""
    _setup(); _setup_sigterm()
    import torch
    print(f"=== id_steering (1C): {model_name} | GPU: {torch.cuda.get_device_name(0)} ===")
    results_vol.reload()
    _run([
        "python", "-m", "src.poc.exp08_multimodel_steering_phase0.id_under_steering",
        "--model-name", model_name,
        "--device", "cuda:0",
    ], timeout=6000)
    _commit_all()
    return f"id_steering {model_name} done"


@app.function(
    gpu="B200", timeout=7200, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def bootstrap_one(model_name: str) -> str:
    """Direction stability bootstrap."""
    _setup(); _setup_sigterm()
    import torch
    print(f"=== bootstrap: {model_name} | GPU: {torch.cuda.get_device_name(0)} ===")
    results_vol.reload()
    _run([
        "python", "-m", "src.poc.exp08_multimodel_steering_phase0.direction_bootstrap",
        "--model-name", model_name,
        "--device", "cuda:0",
        "--max-records", "200",
        "--n-resamples", "100",
    ], timeout=6000)
    _commit_all()
    return f"bootstrap {model_name} done"


@app.function(
    timeout=1800, image=image, volumes=VOLUME_MOUNTS, memory=8192,
)
def commitment_one(model_name: str, dir_type: str) -> str:
    """1B: Commitment delay vs alpha (CPU post-processing)."""
    _setup()
    print(f"=== commitment (1B): {model_name} {dir_type} ===")
    results_vol.reload()
    _run([
        "python", "-m", "src.poc.exp08_multimodel_steering_phase0.commitment_vs_alpha",
        "--model-name", model_name,
    ], timeout=1200)
    _commit_all()
    return f"commitment {model_name} {dir_type} done"


@app.function(
    gpu="B200", timeout=7200, retries=GPU_RETRIES, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def exp10_patch_one(model_name: str) -> str:
    """Phase 3 patching: causal activation patching validation."""
    _setup(); _setup_sigterm()
    import torch
    print(f"=== exp10_patch: {model_name} | GPU: {torch.cuda.get_device_name(0)} ===")
    results_vol.reload()
    exp10_vol.reload()

    probes_dir = f"/root/exp10_results/{model_name}/probes"
    paired_dir = f"/root/exp10_results/{model_name}/paired_data"
    output_dir = f"/root/exp10_results/{model_name}/patching"
    mean_dir = f"/root/results/cross_model/{model_name}/directions/corrective_directions.npz"

    _run([
        "python", "-m", "src.poc.exp10_contrastive_activation_patching.patching",
        "--model", model_name,
        "--device", "cuda:0",
        "--probes-dir", probes_dir,
        "--paired-data-dir", paired_dir,
        "--output-dir", output_dir,
        "--mean-dir-path", mean_dir,
        "--tuned-lens-dir", "/root/probes",
    ], timeout=6000)

    _commit_all()
    return f"exp10_patch {model_name} done"


# ══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ══════════════════════════════════════════════════════════════════════════════

@app.function(
    gpu="B200", timeout=600, image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=32768,
)
def smoke_test() -> str:
    """Verify image, imports, GPU, volumes, chat template, dataset."""
    _setup()
    import json
    import torch

    checks = []

    # 1. GPU
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    checks.append(f"GPU: {gpu_name} ({vram:.0f} GB)")

    # 2. Core imports
    from src.poc.cross_model.config import MODEL_REGISTRY, get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.exp06_corrective_direction_steering.model_adapter import get_steering_adapter
    from src.poc.exp10_contrastive_activation_patching.collect_paired import collect_paired_data, commitment_continuous
    from src.poc.exp10_contrastive_activation_patching.train_probes import train_probes
    checks.append("Imports: all OK")

    # 3. Dataset
    ds_path = Path("/root/data/eval_dataset_v2.jsonl")
    n_records = sum(1 for _ in open(ds_path))
    checks.append(f"Dataset: {n_records} records")

    # 4. Volumes
    results_ok = Path("/root/results").is_dir()
    exp10_ok = Path("/root/exp10_results").is_dir()
    probes_ok = Path("/root/probes").is_dir()
    checks.append(f"Volumes: results={results_ok} exp10={exp10_ok} probes={probes_ok}")

    # 5. Tuned lens probes
    n_probe_models = 0
    for m in MODELS:
        it_path = Path(f"/root/probes/{m}/it")
        if it_path.exists():
            n_probe_models += 1
    checks.append(f"Tuned lens probes: {n_probe_models}/6 models available")

    # 6. Chat template test
    from transformers import AutoTokenizer
    spec = get_spec("olmo2_7b")
    adapter = get_adapter("olmo2_7b")
    tok = AutoTokenizer.from_pretrained(spec.it_id, revision=spec.it_revision)
    raw = "Hello world"
    templated = adapter.apply_template(tok, raw, is_it=True)
    raw_len = len(tok.encode(raw))
    tmpl_len = len(tok.encode(templated))
    checks.append(f"Chat template: raw={raw_len} tokens, templated={tmpl_len} tokens (OK)")

    # 7. Smoke marker
    marker = Path("/root/results/_smoke_test_v2.txt")
    marker.write_text("unified pipeline smoke test passed\n")
    _commit_all()
    checks.append("Smoke marker written to volume")

    summary = "\n".join(f"  [{i+1}] {c}" for i, c in enumerate(checks))
    print(f"\n=== SMOKE TEST PASSED ===\n{summary}\n")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Phase entrypoints (for running individual phases)
# ══════════════════════════════════════════════════════════════════════════════

@app.local_entrypoint()
def run_phase_a():
    """Phase A only: precompute directions for all 6 models."""
    print("=== Phase A: Precompute directions (6 models) ===")

    # A.1: gen (12 GPU jobs: 6 models x 2 workers)
    print("  A.1: gen...")
    list(precompute_gen.starmap([(m, wi, N_WORKERS) for m in MODELS for wi in range(N_WORKERS)]))
    print("  A.1: gen complete")

    # A.2: score (6 CPU jobs, needs API key)
    print("  A.2: score...")
    list(precompute_score.map(MODELS))
    print("  A.2: score complete")

    # A.3: acts (12 GPU jobs)
    print("  A.3: acts...")
    list(precompute_acts.starmap([(m, wi, N_WORKERS) for m in MODELS for wi in range(N_WORKERS)]))
    print("  A.3: acts complete")

    # A.4: merge (6 CPU jobs)
    print("  A.4: merge...")
    list(precompute_merge.map(MODELS))
    print("  A.4: merge complete")

    print("=== Phase A DONE ===")


@app.local_entrypoint()
def run_phase_b():
    """Phase B only: exp10 paired data + probes (requires Phase A complete)."""
    print("=== Phase B: Exp10 collect + probes ===")

    print("  B.1: collect paired data...")
    list(exp10_collect.map(MODELS))
    print("  B.1: collect complete")

    print("  B.2: train probes...")
    list(exp10_probes.map(MODELS))
    print("  B.2: probes complete")

    print("=== Phase B DONE ===")


@app.local_entrypoint()
def run_phase_c():
    """Phase C only: dual steering (requires Phases A+B complete)."""
    print("=== Phase C: Dual steering (exp8 + exp10) ===")

    # Read go/no-go from exp10 probes to decide which models get exp10 steering
    dir_types_per_model = {}
    for m in MODELS:
        dir_types_per_model[m] = ["exp8"]  # always run exp8
        # Check exp10 go/no-go (if probes exist)
        probe_summary = Path(f"results/exp10_contrastive_activation_patching/{m}/probes/probe_summary.json")
        if probe_summary.exists():
            import json
            summary = json.loads(probe_summary.read_text())
            if summary.get("go_nogo") == "proceed":
                dir_types_per_model[m].append("exp10")
                print(f"  {m}: exp8 + exp10 (go/no-go = proceed)")
            else:
                print(f"  {m}: exp8 only (go/no-go = redundant)")
        else:
            # If no local probe summary, run both and let it fail gracefully
            dir_types_per_model[m].append("exp10")
            print(f"  {m}: exp8 + exp10 (no local probe summary, running both)")

    steer_args = [
        (m, dt, wi, N_WORKERS)
        for m in MODELS
        for dt in dir_types_per_model[m]
        for wi in range(N_WORKERS)
    ]
    print(f"  Launching {len(steer_args)} steer jobs...")
    list(steer_one.starmap(steer_args))
    print("=== Phase C DONE ===")


@app.local_entrypoint()
def run_phase_d():
    """Phase D only: merge + judge (requires Phase C complete)."""
    print("=== Phase D: Merge + Judge ===")

    # Merge
    merge_args = [(m, dt) for m in MODELS for dt in ["exp8", "exp10"]]
    print(f"  Merging {len(merge_args)} result sets...")
    list(merge_workers.starmap(merge_args))

    # Judge
    print(f"  Judging {len(merge_args)} result sets...")
    list(judge_one.starmap(merge_args))

    print("=== Phase D DONE ===")


@app.local_entrypoint()
def run_phase_e():
    """Phase E only: piggybacked analyses (requires Phases C+D complete)."""
    print("=== Phase E: Piggybacked analyses ===")

    print("  1A PCA...")
    list(pca_one.map(MODELS))

    print("  1C ID under steering...")
    list(id_steering_one.map(MODELS))

    print("  Bootstrap stability...")
    list(bootstrap_one.map(MODELS))

    print("  Exp10 patching...")
    list(exp10_patch_one.map(MODELS))

    print("  1B Commitment vs alpha...")
    commitment_args = [(m, dt) for m in MODELS for dt in ["exp8", "exp10"]]
    list(commitment_one.starmap(commitment_args))

    print("=== Phase E DONE ===")


@app.local_entrypoint()
def main():
    """Full pipeline: A -> B -> C -> D -> E."""
    print("=" * 60)
    print("UNIFIED PIPELINE: directions + exp10 + dual steering")
    print("=" * 60)

    run_phase_a()
    run_phase_b()
    run_phase_c()
    run_phase_d()
    run_phase_e()

    print("\n=== ALL PHASES COMPLETE ===")
    print("Download results:")
    print("  modal volume get phase0-results-v2 cross_model results/cross_model")
    print("  modal volume get exp10-results exp10 results/exp10_contrastive_activation_patching")
