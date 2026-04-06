"""
Modal script for Exp10: Contrastive Activation Patching.

Runs the full Exp10 pipeline on Modal cloud H100 GPUs:
  Phase 1: Forced-decoding paired data collection (6 models in parallel)
            Regression target: per-layer KL excess (Δkl_ℓ), winsorized.
  Phase 2: Ridge regression probes + direction comparison (CPU-heavy)
            Outputs: convergence_directions.npz (+ backward-compat commitment_directions.npz)
  Phase 3: Causal activation patching (6 models in parallel)
            Metric: ΔKL at downstream layers (not scalar commitment)
  Phase 4B: Steering with convergence-gap directions (conditional on go/no-go)

Setup:
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

# ── Constants ────────────────────────────────────────────────────────────��───

MODELS = [
    "gemma3_4b", "llama31_8b", "qwen3_4b",
    "mistral_7b", "deepseek_v2_lite", "olmo2_7b",
]

VOLUME_MOUNTS = {
    "/root/results": results_vol,
    "/root/tuned_lens_probes": tuned_lens_vol,
    "/root/phase0_results": phase0_vol,  # corrective_directions.npz live here
}

GPU_RETRIES = modal.Retries(max_retries=3, initial_delay=15.0, backoff_coefficient=2.0)


# ── Helpers ─────────────���─────────────────────────────────────────────────────

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


# ── Phase 1: Forced-decoding data collection ─────────────────────────────────

@app.function(
    gpu="H100",
    timeout=3600,  # 60 min
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def collect_one(model_name: str) -> str:
    """Phase 1: Collect forced-decoding paired data for one model."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== Phase 1 collect: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    from src.poc.exp10.collect_paired import collect_paired_data

    output_dir = f"/root/results/exp10/{model_name}/paired_data"

    # Tuned lens probes (v3, IT trained with chat template) on 0g-probes-v3 volume
    # Mounted at /root/tuned_lens_probes, with structure:
    # {model}/{variant}/probe_layer_*.pt
    tuned_lens_dir = "/root/tuned_lens_probes"

    metadata = collect_paired_data(
        model_name=model_name,
        device="cuda:0",
        output_dir=output_dir,
        n_prompts=600,
        max_gen_tokens=128,
        dataset_path="/root/data/eval_dataset_v2.jsonl",
        tuned_lens_dir=tuned_lens_dir,
        compute_kl_gradient=True,
    )

    _commit_volumes()
    return f"collect {model_name}: {metadata['n_prompts_processed']} prompts, {metadata['total_tokens']} tokens"


# ── Phase 2+2.5: Probe training + direction comparison ───────────────────────

@app.function(
    gpu="H100",  # GPU needed for PCA subsample SVD on large matrices
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

    # Existing mean directions are on the phase0-results volume
    # Mounted at /root/phase0_results/cross_model/{model}/directions/
    mean_dir_path = f"/root/phase0_results/cross_model/{model_name}/directions/corrective_directions.npz"

    import os
    if not os.path.exists(mean_dir_path):
        # Fallback: check tuned-lens volume
        alt_path = f"/root/tuned_lens_probes/cross_model/{model_name}/directions/corrective_directions.npz"
        if os.path.exists(alt_path):
            mean_dir_path = alt_path
            print(f"[warn] Using fallback mean_dir_path: {alt_path}")
        else:
            print(f"[ERROR] No corrective_directions.npz found for {model_name}")
            return f"probes {model_name}: FAILED — no mean directions found"

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
    gpu="H100",
    timeout=5400,  # 90 min
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def patch_one(model_name: str) -> str:
    """Phase 3: Causal activation patching for one model."""
    _setup()
    _setup_sigterm_handler()

    import torch
    print(f"=== Phase 3 patching: {model_name} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    results_vol.reload()

    from src.poc.exp10.patching import validate_patching

    tuned_lens_dir = "/root/tuned_lens_probes"
    mean_dir_path = f"/root/phase0_results/cross_model/{model_name}/directions/corrective_directions.npz"

    validate_patching(
        model_name=model_name,
        device="cuda:0",
        probes_dir=f"/root/results/exp10/{model_name}/probes",
        paired_data_dir=f"/root/results/exp10/{model_name}/paired_data",
        output_dir=f"/root/results/exp10/{model_name}/patching",
        mean_dir_path=mean_dir_path,
        n_test_prompts=120,
        max_tokens_per_prompt=5,
        top_k_layers=10,
        dataset_path="/root/data/eval_dataset_v2.jsonl",
        tuned_lens_dir=tuned_lens_dir,
    )

    _commit_volumes()
    return f"patching {model_name} done"


# ── Phase 4B: Steering with commitment directions ────────────────────────────

@app.function(
    gpu="H100",
    timeout=10800,  # 3h
    retries=GPU_RETRIES,
    image=image,
    volumes=VOLUME_MOUNTS,
    secrets=[modal.Secret.from_name("huggingface-token")],
    memory=65536,
)
def steer_one(model_name: str) -> str:
    """Phase 4B: A1 α-sweep using commitment directions (conditional on go/no-go)."""
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

    # Run exp6 A1 sweep with commitment directions
    import subprocess as sp
    directions_path = f"/root/results/exp10/{model_name}/probes/commitment_directions.npz"
    run_name = f"A1_commitment_dir_{model_name}"
    output_dir = f"/root/results/exp10/{model_name}/steering"

    cmd = [
        "python", "-m", "src.poc.exp6.run",
        "--experiment", "A1",
        "--variant", "it",
        "--device", "cuda:0",
        "--model-name", model_name,
        "--corrective-direction-path", directions_path,
        "--run-name", run_name,
        "--output-dir", output_dir,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = "/root"
    result = sp.run(cmd, capture_output=True, text=True, cwd="/root", env=env, timeout=10000)

    if result.stdout:
        print(result.stdout[-3000:])
    if result.returncode != 0:
        err = result.stderr[-2000:] if result.stderr else "(no stderr)"
        print(f"STDERR:\n{err}")
        # Don't raise — steering is optional. Log the error.
        _commit_volumes()
        return f"steer {model_name}: FAILED — {err[:200]}"

    _commit_volumes()
    return f"steer {model_name} done"


# ── GCS sync helper ───��──────────────────────────────────────────────────────

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


# ── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    import json as _json

    print("=" * 70)
    print("EXP10: CONTRASTIVE ACTIVATION PATCHING")
    print("=" * 70)

    # ── Phase 1: 6 parallel collection jobs ───────────────────────────────────
    print("\n[Phase 1] Collecting forced-decoding paired data (6 models)...")
    results_1 = list(collect_one.map(MODELS))
    for r in results_1:
        print(f"  {r}")

    # ── Phase 2+2.5: 6 parallel probe training ───���───────────────────────────
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
