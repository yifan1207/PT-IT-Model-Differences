"""
Modal-based evaluation for 0G Tuned-Lens commitment delay.

Runs 12 independent GPU jobs (6 models × 2 variants) on Modal cloud GPUs.
Collects all commitment metrics (7 methods, multiple thresholds, qualified variants).

Features:
- Periodic Modal Volume commits (every 30s + every 10 prompts)
- Resume after preemption: reads done_ids from existing JSONL on Volume
- 10 retries with 10s delay between attempts
- Graceful shutdown: commits Volume on SIGTERM
- HF model cache persisted across retries via Volume

Setup (run once):
    # 1. Create Modal secret for HuggingFace:
    modal secret create huggingface-token HF_TOKEN=hf_YOUR_TOKEN_HERE

    # 2. Upload probes to Modal Volume:
    bash scripts/modal_upload_probes.sh

    # 3. Run evaluation:
    modal run --detach scripts/modal_eval_0G.py

    # 4. After completion, download results:
    bash scripts/modal_download_results.sh
"""
from __future__ import annotations

import itertools

import modal

# ── Modal app & infrastructure ─────────────────────────────────────────────

app = modal.App("0g-tuned-lens-eval")

# Persistent volumes
probes_vol = modal.Volume.from_name("0g-probes", create_if_missing=True)
results_vol = modal.Volume.from_name("0g-results", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

# Container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0,<2.7.0",  # pin to CUDA 12.x compatible range
        "transformers==4.57.3",  # pin exact — DeepSeek breaks on >=5.0
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

# ── Model registry (mirrors src/poc/cross_model/config.py) ────────────────
# Only the fields needed for GPU/memory selection.
MODEL_INFO = {
    "gemma3_4b":        {"n_layers": 34, "d_model": 2560, "is_moe": False},
    "llama31_8b":       {"n_layers": 32, "d_model": 4096, "is_moe": False},
    "qwen3_4b":         {"n_layers": 36, "d_model": 2560, "is_moe": False},
    "mistral_7b":       {"n_layers": 32, "d_model": 4096, "is_moe": False},
    "deepseek_v2_lite": {"n_layers": 27, "d_model": 2048, "is_moe": True},
    "olmo2_7b":         {"n_layers": 32, "d_model": 4096, "is_moe": False},
}


# ── Main eval function ────────────────────────────────────────────────────

@app.function(
    gpu=["H100", "A100-80GB"],           # H100 preferred; B200 needs torch>=2.7
    timeout=86400,                    # 24 hours max per attempt
    retries=modal.Retries(
        max_retries=10,               # max 10 retries per job
        backoff_coefficient=1.0,      # fixed delay
        initial_delay=10.0,           # 10s before retry
    ),
    image=image,
    volumes={
        "/probes": probes_vol,
        "/results": results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=10,                # use all 10 GPU slots
    memory=65536,                     # 64 GiB RAM
    cpu=8.0,
)
def eval_model_variant(model_name: str, variant: str) -> dict:
    """Evaluate one model+variant on 2,936 prompts with all commitment metrics.

    Saves results to /results/{model_name}/tuned_lens_commitment_{variant}.jsonl
    on the Modal Volume. Resume-safe: skips already-evaluated prompts.
    """
    import atexit
    import json
    import logging
    import signal
    import threading
    import time
    from pathlib import Path

    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("modal_eval")
    log.info("=== Starting eval: %s/%s ===", model_name, variant)
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    log.info("VRAM: %.1f GB", vram / 1e9)

    # ── Set up paths ──────────────────────────────────────────────────────
    probe_src = Path(f"/probes/{model_name}/{variant}")
    results_dir = Path(f"/results/{model_name}/tuned_lens/commitment")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"tuned_lens_commitment_{variant}.jsonl"
    summary_path = results_dir / f"summary_{variant}.json"

    # ── Check for already-complete evaluation ─────────────────────────────
    results_vol.reload()
    arrays_complete = (results_dir / "arrays" / "step_index.jsonl").exists()
    if summary_path.exists() and arrays_complete:
        log.info("Summary + arrays already exist — skipping")
        return json.loads(summary_path.read_text())
    elif summary_path.exists():
        log.info("Summary exists but arrays missing — re-running with collect_full")

    # ── Reload probes volume ──────────────────────────────────────────────
    probes_vol.reload()
    probe_files = list(probe_src.glob("probe_layer_*.pt"))
    if not probe_files:
        raise RuntimeError(
            f"No probes found at {probe_src}. "
            f"Upload probes first: bash scripts/modal_upload_probes.sh"
        )
    log.info("Found %d probe files in %s", len(probe_files), probe_src)

    # ── Check resume state ────────────────────────────────────────────────
    from src.poc.cross_model.utils import read_done_ids

    done_ids = read_done_ids(out_path) if out_path.exists() else set()
    log.info("Resume state: %d prompts already done", len(done_ids))

    # ── Background Volume commit thread ───────────────────────────────────
    stop_event = threading.Event()
    commit_counter = {"n": 0}  # mutable counter for prompt-based commits

    def _periodic_commit():
        """Commit Volume every 30 seconds."""
        while not stop_event.is_set():
            stop_event.wait(timeout=30)
            if stop_event.is_set():
                break
            try:
                results_vol.commit()
            except Exception as e:
                log.warning("[volume] Periodic commit failed: %s", e)

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    # Graceful shutdown: commit on SIGTERM/exit
    def _shutdown_commit(*_args):
        log.info("[shutdown] Committing Volume before exit...")
        try:
            results_vol.commit()
            log.info("[shutdown] Volume committed successfully")
        except Exception as e:
            log.warning("[shutdown] Commit failed: %s", e)
        if _args:  # called from signal handler
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _shutdown_commit)
    atexit.register(_shutdown_commit)

    # ── Import project code ───────────────────────────────────────────────
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer, load_dataset
    from src.poc.cross_model.tuned_lens import _load_probes, eval_commitment

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    # ── Load probes ───────────────────────────────────────────────────────
    probes = _load_probes(probe_src, spec.d_model, device)
    expected = spec.n_layers
    if len(probes) != expected:
        log.warning(
            "Expected %d probes, got %d — some layers will use identity",
            expected, len(probes),
        )
    log.info("Loaded %d/%d probes", len(probes), expected)

    # ── Load model ────────────────────────────────────────────────────────
    model_id = model_id_for_variant(spec, variant)
    eager_attn = spec.is_moe  # DeepSeek V2 needs eager attention
    log.info("Loading model %s (eager_attn=%s)", model_id, eager_attn)

    model, tokenizer = load_model_and_tokenizer(
        model_id, device, eager_attn=eager_attn, multi_gpu=False,
    )
    # Disable gradients for memory savings
    model.requires_grad_(False)

    # Commit HF cache so next retry doesn't re-download
    try:
        hf_cache_vol.commit()
        log.info("HF model cache committed to Volume")
    except Exception:
        pass

    # ── Load dataset ──────────────────────────────────────────────────────
    records = load_dataset("/root/data/exp3_dataset.jsonl")
    log.info("Dataset: %d records (will skip %d already done)", len(records), len(done_ids))

    # ── Monkey-patch eval_commitment to commit Volume every 10 prompts ────
    # The original function flushes JSONL after each prompt. We hook into
    # the logging call (every 50 prompts) and also commit more frequently.
    _orig_log_info = log.info

    def _patched_log_info(msg, *args, **kwargs):
        _orig_log_info(msg, *args, **kwargs)
        if isinstance(msg, str) and "prompts done" in msg:
            try:
                results_vol.commit()
                _orig_log_info("[volume] Committed after %s", args[0] if args else "?")
            except Exception as e:
                _orig_log_info("[volume] Commit failed: %s", e)

    # Apply patch to the tuned_lens module's logger
    import src.poc.cross_model.tuned_lens as tl_module
    tl_module.log.info = _patched_log_info

    # ── Run evaluation ────────────────────────────────────────────────────
    max_new_tokens = 64 if spec.is_moe else 512
    log.info(
        "Starting eval: %d records, max_new_tokens=%d",
        len(records), max_new_tokens,
    )

    arrays_dir = results_dir / "arrays"
    try:
        summary = eval_commitment(
            model, tokenizer, adapter, spec, probes, records, device,
            output_path=out_path,
            variant=variant,
            max_new_tokens=max_new_tokens,
            collect_full=True,
            collect_top5=True,
            top5_max_prompts=200,
            arrays_dir=arrays_dir,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    # ── Save summary ──────────────────────────────────────────────────────
    summary_path.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    log.info("=== Evaluation COMPLETE: %s/%s ===", model_name, variant)
    log.info("  Prompts: %d", summary.get("n_prompts", "?"))
    log.info("  Median raw commitment: %s", summary.get("median_commitment_raw", "?"))
    log.info("  Results: %s", out_path)
    log.info("  Summary: %s", summary_path)

    return summary


# ── Entrypoint ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Launch all 12 eval jobs in parallel."""
    # Order: slowest first. With 10-GPU limit, the 11th/12th jobs queue.
    # 7B models (d=4096) are slowest → start first to minimize wall time.
    # DeepSeek (smallest, max_new_tokens=64) is fastest → queued last.
    models = [
        "llama31_8b", "mistral_7b", "olmo2_7b",   # 7B models (slowest)
        "gemma3_4b", "qwen3_4b",                   # 4B models
        "deepseek_v2_lite",                         # MoE (fastest)
    ]
    variants = ["pt", "it"]
    configs = list(itertools.product(models, variants))

    print(f"Launching {len(configs)} eval jobs in parallel...")
    print("Each job: 2,936 prompts × all commitment metrics")
    print("GPU preference: H100 > A100-80GB (max 10 containers)")
    print()

    results = list(eval_model_variant.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    for (model, variant), summary in zip(configs, results):
        if summary:
            n = summary.get("n_prompts", "?")
            med = summary.get("median_commitment_raw", "?")
            print(f"  {model:20s} {variant:3s}: {n:>5} prompts, median_raw={med}")
        else:
            print(f"  {model:20s} {variant:3s}: FAILED")

    print()
    print("Download results with:")
    print("  bash scripts/modal_download_results.sh")
