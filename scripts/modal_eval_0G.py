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
probes_vol = modal.Volume.from_name("0g-probes-v2", create_if_missing=True)
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
def eval_model_variant(model_name: str, variant: str, worker_index: int = 0, n_workers: int = 1) -> dict:
    """Evaluate one model+variant (or a worker slice) with all commitment metrics.

    When n_workers > 1, processes records[worker_index::n_workers] and saves to
    worker-specific output paths (e.g. tuned_lens_commitment_pt_w0.jsonl, arrays_w0/).
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
    worker_tag = f"w{worker_index}" if n_workers > 1 else ""
    log.info("=== Starting eval: %s/%s %s===", model_name, variant,
             f"(worker {worker_index}/{n_workers}) " if n_workers > 1 else "")
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
    log.info("VRAM: %.1f GB", vram / 1e9)

    # ── Set up paths (per-worker when parallelized) ───────────────────────
    probe_src = Path(f"/probes/{model_name}/{variant}")
    results_dir = Path(f"/results/{model_name}/tuned_lens/commitment")
    results_dir.mkdir(parents=True, exist_ok=True)
    if n_workers > 1:
        out_path = results_dir / f"tuned_lens_commitment_{variant}_{worker_tag}.jsonl"
        arrays_subdir = f"arrays_{variant}_{worker_tag}"
    else:
        out_path = results_dir / f"tuned_lens_commitment_{variant}.jsonl"
        arrays_subdir = f"arrays_{variant}"
    summary_path = results_dir / f"summary_{variant}.json"

    # ── Check for already-complete evaluation ─────────────────────────────
    results_vol.reload()
    arrays_complete = (results_dir / arrays_subdir / "step_index.jsonl").exists()
    jsonl_complete = out_path.exists() and sum(1 for _ in open(out_path)) > 0 if out_path.exists() else False
    if n_workers == 1 and summary_path.exists() and arrays_complete:
        log.info("Summary + arrays already exist — skipping")
        return json.loads(summary_path.read_text())
    elif jsonl_complete and arrays_complete:
        log.info("Worker %d output already complete — skipping", worker_index)
        return {"status": "skipped", "worker": worker_index}
    elif summary_path.exists() and n_workers == 1:
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

    # ── Load dataset (slice for data parallelism) ──────────────────────────
    all_records = load_dataset("/root/data/exp3_dataset.jsonl")
    if n_workers > 1:
        records = all_records[worker_index::n_workers]
        log.info("Worker %d/%d: %d records (slice of %d total)",
                 worker_index, n_workers, len(records), len(all_records))
    else:
        records = all_records
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

    arrays_dir = results_dir / arrays_subdir
    # Limit top5 collection proportionally for workers
    top5_limit = 200 // n_workers if n_workers > 1 else 200
    try:
        summary = eval_commitment(
            model, tokenizer, adapter, spec, probes, records, device,
            output_path=out_path,
            variant=variant,
            max_new_tokens=max_new_tokens,
            collect_full=True,
            collect_top5=True,
            top5_max_prompts=top5_limit,
            arrays_dir=arrays_dir,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    # ── Save summary (only for single-worker or we save per-worker) ───────
    if n_workers == 1:
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
def main(models: str = "", n_workers: int = 1):
    """Launch eval jobs in parallel, with optional data parallelism.

    Usage:
        modal run --detach scripts/modal_eval_0G.py                             # all 6, 1 worker each
        modal run --detach scripts/modal_eval_0G.py --models gemma3_4b,qwen3_4b # specific models
        modal run --detach scripts/modal_eval_0G.py --n-workers 3               # 3 workers per model
    """
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = [
            "llama31_8b", "mistral_7b", "olmo2_7b",
            "gemma3_4b", "qwen3_4b",
            "deepseek_v2_lite",
        ]
    variants = ["pt", "it"]

    # Build config tuples: (model, variant, worker_index, n_workers)
    configs = []
    for model in model_list:
        for variant in variants:
            for wi in range(n_workers):
                configs.append((model, variant, wi, n_workers))

    print(f"Launching {len(configs)} eval jobs ({len(model_list)} models × 2 variants × {n_workers} workers)")
    print("GPU preference: H100 > A100-80GB (max 10 containers)")
    print()

    results = list(eval_model_variant.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print("=" * 60)
    for cfg, summary in zip(configs, results):
        model, variant, wi, nw = cfg
        if summary and nw == 1:
            n = summary.get("n_prompts", "?")
            med = summary.get("median_commitment_raw", "?")
            print(f"  {model:20s} {variant:3s}: {n:>5} prompts, median_raw={med}")
        elif summary:
            n = summary.get("n_prompts", "?")
            print(f"  {model:20s} {variant:3s} w{wi}: {n:>5} prompts")
        else:
            print(f"  {model:20s} {variant:3s} w{wi}: FAILED")

    if n_workers > 1:
        print()
        print("Workers finished. Run merge locally:")
        print(f"  uv run python scripts/merge_tuned_lens_workers.py --models {','.join(model_list)} --n-workers {n_workers}")

    print()
    print("Download results with:")
    print("  bash scripts/modal_download_results.sh")
