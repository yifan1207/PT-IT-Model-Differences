"""
Modal-based training for 0G Tuned-Lens probes.

Trains tuned-lens probes for specified models on Modal cloud GPUs.
Uses the Belrose et al. (2023) exact recipe: SGD Nesterov, 250 steps,
262K tokens/step, linear LR decay.

Features:
- Periodic checkpoint commits to Modal Volume (every 50 steps)
- Resume: detects existing checkpoint.json and skips completed training
- HF model cache persisted across retries
- Configurable lr_scale (default 1.0; use 0.25 for Gemma 3 4B)

Setup:
    # 1. Create Modal secret for HuggingFace:
    modal secret create huggingface-token HF_TOKEN=hf_YOUR_TOKEN_HERE

    # 2. Run training (all 3 models that need retraining):
    modal run --detach scripts/modal_train_0G.py

    # 3. Run training for a specific model:
    modal run --detach scripts/modal_train_0G.py --model gemma3_4b --lr-scale 0.25

    # 4. After completion, download probes:
    bash scripts/modal_download_probes.sh
"""
from __future__ import annotations

import itertools

import modal

# ── Modal app & infrastructure ─────────────────────────────────────────────

app = modal.App("0g-tuned-lens-train")

# Persistent volumes — fresh volume for retrained probes
probes_vol = modal.Volume.from_name("0g-probes-v2", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

# Container image — matches eval script exactly
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.5.0,<2.7.0",
        "transformers==4.57.3",
        "accelerate",
        "numpy",
        "scipy",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "datasets",  # needed for C4 streaming
    )
    .add_local_python_source("src")
)

# Models to retrain and their lr_scale overrides.
# Gemma 3 has extreme raw_kl range (85 nats at L0) → lr_scale=0.25
# DeepSeek/Qwen have minor last-layer drift → lr_scale=0.5
TRAIN_CONFIGS = {
    "gemma3_4b":        {"lr_scale": 0.25},
    "qwen3_4b":         {"lr_scale": 0.5},
    "deepseek_v2_lite": {"lr_scale": 0.5},
}


# ── Training function ────────────────────────────────────────────────────

@app.function(
    gpu=["H100", "A100-80GB"],           # B200 needs torch>=2.7 (Blackwell); H100 is fastest compatible
    timeout=86400,                    # 24h max
    retries=modal.Retries(
        max_retries=5,
        backoff_coefficient=1.0,
        initial_delay=10.0,
    ),
    image=image,
    volumes={
        "/probes": probes_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=7,                 # 3 models × 2 variants = 6 jobs; 7th slot for headroom
    memory=65536,
    cpu=8.0,
)
def train_model_variant(model_name: str, variant: str, lr_scale: float = 1.0) -> dict:
    """Train tuned-lens probes for one model+variant.

    Saves probes to /probes/{model_name}/{variant}/ on the Modal Volume.
    Resume-safe: skips if checkpoint.json shows 250/250 steps.
    """
    import json
    import logging
    import signal
    import threading
    from pathlib import Path

    import torch

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("modal_train")
    log.info("=== Starting training: %s/%s (lr_scale=%.2f) ===", model_name, variant, lr_scale)
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Set up paths ──────────────────────────────────────────────────────
    output_dir = Path(f"/probes/{model_name}/{variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Check for already-complete training ───────────────────────────────
    probes_vol.reload()
    ckpt_path = output_dir / "checkpoint.json"
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if ckpt.get("checkpoint_step", 0) >= 250:
            log.info("Training already complete (step %d/%d) — skipping",
                     ckpt["checkpoint_step"], ckpt["n_steps"])
            summary_path = output_dir / "training_summary.json"
            if summary_path.exists():
                return json.loads(summary_path.read_text())
            return {"status": "already_complete", "model": model_name, "variant": variant}

    # ── Graceful shutdown: commit on SIGTERM ──────────────────────────────
    def _shutdown(*_args):
        log.info("[shutdown] Committing Volume before exit...")
        try:
            probes_vol.commit()
        except Exception as e:
            log.warning("[shutdown] Commit failed: %s", e)
        if _args:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _shutdown)

    # ── Background Volume commit (every 60s) ──────────────────────────────
    stop_event = threading.Event()

    def _periodic_commit():
        while not stop_event.is_set():
            stop_event.wait(timeout=60)
            if stop_event.is_set():
                break
            try:
                probes_vol.commit()
            except Exception:
                pass

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    # ── Import project code ───────────────────────────────────────────────
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer
    from src.poc.cross_model.tuned_lens import train_probes, load_training_texts

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    # ── Load model ────────────────────────────────────────────────────────
    model_id = model_id_for_variant(spec, variant)
    log.info("Loading model %s", model_id)
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    model.requires_grad_(False)

    # Commit HF cache
    try:
        hf_cache_vol.commit()
    except Exception:
        pass

    # ── Load training data (C4 validation, Belrose recipe) ────────────────
    log.info("Loading C4 validation texts (~70M tokens)...")
    all_texts = load_training_texts(n_tokens_target=2_000_000)
    # 80/20 train/val split (same as local training)
    split = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split]
    val_texts = all_texts[split:]
    log.info("Loaded %d train texts, %d val texts", len(train_texts), len(val_texts))

    # ── Monkey-patch checkpoint saves to commit volume ─────────────────────
    import src.poc.cross_model.tuned_lens as tl_module
    _orig_log_info = tl_module.log.info

    def _patched_log_info(msg, *args, **kwargs):
        _orig_log_info(msg, *args, **kwargs)
        if isinstance(msg, str) and "[checkpoint]" in msg:
            try:
                probes_vol.commit()
                _orig_log_info("[volume] Committed checkpoint to Volume")
            except Exception as e:
                _orig_log_info("[volume] Commit failed: %s", e)

    tl_module.log.info = _patched_log_info

    # ── Train ─────────────────────────────────────────────────────────────
    log.info("Starting training: lr_scale=%.2f → lr_passed=%.4f", lr_scale, lr_scale * 0.1)
    try:
        summary = train_probes(
            model, tokenizer, adapter, spec,
            train_texts, val_texts,
            device, output_dir,
            n_steps=250,
            lr=lr_scale,  # interpreted as lr_scale for SGD
            batch_size=262144,
            micro_batch_size=2048,  # 4× larger than default; H100 80GB can handle it
            optimizer_type="sgd_nesterov",
            streaming=True,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    # ── Final commit ──────────────────────────────────────────────────────
    probes_vol.commit()
    log.info("=== Training COMPLETE: %s/%s ===", model_name, variant)
    return summary


# ── Entrypoint ─────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    model: str = "",
    lr_scale: float = 0.0,
):
    """Launch training jobs.

    Usage:
        modal run --detach scripts/modal_train_0G.py                    # all 3 models
        modal run --detach scripts/modal_train_0G.py --model gemma3_4b  # one model
        modal run --detach scripts/modal_train_0G.py --model gemma3_4b --lr-scale 0.25
    """
    if model:
        # Single model
        models = {model: TRAIN_CONFIGS.get(model, {})}
    else:
        models = TRAIN_CONFIGS

    configs = []
    for m, cfg in models.items():
        scale = lr_scale if lr_scale > 0 else cfg.get("lr_scale", 1.0)
        for v in ["pt", "it"]:
            configs.append((m, v, scale))

    print(f"Launching {len(configs)} training jobs:")
    for m, v, s in configs:
        print(f"  {m}/{v} lr_scale={s}")
    print()

    results = list(train_model_variant.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE")
    print("=" * 60)
    for (m, v, s), summary in zip(configs, results):
        if isinstance(summary, dict):
            status = summary.get("status", "trained")
            print(f"  {m:20s} {v:3s} lr_scale={s:.2f}: {status}")
        else:
            print(f"  {m:20s} {v:3s}: FAILED")

    print()
    print("Next steps:")
    print("  1. Download probes: modal volume get 0g-probes-v2 / results/cross_model/")
    print("  2. Run eval: modal run --detach scripts/modal_eval_0G.py")
