"""
Modal v3: Retrain ALL 12 tuned-lens probes (6 models × 2 variants) with
IT models using chat template during training data collection.

Key change from v2: IT variant applies adapter.apply_template() to C4 texts
before forward pass, so probes train on activations matching IT inference.
PT variant unchanged (raw text).

    modal run --detach scripts/modal_train_0G_v3.py
"""
from __future__ import annotations
import modal

app = modal.App("0g-tuned-lens-train-v3")

probes_vol = modal.Volume.from_name("0g-probes-v3", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7.0",
        "transformers==4.57.3",
        "accelerate",
        "numpy",
        "scipy",
        "sentencepiece",
        "protobuf",
        "huggingface_hub",
        "safetensors",
        "einops",
        "datasets",
    )
    .add_local_python_source("src")
)

# lr_scale per model (from CLAUDE.md / prior experiments)
LR_SCALES = {
    "gemma3_4b": 0.25,
    "qwen3_4b": 0.5,
    "deepseek_v2_lite": 0.5,
    "llama31_8b": 1.0,
    "mistral_7b": 1.0,
    "olmo2_7b": 1.0,
}


@app.function(
    gpu="B200",
    timeout=86400,
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0),
    image=image,
    volumes={
        "/probes": probes_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=12,
    memory=65536,
    cpu=8.0,
)
def train_model_variant(model_name: str, variant: str, lr_scale: float = 1.0) -> dict:
    """Train tuned-lens probes for one model+variant.

    IT variant: C4 texts wrapped in chat template before forward pass.
    PT variant: raw C4 text (unchanged from v2).
    """
    import json
    import logging
    import signal
    import threading
    from pathlib import Path
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("modal_train_v3")
    log.info("=== Training: %s/%s (lr_scale=%.2f, chat_template=%s) ===",
             model_name, variant, lr_scale, variant == "it")
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    output_dir = Path(f"/probes/{model_name}/{variant}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already complete
    probes_vol.reload()
    ckpt_path = output_dir / "checkpoint.json"
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        if ckpt.get("checkpoint_step", 0) >= 250:
            log.info("Already complete (step %d/250) — skipping", ckpt["checkpoint_step"])
            summary_path = output_dir / "training_summary.json"
            if summary_path.exists():
                return json.loads(summary_path.read_text())
            return {"status": "already_complete", "model": model_name, "variant": variant}

    # Graceful shutdown
    def _shutdown(*_args):
        log.info("[shutdown] Committing...")
        try:
            probes_vol.commit()
        except Exception:
            pass
        if _args:
            raise SystemExit(1)
    signal.signal(signal.SIGTERM, _shutdown)

    stop_event = threading.Event()
    def _periodic_commit():
        while not stop_event.is_set():
            stop_event.wait(timeout=60)
            if not stop_event.is_set():
                try:
                    probes_vol.commit()
                except Exception:
                    pass
    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    # Load model
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer
    from src.poc.cross_model.tuned_lens import train_probes, load_training_texts

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    model_id = model_id_for_variant(spec, variant)
    log.info("Loading %s", model_id)
    model, tokenizer = load_model_and_tokenizer(model_id, device)
    model.requires_grad_(False)

    try:
        hf_cache_vol.commit()
    except Exception:
        pass

    # Load C4 training data
    all_texts = load_training_texts(n_tokens_target=2_000_000)
    split = int(len(all_texts) * 0.8)
    train_texts, val_texts = all_texts[:split], all_texts[split:]
    log.info("%d train + %d val texts", len(train_texts), len(val_texts))

    # Patch checkpoint commits
    import src.poc.cross_model.tuned_lens as tl_module
    _orig = tl_module.log.info
    def _patched(msg, *a, **kw):
        _orig(msg, *a, **kw)
        if isinstance(msg, str) and "[checkpoint]" in msg:
            try:
                probes_vol.commit()
            except Exception:
                pass
    tl_module.log.info = _patched

    # Train — variant passed through to collect_hidden_states_prefill
    try:
        summary = train_probes(
            model, tokenizer, adapter, spec,
            train_texts, val_texts,
            device, output_dir,
            n_steps=250,
            lr=lr_scale,
            batch_size=262144,
            micro_batch_size=8192,  # B200 192GB: 4× vs H100 default, ~4 seqs per forward pass
            optimizer_type="sgd_nesterov",
            streaming=True,
            variant=variant,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    probes_vol.commit()
    log.info("=== COMPLETE: %s/%s ===", model_name, variant)
    return summary


@app.local_entrypoint()
def main(models: str = ""):
    """Train all 12 probes (6 models × PT/IT) on B200 GPUs.

        modal run --detach scripts/modal_train_0G_v3.py
        modal run --detach scripts/modal_train_0G_v3.py --models llama31_8b
    """
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = list(LR_SCALES.keys())

    configs = []
    for m in model_list:
        scale = LR_SCALES.get(m, 1.0)
        for v in ["pt", "it"]:
            configs.append((m, v, scale))

    print(f"Launching {len(configs)} training jobs (v3: IT uses chat template)")
    for m, v, s in configs:
        print(f"  {m}/{v} lr_scale={s}")
    print()

    results = list(train_model_variant.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL TRAINING COMPLETE (v3)")
    print("=" * 60)
    for (m, v, s), summary in zip(configs, results):
        status = summary.get("status", "trained") if isinstance(summary, dict) else "FAILED"
        print(f"  {m:20s} {v}: {status}")
    print("\nProbes on volume: 0g-probes-v3")
