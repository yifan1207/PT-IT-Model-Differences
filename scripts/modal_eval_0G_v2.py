"""
Modal eval v2: Collect per-variant per-layer KL arrays for all 6 models × 2 variants.

Fixes the variant-overwrite bug: saves arrays to arrays_{variant}/ (not shared arrays/).
Uses B200 GPUs for max throughput. Full dataset (2936 prompts), 512 max tokens, greedy.

    modal run --detach scripts/modal_eval_0G_v2.py

Download results after completion:
    modal volume get 0g-results-v2 . --force
"""
from __future__ import annotations

import modal

app = modal.App("0g-tuned-lens-eval-v2")

# Persistent volumes
probes_vol = modal.Volume.from_name("0g-probes-v2", create_if_missing=True)
results_vol = modal.Volume.from_name("0g-results-v2", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

# Container image — B200 needs torch>=2.7
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
    )
    .add_local_python_source("src")
    .add_local_file(
        "data/exp3_dataset.jsonl",
        remote_path="/root/data/exp3_dataset.jsonl",
    )
)


@app.function(
    gpu="B200",
    timeout=86400,
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=10.0),
    image=image,
    volumes={
        "/probes": probes_vol,
        "/results": results_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=12,
    memory=65536,
    cpu=8.0,
)
def eval_model_variant(model_name: str, variant: str) -> dict:
    """Evaluate one model+variant: full dataset, collect per-layer KL arrays."""
    import atexit
    import json
    import logging
    import signal
    import threading
    from pathlib import Path

    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("modal_eval_v2")
    log.info("=== Starting eval: %s/%s ===", model_name, variant)
    log.info("GPU: %s", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    log.info("VRAM: %.1f GB", getattr(props, 'total_memory', 0) / 1e9)

    # ── Paths ────────────────────────────────────────────────────────────────
    probe_src = Path(f"/probes/{model_name}/{variant}")
    results_dir = Path(f"/results/{model_name}/tuned_lens/commitment")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"tuned_lens_commitment_{variant}.jsonl"
    arrays_dir = results_dir / f"arrays_{variant}"
    summary_path = results_dir / f"summary_{variant}.json"

    # ── Skip if already complete ─────────────────────────────────────────────
    results_vol.reload()
    arrays_kl = arrays_dir / "raw_kl_final.npy"
    if arrays_kl.exists() and summary_path.exists():
        log.info("arrays_%s/raw_kl_final.npy + summary already exist — skipping", variant)
        return json.loads(summary_path.read_text())

    # ── Load probes ──────────────────────────────────────────────────────────
    probes_vol.reload()
    probe_files = list(probe_src.glob("probe_layer_*.pt"))
    if not probe_files:
        raise RuntimeError(f"No probes at {probe_src}. Run: bash scripts/modal_upload_probes.sh")
    log.info("Found %d probe files in %s", len(probe_files), probe_src)

    # ── Background Volume commits ────────────────────────────────────────────
    stop_event = threading.Event()

    def _periodic_commit():
        while not stop_event.is_set():
            stop_event.wait(timeout=30)
            if not stop_event.is_set():
                try:
                    results_vol.commit()
                except Exception:
                    pass

    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()

    def _shutdown(*_args):
        log.info("[shutdown] Committing Volume...")
        try:
            results_vol.commit()
        except Exception:
            pass
        if _args:
            raise SystemExit(1)

    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)

    # ── Load model + probes ──────────────────────────────────────────────────
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer, load_dataset
    from src.poc.cross_model.tuned_lens import _load_probes, eval_commitment

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    probes = _load_probes(probe_src, spec.d_model, device)
    log.info("Loaded %d/%d probes", len(probes), spec.n_layers)

    model_id = model_id_for_variant(spec, variant)
    log.info("Loading model %s", model_id)
    model, tokenizer = load_model_and_tokenizer(
        model_id, device, eager_attn=spec.is_moe, multi_gpu=False,
    )
    model.requires_grad_(False)

    try:
        hf_cache_vol.commit()
    except Exception:
        pass

    # ── Load dataset ─────────────────────────────────────────────────────────
    records = load_dataset("/root/data/exp3_dataset.jsonl")
    log.info("Dataset: %d records", len(records))

    # ── Patch logger to commit Volume on progress ────────────────────────────
    _orig_log_info = log.info

    def _patched_log_info(msg, *args, **kwargs):
        _orig_log_info(msg, *args, **kwargs)
        if isinstance(msg, str) and "prompts done" in msg:
            try:
                results_vol.commit()
            except Exception:
                pass

    import src.poc.cross_model.tuned_lens as tl_module
    tl_module.log.info = _patched_log_info

    # ── Run evaluation ───────────────────────────────────────────────────────
    max_new_tokens = 64 if spec.is_moe else 512
    log.info("Starting eval: %d records, max_new_tokens=%d, greedy (do_sample=False)",
             len(records), max_new_tokens)

    try:
        summary = eval_commitment(
            model, tokenizer, adapter, spec, probes, records, device,
            output_path=out_path,
            variant=variant,
            max_new_tokens=max_new_tokens,
            collect_full=True,
            collect_top5=False,  # skip top5 to save time
            arrays_dir=arrays_dir,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    summary_path.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    log.info("=== COMPLETE: %s/%s — %d prompts ===", model_name, variant,
             summary.get("n_prompts", "?"))
    return summary


@app.local_entrypoint()
def main(models: str = ""):
    """Launch 12 parallel eval jobs (6 models × 2 variants) on B200 GPUs.

    Usage:
        modal run --detach scripts/modal_eval_0G_v2.py
        modal run --detach scripts/modal_eval_0G_v2.py --models llama31_8b,mistral_7b
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

    configs = [(m, v) for m in model_list for v in variants]

    print(f"Launching {len(configs)} eval jobs ({len(model_list)} models × 2 variants)")
    print("GPU: B200 | max_containers: 12 | greedy generation | collect_full=True")
    print(f"Arrays saved to: arrays_pt/ and arrays_it/ (per-variant, no overwrite)")
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
    print("Download results:")
    print("  modal volume get 0g-results-v2 results/cross_model --force")
