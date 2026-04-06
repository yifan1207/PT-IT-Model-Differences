"""
Modal v3: Full eval with chat template for IT models.

Uses v3 probes (trained with chat template for IT).
IT eval prompts wrapped in chat template.
Collects ALL arrays (raw_kl_final, tuned_kl_final, entropy, etc.)
to per-variant directories (arrays_pt/, arrays_it/).

    # After training completes:
    modal run --detach scripts/modal_eval_0G_v3.py
"""
from __future__ import annotations
import modal

app = modal.App("0g-tuned-lens-eval-v3")

probes_vol = modal.Volume.from_name("0g-probes-v3", create_if_missing=True)
results_vol = modal.Volume.from_name("0g-results-v3", create_if_missing=True)
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
    )
    .add_local_python_source("src")
    .add_local_file("data/exp3_dataset.jsonl", remote_path="/root/data/exp3_dataset.jsonl")
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
    max_containers=6,  # leave GPUs for training + other tasks
    memory=65536,
    cpu=8.0,
)
def eval_model_variant(model_name: str, variant: str) -> dict:
    """Eval one model+variant with chat template for IT.

    Collects all commitment metrics + per-layer arrays.
    IT prompts use adapter.apply_template() — handled inside eval_commitment().
    """
    import atexit
    import json
    import logging
    import signal
    import threading
    from pathlib import Path
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("modal_eval_v3")
    log.info("=== Eval v3: %s/%s (chat_template=%s) ===", model_name, variant, variant == "it")
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    probe_src = Path(f"/probes/{model_name}/{variant}")
    results_dir = Path(f"/results/{model_name}/tuned_lens/commitment")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"tuned_lens_commitment_{variant}.jsonl"
    arrays_dir = results_dir / f"arrays_{variant}"
    summary_path = results_dir / f"summary_{variant}.json"

    # Skip if complete
    results_vol.reload()
    if (arrays_dir / "raw_kl_final.npy").exists() and summary_path.exists():
        log.info("Already complete — skipping")
        return json.loads(summary_path.read_text())

    # Load probes
    probes_vol.reload()
    probe_files = list(probe_src.glob("probe_layer_*.pt"))
    if not probe_files:
        raise RuntimeError(f"No probes at {probe_src}. Train first with modal_train_0G_v3.py")
    log.info("Found %d probes", len(probe_files))

    # Volume commits
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
        log.info("[shutdown] Committing...")
        try:
            results_vol.commit()
        except Exception:
            pass
        if _args:
            raise SystemExit(1)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)

    # Load model + probes
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
    log.info("Loading %s", model_id)
    model, tokenizer = load_model_and_tokenizer(
        model_id, device, eager_attn=spec.is_moe, multi_gpu=False,
    )
    model.requires_grad_(False)

    try:
        hf_cache_vol.commit()
    except Exception:
        pass

    records = load_dataset("/root/data/exp3_dataset.jsonl")
    log.info("Dataset: %d prompts", len(records))

    # Patch logger for volume commits on progress
    _orig = log.info
    def _patched(msg, *a, **kw):
        _orig(msg, *a, **kw)
        if isinstance(msg, str) and "prompts done" in msg:
            try:
                results_vol.commit()
            except Exception:
                pass
    import src.poc.cross_model.tuned_lens as tl_module
    tl_module.log.info = _patched

    # Run eval — variant is passed through, IT gets chat template inside eval_commitment
    max_new_tokens = 64 if spec.is_moe else 512
    log.info("Starting eval: %d prompts, max_new_tokens=%d, greedy", len(records), max_new_tokens)

    use_chat_template = variant == "it"
    log.info("Chat template: %s (variant=%s)", use_chat_template, variant)

    try:
        summary = eval_commitment(
            model, tokenizer, adapter, spec, probes, records, device,
            output_path=out_path,
            variant=variant,
            max_new_tokens=max_new_tokens,
            collect_full=True,
            collect_top5=False,
            arrays_dir=arrays_dir,
            apply_chat_template=use_chat_template,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    summary_path.write_text(json.dumps(summary, indent=2))
    results_vol.commit()
    log.info("=== COMPLETE: %s/%s — %d prompts ===",
             model_name, variant, summary.get("n_prompts", "?"))
    return summary


@app.local_entrypoint()
def main(models: str = ""):
    """Launch eval for all models.

        modal run --detach scripts/modal_eval_0G_v3.py
    """
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = ["llama31_8b", "mistral_7b", "olmo2_7b", "gemma3_4b", "qwen3_4b", "deepseek_v2_lite"]

    configs = [(m, v) for m in model_list for v in ["pt", "it"]]

    print(f"Launching {len(configs)} eval jobs (v3: IT uses chat template)")
    print("GPU: B200 | greedy | collect_full | arrays_{variant}/")
    print()

    results = list(eval_model_variant.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL EVAL COMPLETE (v3)")
    print("=" * 60)
    for (m, v), r in zip(configs, results):
        if isinstance(r, dict):
            print(f"  {m:20s} {v}: {r.get('n_prompts', '?')} prompts")
        else:
            print(f"  {m:20s} {v}: FAILED")
    print("\nResults on volume: 0g-results-v3")
    print("Download: modal volume get 0g-results-v3 results_v3 --force")
