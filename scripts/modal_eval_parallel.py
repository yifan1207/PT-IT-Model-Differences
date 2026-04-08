"""
Data-parallel eval for gemma3_4b/pt and mistral_7b/pt.
5 workers per model, each processes 1/5 of prompts.
Arrays saved to arrays_pt_w{i}/, merged after.

    modal run --detach scripts/modal_eval_parallel.py
"""
from __future__ import annotations
import modal

app = modal.App("0g-eval-parallel")

probes_vol = modal.Volume.from_name("0g-probes-v3", create_if_missing=True)
results_vol = modal.Volume.from_name("0g-results-v3", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("0g-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7.0", "transformers==4.57.3", "accelerate",
        "numpy", "scipy", "sentencepiece", "protobuf",
        "huggingface_hub", "safetensors", "einops",
    )
    .add_local_python_source("src")
    .add_local_file("data/exp3_dataset.jsonl", remote_path="/root/data/exp3_dataset.jsonl")
)


@app.function(
    gpu="B200", timeout=86400,
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0),
    image=image,
    volumes={"/probes": probes_vol, "/results": results_vol, "/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-token")],
    max_containers=10, memory=65536, cpu=8.0,
)
def eval_worker(model_name: str, variant: str, worker_index: int, n_workers: int) -> dict:
    import atexit, json, logging, signal, threading
    from pathlib import Path
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("eval_worker")
    log.info("=== Worker %d/%d: %s/%s ===", worker_index, n_workers, model_name, variant)
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    probe_src = Path(f"/probes/{model_name}/{variant}")
    results_dir = Path(f"/results/{model_name}/tuned_lens/commitment")
    results_dir.mkdir(parents=True, exist_ok=True)

    out_path = results_dir / f"tuned_lens_commitment_{variant}_w{worker_index}.jsonl"
    arrays_dir = results_dir / f"arrays_{variant}_w{worker_index}"

    # Skip if done
    results_vol.reload()
    if (arrays_dir / "step_index.jsonl").exists():
        log.info("Worker %d already complete — skipping", worker_index)
        return {"status": "skipped", "worker": worker_index}

    # Volume commits
    stop_event = threading.Event()
    def _periodic_commit():
        while not stop_event.is_set():
            stop_event.wait(timeout=30)
            if not stop_event.is_set():
                try: results_vol.commit()
                except: pass
    commit_thread = threading.Thread(target=_periodic_commit, daemon=True)
    commit_thread.start()
    def _shutdown(*a):
        try: results_vol.commit()
        except: pass
        if a: raise SystemExit(1)
    signal.signal(signal.SIGTERM, _shutdown)
    atexit.register(_shutdown)

    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer, load_dataset
    from src.poc.cross_model.tuned_lens import _load_probes, eval_commitment

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    probes_vol.reload()
    probes = _load_probes(probe_src, spec.d_model, device)
    log.info("Loaded %d probes", len(probes))

    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(model_id, device, eager_attn=spec.is_moe)
    model.requires_grad_(False)
    try: hf_cache_vol.commit()
    except: pass

    all_records = load_dataset("/root/data/exp3_dataset.jsonl")
    records = all_records[worker_index::n_workers]
    log.info("Worker %d: %d/%d records", worker_index, len(records), len(all_records))

    # Patch logger for volume commits
    import src.poc.cross_model.tuned_lens as tl
    _orig = tl.log.info
    def _p(msg, *a, **kw):
        _orig(msg, *a, **kw)
        if isinstance(msg, str) and "prompts done" in msg:
            try: results_vol.commit()
            except: pass
    tl.log.info = _p

    max_new_tokens = 64 if spec.is_moe else 512
    use_template = variant == "it"

    try:
        summary = eval_commitment(
            model, tokenizer, adapter, spec, probes, records, device,
            output_path=out_path, variant=variant,
            max_new_tokens=max_new_tokens,
            collect_full=True, collect_top5=False,
            arrays_dir=arrays_dir,
            apply_chat_template=use_template,
        )
    finally:
        stop_event.set()
        commit_thread.join(timeout=5)

    results_vol.commit()
    log.info("=== Worker %d COMPLETE: %d prompts ===", worker_index, summary.get("n_prompts", "?"))
    return summary


@app.local_entrypoint()
def main():
    n_workers = 5
    configs = []
    for model in ["gemma3_4b", "mistral_7b"]:
        for wi in range(n_workers):
            configs.append((model, "pt", wi, n_workers))

    print(f"Launching {len(configs)} workers (2 models × {n_workers} workers)")
    print("Each worker processes 1/5 of prompts → 5x faster")
    print()

    results = list(eval_worker.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL WORKERS COMPLETE")
    for (m, v, wi, nw), r in zip(configs, results):
        print(f"  {m}/{v} w{wi}: {r}")
    print("\nMerge arrays locally after download")
