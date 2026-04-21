"""
Modal: Lightweight per-layer KL collection only.

Collects ONLY raw_kl_final and tuned_kl_final arrays per variant.
No commitment computation, no other arrays — pure forward pass + KL.
~3-5x faster than full eval_commitment with collect_full=True.

    modal run --detach scripts/modal_collect_kl_only.py
"""
from __future__ import annotations
import modal

app = modal.App("kl-per-layer-collect")

probes_vol = modal.Volume.from_name("0g-probes-v2", create_if_missing=True)
results_vol = modal.Volume.from_name("0g-results-kl", create_if_missing=True)
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
    retries=modal.Retries(max_retries=5, backoff_coefficient=1.0, initial_delay=10.0),
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
def collect_kl(model_name: str, variant: str) -> dict:
    """Collect per-layer KL(ℓ || final) for raw and tuned logit-lens.

    Autoregressive generation (greedy, deterministic), captures hidden states
    at every layer at every step, computes raw and tuned KL to final layer.
    Saves two NPY arrays: [total_steps, n_layers].
    """
    import atexit
    import json
    import logging
    import signal
    import threading
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("kl_collect")
    log.info("=== KL collection: %s/%s ===", model_name, variant)
    log.info("GPU: %s", torch.cuda.get_device_name(0))

    # ── Paths ────────────────────────────────────────────────────────────────
    probe_src = Path(f"/probes/{model_name}/{variant}")
    out_dir = Path(f"/results/{model_name}/arrays_{variant}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Skip if done ─────────────────────────────────────────────────────────
    results_vol.reload()
    if (out_dir / "raw_kl_final.npy").exists() and (out_dir / "step_index.jsonl").exists():
        log.info("Already complete — skipping")
        return {"status": "skipped", "model": model_name, "variant": variant}

    # ── Volume commits ───────────────────────────────────────────────────────
    stop_event = threading.Event()

    def _periodic_commit():
        while not stop_event.is_set():
            stop_event.wait(timeout=60)
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

    # ── Load model + probes ──────────────────────────────────────────────────
    from src.poc.cross_model.config import get_spec, model_id_for_variant
    from src.poc.cross_model.adapters import get_adapter
    from src.poc.cross_model.utils import load_model_and_tokenizer, load_dataset
    from src.poc.cross_model.tuned_lens import _load_probes

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    device = torch.device("cuda:0")

    probes_vol.reload()
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

    # ── Setup ────────────────────────────────────────────────────────────────
    n_layers = spec.n_layers
    final_norm_mod = adapter.final_norm(model)
    model_dtype = next(final_norm_mod.parameters()).dtype
    W_U = adapter.lm_head(model).weight.detach().float().T.to(device)  # [d_model, vocab]
    layer_modules = adapter.layers(model)
    stop_ids = list(adapter.stop_token_ids(tokenizer))

    records = load_dataset("/root/data/exp3_dataset.jsonl")
    log.info("Dataset: %d prompts", len(records))
    max_new_tokens = 64 if spec.is_moe else 512

    # ── Accumulators ─────────────────────────────────────────────────────────
    all_raw_kl: list[list[float]] = []
    all_tuned_kl: list[list[float]] = []
    step_index: list[dict] = []
    global_step = 0

    for ri, rec in enumerate(records):
        pid = rec.get("id", f"rec_{ri}")
        raw_prompt = rec.get("raw_prompt") or rec.get("prompt", "")

        # Per-step KL for this prompt
        prompt_raw_kl: list[list[float]] = []
        prompt_tuned_kl: list[list[float]] = []

        # Hook: capture hidden states at each layer
        step_buf = [None] * n_layers

        def _process_step():
            """Called after all layers captured for one generation step."""
            all_h = torch.stack([step_buf[i] for i in range(n_layers)], dim=0)  # [n_layers, d_model]

            # Final layer logits (ground truth)
            h_final = all_h[-1]
            normed_final = final_norm_mod(h_final.unsqueeze(0)).squeeze(0)
            logits_final = normed_final.float() @ W_U
            log_p_final = F.log_softmax(logits_final, dim=-1)

            # Raw logit-lens KL
            all_normed_raw = final_norm_mod(all_h.unsqueeze(1)).squeeze(1)
            all_raw_logits = all_normed_raw.float() @ W_U
            all_raw_log_q = F.log_softmax(all_raw_logits, dim=-1)
            raw_kl = F.kl_div(
                all_raw_log_q, log_p_final.unsqueeze(0).expand_as(all_raw_log_q),
                reduction="none", log_target=True,
            ).sum(dim=-1)
            raw_kl_list = [max(raw_kl[i].item(), 0.0) for i in range(n_layers)]

            # Tuned logit-lens KL
            tuned_parts = []
            for li in range(n_layers):
                if li in probes:
                    tuned_parts.append(probes[li](all_h[li].float().unsqueeze(0)).squeeze(0).to(model_dtype))
                else:
                    tuned_parts.append(all_h[li])
            all_h_tuned = torch.stack(tuned_parts, dim=0)
            all_normed_tuned = final_norm_mod(all_h_tuned.unsqueeze(1)).squeeze(1)
            all_tuned_logits = all_normed_tuned.float() @ W_U
            all_tuned_log_q = F.log_softmax(all_tuned_logits, dim=-1)
            tuned_kl = F.kl_div(
                all_tuned_log_q, log_p_final.unsqueeze(0).expand_as(all_tuned_log_q),
                reduction="none", log_target=True,
            ).sum(dim=-1)
            tuned_kl_list = [max(tuned_kl[i].item(), 0.0) for i in range(n_layers)]

            prompt_raw_kl.append(raw_kl_list)
            prompt_tuned_kl.append(tuned_kl_list)

        def make_hook(layer_idx):
            def hook(module, input, output):
                vec = adapter.residual_from_output(output).detach().squeeze(0)
                if vec.dim() > 1:
                    vec = vec[-1]
                step_buf[layer_idx] = vec
                if layer_idx == n_layers - 1:
                    _process_step()
            return hook

        handles = [layer_modules[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

        input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(device)
        try:
            model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,
                eos_token_id=stop_ids or None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        except Exception as e:
            log.warning("Prompt %s failed: %s", pid, e)
        finally:
            for h in handles:
                h.remove()

        n_steps = len(prompt_raw_kl)
        if n_steps > 0:
            step_index.append({
                "prompt_id": pid,
                "start_step": global_step,
                "end_step": global_step + n_steps,
                "n_steps": n_steps,
            })
            all_raw_kl.extend(prompt_raw_kl)
            all_tuned_kl.extend(prompt_tuned_kl)
            global_step += n_steps

        if (ri + 1) % 50 == 0:
            log.info("[kl] %d/%d prompts done (%d steps)", ri + 1, len(records), global_step)

    # ── Save ─────────────────────────────────────────────────────────────────
    log.info("Saving arrays: %d steps × %d layers", len(all_raw_kl), n_layers)
    np.save(out_dir / "raw_kl_final.npy", np.array(all_raw_kl, dtype=np.float16))
    np.save(out_dir / "tuned_kl_final.npy", np.array(all_tuned_kl, dtype=np.float16))
    with open(out_dir / "step_index.jsonl", "w") as f:
        for entry in step_index:
            f.write(json.dumps(entry) + "\n")

    stop_event.set()
    commit_thread.join(timeout=5)
    results_vol.commit()

    summary = {
        "model": model_name, "variant": variant,
        "n_prompts": len(step_index), "n_steps": global_step,
        "n_layers": n_layers,
    }
    log.info("=== COMPLETE: %s/%s — %d prompts, %d steps ===",
             model_name, variant, len(step_index), global_step)
    return summary


@app.local_entrypoint()
def main(models: str = "", skip: str = ""):
    """Launch KL collection for all models.

    Usage:
        modal run --detach scripts/modal_collect_kl_only.py
        modal run --detach scripts/modal_collect_kl_only.py --skip olmo2_7b/it
    """
    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "deepseek_v2_lite", "olmo2_7b"]

    skip_set = set(skip.split(",")) if skip else set()

    configs = []
    for m in model_list:
        for v in ["pt", "it"]:
            if f"{m}/{v}" not in skip_set:
                configs.append((m, v))

    print(f"Launching {len(configs)} KL collection jobs")
    print("Collects ONLY: raw_kl_final.npy + tuned_kl_final.npy + step_index.jsonl")
    print("GPU: B200 | greedy | 512 max tokens (64 for DeepSeek)")
    print()

    results = list(collect_kl.starmap(configs))

    print("\n" + "=" * 60)
    print("ALL KL COLLECTION COMPLETE")
    print("=" * 60)
    for (m, v), r in zip(configs, results):
        print(f"  {m:20s} {v}: {r}")

    print("\nDownload: modal volume get 0g-results-kl . --force")
