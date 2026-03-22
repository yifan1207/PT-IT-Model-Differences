from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from nnsight import save as nnsave

from src.poc.collect import load_dataset_records
from src.poc.exp5.config import Exp5Config
from src.poc.exp5.utils import ensure_dir, navigate_path, save_json
from src.poc.shared.collect_config import hooks_for_model
from src.poc.shared.model import load_model


def _prepare_input_ids(rec: dict, tokenizer: Any, cfg: Exp5Config) -> torch.Tensor:
    """Mirror runtime._prepare_input_ids so precomputed stats use the same tokens.

    When cfg.apply_chat_template is True and the model is instruction-tuned, both
    the precompute pass and the runtime generation pass must see identical token
    sequences; otherwise the mean activations / corrective directions are computed
    on a different distribution than they are applied to.
    """
    prompt = rec["formats"][cfg.prompt_format]
    if cfg.model_variant == "it" and cfg.apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Complete the following: {prompt}"}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(cfg.device)
    return tokenizer.encode(prompt, return_tensors="pt").to(cfg.device)


def _sample_records(records: list[dict], n: int) -> list[dict]:
    if len(records) <= n:
        return records
    rng = random.Random(42)
    return rng.sample(records, n)


def compute_mean_mlp_outputs(cfg: Exp5Config, n_records: int = 1000) -> Path:
    """Compute layerwise mean MLP outputs on a neutral reference set.

    Following Li & Janson (2024, arXiv 2409.09951): the mean ablation constant
    a* = E[component(X)] is the expectation over the subtask distribution.
    We average over ALL token positions across ALL reference prompts, giving a
    position-agnostic mean per layer.  This is more principled than a last-token-
    only mean because:
      (a) Wang et al. (2023, IOI) compute per-position means for each patched
          position; a global mean is the natural generalisation when patching all
          positions simultaneously.
      (b) During generation, MLP outputs at non-last positions still participate
          in attention for later steps; using a last-token mean at all positions
          would introduce a distributional mismatch at those earlier positions.
    """
    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)
    sampled = _sample_records(records, n_records)
    loaded = load_model(cfg)
    hooks = hooks_for_model(cfg.model_id)
    sums = [torch.zeros(cfg.d_model, device=cfg.device, dtype=torch.float64) for _ in range(cfg.n_layers)]
    # Track total number of (prompt, position) pairs seen per layer.
    token_counts = [0] * cfg.n_layers

    for rec in sampled:
        prompt_ids = _prepare_input_ids(rec, loaded.tokenizer, cfg)
        with loaded.model.trace(prompt_ids):
            mlp_saves = []
            layers_root = navigate_path(loaded.model, hooks.layers_root)
            for i in range(cfg.n_layers):
                mlp_saves.append(nnsave(navigate_path(layers_root[i], hooks.mlp_output)))
        for i, saved in enumerate(mlp_saves):
            # saved shape: [B, T, d_model] = [1, T, d_model]
            # Sum over all T token positions; divide by total token count at the end.
            T = saved.shape[1]
            sums[i] += saved[0, :, :].float().to(torch.float64).sum(dim=0)
            token_counts[i] += T

    out_dir = ensure_dir(cfg.run_dir / "precompute")
    out_path = out_dir / "mean_mlp_outputs.npz"
    payload = {
        f"layer_{i}": (sums[i] / max(token_counts[i], 1)).cpu().to(torch.float32).numpy()
        for i in range(cfg.n_layers)
    }
    np.savez_compressed(out_path, **payload)
    n_records_used = len(sampled)
    save_json(
        out_dir / "mean_mlp_outputs.meta.json",
        {
            "model_id": cfg.model_id,
            "dataset_path": cfg.dataset_path,
            "prompt_format": cfg.prompt_format,
            "n_records": n_records_used,
            "total_tokens_per_layer": token_counts,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "averaging": "all_positions",  # as opposed to last_token_only
        },
    )
    return out_path


def compute_corrective_directions(cfg: Exp5Config, n_records: int = 1000) -> Path:
    """Compute late-layer IT-PT mean residual difference directions.

    Direction per layer = mean(mlp_IT[layer, -1] - mlp_PT[layer, -1]) over prompts,
    then L2-normalised.  We capture MLP output (before the residual add), not the
    full residual stream, so the direction lives in the same activation space where
    InterventionSpec.apply_in_trace() applies it.  Last-token position follows
    Arditi et al. (2024, arXiv 2406.11717).

    Models are loaded and run SEQUENTIALLY (PT first, then IT) to avoid holding
    two 4B-parameter models in VRAM simultaneously (~16 GB peak with concurrent
    loading).  PT residuals are cached to CPU RAM before the PT model is unloaded;
    ~143 MB for 1000 prompts × 14 corrective layers × 2560 dims × 4 bytes.
    """
    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)
    sampled = _sample_records(records, n_records)

    pt_cfg = Exp5Config(**{**cfg.to_dict(), "model_variant": "pt", "run_name": cfg.run_name + "_pt_tmp"})
    it_cfg = Exp5Config(**{**cfg.to_dict(), "model_variant": "it", "run_name": cfg.run_name + "_it_tmp"})

    corrective_layer_ids = list(cfg.corrective_layers)

    # ── Pass 1: PT model — collect last-token residuals for corrective layers ──
    print("  [corrective directions] loading PT model for pass 1 ...")
    pt_loaded = load_model(pt_cfg)
    pt_hooks = hooks_for_model(pt_cfg.model_id)

    # pt_cache[layer_idx] = list of [d_model] float32 tensors, one per prompt
    pt_cache: dict[int, list[torch.Tensor]] = {i: [] for i in corrective_layer_ids}

    for rec in sampled:
        pt_ids = _prepare_input_ids(rec, pt_loaded.tokenizer, pt_cfg)
        with pt_loaded.model.trace(pt_ids):
            pt_saves = []
            pt_layers = navigate_path(pt_loaded.model, pt_hooks.layers_root)
            for i in corrective_layer_ids:
                # Collect MLP output (before residual add) — the same activation
                # space where directional ablation is applied in InterventionSpec.
                pt_saves.append((i, nnsave(navigate_path(pt_layers[i], pt_hooks.mlp_output))))
        for li, saved in pt_saves:
            # Store on CPU immediately to free GPU memory after each prompt.
            pt_cache[li].append(saved[0, -1, :].float().cpu())

    del pt_loaded
    import torch as _torch
    _torch.cuda.empty_cache()

    # ── Pass 2: IT model — collect residuals and compute deltas on the fly ────
    print("  [corrective directions] loading IT model for pass 2 ...")
    it_loaded = load_model(it_cfg)
    it_hooks = hooks_for_model(it_cfg.model_id)

    sums = {i: torch.zeros(cfg.d_model, dtype=torch.float64) for i in corrective_layer_ids}
    count = 0
    for rec_idx, rec in enumerate(sampled):
        it_ids = _prepare_input_ids(rec, it_loaded.tokenizer, it_cfg)
        with it_loaded.model.trace(it_ids):
            it_saves = []
            it_layers = navigate_path(it_loaded.model, it_hooks.layers_root)
            for i in corrective_layer_ids:
                it_saves.append((i, nnsave(navigate_path(it_layers[i], it_hooks.mlp_output))))
        for li, it_saved in it_saves:
            it_vec = it_saved[0, -1, :].float().cpu()
            pt_vec = pt_cache[li][rec_idx]
            sums[li] += (it_vec - pt_vec).to(torch.float64)
        count += 1

    out_dir = ensure_dir(cfg.run_dir / "precompute")
    out_path = out_dir / "corrective_directions.npz"
    payload = {}
    for li in cfg.corrective_layers:
        vec = sums[li] / max(count, 1)
        vec = vec / (vec.norm() + 1e-12)
        payload[f"layer_{li}"] = vec.cpu().to(torch.float32).numpy()
    np.savez_compressed(out_path, **payload)
    save_json(
        out_dir / "corrective_directions.meta.json",
        {
            "pt_model_id": pt_cfg.model_id,
            "it_model_id": it_cfg.model_id,
            "dataset_path": cfg.dataset_path,
            "prompt_format": cfg.prompt_format,
            "n_records": count,
            "layers": corrective_layer_ids,
            "token_position": "last",      # Arditi et al. 2024: last-token position
            "activation_space": "mlp_output",  # before residual add — matches apply_in_trace
            "loading": "sequential",   # PT then IT, to avoid 2× VRAM peak
        },
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute exp5 ablation artifacts.")
    p.add_argument("--variant", choices=["pt", "it"], default="it")
    p.add_argument("--dataset", default="data/exp3_dataset.jsonl")
    p.add_argument("--prompt-format", choices=["A", "B"], default="B")
    p.add_argument("--n-records", type=int, default=1000)
    p.add_argument("--compute", choices=["mean", "directions", "both"], default="both")
    p.add_argument("--run-name", default="precompute")
    args = p.parse_args()

    cfg = Exp5Config(
        experiment="baseline",
        model_variant=args.variant,
        dataset_path=args.dataset,
        prompt_format=args.prompt_format,
        run_name=args.run_name,
    )
    if args.compute in {"mean", "both"}:
        path = compute_mean_mlp_outputs(cfg, n_records=args.n_records)
        print(f"Saved mean MLP outputs → {path}")
    if args.compute in {"directions", "both"}:
        path = compute_corrective_directions(cfg, n_records=args.n_records)
        print(f"Saved corrective directions → {path}")


if __name__ == "__main__":
    main()
