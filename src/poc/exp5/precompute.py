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
    """Mirror runtime._prepare_input_ids so precomputed stats use the same tokens."""
    prompt = rec["formats"][cfg.prompt_format]
    if cfg.model_variant == "it" and cfg.apply_chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Complete the following: {prompt}"}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(cfg.device)
    return tokenizer.encode(prompt, return_tensors="pt").to(cfg.device)


def _sample_records(records: list[dict], n: int) -> list[dict]:
    """Fixed-seed sample so every worker sees the same ordered list and slices it."""
    if len(records) <= n:
        return records
    return random.Random(42).sample(records, n)


# ── Mean MLP outputs ──────────────────────────────────────────────────────────

def _compute_mean_partial(
    cfg: Exp5Config,
    sampled: list[dict],
    worker_index: int,
    n_workers: int,
    out_dir: Path,
) -> Path:
    """Accumulate partial sums for one worker's record slice.

    Each worker processes sampled[worker_index::n_workers] and saves raw
    float64 sums + token counts.  The merge step adds all partials together
    before dividing — so the math is identical to the single-worker case.
    """
    worker_records = sampled[worker_index::n_workers]
    print(f"  [mean partial w{worker_index}/{n_workers}] {len(worker_records)} records on {cfg.device}", flush=True)

    loaded = load_model(cfg)
    hooks = hooks_for_model(cfg.model_id)
    sums = [torch.zeros(cfg.d_model, dtype=torch.float64) for _ in range(cfg.n_layers)]
    token_counts = [0] * cfg.n_layers

    for rec in worker_records:
        prompt_ids = _prepare_input_ids(rec, loaded.tokenizer, cfg)
        mlp_saves = []
        with loaded.model.trace(prompt_ids):
            layers_root = navigate_path(loaded.model, hooks.layers_root)
            for i in range(cfg.n_layers):
                mlp_saves.append(nnsave(navigate_path(layers_root[i], hooks.mlp_output)))
        for i, saved in enumerate(mlp_saves):
            T = saved.shape[1]
            sums[i] += saved[0, :, :].float().cpu().to(torch.float64).sum(dim=0)
            token_counts[i] += T

    partial_dir = ensure_dir(out_dir / "partials" / f"w{worker_index}")
    out_path = partial_dir / "mean_partial.npz"
    payload = {f"sum_layer_{i}": sums[i].to(torch.float32).numpy() for i in range(cfg.n_layers)}
    payload.update({f"count_layer_{i}": np.array(token_counts[i], dtype=np.int64) for i in range(cfg.n_layers)})
    np.savez_compressed(out_path, **payload)
    print(f"  [mean partial w{worker_index}] saved → {out_path}", flush=True)
    return out_path


def _merge_mean_partials(cfg: Exp5Config, out_dir: Path, n_workers: int) -> Path:
    """Combine all worker partials into the final mean_mlp_outputs.npz."""
    sums = [np.zeros(cfg.d_model, dtype=np.float64) for _ in range(cfg.n_layers)]
    token_counts = [0] * cfg.n_layers

    for w in range(n_workers):
        partial_path = out_dir / "partials" / f"w{w}" / "mean_partial.npz"
        if not partial_path.exists():
            raise FileNotFoundError(f"Missing partial from worker {w}: {partial_path}")
        with np.load(partial_path) as d:
            for i in range(cfg.n_layers):
                sums[i] += d[f"sum_layer_{i}"].astype(np.float64)
                token_counts[i] += int(d[f"count_layer_{i}"])

    out_path = out_dir / "mean_mlp_outputs.npz"
    payload = {
        f"layer_{i}": (sums[i] / max(token_counts[i], 1)).astype(np.float32)
        for i in range(cfg.n_layers)
    }
    np.savez_compressed(out_path, **payload)
    save_json(out_dir / "mean_mlp_outputs.meta.json", {
        "model_id": cfg.model_id,
        "dataset_path": cfg.dataset_path,
        "prompt_format": cfg.prompt_format,
        "total_tokens_per_layer": token_counts,
        "n_layers": cfg.n_layers,
        "d_model": cfg.d_model,
        "n_workers": n_workers,
        "averaging": "all_positions",
    })
    print(f"  [mean merge] wrote {out_path}", flush=True)
    return out_path


def compute_mean_mlp_outputs(
    cfg: Exp5Config,
    n_records: int = 1000,
    worker_index: int = 0,
    n_workers: int = 1,
) -> Path:
    """Compute layerwise mean MLP outputs; supports data-parallel workers.

    With n_workers=1 (default) this is identical to the original single-GPU
    path.  With n_workers>1 each call saves a partial and the caller must
    invoke this once more with compute="merge-mean" to produce the final npz.

    Following Li & Janson (2024): mean ablation constant a* = E[component(X)],
    averaged over all token positions across all reference prompts.
    """
    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)
    sampled = _sample_records(records, n_records)
    out_dir = ensure_dir(cfg.run_dir / "precompute")

    if n_workers == 1:
        # Single-worker fast path: compute and write final artifact directly.
        _compute_mean_partial(cfg, sampled, 0, 1, out_dir)
        return _merge_mean_partials(cfg, out_dir, n_workers=1)

    # Parallel path: each worker saves its partial; caller merges separately.
    _compute_mean_partial(cfg, sampled, worker_index, n_workers, out_dir)
    return out_dir / "partials" / f"w{worker_index}" / "mean_partial.npz"


def merge_mean_mlp_outputs(cfg: Exp5Config, n_workers: int) -> Path:
    out_dir = ensure_dir(cfg.run_dir / "precompute")
    return _merge_mean_partials(cfg, out_dir, n_workers)


# ── Corrective directions ─────────────────────────────────────────────────────

def _compute_directions_partial(
    cfg: Exp5Config,
    sampled: list[dict],
    worker_index: int,
    n_workers: int,
    out_dir: Path,
    layer_ids_override: list[int] | None = None,
) -> Path:
    """Accumulate partial IT-PT MLP-output difference sums for one worker's slice.

    Both PT and IT models are loaded sequentially on the same GPU (PT first,
    cache activations to CPU, unload, then load IT) — same as the single-GPU
    path, but on a smaller record subset so each GPU finishes faster.
    """
    worker_records = sampled[worker_index::n_workers]
    print(f"  [dir partial w{worker_index}/{n_workers}] {len(worker_records)} records on {cfg.device}", flush=True)

    # Clear model_id so __post_init__ derives it from model_variant.
    pt_cfg = Exp5Config(**{**cfg.to_dict(), "model_variant": "pt", "model_id": "", "run_name": cfg.run_name + "_pt_tmp"})
    it_cfg = Exp5Config(**{**cfg.to_dict(), "model_variant": "it", "model_id": "", "run_name": cfg.run_name + "_it_tmp"})

    corrective_layer_ids = layer_ids_override if layer_ids_override is not None else list(cfg.corrective_layers)

    # Pass 1: PT
    print(f"  [dir partial w{worker_index}] loading PT ...", flush=True)
    pt_loaded = load_model(pt_cfg)
    pt_hooks = hooks_for_model(pt_cfg.model_id)
    pt_cache: dict[int, list[torch.Tensor]] = {i: [] for i in corrective_layer_ids}
    for rec in worker_records:
        pt_ids = _prepare_input_ids(rec, pt_loaded.tokenizer, pt_cfg)
        pt_saves = []
        with pt_loaded.model.trace(pt_ids):
            pt_layers = navigate_path(pt_loaded.model, pt_hooks.layers_root)
            for i in corrective_layer_ids:
                pt_saves.append((i, nnsave(navigate_path(pt_layers[i], pt_hooks.mlp_output))))
        for li, saved in pt_saves:
            pt_cache[li].append(saved[0, -1, :].float().cpu())
    del pt_loaded
    torch.cuda.empty_cache()

    # Pass 2: IT
    print(f"  [dir partial w{worker_index}] loading IT ...", flush=True)
    it_loaded = load_model(it_cfg)
    it_hooks = hooks_for_model(it_cfg.model_id)
    sums = {i: torch.zeros(cfg.d_model, dtype=torch.float64) for i in corrective_layer_ids}
    count = 0
    for rec_idx, rec in enumerate(worker_records):
        it_ids = _prepare_input_ids(rec, it_loaded.tokenizer, it_cfg)
        it_saves = []
        with it_loaded.model.trace(it_ids):
            it_layers = navigate_path(it_loaded.model, it_hooks.layers_root)
            for i in corrective_layer_ids:
                it_saves.append((i, nnsave(navigate_path(it_layers[i], it_hooks.mlp_output))))
        for li, it_saved in it_saves:
            sums[li] += (it_saved[0, -1, :].float().cpu() - pt_cache[li][rec_idx]).to(torch.float64)
        count += 1

    partial_dir = ensure_dir(out_dir / "partials" / f"w{worker_index}")
    out_path = partial_dir / "directions_partial.npz"
    payload = {f"sum_layer_{li}": sums[li].to(torch.float32).numpy() for li in corrective_layer_ids}
    payload["count"] = np.array(count, dtype=np.int64)
    np.savez_compressed(out_path, **payload)
    print(f"  [dir partial w{worker_index}] saved → {out_path}", flush=True)
    return out_path


def _merge_directions_partials(cfg: Exp5Config, out_dir: Path, n_workers: int) -> Path:
    corrective_layer_ids = list(cfg.corrective_layers)
    sums = {li: np.zeros(cfg.d_model, dtype=np.float64) for li in corrective_layer_ids}
    total_count = 0

    for w in range(n_workers):
        partial_path = out_dir / "partials" / f"w{w}" / "directions_partial.npz"
        if not partial_path.exists():
            raise FileNotFoundError(f"Missing partial from worker {w}: {partial_path}")
        with np.load(partial_path) as d:
            for li in corrective_layer_ids:
                sums[li] += d[f"sum_layer_{li}"].astype(np.float64)
            total_count += int(d["count"])

    out_path = out_dir / "corrective_directions.npz"
    payload = {}
    for li in corrective_layer_ids:
        vec = sums[li] / max(total_count, 1)
        norm = float(np.linalg.norm(vec))
        payload[f"layer_{li}"] = (vec / (norm + 1e-12)).astype(np.float32)
    np.savez_compressed(out_path, **payload)
    save_json(out_dir / "corrective_directions.meta.json", {
        "pt_model_id": f"google/gemma-3-4b-pt",
        "it_model_id": f"google/gemma-3-4b-it",
        "dataset_path": cfg.dataset_path,
        "prompt_format": cfg.prompt_format,
        "n_records": total_count,
        "layers": corrective_layer_ids,
        "token_position": "last",
        "activation_space": "mlp_output",
        "n_workers": n_workers,
    })
    print(f"  [dir merge] wrote {out_path}", flush=True)
    return out_path


def compute_corrective_directions(
    cfg: Exp5Config,
    n_records: int = 1000,
    worker_index: int = 0,
    n_workers: int = 1,
    layer_ids_override: list[int] | None = None,
) -> Path:
    """Compute IT-PT corrective directions; supports data-parallel workers."""
    records = load_dataset_records(cfg.dataset_path, prompt_format=cfg.prompt_format)
    sampled = _sample_records(records, n_records)
    out_dir = ensure_dir(cfg.run_dir / "precompute")

    if n_workers == 1:
        _compute_directions_partial(cfg, sampled, 0, 1, out_dir, layer_ids_override)
        return _merge_directions_partials(cfg, out_dir, n_workers=1)

    _compute_directions_partial(cfg, sampled, worker_index, n_workers, out_dir, layer_ids_override)
    return out_dir / "partials" / f"w{worker_index}" / "directions_partial.npz"


def merge_corrective_directions(cfg: Exp5Config, n_workers: int) -> Path:
    out_dir = ensure_dir(cfg.run_dir / "precompute")
    return _merge_directions_partials(cfg, out_dir, n_workers)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Precompute exp5 ablation artifacts.")
    p.add_argument("--variant", choices=["pt", "it"], default="it")
    p.add_argument("--dataset", default="data/exp3_dataset.jsonl")
    p.add_argument("--prompt-format", choices=["A", "B"], default="B")
    p.add_argument("--n-records", type=int, default=1000)
    p.add_argument("--compute", choices=["mean", "directions", "both",
                                         "merge-mean", "merge-directions", "merge-both"],
                   default="both")
    p.add_argument("--run-name", default="precompute")
    p.add_argument("--device", default="cuda")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--proposal-boundary", type=int, default=20,
                   help="First layer of the corrective phase. Directions are computed "
                        "for layers [proposal_boundary, n_layers). Default: 20.")
    p.add_argument("--direction-layers", default="",
                   help="Comma-separated list of specific layer indices to compute directions for, "
                        "overriding proposal-boundary. E.g. '5,15' to compute only layers 5 and 15.")
    args = p.parse_args()

    cfg = Exp5Config(
        experiment="baseline",
        model_variant=args.variant,
        dataset_path=args.dataset,
        prompt_format=args.prompt_format,
        run_name=args.run_name,
        device=args.device,
        proposal_boundary=args.proposal_boundary,
    )

    if args.compute in {"mean", "both"}:
        path = compute_mean_mlp_outputs(cfg, args.n_records, args.worker_index, args.n_workers)
        print(f"Saved mean partial/output → {path}")

    if args.compute in {"directions", "both"}:
        layer_ids_override = None
        if args.direction_layers:
            layer_ids_override = [int(x.strip()) for x in args.direction_layers.split(",") if x.strip()]
        path = compute_corrective_directions(cfg, args.n_records, args.worker_index, args.n_workers, layer_ids_override)
        print(f"Saved directions partial/output → {path}")

    if args.compute in {"merge-mean", "merge-both"}:
        path = merge_mean_mlp_outputs(cfg, args.n_workers)
        print(f"Merged mean MLP outputs → {path}")

    if args.compute in {"merge-directions", "merge-both"}:
        path = merge_corrective_directions(cfg, args.n_workers)
        print(f"Merged corrective directions → {path}")


if __name__ == "__main__":
    main()
