"""
L8: Intrinsic dimensionality profile via TwoNN estimator.

For each model (PT and IT), runs a single forward pass per prompt and collects
the last-token residual stream at every layer.  Then estimates intrinsic
dimensionality (ID) using the Two-Nearest-Neighbour method (Facco et al. 2017).

Expected finding: ID minimum at ~1/3 depth (phase boundary) in all models,
matching the Gemma 3 4B result from exp4.

═══ Key design decisions ═════════════════════════════════════════════════
  • NO CHAT TEMPLATE — same raw text for PT and IT (plan §4.L8 Step 1).
  • 8-GPU parallel workers (same pattern as L1L2), merged with TwoNN
    run on the merged residual matrix.
  • Residuals stored as float32 numpy arrays (not bfloat16) so TwoNN
    distances are numerically stable.
  • Reuses skdim.id.TwoNN — same library as src/poc/exp4/analysis/intrinsic_dim.py.

═══ Output ═══════════════════════════════════════════════════════════════
  results/cross_model/{model}/{variant}/
    L8_residuals_w{worker}.npz   per-worker {prompt_id: [n_layers, d_model]}
    L8_residuals.npz             merged residuals (after --merge-only)
    L8_id_profile.json           ID estimates per layer

  JSON format (matches plan §4.L8 Step 4):
    {
      "model": str, "variant": str, "n_prompts": int, "n_layers": int,
      "intrinsic_dim":      [float * n_layers],
      "intrinsic_dim_ci_low":  [float * n_layers],
      "intrinsic_dim_ci_high": [float * n_layers]
    }
"""
from __future__ import annotations

import json
import logging
import argparse
from pathlib import Path

import torch
import numpy as np

from src.poc.cross_model.config import get_spec, MODEL_REGISTRY, model_id_for_variant
from src.poc.cross_model.adapters import get_adapter
from src.poc.cross_model.utils import (
    load_model_and_tokenizer,
    load_dataset,
    get_raw_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import skdim
    _SKDIM_AVAILABLE = True
except ImportError:
    _SKDIM_AVAILABLE = False
    log.warning("scikit-dimension not installed. Run: uv add scikit-dimension")


# ── residual collection ───────────────────────────────────────────────────────

def collect_residuals(
    records: list[dict],
    model,
    tokenizer,
    adapter,
    spec,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Single forward pass per prompt; collect last-token residual at every layer.

    Returns dict: {prompt_id: np.ndarray [n_layers, d_model] float32}
    """
    n_layers = spec.n_layers
    results:  dict[str, np.ndarray] = {}
    step_buf: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, output):
            h = adapter.residual_from_output(output)
            # Last token position, float32 for numerical stability in TwoNN
            step_buf[layer_idx] = h[0, -1, :].detach().float().cpu()
        return hook

    layer_modules = adapter.layers(model)
    handles = [layer_modules[i].register_forward_hook(make_hook(i)) for i in range(n_layers)]

    for i, rec in enumerate(records):
        step_buf.clear()
        raw_prompt = get_raw_prompt(rec)
        pid        = rec.get("id", f"rec_{i}")

        input_ids = tokenizer.encode(raw_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(input_ids)

        # Stack layer residuals into [n_layers, d_model]
        stacked = np.stack([
            step_buf[ℓ].numpy() if ℓ in step_buf
            else np.zeros(spec.d_model, dtype=np.float32)
            for ℓ in range(n_layers)
        ])
        results[pid] = stacked

        if (i + 1) % 100 == 0:
            log.info("  %d/%d prompts", i + 1, len(records))

    for h in handles:
        h.remove()

    return results


# ── TwoNN estimation ──────────────────────────────────────────────────────────

def estimate_id_profile(
    residuals_npz: dict[str, np.ndarray],
    n_layers: int,
) -> dict:
    """Run TwoNN ID estimation per layer from a merged residual dict.

    residuals_npz: {prompt_id: [n_layers, d_model]} — all prompts, all layers.
    """
    if not _SKDIM_AVAILABLE:
        raise RuntimeError("scikit-dimension required. Run: uv add scikit-dimension")

    # Build per-layer matrices [n_prompts, d_model]
    pids = list(residuals_npz.keys())
    n_prompts = len(pids)
    layer_matrices: list[np.ndarray] = []
    for ℓ in range(n_layers):
        mat = np.stack([residuals_npz[pid][ℓ] for pid in pids])  # [n_prompts, d_model]
        layer_matrices.append(mat)

    id_vals:  list[float] = []
    ci_lows:  list[float] = []
    ci_highs: list[float] = []
    rng = np.random.default_rng(42)

    for ℓ, X in enumerate(layer_matrices):
        try:
            id_est = float(skdim.id.TwoNN().fit_transform(X))
        except Exception as e:
            log.warning("TwoNN failed at layer %d: %s", ℓ, e)
            id_vals.append(float("nan"))
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))
            continue

        # Bootstrap CI (100 subsamples, 5th/95th percentile)
        # Use 80% subsampling WITHOUT replacement to avoid duplicate rows
        # which cause TwoNN to fail on degenerate distance matrices.
        boot_estimates: list[float] = []
        sub_size = max(int(n_prompts * 0.8), 2)
        for _ in range(100):
            idx = rng.choice(n_prompts, size=sub_size, replace=False)
            try:
                boot_estimates.append(float(skdim.id.TwoNN().fit_transform(X[idx])))
            except Exception:
                pass

        id_vals.append(id_est)
        if boot_estimates:
            ci_lows.append(float(np.percentile(boot_estimates, 5)))
            ci_highs.append(float(np.percentile(boot_estimates, 95)))
        else:
            ci_lows.append(float("nan"))
            ci_highs.append(float("nan"))

        if (ℓ + 1) % 8 == 0 or ℓ == n_layers - 1:
            log.info("  layer %d/%d: ID=%.1f", ℓ + 1, n_layers, id_est)

    return {
        "n_prompts":          n_prompts,
        "intrinsic_dim":      id_vals,
        "intrinsic_dim_ci_low":  ci_lows,
        "intrinsic_dim_ci_high": ci_highs,
    }


# ── worker ────────────────────────────────────────────────────────────────────

def run_worker(
    model_name: str,
    variant: str,
    dataset_path: str,
    out_dir: str,
    device_str: str,
    worker_index: int,
    n_workers: int,
    n_eval_examples: int | None,
) -> None:
    """Single-GPU worker: collect residuals for this worker's slice."""
    spec    = get_spec(model_name)
    adapter = get_adapter(model_name)
    device  = torch.device(device_str)

    model_id = model_id_for_variant(spec, variant)
    model, tokenizer = load_model_and_tokenizer(model_id, device_str)

    records = load_dataset(
        dataset_path,
        worker_index=worker_index,
        n_workers=n_workers,
        n_examples=n_eval_examples,
    )

    out_path = Path(out_dir) / f"L8_residuals_w{worker_index}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        log.info("[w%d] Residuals already exist at %s — skipping.", worker_index, out_path)
        return

    log.info("[w%d] Collecting residuals for %d prompts...", worker_index, len(records))
    residuals = collect_residuals(records, model, tokenizer, adapter, spec, device)

    np.savez_compressed(out_path, **residuals)
    log.info("[w%d] Saved %d residuals → %s", worker_index, len(residuals), out_path)


# ── merge + estimate ──────────────────────────────────────────────────────────

def merge_and_estimate(
    out_dir: Path,
    model_name: str,
    variant: str,
    n_workers: int,
) -> None:
    """Merge per-worker NPZ files, run TwoNN, save ID profile."""
    spec = get_spec(model_name)

    merged_path = out_dir / "L8_residuals.npz"
    if not merged_path.exists():
        log.info("Merging %d worker NPZ files...", n_workers)
        all_residuals: dict[str, np.ndarray] = {}
        for w in range(n_workers):
            wpath = out_dir / f"L8_residuals_w{w}.npz"
            if not wpath.exists():
                log.warning("Missing worker file: %s", wpath)
                continue
            data = np.load(wpath)
            for k in data.files:
                all_residuals[k] = data[k]
        np.savez_compressed(merged_path, **all_residuals)
        log.info("Merged %d prompts → %s", len(all_residuals), merged_path)
    else:
        log.info("Loading existing merged residuals from %s", merged_path)
        data = np.load(merged_path)
        all_residuals = {k: data[k] for k in data.files}

    id_path = out_dir / "L8_id_profile.json"
    if id_path.exists():
        log.info("ID profile already exists: %s — skipping.", id_path)
        return

    log.info("Estimating ID profile for %d prompts × %d layers...",
             len(all_residuals), spec.n_layers)
    profile = estimate_id_profile(all_residuals, spec.n_layers)
    profile.update({
        "model":    model_name,
        "variant":  variant,
        "n_layers": spec.n_layers,
    })

    with open(id_path, "w") as f:
        json.dump(profile, f, indent=2)
    log.info("ID profile saved → %s", id_path)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="L8: Intrinsic dimensionality profile.")
    parser.add_argument("--model",           required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--variant",         required=True, choices=["pt", "it"])
    parser.add_argument("--dataset",         default="data/eval_dataset_v2.jsonl")
    parser.add_argument("--n-eval-examples", type=int, default=None)
    parser.add_argument("--device",          default="cuda:0")
    parser.add_argument("--worker-index",    type=int, default=0)
    parser.add_argument("--n-workers",       type=int, default=1)
    parser.add_argument("--out-dir",         default=None)
    parser.add_argument(
        "--merge-only", action="store_true",
        help="Skip collection; merge NPZ files and run TwoNN.",
    )
    args = parser.parse_args()

    spec    = get_spec(args.model)
    out_dir = Path(args.out_dir) if args.out_dir else spec.result_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        merge_and_estimate(out_dir, args.model, args.variant, args.n_workers)
        return

    run_worker(
        model_name=args.model,
        variant=args.variant,
        dataset_path=args.dataset,
        out_dir=str(out_dir),
        device_str=args.device,
        worker_index=args.worker_index,
        n_workers=args.n_workers,
        n_eval_examples=args.n_eval_examples,
    )


if __name__ == "__main__":
    main()
