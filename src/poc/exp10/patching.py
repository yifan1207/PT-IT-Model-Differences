"""
Exp10 — Phase 3: Causal Validation via Activation Patching.

For the top-k layers by probe R², patches the convergence-gap direction
from IT into PT and measures whether KL-to-final increases at downstream
layers (i.e., convergence slows).

5 conditions per (layer, token):
  - commit:      project Δh onto convergence-gap direction, patch that component
  - full:        replace h_pt with h_it entirely at this layer
  - random:      same magnitude along a random unit vector
  - mean:        project Δh onto existing mean IT-PT direction
  - orthogonal:  everything in Δh EXCEPT the convergence-gap component

Causal metric: ΔKL_causal(ℓ') = KL_patched(ℓ') - KL_baseline(ℓ') at
downstream layers. Positive = patch slowed convergence.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.poc.cross_model.config import get_spec, model_id_for_variant
from src.poc.cross_model.adapters import get_adapter, ModelAdapter
from src.poc.cross_model.utils import (
    load_model_and_tokenizer,
    load_dataset,
    get_raw_prompt,
)
from src.poc.exp10.collect_paired import (
    commitment_continuous,
    _forward_capture_all_layers,
    _compute_kl_to_final_raw,
    _compute_kl_to_final_tuned,
    KL_THRESHOLD,
)

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
CONDITIONS = ["commit", "full", "random", "mean", "orthogonal"]


# ── Direction loading ──────────────────────────────────────────────────────────

def _load_directions_npz(path: Path) -> dict[int, torch.Tensor]:
    """Load directions from NPZ file. Keys: layer_0, layer_1, ..."""
    dirs: dict[int, torch.Tensor] = {}
    if not path.exists():
        return dirs
    with np.load(path) as data:
        for k in data.files:
            if k.startswith("layer_"):
                li = int(k.split("_", 1)[1])
                dirs[li] = torch.tensor(data[k], dtype=torch.float32)
    return dirs


def _load_probe_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _select_top_layers(probe_summary: dict, top_k: int = 10) -> list[int]:
    """Select top-k layers by R² from probe summary."""
    layers_r2 = [(r["layer"], r["r2_test"]) for r in probe_summary["per_layer"]]
    layers_r2.sort(key=lambda x: x[1], reverse=True)
    return [li for li, _ in layers_r2[:top_k]]


# ── Patching hook ──────────────────────────────────────────────────────────────

def _make_patch_hook(
    adapter: ModelAdapter,
    condition: str,
    target_pos: int,
    delta_h_token: torch.Tensor,   # [d_model] — Δh at this layer for this token
    h_it_token: torch.Tensor,      # [d_model] — IT activation for full replacement
    d_commit: torch.Tensor,        # [d_model] unit vector
    d_mean: torch.Tensor,          # [d_model] unit vector
    d_random: torch.Tensor,        # [d_model] unit vector
):
    """Create a forward hook that patches residual stream at target_pos.

    The hook modifies the PT model's activation at layer ℓ, position target_pos.
    """
    # Cast all vectors to model dtype (bfloat16) to avoid dtype mismatches in hook
    model_dtype = delta_h_token.dtype
    d_commit = d_commit.to(model_dtype)
    d_mean = d_mean.to(model_dtype)
    d_random = d_random.to(model_dtype)

    def hook(module, inp, output):
        h = adapter.residual_from_output(output)
        # h: [1, seq_len, d_model]

        with torch.no_grad():
            if condition == "commit":
                # Patch commitment-direction component of Δh
                proj_scalar = delta_h_token @ d_commit
                h[0, target_pos] = h[0, target_pos] + proj_scalar * d_commit

            elif condition == "full":
                # Full replacement: h_pt → h_it
                h[0, target_pos] = h_it_token

            elif condition == "random":
                # Same magnitude as d_commit projection, random direction
                proj_scalar = delta_h_token @ d_commit
                h[0, target_pos] = h[0, target_pos] + proj_scalar * d_random

            elif condition == "mean":
                # Mean-direction component of Δh
                proj_scalar = delta_h_token @ d_mean
                h[0, target_pos] = h[0, target_pos] + proj_scalar * d_mean

            elif condition == "orthogonal":
                # Everything EXCEPT d_commit component
                proj_scalar = delta_h_token @ d_commit
                ortho = delta_h_token - proj_scalar * d_commit
                h[0, target_pos] = h[0, target_pos] + ortho

        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h

    return hook


# ── Commitment measurement (for patched model) ────────────────────────────────

@torch.no_grad()
def _measure_kl_profile_at_position(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,       # [1, seq_len]
    target_pos: int,
    n_layers: int,
    final_norm: nn.Module,
    W_U: torch.Tensor,
    probes: dict[int, nn.Module] | None = None,
) -> torch.Tensor:
    """Run forward pass and return per-layer KL-to-final at target_pos.

    Returns: [n_layers] float tensor (KL in nats, non-negative).
    """
    captured = _forward_capture_all_layers(model, adapter, input_ids, n_layers, model.device)

    # Stack hidden states at target_pos: [n_layers, d_model]
    h_layers = torch.stack([captured[li][target_pos] for li in range(n_layers)])
    h_final = captured[n_layers - 1][target_pos]

    # Reshape to [n_layers, 1, d_model] for the compute functions
    h_layers_3d = h_layers.unsqueeze(1)
    h_final_2d = h_final.unsqueeze(0)

    if probes:
        kl = _compute_kl_to_final_tuned(h_layers_3d, h_final_2d, final_norm, W_U, probes)
    else:
        kl = _compute_kl_to_final_raw(h_layers_3d, h_final_2d, final_norm, W_U)

    return kl[:, 0]  # [n_layers]


@torch.no_grad()
def _measure_commitment_at_position(
    model: nn.Module,
    adapter: ModelAdapter,
    input_ids: torch.Tensor,
    target_pos: int,
    n_layers: int,
    final_norm: nn.Module,
    W_U: torch.Tensor,
    probes: dict[int, nn.Module] | None = None,
    kl_threshold: float = KL_THRESHOLD,
) -> float:
    """Backward-compat wrapper: returns scalar commitment layer index."""
    kl = _measure_kl_profile_at_position(
        model, adapter, input_ids, target_pos, n_layers, final_norm, W_U, probes,
    )
    return commitment_continuous(kl.cpu().tolist(), kl_threshold)


# ── Main patching function ─────────────────────────────────────────────────────

def validate_patching(
    model_name: str,
    device: torch.device | str,
    probes_dir: str | Path,
    paired_data_dir: str | Path,
    output_dir: str | Path,
    *,
    mean_dir_path: str | Path | None = None,
    n_test_prompts: int = 120,
    max_tokens_per_prompt: int = 5,
    top_k_layers: int = 10,
    max_gen_tokens: int = 128,
    dataset_path: str | Path = "data/eval_dataset_v2.jsonl",
    tuned_lens_dir: str | Path | None = None,
    kl_threshold: float = KL_THRESHOLD,
):
    """Phase 3: Causal patching validation.

    For the top-k layers by probe R², patches d_commit into PT and measures
    whether commitment shifts.

    Args:
        model_name: Key in MODEL_REGISTRY.
        device: GPU device.
        probes_dir: Phase 2 output (commitment_directions.npz, probe_summary.json).
        paired_data_dir: Phase 1 output (for re-running forced decoding on test set).
        output_dir: Where to save patching results.
        n_test_prompts: Number of test prompts (last N from dataset).
        max_tokens_per_prompt: Max generated tokens to patch per prompt.
        top_k_layers: Number of top layers to test.
        max_gen_tokens: Max generation tokens for IT.
        dataset_path: Path to eval dataset.
        tuned_lens_dir: Path to tuned lens probes (for commitment measurement).
        kl_threshold: KL threshold for commitment.
    """
    device = torch.device(device) if isinstance(device, str) else device
    probes_dir = Path(probes_dir)
    paired_data_dir = Path(paired_data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = get_spec(model_name)
    adapter = get_adapter(model_name)
    n_layers = spec.n_layers
    d_model = spec.d_model
    is_moe = spec.is_moe

    if is_moe and max_gen_tokens > 64:
        max_gen_tokens = 64

    # ── Load directions ───────────────────────────────────────────────────────
    commit_dirs = _load_directions_npz(probes_dir / "commitment_directions.npz")
    if mean_dir_path is None:
        mean_dir_path = Path(f"results/cross_model/{model_name}/directions/corrective_directions.npz")
    mean_dirs = _load_directions_npz(Path(mean_dir_path))
    log.info("Loaded %d commitment dirs, %d mean dirs", len(commit_dirs), len(mean_dirs))

    # ── Select top layers ─────────────────────────────────────────────────────
    summary = _load_probe_summary(probes_dir / "probe_summary.json")
    top_layers = _select_top_layers(summary, top_k_layers)
    log.info("Top %d layers by R²: %s", top_k_layers, top_layers)

    # ── Load PT model (and IT for forced decoding) ────────────────────────────
    log.info("Loading IT model for generation...")
    model_it, tokenizer_it = load_model_and_tokenizer(
        spec.it_id, device, eager_attn=is_moe,
    )
    log.info("Loading PT model for patching...")
    model_pt, tokenizer_pt = load_model_and_tokenizer(
        spec.pt_id, device, eager_attn=is_moe,
    )

    # ── Load tuned lens probes for PT (for commitment measurement) ────────────
    from src.poc.cross_model.tuned_lens import _load_probes
    probes_pt_tuned: dict = {}
    use_tuned = False
    if model_name != "gemma3_4b" and tuned_lens_dir is not None:
        pt_probe_dir = Path(tuned_lens_dir) / model_name / "tuned_lens" / "pt"
        if not pt_probe_dir.exists():
            pt_probe_dir = Path(tuned_lens_dir) / model_name / "pt"
        if pt_probe_dir.exists():
            probes_pt_tuned = _load_probes(pt_probe_dir, d_model, device)
            if probes_pt_tuned:
                use_tuned = True

    # ── Logit lens components ─────────────────────────────────────────────────
    final_norm_pt = adapter.final_norm(model_pt)
    W_U_pt = adapter.lm_head(model_pt).weight.T.float()
    final_norm_it = adapter.final_norm(model_it)
    W_U_it = adapter.lm_head(model_it).weight.T.float()

    # ── Load test prompts (last n_test_prompts from dataset) ──────────────────
    all_records = load_dataset(dataset_path)
    # Use last n_test_prompts as test set (first 80% used for training)
    test_records = all_records[-n_test_prompts:]
    log.info("Using %d test prompts", len(test_records))

    # ── Stop tokens ───────────────────────────────────────────────────────────
    stop_ids = list(adapter.stop_token_ids(tokenizer_it))

    # ── Random directions (seeded) ────────────────────────────────────────────
    rng = torch.Generator().manual_seed(RANDOM_SEED)
    random_dirs: dict[int, torch.Tensor] = {}
    for li in top_layers:
        d_rand = torch.randn(d_model, generator=rng)
        random_dirs[li] = (d_rand / d_rand.norm()).to(device)

    # ── Normalise directions and move to device ───────────────────────────────
    for li in top_layers:
        if li in commit_dirs:
            v = commit_dirs[li].to(device)
            commit_dirs[li] = v / (v.norm() + 1e-12)
        if li in mean_dirs:
            v = mean_dirs[li].to(device)
            mean_dirs[li] = v / (v.norm() + 1e-12)

    # ── Resume support ────────────────────────────────────────────────────────
    results_path = output_dir / "patching_results.jsonl"
    done_keys: set[str] = set()
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_keys.add(f"{rec['prompt_id']}_{rec['token_idx']}_{rec['layer']}_{rec['condition']}")

    # ── Main patching loop ────────────────────────────────────────────────────
    t0 = time.time()
    n_patched = 0
    layer_modules_pt = adapter.layers(model_pt)

    with open(results_path, "a") as f_out:
        for ri, record in enumerate(test_records):
            prompt_id = record.get("id", f"test_{ri}")
            raw_prompt = get_raw_prompt(record)
            if not raw_prompt.strip():
                continue

            # Step 1: IT generates (with chat template for IT)
            it_prompt = adapter.apply_template(tokenizer_it, raw_prompt, is_it=True)
            prompt_ids = tokenizer_it.encode(it_prompt, return_tensors="pt").to(device)
            n_prompt_tokens = prompt_ids.shape[1]

            with torch.no_grad():
                gen_output = model_it.generate(
                    prompt_ids,
                    max_new_tokens=max_gen_tokens,
                    do_sample=False,
                    temperature=1.0,
                    eos_token_id=stop_ids or None,
                    pad_token_id=tokenizer_it.pad_token_id or tokenizer_it.eos_token_id,
                )
            n_gen = gen_output.shape[1] - n_prompt_tokens
            if n_gen < 2:
                continue

            full_ids = gen_output  # [1, total_len]

            # Step 2: Forced-decode both to get Δh and baseline commitment
            captured_it = _forward_capture_all_layers(model_it, adapter, full_ids, n_layers, device)
            captured_pt = _forward_capture_all_layers(model_pt, adapter, full_ids, n_layers, device)

            # Select token positions to patch (up to max_tokens_per_prompt)
            gen_positions = list(range(n_prompt_tokens, min(n_prompt_tokens + n_gen, full_ids.shape[1])))
            patch_positions = gen_positions[:max_tokens_per_prompt]

            for t_pos in patch_positions:
                t_idx = t_pos - n_prompt_tokens

                # Baseline KL profile for PT at this position
                h_pt_layers = torch.stack([captured_pt[li][t_pos] for li in range(n_layers)])
                h_pt_final = captured_pt[n_layers - 1][t_pos]
                h_pt_3d = h_pt_layers.unsqueeze(1)
                h_pt_final_2d = h_pt_final.unsqueeze(0)
                if use_tuned:
                    kl_baseline = _compute_kl_to_final_tuned(
                        h_pt_3d, h_pt_final_2d, final_norm_pt, W_U_pt, probes_pt_tuned,
                    )[:, 0]  # [n_layers]
                else:
                    kl_baseline = _compute_kl_to_final_raw(
                        h_pt_3d, h_pt_final_2d, final_norm_pt, W_U_pt,
                    )[:, 0]  # [n_layers]
                c_baseline = commitment_continuous(kl_baseline.cpu().tolist(), kl_threshold)

                for layer in top_layers:
                    # Δh at this layer and position
                    delta_h_token = (captured_it[layer][t_pos] - captured_pt[layer][t_pos]).to(device)
                    h_it_token = captured_it[layer][t_pos].to(device)

                    d_commit_l = commit_dirs.get(layer, torch.zeros(d_model, device=device))
                    d_mean_l = mean_dirs.get(layer, torch.zeros(d_model, device=device))
                    d_random_l = random_dirs.get(layer, torch.zeros(d_model, device=device))

                    for condition in CONDITIONS:
                        key = f"{prompt_id}_{t_idx}_{layer}_{condition}"
                        if key in done_keys:
                            continue

                        # Register patching hook on this layer
                        hook_fn = _make_patch_hook(
                            adapter, condition, t_pos,
                            delta_h_token, h_it_token,
                            d_commit_l, d_mean_l, d_random_l,
                        )
                        handle = layer_modules_pt[layer].register_forward_hook(hook_fn)

                        try:
                            # Forward PT with patch active → measure KL profile
                            kl_patched = _measure_kl_profile_at_position(
                                model_pt, adapter, full_ids, t_pos,
                                n_layers, final_norm_pt, W_U_pt,
                                probes_pt_tuned if use_tuned else None,
                            )
                        finally:
                            handle.remove()

                        # Per-layer causal effect
                        delta_kl_causal = (kl_patched - kl_baseline).cpu().tolist()
                        c_patched = commitment_continuous(kl_patched.cpu().tolist(), kl_threshold)
                        delta_c_causal = c_patched - c_baseline

                        # Aggregate: mean ΔKL at downstream layers (ℓ' > patched layer)
                        downstream = [delta_kl_causal[l] for l in range(layer + 1, n_layers)]
                        mean_delta_kl_downstream = float(np.mean(downstream)) if downstream else 0.0

                        result = {
                            "prompt_id": prompt_id,
                            "token_idx": t_idx,
                            "layer": layer,
                            "condition": condition,
                            "c_baseline": c_baseline,
                            "c_patched": c_patched,
                            "delta_c_causal": delta_c_causal,
                            "delta_kl_causal": delta_kl_causal,
                            "mean_delta_kl_downstream": mean_delta_kl_downstream,
                        }
                        f_out.write(json.dumps(result) + "\n")
                        n_patched += 1

                        if n_patched % 100 == 0:
                            f_out.flush()

            # Free memory
            del captured_it, captured_pt
            torch.cuda.empty_cache()

            if (ri + 1) % 10 == 0:
                elapsed = time.time() - t0
                log.info(
                    "[Phase 3] %s: %d/%d prompts, %d patches, %.1f sec",
                    model_name, ri + 1, len(test_records), n_patched, elapsed,
                )

    # ── Aggregate results ─────────────────────────────────────────────────────
    _aggregate_patching_results(results_path, output_dir / "patching_summary.json", top_layers)

    elapsed = time.time() - t0
    log.info("[Phase 3] %s COMPLETE: %d patches in %.1f sec", model_name, n_patched, elapsed)


def _aggregate_patching_results(results_path: Path, summary_path: Path, top_layers: list[int]):
    """Aggregate per-(layer, condition) statistics from patching results."""
    from collections import defaultdict

    # Collect per (layer, condition)
    dc_by_lc: dict[tuple[int, str], list[float]] = defaultdict(list)
    dkl_by_lc: dict[tuple[int, str], list[float]] = defaultdict(list)

    with open(results_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["layer"], rec["condition"])
            dc_by_lc[key].append(rec["delta_c_causal"])
            dkl_by_lc[key].append(rec.get("mean_delta_kl_downstream", 0.0))

    # Compute summary statistics
    summary: dict[str, dict] = {}
    for layer in top_layers:
        layer_summary: dict[str, dict] = {}
        for cond in CONDITIONS:
            dc_vals = dc_by_lc.get((layer, cond), [])
            dkl_vals = dkl_by_lc.get((layer, cond), [])
            if not dc_vals:
                layer_summary[cond] = {
                    "mean_delta_c": 0, "mean_delta_kl_downstream": 0,
                    "std_delta_kl_downstream": 0, "std": 0, "n": 0,
                }
                continue

            dc_arr = np.array(dc_vals)
            dkl_arr = np.array(dkl_vals)
            layer_summary[cond] = {
                "mean_delta_c": float(dc_arr.mean()),
                "median_delta_c": float(np.median(dc_arr)),
                "std": float(dc_arr.std()),
                "mean_delta_kl_downstream": float(dkl_arr.mean()),
                "std_delta_kl_downstream": float(dkl_arr.std()),
                "median_delta_kl_downstream": float(np.median(dkl_arr)),
                "n": len(dc_vals),
            }

        # Permutation test: d_commit vs random (on ΔKL metric)
        commit_dkl = dkl_by_lc.get((layer, "commit"), [])
        random_dkl = dkl_by_lc.get((layer, "random"), [])
        p_value = _permutation_test(commit_dkl, random_dkl) if commit_dkl and random_dkl else 1.0
        layer_summary["p_value_commit_vs_random"] = p_value

        summary[str(layer)] = layer_summary

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved patching summary → %s", summary_path)


def _permutation_test(a: list[float], b: list[float], n_perms: int = 10000) -> float:
    """Two-sample permutation test on mean difference."""
    a_arr = np.array(a)
    b_arr = np.array(b)
    observed = abs(a_arr.mean() - b_arr.mean())
    combined = np.concatenate([a_arr, b_arr])
    n_a = len(a_arr)
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        rng.shuffle(combined)
        perm_diff = abs(combined[:n_a].mean() - combined[n_a:].mean())
        if perm_diff >= observed:
            count += 1
    return (count + 1) / (n_perms + 1)


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Exp10 Phase 3: Causal patching")
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--probes-dir", required=True)
    parser.add_argument("--paired-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-test-prompts", type=int, default=120)
    parser.add_argument("--max-tokens-per-prompt", type=int, default=5)
    parser.add_argument("--top-k-layers", type=int, default=10)
    parser.add_argument("--tuned-lens-dir", default=None)
    parser.add_argument("--mean-dir-path", default=None,
                        help="Path to corrective_directions.npz")
    args = parser.parse_args()

    validate_patching(
        model_name=args.model,
        device=args.device,
        probes_dir=args.probes_dir,
        paired_data_dir=args.paired_data_dir,
        output_dir=args.output_dir,
        mean_dir_path=args.mean_dir_path,
        n_test_prompts=args.n_test_prompts,
        max_tokens_per_prompt=args.max_tokens_per_prompt,
        top_k_layers=args.top_k_layers,
        tuned_lens_dir=args.tuned_lens_dir,
    )
