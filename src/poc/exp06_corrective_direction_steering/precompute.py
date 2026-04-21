"""Exp6 precomputation:

  1. Aggregate content-layer corrective direction (IT-PT diff at layers 0-11).
     Run via scripts/precompute_extra_directions.py --layers 0,1,...,11
     then this script aggregates them into a single unit vector.

  2. Mean feature activations ā[f] for governance features (for B1 γ scaling).

  3. Governance direction from W_dec projection (for B2).

CLI:
    python src/poc/exp06_corrective_direction_steering/precompute.py --task aggregate-content-direction
    python src/poc/exp06_corrective_direction_steering/precompute.py --task mean-feature-acts --device cuda:7
    python src/poc/exp06_corrective_direction_steering/precompute.py --task governance-direction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


# ── Task 1: Aggregate content-layer directions ────────────────────────────────

def aggregate_content_directions(
    content_directions_npz: str,
    out_path: str,
    proposal_boundary: int = 20,
) -> None:
    """Compute a single aggregate direction from per-layer content-layer directions.

    Loads layer_0 through layer_{proposal_boundary-1} from the NPZ (produced by
    scripts/precompute_extra_directions.py --layers 0,1,...,11), computes the
    mean direction vector, and re-normalizes it.

    Output: aggregate_direction.npz with key "aggregate" — a single [d_model]
    unit vector representing the average content-layer IT-PT difference.
    This is mapped to all corrective layers in interventions.build_intervention().
    """
    src = Path(content_directions_npz)
    if not src.exists():
        raise FileNotFoundError(f"Content directions not found: {src}")

    with np.load(src) as data:
        vecs = []
        for l in range(proposal_boundary):
            key = f"layer_{l}"
            if key in data.files:
                vecs.append(data[key].astype(np.float64))

    if not vecs:
        raise ValueError(f"No content-layer direction vectors found in {src}")

    agg = np.stack(vecs, axis=0).mean(axis=0)
    norm = float(np.linalg.norm(agg))
    agg = (agg / (norm + 1e-12)).astype(np.float32)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), aggregate=agg)
    print(f"Saved aggregate content direction (from {len(vecs)} layers, norm={norm:.4f}) → {out}")


# ── Task 2: Mean feature activations ─────────────────────────────────────────

def compute_mean_feature_acts(
    model_id: str,
    dataset_path: str,
    n_records: int,
    corrective_layers: list[int],
    out_dir: str,
    device: str,
    transcoder_release: str,
    transcoder_variant: str,
) -> None:
    """Compute mean transcoder feature activations over N prompts.

    Runs the IT model with plain forward hooks on pre_feedforward_layernorm
    at each corrective layer, collects transcoder.encode() outputs, and
    computes per-feature means.

    Output: {out_dir}/layer_{l}.npy for each corrective layer, shape [n_features].
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from huggingface_hub import snapshot_download  # type: ignore
    import random

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {model_id} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    print("Model loaded.", flush=True)

    # Load transcoders for corrective layers using the same path as shared/model.py
    print(f"Loading transcoders from {transcoder_release}...", flush=True)
    from circuit_tracer.transcoder.single_layer_transcoder import load_gemma_scope_2_transcoder
    pattern = f"transcoder_all/layer_*_{transcoder_variant}/params.safetensors"
    local_dir = snapshot_download(transcoder_release, allow_patterns=[pattern])
    transcoders = {}
    for l in corrective_layers:
        path = Path(local_dir) / "transcoder_all" / f"layer_{l}_{transcoder_variant}" / "params.safetensors"
        if path.exists():
            transcoders[l] = load_gemma_scope_2_transcoder(
                str(path), layer=l, device=torch.device(device), dtype=torch.bfloat16,
                lazy_encoder=False, lazy_decoder=False,
            )
        else:
            print(f"  Warning: transcoder not found for layer {l}: {path}", flush=True)

    # Load dataset
    print(f"Loading dataset {dataset_path}...", flush=True)
    records = []
    with open(dataset_path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    sampled = random.Random(42).sample(records, min(n_records, len(records)))
    print(f"Using {len(sampled)} records.", flush=True)

    # Collect feature activations with hooks
    cache: dict[int, list[torch.Tensor]] = {l: [] for l in corrective_layers if l in transcoders}
    hooks = []
    input_bufs: dict[int, list[torch.Tensor | None]] = {l: [None] for l in corrective_layers}

    for l_idx in corrective_layers:
        if l_idx not in transcoders:
            continue
        pre_ln = model.language_model.layers[l_idx].pre_feedforward_layernorm

        def make_hook(li: int):
            def hook(mod, inp, out):
                # Save last-token activation
                input_bufs[li][0] = out[0, -1:, :].float().cpu()
            return hook

        hooks.append(pre_ln.register_forward_hook(make_hook(l_idx)))

    try:
        for i, rec in enumerate(sampled):
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(sampled)}", flush=True)
            prompt = rec["formats"]["B"]
            ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(ids)
            # Encode saved activations
            for l_idx in corrective_layers:
                if l_idx not in transcoders or input_bufs[l_idx][0] is None:
                    continue
                x = input_bufs[l_idx][0].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    feats = transcoders[l_idx].encode(x)  # [1, n_features]
                cache[l_idx].append(feats.squeeze(0).float().cpu())
                input_bufs[l_idx][0] = None
    finally:
        for h in hooks:
            h.remove()

    # Compute means and save
    for l_idx in corrective_layers:
        if l_idx not in cache or not cache[l_idx]:
            print(f"  No data for layer {l_idx}, skipping.", flush=True)
            continue
        stack = torch.stack(cache[l_idx], dim=0)  # [N, n_features]
        means = stack.mean(dim=0).numpy().astype(np.float32)
        np.save(str(out_path / f"layer_{l_idx}.npy"), means)
        print(f"  layer {l_idx}: mean activation saved, shape {means.shape}", flush=True)

    print(f"Mean feature activations saved → {out_dir}")


# ── Task 3: Governance direction from W_dec ───────────────────────────────────

def compute_governance_direction(
    governance_features_path: str,
    mean_feature_acts_dir: str,
    transcoder_release: str,
    transcoder_variant: str,
    corrective_layers: list[int],
    feature_set_key: str,
    out_path: str,
    device: str,
) -> None:
    """Compute governance direction v_gov[l] = Σ_f (ā[f] × W_dec[f]) for each corrective layer.

    This is the W_dec-weighted sum of decoder vectors for governance features,
    giving a direction in residual stream space (d_model = 2560) that represents
    what governance features collectively push toward.

    v_gov[l] is then normalized to a unit vector and injected into PT in B2.
    """
    # Load feature sets
    with open(governance_features_path) as f:
        all_sets = json.load(f)

    # Load mean activations
    mean_acts_dir = Path(mean_feature_acts_dir)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}

    # Download transcoder safetensors once for all layers
    from circuit_tracer.transcoder.single_layer_transcoder import load_gemma_scope_2_transcoder
    from huggingface_hub import snapshot_download
    pattern = f"transcoder_all/layer_*_{transcoder_variant}/params.safetensors"
    tc_local_dir = snapshot_download(transcoder_release, allow_patterns=[pattern])
    print(f"Transcoders downloaded to {tc_local_dir}", flush=True)

    for l_idx in corrective_layers:
        layer_key = f"layer_{l_idx}"
        feat_indices = all_sets.get(layer_key, {}).get(feature_set_key, [])
        if not feat_indices:
            print(f"  layer {l_idx}: no features for {feature_set_key!r}, skipping.", flush=True)
            continue

        mean_acts_path = mean_acts_dir / f"layer_{l_idx}.npy"
        if not mean_acts_path.exists():
            print(f"  layer {l_idx}: mean_acts not found, skipping.", flush=True)
            continue
        mean_acts = np.load(str(mean_acts_path)).astype(np.float64)

        # Load W_dec for this layer from the transcoder
        try:
            tc_path = Path(tc_local_dir) / "transcoder_all" / f"layer_{l_idx}_{transcoder_variant}" / "params.safetensors"
            tc = load_gemma_scope_2_transcoder(str(tc_path), layer=l_idx, device=torch.device(device), dtype=torch.bfloat16,
                                               lazy_encoder=True, lazy_decoder=False)
            W_dec = tc.W_dec.detach().float().cpu().numpy()  # [n_features, d_model]
        except Exception as e:
            print(f"  layer {l_idx}: could not load W_dec: {e}", flush=True)
            continue

        # v_gov[l] = Σ_f (ā[f] × W_dec[f])
        v_gov = np.zeros(W_dec.shape[1], dtype=np.float64)
        for f in feat_indices:
            if f < len(mean_acts) and f < W_dec.shape[0]:
                v_gov += mean_acts[f] * W_dec[f].astype(np.float64)

        norm = float(np.linalg.norm(v_gov))
        if norm < 1e-8:
            print(f"  layer {l_idx}: near-zero governance direction, skipping.", flush=True)
            continue

        payload[layer_key] = (v_gov / norm).astype(np.float32)
        print(f"  layer {l_idx}: governance direction norm={norm:.4f}, {len(feat_indices)} features", flush=True)

    np.savez_compressed(str(out), **payload)
    print(f"Governance directions saved → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=[
        "aggregate-content-direction",
        "mean-feature-acts",
        "governance-direction",
    ], required=True)
    p.add_argument("--content-directions-npz",
                   default="results/exp06_corrective_direction_steering/precompute/content_directions.npz")
    p.add_argument("--out-path",
                   default="results/exp06_corrective_direction_steering/precompute/content_direction_aggregate.npz")
    p.add_argument("--device", default="cuda:7")
    p.add_argument("--dataset", default="data/exp6_dataset.jsonl")
    p.add_argument("--n-records", type=int, default=500)
    p.add_argument("--corrective-layers", default="20,21,22,23,24,25,26,27,28,29,30,31,32,33")
    p.add_argument("--mean-feature-acts-dir",
                   default="results/exp06_corrective_direction_steering/precompute/mean_feature_acts_it")
    p.add_argument("--governance-features-path",
                   default="results/exp06_corrective_direction_steering/governance_feature_sets.json")
    p.add_argument("--feature-set-key", default="method12_top100")
    p.add_argument("--transcoder-release", default="google/gemma-scope-2-4b-it")
    p.add_argument("--transcoder-variant", default="width_16k_l0_big_affine")
    args = p.parse_args()

    layers = [int(x) for x in args.corrective_layers.split(",")]

    if args.task == "aggregate-content-direction":
        aggregate_content_directions(
            args.content_directions_npz, args.out_path
        )

    elif args.task == "mean-feature-acts":
        model_id = "google/gemma-3-4b-it"
        compute_mean_feature_acts(
            model_id=model_id,
            dataset_path=args.dataset,
            n_records=args.n_records,
            corrective_layers=layers,
            out_dir=args.mean_feature_acts_dir,
            device=args.device,
            transcoder_release=args.transcoder_release,
            transcoder_variant=args.transcoder_variant,
        )

    elif args.task == "governance-direction":
        compute_governance_direction(
            governance_features_path=args.governance_features_path,
            mean_feature_acts_dir=args.mean_feature_acts_dir,
            transcoder_release=args.transcoder_release,
            transcoder_variant=args.transcoder_variant,
            corrective_layers=layers,
            feature_set_key=args.feature_set_key,
            out_path=args.out_path.replace("aggregate", "governance") if "aggregate" in args.out_path
                     else "results/exp06_corrective_direction_steering/precompute/content_direction_governance.npz",
            device=args.device,
        )


if __name__ == "__main__":
    main()
