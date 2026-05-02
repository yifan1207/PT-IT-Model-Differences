#!/usr/bin/env python3
"""Materialize small causal-feature crosscoder slices for Exp30/34 mediation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import (
    _load_causal_rank_rows,
    build_feature_selections,
)


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    raise ValueError(f"unsupported dtype={name}")


def _causal_union(
    *,
    causal_csv: Path,
    k_list: list[int],
    random_seeds: list[int],
) -> dict[int, set[int]]:
    causal_rows, causal_pool_rows = _load_causal_rank_rows(causal_csv)
    selections = build_feature_selections(
        latent_rows=causal_pool_rows,
        k_list=k_list,
        random_seeds=random_seeds,
        selection_suite="causal",
        causal_rows=causal_rows,
        causal_pool_rows=causal_pool_rows,
    )
    by_layer: dict[int, set[int]] = {}
    for _feature_set, _k, _seed, selection in selections:
        if selection.mode != "ablate":
            continue
        for layer, latent_ids in selection.by_layer.items():
            by_layer.setdefault(int(layer), set()).update(int(idx) for idx in latent_ids)
    return by_layer


def _write_slice(
    *,
    layer_dir: Path,
    latent_ids: list[int],
    dtype: torch.dtype,
    force: bool,
) -> dict[str, int | str]:
    out_path = layer_dir / "crosscoder_sliced_causal.pt"
    if out_path.exists() and not force:
        return {"path": str(out_path), "n_latents": len(latent_ids), "status": "exists"}

    full_path = layer_dir / "crosscoder.pt"
    payload = torch.load(full_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    state = payload["state_dict"]
    ids = torch.tensor(sorted(set(latent_ids)), dtype=torch.long)
    decoder_full = state["decoder"]
    if bool(config.get("scale_topk_by_decoder_norm", False)):
        score_weights = decoder_full[ids].float().norm(dim=-1).sum(dim=-1).clamp_min(float(config.get("eps", 1e-8)))
    else:
        score_weights = torch.ones(ids.numel(), dtype=torch.float32)
    sliced = {
        "config": config,
        "latent_ids": ids,
        "encoder": state["encoder"][:, :, ids].contiguous().to(dtype=dtype),
        "encoder_bias": state["encoder_bias"][ids].contiguous().to(dtype=dtype),
        "decoder": decoder_full[ids, :, :].contiguous().to(dtype=dtype),
        "input_mean": state["input_mean"].contiguous().to(dtype=dtype),
        "inference_threshold": state["inference_threshold"].detach().cpu().float(),
        "score_weights": score_weights.contiguous().to(dtype=dtype),
    }
    torch.save(sliced, out_path)
    del payload, state, decoder_full, sliced
    gc.collect()
    return {"path": str(out_path), "n_latents": len(ids), "status": "written"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--causal-feature-csv", type=Path, default=None)
    parser.add_argument("--k-list", nargs="+", type=int, default=[25, 50, 100, 200, 500])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    causal_csv = args.causal_feature_csv or args.run_root / "feature_stats" / "causal_feature_scores.csv"
    by_layer = _causal_union(
        causal_csv=causal_csv,
        k_list=list(args.k_list),
        random_seeds=list(args.random_seeds),
    )
    if args.layers:
        keep = set(int(layer) for layer in args.layers)
        by_layer = {layer: ids for layer, ids in by_layer.items() if layer in keep}
    if not by_layer:
        raise RuntimeError(f"No causal features found to slice from {causal_csv}")

    rows = []
    for layer, ids in sorted(by_layer.items()):
        layer_dir = args.run_root / "dictionaries" / f"layer_{layer}"
        row = _write_slice(
            layer_dir=layer_dir,
            latent_ids=sorted(ids),
            dtype=_dtype(args.dtype),
            force=bool(args.force),
        )
        row["layer"] = int(layer)
        rows.append(row)
        print(f"[exp30-slice] layer={layer} n_latents={row['n_latents']} status={row['status']} path={row['path']}")

    summary_path = args.run_root / "feature_stats" / "causal_slice_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[exp30-slice] wrote {summary_path}")


if __name__ == "__main__":
    main()
