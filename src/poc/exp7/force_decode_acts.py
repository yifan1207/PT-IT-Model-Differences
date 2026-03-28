"""Force-decode PT on IT's token sequences to control for KV-cache confound (Exp7 0B).

The concern: IT and PT generate different token sequences. MLP activation differences
may partly reflect different input histories (KV-cache effects) rather than weight
differences. Solution: force PT to process IT's exact tokens via teacher-forcing.

Teacher-forcing method:
  For each token position t = 0..k-1:
    input_ids = [prompt_ids, it_gen_ids[0], ..., it_gen_ids[t]]
    Forward pass → capture MLP output at position -1 (last = forced token)
  Mean over t → per-record per-layer PT-on-IT activation

Output: results/exp7/0B/acts/w{N}.npz
  "record_ids"          : str [n_records_this_worker]
  "it_acts_{L}"         : float32 [n_records, D_MODEL]  (PT model, forced on IT tokens)
  "per_record_k_it"     : int32 [n_records]

Note: The output uses "it_acts_{L}" keys for API compatibility with
bootstrap_directions.py --matched-mode (which loads the second acts dir
under the same key names as IT).

Usage:
  for i in {0..7}; do
      uv run python -m src.poc.exp7.force_decode_acts \\
          --worker-index $i --n-workers 8 --device cuda:$i &
  done; wait
  uv run python -m src.poc.exp7.force_decode_acts \\
      --merge-only --n-workers 8

  # Test
  uv run python -m src.poc.exp7.force_decode_acts \\
      --worker-index 0 --n-workers 1 --device cuda:0 --n-records 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.poc.exp5.config import Exp5Config
from src.poc.shared.model import load_model

ALL_LAYERS = list(range(1, 34))
D_MODEL = 2560
MAX_GEN = 80

WORK_DIR = Path("results/precompute_v2_work")
OUTPUT_DIR = Path("results/exp7/0B/acts")


def _get_raw(loaded):
    return loaded.model._model


def _force_decode_acts(
    model_raw,
    tokenizer,
    rows: list[dict],
    device: str,
    *,
    verbose: bool = True,
) -> tuple[dict[int, list[np.ndarray]], np.ndarray]:
    """Teacher-force PT model on IT token IDs. Return per-record mean per-layer activations.

    For each record:
      For each generated position t (0..k-1):
        input_ids = [prompt_ids, it_gen_ids[:t+1]]
        run forward pass → capture MLP output at the last position
      Mean over positions → per-record per-layer activation

    This is O(k) forward passes per record (not k² — each pass is prefix-length).
    For k=80 and 600 records = 48,000 forward passes. With batch_size=1 on a modern
    GPU and short sequences this takes ~15-20 min per worker.
    """
    n = len(rows)
    acts_by_layer: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}
    per_record_k = []

    for idx, row in enumerate(rows):
        k = min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), MAX_GEN)
        it_gen_ids = row["it_gen_ids"][:k]

        if k == 0 or not it_gen_ids:
            for li in ALL_LAYERS:
                acts_by_layer[li].append(np.zeros(D_MODEL, dtype=np.float32))
            per_record_k.append(0)
            continue

        # Encode prompt once
        prompt_ids = tokenizer.encode(row["prompt"], return_tensors="pt").to(device)
        # prompt_ids: [1, T_prompt]

        # Accumulate per-token per-layer MLP outputs
        step_acts: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}

        def make_hook(li: int):
            def hook(mod, inp, out):
                # Capture last position (the forced token we just added)
                step_acts[li].append(out[0, -1, :].float().cpu())
            return hook

        handles = [
            model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
            for li in ALL_LAYERS
        ]
        try:
            for t in range(len(it_gen_ids)):
                # Build input: [prompt, it_gen_ids[0..t]]
                forced = torch.tensor([it_gen_ids[:t + 1]], dtype=torch.long, device=device)
                input_ids = torch.cat([prompt_ids, forced], dim=1)
                with torch.no_grad():
                    model_raw(input_ids)
        finally:
            for h in handles:
                h.remove()

        actual_k = len(it_gen_ids)
        for li in ALL_LAYERS:
            if step_acts[li]:
                stacked = torch.stack(step_acts[li])  # [k, D_MODEL]
                acts_by_layer[li].append(stacked.mean(dim=0).numpy().astype(np.float32))
            else:
                acts_by_layer[li].append(np.zeros(D_MODEL, dtype=np.float32))
        per_record_k.append(actual_k)

        if verbose and (idx + 1) % 20 == 0:
            print(f"  PT-forced {idx+1}/{n} records (k={actual_k})", flush=True)

    return acts_by_layer, np.array(per_record_k, dtype=np.int32)


def run_worker(
    worker_index: int,
    n_workers: int,
    device: str,
    output_dir: Path,
    n_records: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"w{worker_index}.npz"

    if out_path.exists():
        print(f"[force_decode w{worker_index}] already exists, skipping.", flush=True)
        return

    selected_ids: set[str] = set(json.loads((WORK_DIR / "selected.json").read_text()))
    rows_by_id: dict[str, dict] = {}
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in selected_ids:
                rows_by_id[r["record_id"]] = r

    my_ids = sorted(rows_by_id.keys())[worker_index::n_workers]
    if n_records is not None:
        my_ids = my_ids[:n_records]
    my_rows = [rows_by_id[rid] for rid in my_ids]

    print(
        f"[force_decode w{worker_index}/{n_workers}] {len(my_rows)} records on {device} "
        f"(PT model forced on IT tokens)",
        flush=True,
    )

    pt_cfg = Exp5Config(
        experiment="baseline", model_variant="pt", model_id="",
        run_name=f"exp7_force_decode_pt_w{worker_index}", device=device, skip_transcoders=True,
    )
    pt_loaded = load_model(pt_cfg)
    acts_by_layer, per_record_k = _force_decode_acts(
        _get_raw(pt_loaded), pt_loaded.tokenizer, my_rows, device
    )
    del pt_loaded
    torch.cuda.empty_cache()

    # Save under "it_acts_{L}" keys for API compatibility with bootstrap_directions.py --matched-mode
    payload: dict[str, np.ndarray] = {
        "record_ids": np.array(my_ids, dtype=object),
        "per_record_k_it": per_record_k,   # "it" here = tokens from IT used for forcing
        "per_record_k_pt": per_record_k,   # same k for both (forced on IT tokens)
    }
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.stack(acts_by_layer[li])   # PT-on-IT acts
        payload[f"pt_acts_{li}"] = np.stack(acts_by_layer[li])   # same, unused by caller

    np.savez_compressed(str(out_path), **payload)
    print(f"[force_decode w{worker_index}] saved → {out_path}", flush=True)


def merge_workers(n_workers: int, output_dir: Path) -> None:
    merged_path = output_dir / "merged.npz"
    if merged_path.exists():
        print(f"[merge] {merged_path} already exists, skipping.", flush=True)
        return

    all_ids: list[str] = []
    all_k: list[np.ndarray] = []
    acts: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}

    for w in range(n_workers):
        wp = output_dir / f"w{w}.npz"
        if not wp.exists():
            print(f"[merge] WARNING: missing {wp}", flush=True)
            continue
        with np.load(wp, allow_pickle=True) as d:
            all_ids.extend(d["record_ids"].tolist())
            all_k.append(d["per_record_k_it"])
            for li in ALL_LAYERS:
                acts[li].append(d[f"it_acts_{li}"])

    payload: dict[str, np.ndarray] = {
        "record_ids": np.array(all_ids, dtype=object),
        "per_record_k_it": np.concatenate(all_k),
        "per_record_k_pt": np.concatenate(all_k),
    }
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.concatenate(acts[li], axis=0)
        payload[f"pt_acts_{li}"] = np.concatenate(acts[li], axis=0)

    np.savez_compressed(str(merged_path), **payload)
    print(f"[merge] {len(all_ids)} records → {merged_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Force-decode PT on IT tokens (Exp7 0B)")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--n-records", type=int, default=None)
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    if args.merge_only:
        merge_workers(args.n_workers, output_dir)
    else:
        run_worker(args.worker_index, args.n_workers, args.device, output_dir, args.n_records)


if __name__ == "__main__":
    main()
