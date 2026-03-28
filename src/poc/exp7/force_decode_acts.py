"""Force-decode for matched-token direction validation (Exp7 0B).

The concern: IT and PT generate different token sequences. MLP activation differences
may partly reflect different input histories (KV-cache effects) rather than weight
differences. Solution: force one model to process the other's exact tokens.

Two directions supported:
  --direction forward  (default): PT forced on IT tokens
  --direction reverse:            IT forced on PT tokens

Two record selection modes:
  --record-set governance  (default): governance-selected 600 from selected.json
  --record-set random:                random 600 from 0H split

Teacher-forcing method:
  For each token position t = 0..k-1:
    input_ids = [prompt_ids, gen_ids[0], ..., gen_ids[t]]
    Forward pass: capture MLP output at position -1 (last = forced token)
  Mean over positions: per-record per-layer activation

Output: results/exp7/0B/{direction}_{record_set}/acts/w{N}.npz

Usage:
  # Forward (PT on IT tokens, governance-selected 600)
  for i in {0..7}; do
      uv run python -m src.poc.exp7.force_decode_acts \\
          --worker-index $i --n-workers 8 --device cuda:$i \\
          --direction forward --record-set governance &
  done; wait
  uv run python -m src.poc.exp7.force_decode_acts --merge-only --n-workers 8 \\
      --direction forward --record-set governance

  # Reverse (IT on PT tokens, governance-selected 600)
  for i in {0..7}; do
      uv run python -m src.poc.exp7.force_decode_acts \\
          --worker-index $i --n-workers 8 --device cuda:$i \\
          --direction reverse --record-set governance &
  done; wait
  uv run python -m src.poc.exp7.force_decode_acts --merge-only --n-workers 8 \\
      --direction reverse --record-set governance
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
BASE_OUTPUT_DIR = Path("results/exp7/0B")


def _get_raw(loaded):
    return loaded.model._model


def _output_dir(direction: str, record_set: str) -> Path:
    return BASE_OUTPUT_DIR / f"{direction}_{record_set}" / "acts"


def _force_decode_acts(
    model_raw,
    tokenizer,
    rows: list[dict],
    device: str,
    force_token_key: str,
    *,
    verbose: bool = True,
) -> tuple[dict[int, list[np.ndarray]], np.ndarray]:
    """Teacher-force a model on specified token IDs. Return per-record mean per-layer activations.

    Args:
        force_token_key: which gen_ids to force-feed the model.
            "it_gen_ids" for forward (PT forced on IT tokens)
            "pt_gen_ids" for reverse (IT forced on PT tokens)
    """
    n = len(rows)
    acts_by_layer: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}
    per_record_k = []

    for idx, row in enumerate(rows):
        k = min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), MAX_GEN)
        gen_ids = row[force_token_key][:k]

        if k == 0 or not gen_ids:
            for li in ALL_LAYERS:
                acts_by_layer[li].append(np.zeros(D_MODEL, dtype=np.float32))
            per_record_k.append(0)
            continue

        prompt_ids = tokenizer.encode(row["prompt"], return_tensors="pt").to(device)

        step_acts: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}
        collecting = False  # only collect during forced-decode steps, not prefill

        def make_hook(li: int):
            def hook(mod, inp, out):
                if collecting:
                    step_acts[li].append(out[0, -1, :].float().cpu())
            return hook

        handles = [
            model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
            for li in ALL_LAYERS
        ]
        try:
            # Use KV cache to avoid quadratic recomputation:
            # First pass: process prompt and cache key/values (hooks fire but collecting=False)
            with torch.no_grad():
                out = model_raw(prompt_ids, use_cache=True)
                past_kv = out.past_key_values

            # Then step through forced tokens one at a time (O(K*(P+K)) not O(K*(P+K)^2))
            collecting = True
            for t in range(len(gen_ids)):
                tok = torch.tensor([[gen_ids[t]]], dtype=torch.long, device=device)
                with torch.no_grad():
                    out = model_raw(tok, past_key_values=past_kv, use_cache=True)
                    past_kv = out.past_key_values
        finally:
            for h in handles:
                h.remove()

        actual_k = len(gen_ids)
        for li in ALL_LAYERS:
            if step_acts[li]:
                stacked = torch.stack(step_acts[li])
                acts_by_layer[li].append(stacked.mean(dim=0).numpy().astype(np.float32))
            else:
                acts_by_layer[li].append(np.zeros(D_MODEL, dtype=np.float32))
        per_record_k.append(actual_k)

        if verbose and (idx + 1) % 20 == 0:
            print(f"  forced {idx+1}/{n} records (k={actual_k})", flush=True)

    return acts_by_layer, np.array(per_record_k, dtype=np.int32)


def _load_record_ids(record_set: str) -> set[str]:
    """Load record IDs for the specified set."""
    if record_set == "governance":
        return set(json.loads((WORK_DIR / "selected.json").read_text()))
    elif record_set == "random":
        random_path = Path("results/exp7/0H/random_600_ids.json")
        if not random_path.exists():
            raise FileNotFoundError(
                f"random_600_ids.json not found. Run 0H split first."
            )
        return set(json.loads(random_path.read_text()))
    else:
        raise ValueError(f"Unknown record_set: {record_set}")


def run_worker(
    worker_index: int,
    n_workers: int,
    device: str,
    direction: str,
    record_set: str,
    n_records: int | None = None,
) -> None:
    output_dir = _output_dir(direction, record_set)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"w{worker_index}.npz"

    if out_path.exists():
        print(f"[0B w{worker_index}] already exists, skipping.", flush=True)
        return

    target_ids = _load_record_ids(record_set)
    rows_by_id: dict[str, dict] = {}
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in target_ids:
                rows_by_id[r["record_id"]] = r

    my_ids = sorted(rows_by_id.keys())[worker_index::n_workers]
    if n_records is not None:
        my_ids = my_ids[:n_records]
    my_rows = [rows_by_id[rid] for rid in my_ids]

    # Determine model variant and token key based on direction
    if direction == "forward":
        # PT model forced on IT tokens
        model_variant = "pt"
        force_token_key = "it_gen_ids"
        desc = "PT forced on IT tokens"
    elif direction == "reverse":
        # IT model forced on PT tokens
        model_variant = "it"
        force_token_key = "pt_gen_ids"
        desc = "IT forced on PT tokens"
    else:
        raise ValueError(f"Unknown direction: {direction}")

    print(
        f"[0B w{worker_index}/{n_workers}] {len(my_rows)} records on {device} "
        f"({desc}, {record_set} set)",
        flush=True,
    )

    cfg = Exp5Config(
        experiment="baseline", model_variant=model_variant, model_id="",
        run_name=f"exp7_0B_{direction}_{record_set}_w{worker_index}",
        device=device, skip_transcoders=True,
    )
    loaded = load_model(cfg)
    acts_by_layer, per_record_k = _force_decode_acts(
        _get_raw(loaded), loaded.tokenizer, my_rows, device, force_token_key
    )
    del loaded
    torch.cuda.empty_cache()

    # Save under "it_acts_{L}" keys for API compatibility with bootstrap_directions.py
    payload: dict[str, np.ndarray] = {
        "record_ids": np.array(my_ids, dtype=object),
        "per_record_k_it": per_record_k,
        "per_record_k_pt": per_record_k,
        "direction": np.array(direction),
        "record_set": np.array(record_set),
    }
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.stack(acts_by_layer[li])
        payload[f"pt_acts_{li}"] = np.stack(acts_by_layer[li])

    np.savez_compressed(str(out_path), **payload)
    print(f"[0B w{worker_index}] saved -> {out_path}", flush=True)


def merge_workers(n_workers: int, direction: str, record_set: str) -> None:
    output_dir = _output_dir(direction, record_set)
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
    print(f"[merge] {len(all_ids)} records -> {merged_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Force-decode for matched-token validation (Exp7 0B)")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=None,
                   help="Override output dir (default: auto from direction+record_set)")
    p.add_argument("--n-records", type=int, default=None)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--direction", choices=["forward", "reverse"], default="forward",
                   help="forward=PT on IT tokens, reverse=IT on PT tokens")
    p.add_argument("--record-set", choices=["governance", "random"], default="governance",
                   help="governance=selected 600, random=random 600 from 0H split")
    args = p.parse_args()

    if args.merge_only:
        merge_workers(args.n_workers, args.direction, args.record_set)
    else:
        run_worker(
            args.worker_index, args.n_workers, args.device,
            args.direction, args.record_set, args.n_records
        )


if __name__ == "__main__":
    main()
