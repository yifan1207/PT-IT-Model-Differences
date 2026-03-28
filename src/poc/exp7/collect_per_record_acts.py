"""Collect per-record MLP activations for direction bootstrap (Exp7 0A).

Unlike precompute_directions_v2.py Phase 3 (which saves global sums), this
script saves per-record mean activation vectors so individual records can be
bootstrapped independently.

Each worker processes its slice of the 600 selected records and saves:
  results/exp7/0A/acts/w{N}.npz
    "record_ids"       : str array [n_records_this_worker]
    "it_acts_{L}"      : float32 [n_records, D_MODEL] for L in 1..33
    "pt_acts_{L}"      : float32 [n_records, D_MODEL] for L in 1..33
    "per_record_k"     : int32   [n_records]  (tokens used per record)

After all workers finish, --merge-only concatenates them into:
  results/exp7/0A/acts/merged.npz  (same structure, all 600 records)

Usage:
  # 8 parallel workers
  for i in {0..7}; do
      uv run python -m src.poc.exp7.collect_per_record_acts \\
          --worker-index $i --n-workers 8 --device cuda:$i \\
          --output-dir results/exp7/0A/acts/ &
  done; wait

  # Merge
  uv run python -m src.poc.exp7.collect_per_record_acts \\
      --merge-only --n-workers 8 --output-dir results/exp7/0A/acts/

  # Quick test (10 records)
  uv run python -m src.poc.exp7.collect_per_record_acts \\
      --worker-index 0 --n-workers 1 --device cuda:0 \\
      --n-records 10 --output-dir results/exp7/0A/acts/
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

ALL_LAYERS = list(range(1, 34))   # layers 1-33
D_MODEL = 2560
MAX_GEN = 80  # tokens per record (same as precompute_v2)

WORK_DIR = Path("results/precompute_v2_work")


def _get_raw(loaded):
    return loaded.model._model


def _collect_model_acts(
    model_raw,
    tokenizer,
    rows: list[dict],
    device: str,
    is_it: bool,
    *,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect per-record mean MLP activations for all layers.

    Returns:
        acts: float32 [n_records, D_MODEL] per layer stored as list
        per_record_k: int32 [n_records] tokens used per record
    """
    n = len(rows)
    # Pre-allocate: list of [n_records, D_MODEL] arrays, one per layer
    acts_by_layer: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}
    per_record_k = []

    for idx, row in enumerate(rows):
        k = min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), MAX_GEN)
        if k == 0:
            # Pad with zeros for records with no tokens
            for li in ALL_LAYERS:
                acts_by_layer[li].append(np.zeros(D_MODEL, dtype=np.float32))
            per_record_k.append(0)
            continue

        gen_acts: dict[int, list[torch.Tensor]] = {li: [] for li in ALL_LAYERS}

        def make_hook(li: int):
            def hook(mod, inp, out):
                if out.shape[1] == 1 and len(gen_acts[li]) < k:
                    gen_acts[li].append(out[0, 0, :].float().cpu())
            return hook

        handles = [
            model_raw.language_model.layers[li].mlp.register_forward_hook(make_hook(li))
            for li in ALL_LAYERS
        ]
        try:
            input_ids = tokenizer.encode(row["prompt"], return_tensors="pt").to(device)
            with torch.no_grad():
                model_raw.generate(
                    input_ids,
                    max_new_tokens=k,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    use_cache=True,
                )
        finally:
            for h in handles:
                h.remove()

        actual_k = 0
        for li in ALL_LAYERS:
            if gen_acts[li]:
                stacked = torch.stack(gen_acts[li])  # [T, D_MODEL]
                mean_act = stacked.mean(dim=0).numpy().astype(np.float32)
                acts_by_layer[li].append(mean_act)
                actual_k = max(actual_k, stacked.shape[0])
            else:
                acts_by_layer[li].append(np.zeros(D_MODEL, dtype=np.float32))
        per_record_k.append(actual_k)

        if verbose and (idx + 1) % 50 == 0:
            label = "IT" if is_it else "PT"
            print(f"  {label} {idx+1}/{n} records", flush=True)

    return acts_by_layer, np.array(per_record_k, dtype=np.int32)


def run_worker(worker_index: int, n_workers: int, device: str, output_dir: Path, n_records: int | None = None) -> None:
    """Collect per-record IT and PT activations for this worker's slice."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"w{worker_index}.npz"

    if out_path.exists():
        print(f"[collect w{worker_index}] already exists, skipping.", flush=True)
        return

    # Load selected IDs + gen data
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
        f"[collect w{worker_index}/{n_workers}] {len(my_rows)} records on {device}",
        flush=True,
    )

    # IT activations
    print(f"[collect w{worker_index}] loading IT model...", flush=True)
    it_cfg = Exp5Config(
        experiment="baseline", model_variant="it", model_id="",
        run_name=f"exp7_acts_it_w{worker_index}", device=device, skip_transcoders=True,
    )
    it_loaded = load_model(it_cfg)
    it_acts, it_k = _collect_model_acts(_get_raw(it_loaded), it_loaded.tokenizer, my_rows, device, is_it=True)
    del it_loaded
    torch.cuda.empty_cache()

    # PT activations
    print(f"[collect w{worker_index}] loading PT model...", flush=True)
    pt_cfg = Exp5Config(
        experiment="baseline", model_variant="pt", model_id="",
        run_name=f"exp7_acts_pt_w{worker_index}", device=device, skip_transcoders=True,
    )
    pt_loaded = load_model(pt_cfg)
    pt_acts, pt_k = _collect_model_acts(_get_raw(pt_loaded), pt_loaded.tokenizer, my_rows, device, is_it=False)
    del pt_loaded
    torch.cuda.empty_cache()

    # Save
    payload: dict[str, np.ndarray] = {
        "record_ids": np.array(my_ids, dtype=object),
        "per_record_k_it": it_k,
        "per_record_k_pt": pt_k,
    }
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.stack(it_acts[li])  # [n_records, D_MODEL]
        payload[f"pt_acts_{li}"] = np.stack(pt_acts[li])

    np.savez_compressed(str(out_path), **payload)
    print(
        f"[collect w{worker_index}] saved {len(my_ids)} records → {out_path}",
        flush=True,
    )


def merge_workers(n_workers: int, output_dir: Path) -> None:
    """Concatenate all worker NPZ files into merged.npz."""
    merged_path = output_dir / "merged.npz"
    if merged_path.exists():
        print(f"[merge] {merged_path} already exists, skipping.", flush=True)
        return

    all_record_ids: list[str] = []
    all_k_it: list[np.ndarray] = []
    all_k_pt: list[np.ndarray] = []
    acts_it: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}
    acts_pt: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}

    for w in range(n_workers):
        wp = output_dir / f"w{w}.npz"
        if not wp.exists():
            print(f"[merge] WARNING: missing {wp}", flush=True)
            continue
        with np.load(wp, allow_pickle=True) as d:
            ids = d["record_ids"].tolist()
            all_record_ids.extend(ids)
            all_k_it.append(d["per_record_k_it"])
            all_k_pt.append(d["per_record_k_pt"])
            for li in ALL_LAYERS:
                acts_it[li].append(d[f"it_acts_{li}"])
                acts_pt[li].append(d[f"pt_acts_{li}"])

    payload: dict[str, np.ndarray] = {
        "record_ids": np.array(all_record_ids, dtype=object),
        "per_record_k_it": np.concatenate(all_k_it),
        "per_record_k_pt": np.concatenate(all_k_pt),
    }
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.concatenate(acts_it[li], axis=0)
        payload[f"pt_acts_{li}"] = np.concatenate(acts_pt[li], axis=0)

    n_total = len(all_record_ids)
    np.savez_compressed(str(merged_path), **payload)
    print(f"[merge] {n_total} records → {merged_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Collect per-record MLP activations (Exp7 0A)")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default="results/exp7/0A/acts/")
    p.add_argument("--n-records", type=int, default=None, help="Cap records per worker (for testing)")
    p.add_argument("--merge-only", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)

    if args.merge_only:
        merge_workers(args.n_workers, output_dir)
    else:
        run_worker(args.worker_index, args.n_workers, args.device, output_dir, args.n_records)


if __name__ == "__main__":
    main()
