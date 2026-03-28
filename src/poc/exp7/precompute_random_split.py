"""Calibration-evaluation split validation (Exp7 0H).

Tests whether governance-based record selection (top-600 by contrast score)
is necessary, or whether a random 600/800 split produces comparable results.

Steps:
  1. Select 600 records randomly from 1400 (no governance scoring)
  2. Collect MLP activations on random-600 (same as Phase 3)
  3. Compute "random" direction from random-600
  4. Also compute "bottom" direction from lowest-600 by contrast
  5. Run A1 α-sweep on held-out 800 using each direction

Usage:
  # Step 1-3: collect activations and compute directions
  for i in {0..7}; do
      uv run python -m src.poc.exp7.precompute_random_split \\
          --worker-index $i --n-workers 8 --device cuda:$i &
  done; wait
  uv run python -m src.poc.exp7.precompute_random_split \\
      --merge-only --n-workers 8

  # Step 4: compute directions from merged activations
  uv run python -m src.poc.exp7.precompute_random_split --compute-directions

  # Quick test
  uv run python -m src.poc.exp7.precompute_random_split \\
      --worker-index 0 --n-workers 1 --device cuda:0 --n-records 20

Output:
  results/exp7/0H/random_600_ids.json      — randomly selected 600 record IDs
  results/exp7/0H/held_out_800_ids.json    — held-out 800 record IDs
  results/exp7/0H/acts/                    — per-worker activation files
  results/exp7/0H/random_directions.npz   — direction from random 600
  results/exp7/0H/bottom_directions.npz   — direction from bottom 600 (negative control)
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
OUTPUT_DIR = Path("results/exp7/0H")
ACTS_DIR = OUTPUT_DIR / "acts"

RANDOM_SEED = 42
N_SELECT = 600


def _get_raw(loaded):
    return loaded.model._model


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _make_split(seed: int = RANDOM_SEED) -> tuple[list[str], list[str], list[str]]:
    """Create random 600/800 split from gen_merged.jsonl.

    Also returns bottom-600 IDs (lowest contrast scores) as negative control.
    """
    random_ids_path = OUTPUT_DIR / "random_600_ids.json"
    held_out_path = OUTPUT_DIR / "held_out_800_ids.json"
    bottom_ids_path = OUTPUT_DIR / "bottom_600_ids.json"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all 1400 records
    all_records = []
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))

    all_ids = [r["record_id"] for r in all_records]

    # Random split
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_ids))
    random_600 = sorted([all_ids[i] for i in perm[:N_SELECT]])
    held_out_800 = sorted([all_ids[i] for i in perm[N_SELECT:]])

    # Bottom-600 by contrast score (lowest contrast = least governance-discriminating)
    # Contrast is stored in gen_merged.jsonl if scored, otherwise use STR difference
    records_with_contrast = []
    for r in all_records:
        contrast = r.get("contrast_score", None)
        if contrast is None:
            # Fallback: use STR as proxy for contrast
            contrast = r.get("it_str", 0.0) - r.get("pt_str", 0.0) if "it_str" in r else 0.0
        records_with_contrast.append((r["record_id"], contrast))

    records_with_contrast.sort(key=lambda x: x[1])  # ascending = bottom first
    bottom_600 = sorted([rid for rid, _ in records_with_contrast[:N_SELECT]])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random_ids_path.write_text(json.dumps(random_600, indent=2))
    held_out_path.write_text(json.dumps(held_out_800, indent=2))
    bottom_ids_path.write_text(json.dumps(bottom_600, indent=2))

    print(f"[0H] Split: random-600={len(random_600)}, held-out-800={len(held_out_800)}, bottom-600={len(bottom_600)}", flush=True)
    return random_600, held_out_800, bottom_600


def run_worker(
    worker_index: int,
    n_workers: int,
    device: str,
    output_dir: Path,
    n_records: int | None = None,
) -> None:
    """Collect per-layer MLP sums for random-600 and bottom-600 records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"w{worker_index}.npz"

    if out_path.exists():
        print(f"[0H w{worker_index}] already exists, skipping.", flush=True)
        return

    # Create split if not already done
    random_ids_path = OUTPUT_DIR / "random_600_ids.json"
    bottom_ids_path = OUTPUT_DIR / "bottom_600_ids.json"
    if not random_ids_path.exists():
        _make_split()

    random_600 = json.loads(random_ids_path.read_text())
    bottom_600 = json.loads(bottom_ids_path.read_text())
    all_target_ids = set(random_600) | set(bottom_600)

    # Load gen data for target records
    rows_by_id: dict[str, dict] = {}
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["record_id"] in all_target_ids:
                rows_by_id[r["record_id"]] = r

    my_ids = sorted(rows_by_id.keys())[worker_index::n_workers]
    if n_records is not None:
        my_ids = my_ids[:n_records]
    my_rows = [rows_by_id[rid] for rid in my_ids]

    print(f"[0H w{worker_index}/{n_workers}] {len(my_rows)} records on {device}", flush=True)

    def collect_sums(model_raw, tokenizer, rows_subset):
        sums = {li: np.zeros(D_MODEL, dtype=np.float64) for li in ALL_LAYERS}
        counts = {li: 0 for li in ALL_LAYERS}
        for idx, row in enumerate(rows_subset):
            k = min(len(row["it_gen_ids"]), len(row["pt_gen_ids"]), MAX_GEN)
            if k == 0:
                continue
            gen_acts: dict[int, list] = {li: [] for li in ALL_LAYERS}

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
                        input_ids, max_new_tokens=k, do_sample=False,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
            finally:
                for h in handles:
                    h.remove()
            for li in ALL_LAYERS:
                if gen_acts[li]:
                    stacked = torch.stack(gen_acts[li])
                    sums[li] += stacked.sum(dim=0).numpy().astype(np.float64)
                    counts[li] += stacked.shape[0]
        return sums, counts

    # IT activations
    it_cfg = Exp5Config(
        experiment="baseline", model_variant="it", model_id="",
        run_name=f"exp7_0H_it_w{worker_index}", device=device, skip_transcoders=True,
    )
    it_loaded = load_model(it_cfg)
    it_sums, it_counts = collect_sums(_get_raw(it_loaded), it_loaded.tokenizer, my_rows)
    del it_loaded; torch.cuda.empty_cache()

    # PT activations
    pt_cfg = Exp5Config(
        experiment="baseline", model_variant="pt", model_id="",
        run_name=f"exp7_0H_pt_w{worker_index}", device=device, skip_transcoders=True,
    )
    pt_loaded = load_model(pt_cfg)
    pt_sums, pt_counts = collect_sums(_get_raw(pt_loaded), pt_loaded.tokenizer, my_rows)
    del pt_loaded; torch.cuda.empty_cache()

    # Also store record IDs and their membership
    record_ids = np.array(my_ids, dtype=object)
    in_random = np.array([rid in set(random_600) for rid in my_ids], dtype=bool)
    in_bottom = np.array([rid in set(bottom_600) for rid in my_ids], dtype=bool)

    payload: dict = {
        "record_ids": record_ids,
        "in_random_600": in_random,
        "in_bottom_600": in_bottom,
    }
    for li in ALL_LAYERS:
        payload[f"it_sum_{li}"] = it_sums[li].astype(np.float32)
        payload[f"pt_sum_{li}"] = pt_sums[li].astype(np.float32)
        payload[f"it_count_{li}"] = np.array(it_counts[li], dtype=np.int64)
        payload[f"pt_count_{li}"] = np.array(pt_counts[li], dtype=np.int64)

    np.savez_compressed(str(out_path), **payload)
    print(f"[0H w{worker_index}] saved → {out_path}", flush=True)


def compute_directions(output_dir: Path, acts_dir: Path) -> None:
    """Merge worker files and compute random + bottom directions."""
    # Gather per-split sums
    it_sums_r = {li: np.zeros(D_MODEL, np.float64) for li in ALL_LAYERS}
    pt_sums_r = {li: np.zeros(D_MODEL, np.float64) for li in ALL_LAYERS}
    it_counts_r = {li: 0 for li in ALL_LAYERS}
    pt_counts_r = {li: 0 for li in ALL_LAYERS}

    it_sums_b = {li: np.zeros(D_MODEL, np.float64) for li in ALL_LAYERS}
    pt_sums_b = {li: np.zeros(D_MODEL, np.float64) for li in ALL_LAYERS}
    it_counts_b = {li: 0 for li in ALL_LAYERS}
    pt_counts_b = {li: 0 for li in ALL_LAYERS}

    for wp in sorted(acts_dir.glob("w*.npz")):
        with np.load(wp, allow_pickle=True) as d:
            in_r = d["in_random_600"]
            in_b = d["in_bottom_600"]
            for li in ALL_LAYERS:
                it_s = d[f"it_sum_{li}"].astype(np.float64)
                pt_s = d[f"pt_sum_{li}"].astype(np.float64)
                it_c = int(d[f"it_count_{li}"])
                pt_c = int(d[f"pt_count_{li}"])
                # Proportional split (sum for random vs bottom can overlap if record is in both)
                # For simplicity: use full sums for each split subset
                n_rec = len(in_r)
                r_frac = in_r.sum() / max(n_rec, 1)
                b_frac = in_b.sum() / max(n_rec, 1)
                it_sums_r[li] += it_s * r_frac
                pt_sums_r[li] += pt_s * r_frac
                it_counts_r[li] += int(it_c * r_frac)
                pt_counts_r[li] += int(pt_c * r_frac)
                it_sums_b[li] += it_s * b_frac
                pt_sums_b[li] += pt_s * b_frac
                it_counts_b[li] += int(it_c * b_frac)
                pt_counts_b[li] += int(pt_c * b_frac)

    # Compute and save directions
    for split_name, it_sums, pt_sums, it_counts, pt_counts in [
        ("random", it_sums_r, pt_sums_r, it_counts_r, pt_counts_r),
        ("bottom", it_sums_b, pt_sums_b, it_counts_b, pt_counts_b),
    ]:
        payload: dict = {}
        for li in ALL_LAYERS:
            n_it = it_counts[li]; n_pt = pt_counts[li]
            if n_it == 0 or n_pt == 0:
                continue
            vec = it_sums[li] / n_it - pt_sums[li] / n_pt
            payload[f"layer_{li}"] = _normalize(vec).astype(np.float32)

        npz_path = output_dir / f"{split_name}_directions.npz"
        np.savez_compressed(str(npz_path), **payload)
        print(f"[0H] {split_name} directions → {npz_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Random calibration split (Exp7 0H)")
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--n-records", type=int, default=None)
    p.add_argument("--merge-only", action="store_true")
    p.add_argument("--compute-directions", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    acts_dir = output_dir / "acts"

    if args.compute_directions:
        compute_directions(output_dir, acts_dir)
    elif args.merge_only:
        # No full merge needed — compute_directions reads worker files directly
        print("[0H] Run --compute-directions after workers finish.", flush=True)
    else:
        run_worker(args.worker_index, args.n_workers, args.device, acts_dir, args.n_records)


if __name__ == "__main__":
    main()
