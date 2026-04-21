"""Unified MLP activation collector for all Exp7 experiments (0A + 0H combined).

Collects per-record per-layer MLP activations for ALL 1400 records in a single
8-GPU inference pass. Both IT and PT models are run. This replaces running
collect_per_record_acts.py (0A, 600 records) and precompute_random_split.py
(0H, random-600 + bottom-600) separately — saving ~1 hour of GPU time.

After this single collection, downstream experiments slice the relevant subsets:
  0A  uses the 600 governance-selected records (selected.json)
  0H  uses random-600 and bottom-600 subsets (created from split logic here)
      plus the held-out 800 for evaluation

Data collected per record:
  it_acts_{L}  [n_records_this_worker, D_MODEL] float32  — IT MLP output mean over generated steps
  pt_acts_{L}  [n_records_this_worker, D_MODEL] float32  — PT MLP output mean over generated steps

Output layout:
  results/exp7/unified_acts/acts/w{N}.npz     — per-worker activation files
  results/exp7/unified_acts/merged.npz        — merged 1400-record file
  results/exp7/0H/random_600_ids.json         — random-600 split (seed=42)
  results/exp7/0H/held_out_800_ids.json       — held-out 800
  results/exp7/0H/bottom_600_ids.json         — bottom-600 (lowest contrast)

Usage:
  # Step 1: 8 parallel workers (load IT on GPU, then PT on same GPU sequentially)
  for i in {0..7}; do
      uv run python -m src.poc.exp07_methodology_validation_tier0.collect_unified_acts \\
          --worker-index $i --n-workers 8 --device cuda:$i &
  done; wait

  # Step 2: merge all workers
  uv run python -m src.poc.exp07_methodology_validation_tier0.collect_unified_acts --merge-only --n-workers 8

  # Quick test (20 records)
  uv run python -m src.poc.exp07_methodology_validation_tier0.collect_unified_acts \\
      --worker-index 0 --n-workers 1 --device cuda:0 --n-records 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.poc.exp05_corrective_direction_ablation_cartography.config import Exp5Config
from src.poc.shared.model import load_model

ALL_LAYERS = list(range(1, 34))  # layers 1..33
D_MODEL    = 2560
MAX_GEN    = 80

WORK_DIR   = Path("results/precompute_v2_work")
OUTPUT_DIR = Path("results/exp7/unified_acts")
ACTS_DIR   = OUTPUT_DIR / "acts"

# Selected 600 records path (governance-ranked, used for 0A direction)
SELECTED_JSON = WORK_DIR / "selected.json"

# 0H split parameters
RANDOM_SEED = 42
N_SELECT    = 600


def _get_raw(loaded):
    return loaded.model._model


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


def _load_all_records() -> tuple[list[str], list[dict]]:
    """Load all 1400 records from gen_merged.jsonl."""
    all_records = []
    with open(WORK_DIR / "gen_merged.jsonl") as f:
        for line in f:
            if line.strip():
                all_records.append(json.loads(line))
    all_ids = [r["record_id"] for r in all_records]
    return all_ids, all_records


def _make_0H_split(all_ids: list[str], all_records: list[dict]) -> None:
    """Create random-600/held-out-800/bottom-600 split for 0H (saved to results/exp7/0H/)."""
    from pathlib import Path as P
    h_dir = P("results/exp7/0H")
    h_dir.mkdir(parents=True, exist_ok=True)

    random_ids_path = h_dir / "random_600_ids.json"
    held_out_path   = h_dir / "held_out_800_ids.json"
    bottom_ids_path = h_dir / "bottom_600_ids.json"

    if random_ids_path.exists() and held_out_path.exists() and bottom_ids_path.exists():
        print("[unified] 0H split already exists, skipping.", flush=True)
        return

    rng = np.random.default_rng(RANDOM_SEED)
    perm = rng.permutation(len(all_ids))
    random_600  = sorted([all_ids[i] for i in perm[:N_SELECT]])
    held_out_800 = sorted([all_ids[i] for i in perm[N_SELECT:]])

    # Bottom-600: lowest contrast score
    records_with_contrast = []
    for r in all_records:
        contrast = r.get("contrast_score")
        if contrast is None:
            contrast = (r.get("it_str", 0.0) - r.get("pt_str", 0.0)) if "it_str" in r else 0.0
        records_with_contrast.append((r["record_id"], contrast))
    records_with_contrast.sort(key=lambda x: x[1])
    bottom_600 = sorted([rid for rid, _ in records_with_contrast[:N_SELECT]])

    random_ids_path.write_text(json.dumps(random_600, indent=2))
    held_out_path.write_text(json.dumps(held_out_800, indent=2))
    bottom_ids_path.write_text(json.dumps(bottom_600, indent=2))
    print(
        f"[unified] 0H split: random-600={len(random_600)}, "
        f"held-out-800={len(held_out_800)}, bottom-600={len(bottom_600)}",
        flush=True,
    )


def _collect_per_record_means(
    model_raw,
    tokenizer,
    rows: list[dict],
    device: str,
    worker_tag: str,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Collect per-record mean MLP output activations over generated tokens.

    Returns:
      sums[layer]   shape [n_records, D_MODEL] float64  (actually per-record means stacked)
      counts[layer] shape [n_records] int64
    """
    n = len(rows)
    per_rec_means: dict[int, np.ndarray] = {li: np.zeros((n, D_MODEL), dtype=np.float64)
                                             for li in ALL_LAYERS}
    per_rec_valid: dict[int, np.ndarray] = {li: np.zeros(n, dtype=bool) for li in ALL_LAYERS}

    for idx, row in enumerate(rows):
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
                stacked = torch.stack(gen_acts[li])  # [k, D_MODEL]
                per_rec_means[li][idx] = stacked.mean(dim=0).numpy().astype(np.float64)
                per_rec_valid[li][idx] = True

        if (idx + 1) % 20 == 0:
            print(f"[unified {worker_tag}] {idx+1}/{n} records", flush=True)

    return per_rec_means, per_rec_valid


def run_worker(
    worker_index: int,
    n_workers: int,
    device: str,
    n_records: int | None = None,
) -> None:
    """Collect IT and PT per-record MLP activations for all 1400 records."""
    ACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ACTS_DIR / f"w{worker_index}.npz"

    if out_path.exists():
        print(f"[unified w{worker_index}] already exists, skipping.", flush=True)
        return

    all_ids, all_records = _load_all_records()

    # Create 0H split if not already done
    _make_0H_split(all_ids, all_records)

    rows_by_id = {r["record_id"]: r for r in all_records}

    # Assign this worker's records
    my_ids = sorted(rows_by_id.keys())[worker_index::n_workers]
    if n_records is not None:
        my_ids = my_ids[:n_records]
    my_rows = [rows_by_id[rid] for rid in my_ids]

    print(f"[unified w{worker_index}/{n_workers}] {len(my_rows)} records on {device}", flush=True)

    # IT activations
    it_cfg = Exp5Config(
        experiment="baseline", model_variant="it", model_id="",
        run_name=f"exp7_unified_it_w{worker_index}", device=device, skip_transcoders=True,
    )
    it_loaded = load_model(it_cfg)
    it_means, it_valid = _collect_per_record_means(
        _get_raw(it_loaded), it_loaded.tokenizer, my_rows, device, f"IT w{worker_index}"
    )
    del it_loaded
    torch.cuda.empty_cache()

    # PT activations
    pt_cfg = Exp5Config(
        experiment="baseline", model_variant="pt", model_id="",
        run_name=f"exp7_unified_pt_w{worker_index}", device=device, skip_transcoders=True,
    )
    pt_loaded = load_model(pt_cfg)
    pt_means, pt_valid = _collect_per_record_means(
        _get_raw(pt_loaded), pt_loaded.tokenizer, my_rows, device, f"PT w{worker_index}"
    )
    del pt_loaded
    torch.cuda.empty_cache()

    # Save
    payload: dict = {"record_ids": np.array(my_ids, dtype=object)}
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"]  = it_means[li].astype(np.float32)
        payload[f"pt_acts_{li}"]  = pt_means[li].astype(np.float32)
        payload[f"it_valid_{li}"] = it_valid[li]
        payload[f"pt_valid_{li}"] = pt_valid[li]

    np.savez_compressed(str(out_path), **payload)
    print(f"[unified w{worker_index}] saved → {out_path}", flush=True)


def merge_workers(n_workers: int) -> None:
    """Merge per-worker NPZ files into a single merged.npz with all 1400 records."""
    merged_path = OUTPUT_DIR / "merged.npz"
    if merged_path.exists():
        print("[unified] merged.npz already exists, skipping.", flush=True)
        return

    all_ids: list[str] = []
    layer_it: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}
    layer_pt: dict[int, list[np.ndarray]] = {li: [] for li in ALL_LAYERS}

    for w in range(n_workers):
        wp = ACTS_DIR / f"w{w}.npz"
        if not wp.exists():
            print(f"[unified] WARNING: {wp} not found, skipping worker {w}", flush=True)
            continue
        with np.load(wp, allow_pickle=True) as d:
            ids = list(d["record_ids"])
            all_ids.extend(ids)
            for li in ALL_LAYERS:
                layer_it[li].append(d[f"it_acts_{li}"])
                layer_pt[li].append(d[f"pt_acts_{li}"])

    print(f"[unified] Merging {len(all_ids)} records from {n_workers} workers...", flush=True)

    payload: dict = {"record_ids": np.array(all_ids, dtype=object)}
    for li in ALL_LAYERS:
        payload[f"it_acts_{li}"] = np.concatenate(layer_it[li], axis=0)
        payload[f"pt_acts_{li}"] = np.concatenate(layer_pt[li], axis=0)

    np.savez_compressed(str(merged_path), **payload)
    print(f"[unified] merged.npz ({len(all_ids)} records) → {merged_path}", flush=True)

    # Write index file: record_id → row index
    idx_path = OUTPUT_DIR / "record_index.json"
    idx = {rid: i for i, rid in enumerate(all_ids)}
    idx_path.write_text(json.dumps(idx, indent=2))
    print(f"[unified] record index → {idx_path}", flush=True)


def extract_subset_acts(subset_ids: list[str], merged_path: Path | None = None) -> dict:
    """Load activations for a specific subset of record IDs from merged.npz.

    Returns: {"record_ids": [...], "it_acts_{L}": array[n, D_MODEL], ...}
    Useful for 0A (600 selected) and 0H (random-600, bottom-600) to slice from merged.
    """
    if merged_path is None:
        merged_path = OUTPUT_DIR / "merged.npz"
    idx_path = OUTPUT_DIR / "record_index.json"

    if not merged_path.exists():
        raise FileNotFoundError(f"merged.npz not found at {merged_path}. Run --merge-only first.")

    idx = json.loads(idx_path.read_text())
    row_indices = [idx[rid] for rid in subset_ids if rid in idx]
    missing = [rid for rid in subset_ids if rid not in idx]
    if missing:
        print(f"[unified] WARNING: {len(missing)} IDs not found in merged.npz", flush=True)

    with np.load(str(merged_path), allow_pickle=True) as d:
        result: dict = {"record_ids": np.array(subset_ids)}
        for li in ALL_LAYERS:
            result[f"it_acts_{li}"] = d[f"it_acts_{li}"][row_indices]
            result[f"pt_acts_{li}"] = d[f"pt_acts_{li}"][row_indices]

    return result


def verify_against_canonical() -> None:
    """Verify that directions derived from unified acts match the canonical precompute_v2 direction.

    The unified collection stores per-record means; precompute_v2 stored aggregate sums.
    They should produce the same direction vector (up to numerical precision).
    Computes: direction from selected-600 via unified acts → cosine vs canonical.
    """
    canonical_path = Path("results/exp5/precompute_v2/precompute/corrective_directions.npz")
    selected_path = SELECTED_JSON

    if not canonical_path.exists():
        print("[verify] canonical corrective_directions.npz not found, skipping.", flush=True)
        return
    if not selected_path.exists():
        print("[verify] selected.json not found, skipping.", flush=True)
        return

    merged_path = OUTPUT_DIR / "merged.npz"
    if not merged_path.exists():
        print("[verify] merged.npz not found, skipping.", flush=True)
        return

    selected_ids = json.loads(selected_path.read_text())
    acts = extract_subset_acts(selected_ids)
    canonical = dict(np.load(str(canonical_path)))

    print("[verify] Comparing unified-derived direction vs canonical (selected-600):", flush=True)
    for li in ALL_LAYERS:
        key = f"layer_{li}"
        if key not in canonical:
            continue
        it_mean = acts[f"it_acts_{li}"].mean(axis=0).astype(np.float64)
        pt_mean = acts[f"pt_acts_{li}"].mean(axis=0).astype(np.float64)
        unified_dir = it_mean - pt_mean
        unified_dir = unified_dir / (np.linalg.norm(unified_dir) + 1e-12)

        canon_dir = canonical[key].astype(np.float64)
        canon_dir = canon_dir / (np.linalg.norm(canon_dir) + 1e-12)

        cosine = float(np.dot(unified_dir, canon_dir))
        status = "OK" if cosine > 0.99 else "WARN" if cosine > 0.90 else "FAIL"
        if li >= 20 or status != "OK":  # only print corrective layers or issues
            print(f"  layer {li:2d}: cosine = {cosine:.6f}  [{status}]", flush=True)

    print(
        "[verify] Note: small deviations are expected because unified collection stores "
        "per-record means (mean of k-step means) while precompute_v2 stored raw sums "
        "(sum of all individual step activations). The direction should still be > 0.99.",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unified per-record MLP activation collector (Exp7 0A+0H combined)"
    )
    p.add_argument("--worker-index", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--n-records", type=int, default=None,
                   help="Cap number of records per worker (for quick tests)")
    p.add_argument("--merge-only", action="store_true",
                   help="Merge existing worker NPZ files into merged.npz (no inference)")
    p.add_argument("--verify", action="store_true",
                   help="Verify unified acts match canonical precompute_v2 direction")
    args = p.parse_args()

    if args.verify:
        verify_against_canonical()
    elif args.merge_only:
        merge_workers(args.n_workers)
    else:
        run_worker(args.worker_index, args.n_workers, args.device, args.n_records)


if __name__ == "__main__":
    main()
