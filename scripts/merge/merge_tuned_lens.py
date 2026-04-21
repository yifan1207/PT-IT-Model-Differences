"""Merge per-worker tuned-lens eval outputs into single files.

After data-parallel eval on Modal, each worker produces:
  tuned_lens_commitment_{variant}_w{i}.jsonl
  arrays_w{i}/*.npy + step_index.jsonl

This script merges them into the canonical single-worker format:
  tuned_lens_commitment_{variant}.jsonl
  arrays/*.npy + step_index.jsonl
  summary_{variant}.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def merge_model_variant(base_dir: Path, variant: str, n_workers: int) -> None:
    """Merge worker outputs for one model+variant."""
    # ── Merge JSONL ──────────────────────────────────────────────────────
    merged_jsonl = base_dir / f"tuned_lens_commitment_{variant}.jsonl"
    all_lines = []
    for wi in range(n_workers):
        wf = base_dir / f"tuned_lens_commitment_{variant}_w{wi}.jsonl"
        if not wf.exists():
            print(f"  WARNING: {wf} missing")
            continue
        with open(wf) as f:
            all_lines.extend(f.readlines())
    with open(merged_jsonl, "w") as f:
        f.writelines(all_lines)
    print(f"  JSONL: {len(all_lines)} prompts → {merged_jsonl.name}")

    # ── Merge NPY arrays ────────────────────────────────────────────────
    merged_arrays_dir = base_dir / f"arrays_{variant}"
    merged_arrays_dir.mkdir(exist_ok=True)

    # Find all .npy files from worker 0
    w0_dir = base_dir / f"arrays_{variant}_w0"
    if not w0_dir.exists():
        print(f"  WARNING: {w0_dir} missing, skipping arrays")
        return
    npy_names = sorted(f.name for f in w0_dir.glob("*.npy"))

    for npy_name in npy_names:
        parts = []
        for wi in range(n_workers):
            p = base_dir / f"arrays_{variant}_w{wi}" / npy_name
            if p.exists():
                parts.append(np.load(p))
        if parts:
            merged = np.concatenate(parts, axis=0)
            np.save(merged_arrays_dir / npy_name, merged)

    print(f"  Arrays: {len(npy_names)} files merged ({sum(np.load(merged_arrays_dir / npy_names[0]).shape[0] for _ in [1])} total steps)")

    # ── Merge step_index.jsonl ───────────────────────────────────────────
    all_step_entries = []
    offset = 0
    for wi in range(n_workers):
        si = base_dir / f"arrays_{variant}_w{wi}" / "step_index.jsonl"
        if not si.exists():
            continue
        with open(si) as f:
            for line in f:
                entry = json.loads(line)
                entry["start_step"] += offset
                entry["end_step"] += offset
                all_step_entries.append(entry)
            # Update offset for next worker
            if all_step_entries:
                offset = all_step_entries[-1]["end_step"]

    with open(merged_arrays_dir / "step_index.jsonl", "w") as f:
        for entry in all_step_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  Step index: {len(all_step_entries)} prompts, {offset} total steps")

    # ── Merge top5_step_index.jsonl (if exists) ──────────────────────────
    all_top5 = []
    t5_offset = 0
    for wi in range(n_workers):
        si = base_dir / f"arrays_{variant}_w{wi}" / "top5_step_index.jsonl"
        if not si.exists():
            continue
        with open(si) as f:
            for line in f:
                entry = json.loads(line)
                entry["start_step"] += t5_offset
                entry["end_step"] += t5_offset
                all_top5.append(entry)
            if all_top5:
                t5_offset = all_top5[-1]["end_step"]
    if all_top5:
        with open(merged_arrays_dir / "top5_step_index.jsonl", "w") as f:
            for entry in all_top5:
                f.write(json.dumps(entry) + "\n")

    # ── Compute summary from merged JSONL ────────────────────────────────
    from src.poc.cross_model.tuned_lens import KL_THRESHOLDS, COSINE_THRESHOLDS

    all_records = [json.loads(l) for l in open(merged_jsonl)]

    def _flatten_median(key):
        flat = []
        for rec in all_records:
            if key in rec:
                flat.extend(rec[key])
        return float(np.median(flat)) if flat else float("nan")

    summary = {
        "model": all_records[0]["model"] if all_records else "?",
        "n_prompts": len(all_records),
        "n_layers": all_records[0].get("n_steps", 0) if all_records else 0,
        "median_commitment_raw": _flatten_median("commitment_layer_raw"),
        "median_commitment_top1_tuned": _flatten_median("commitment_layer_top1_tuned"),
    }
    for t in KL_THRESHOLDS:
        summary[f"median_commitment_raw_kl_{t}"] = _flatten_median(f"commitment_layer_raw_kl_{t}")
        summary[f"median_commitment_tuned_{t}"] = _flatten_median(f"commitment_layer_tuned_{t}")
        summary[f"median_commitment_majority_{t}"] = _flatten_median(f"commitment_layer_majority_{t}")
        summary[f"median_commitment_entropy_{t}"] = _flatten_median(f"commitment_layer_entropy_{t}")
    for t in COSINE_THRESHOLDS:
        summary[f"median_commitment_cosine_{t}"] = _flatten_median(f"commitment_layer_cosine_{t}")

    summary_path = base_dir / f"summary_{variant}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary → {summary_path.name} (median_raw={summary['median_commitment_raw']})")


def main():
    parser = argparse.ArgumentParser(description="Merge tuned-lens worker outputs")
    parser.add_argument("--models", required=True, help="Comma-separated model names")
    parser.add_argument("--n-workers", type=int, required=True)
    args = parser.parse_args()

    for model in args.models.split(","):
        model = model.strip()
        for variant in ["pt", "it"]:
            base = Path(f"results/cross_model/{model}/tuned_lens/commitment")
            print(f"\n=== {model}/{variant} ===")
            if not (base / f"tuned_lens_commitment_{variant}_w0.jsonl").exists():
                print(f"  No worker files found, skipping")
                continue
            merge_model_variant(base, variant, args.n_workers)


if __name__ == "__main__":
    main()
