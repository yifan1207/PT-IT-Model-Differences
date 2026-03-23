#!/usr/bin/env python3
"""Merge worker run directories from a parallel exp5 run into one combined output.

Usage (called automatically by run_exp5_parallel.sh):
    uv run python scripts/merge_exp5_workers.py \
        --experiment phase --variant it --n-workers 8

    # When all workers wrote to the same dir (no --run-name suffix), pass it explicitly:
    uv run python scripts/merge_exp5_workers.py \
        --experiment phase --variant it --n-workers 8 \
        --source-dirs results/exp5/phase_it_none_t200

What it does:
    1. Reads scores.jsonl from each worker's run dir (or --source-dirs)
    2. Concatenates into merged_<experiment>_<variant>/scores.jsonl + scores.json
    3. Writes merged scores.csv
    4. Re-runs phase heatmap and dose-response plots on the merged data
    5. Loads per-condition hidden-state npz files, computes subspace metrics vs
       baseline, and writes checkpoint_metrics.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--n-workers", type=int, required=True)
    p.add_argument("--output-base", default="results/exp5")
    p.add_argument(
        "--baseline-scores", default="",
        help="Path to a scores.jsonl to pull the 'baseline' condition from and inject "
             "into the merge (for single_layer_phase which skips its own baseline run).",
    )
    p.add_argument(
        "--source-dirs", nargs="+", default=[],
        help="Explicit source dirs to merge scores/checkpoints from, instead of "
             "auto-discovering *_w{N} worker dirs. Use when all workers wrote to "
             "the same directory (i.e. --run-name was not passed to run.py).",
    )
    args = p.parse_args()

    base = Path(args.output_base)
    merged_dir = base / f"merged_{args.experiment}_{args.variant}"
    merged_dir.mkdir(parents=True, exist_ok=True)

    all_scores: list[dict] = []
    seen_conditions: list[str] = []
    # Collect all source dirs so we can find hidden-state npz files later.
    source_dirs: list[Path] = []

    if args.source_dirs:
        # Explicit source dirs — used when all workers shared one output directory.
        for sd in args.source_dirs:
            source_dir = Path(sd)
            source_dirs.append(source_dir)
            scores_path = source_dir / "scores.jsonl"
            rows = _read_jsonl(scores_path)
            if not rows:
                print(f"[merge] WARNING: {scores_path} is empty or missing")
                continue
            conditions = sorted({r["condition"] for r in rows})
            print(f"[merge] source {source_dir.name} — {len(rows)} score rows, conditions: {conditions}")
            seen_conditions.extend(conditions)
            all_scores.extend(rows)
    else:
        for w in range(args.n_workers):
            # Worker run dirs use the auto-generated name with _w{N} suffix.
            # Glob for any dir that ends with _w{w} to be robust to name variations.
            pattern = f"*_w{w}"
            candidates = sorted(base.glob(pattern))
            if not candidates:
                print(f"[merge] WARNING: no run dir found for worker {w} (pattern: {base}/{pattern})")
                continue
            worker_dir = candidates[0]
            source_dirs.append(worker_dir)
            scores_path = worker_dir / "scores.jsonl"
            rows = _read_jsonl(scores_path)
            if not rows:
                print(f"[merge] WARNING: {scores_path} is empty or missing")
                continue
            conditions = sorted({r["condition"] for r in rows})
            print(f"[merge] worker {w}: {worker_dir.name} — {len(rows)} score rows, conditions: {conditions}")
            seen_conditions.extend(conditions)
            all_scores.extend(rows)

    # Inject baseline scores from a separate run if requested.
    if args.baseline_scores:
        baseline_rows = [r for r in _read_jsonl(Path(args.baseline_scores)) if r.get("condition") == "baseline"]
        if baseline_rows:
            print(f"[merge] injecting {len(baseline_rows)} baseline rows from {args.baseline_scores}")
            all_scores.extend(baseline_rows)
        else:
            print(f"[merge] WARNING: no 'baseline' rows found in {args.baseline_scores}")

    if not all_scores:
        print("[merge] ERROR: no scores found across any worker. Nothing to merge.")
        return

    # Deduplicate on (condition, benchmark) in case of overlapping resume runs.
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for row in all_scores:
        key = (row.get("condition", ""), row.get("benchmark", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    if len(deduped) < len(all_scores):
        print(f"[merge] deduplicated {len(all_scores) - len(deduped)} duplicate rows")
    all_scores = deduped

    # Write merged outputs.
    _write_jsonl(merged_dir / "scores.jsonl", all_scores)
    with open(merged_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump(all_scores, f, indent=2)
    print(f"[merge] wrote {len(all_scores)} rows → {merged_dir}/scores.jsonl")

    # CSV
    try:
        from src.poc.exp5.analysis.summary import write_scores_csv
        write_scores_csv(merged_dir / "scores.csv", all_scores)
        print(f"[merge] wrote scores.csv")
    except Exception as e:
        print(f"[merge] WARNING: could not write CSV: {e}")

    # Plots
    plots_dir = merged_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    try:
        from src.poc.exp5.plots.heatmap import plot_phase_benchmark_heatmap
        plot_phase_benchmark_heatmap(all_scores, plots_dir / "phase_benchmark_heatmap.png")
        print(f"[merge] wrote phase_benchmark_heatmap.png")
    except Exception as e:
        print(f"[merge] WARNING: heatmap failed: {e}")

    # ── Subspace / checkpoint metrics plot ────────────────────────────────────
    # Collect per-condition hidden-state npz files from all source dirs.
    # Workers save <condition>_hidden_states.npz in their checkpoints/ subdir.
    # When multiple source dirs exist, prefer the first file found per condition
    # (workers don't overlap on conditions in round-robin mode).
    try:
        import numpy as np
        from src.poc.exp5.analysis.subspace import summarise_checkpoint_shift
        from src.poc.exp5.plots.subspace import plot_checkpoint_metrics

        # Build a map: condition_name -> npz path, scanning all source dirs.
        condition_npz: dict[str, Path] = {}
        for sd in source_dirs:
            ckpt_dir = sd / "checkpoints"
            if not ckpt_dir.exists():
                continue
            for npz_path in sorted(ckpt_dir.glob("*_hidden_states.npz")):
                # Strip the "_hidden_states" suffix to get condition name.
                cname = npz_path.stem.replace("_hidden_states", "")
                if cname not in condition_npz:
                    condition_npz[cname] = npz_path

        if condition_npz:
            print(f"[merge] found hidden-state npz for conditions: {sorted(condition_npz)}")

            # Load baseline (prefer "baseline" key, fall back to first available).
            baseline_key = "baseline" if "baseline" in condition_npz else next(iter(condition_npz))
            with np.load(condition_npz[baseline_key]) as d:
                baseline_checkpoints: dict[int, np.ndarray] = {
                    int(k.split("_", 1)[1]): d[k] for k in d.files
                }
            print(f"[merge] loaded baseline checkpoints from '{baseline_key}'")

            subspace_rows: dict[str, dict[int, dict[str, float]]] = {}
            for cname, npz_path in condition_npz.items():
                with np.load(npz_path) as d:
                    ablated: dict[int, np.ndarray] = {
                        int(k.split("_", 1)[1]): d[k] for k in d.files
                    }
                result = summarise_checkpoint_shift(baseline_checkpoints, ablated)
                if result:
                    subspace_rows[cname] = result

            if subspace_rows:
                plot_checkpoint_metrics(subspace_rows, plots_dir / "checkpoint_metrics.png")
                print(f"[merge] wrote checkpoint_metrics.png ({len(subspace_rows)} conditions)")
            else:
                print("[merge] WARNING: subspace computation produced no results")
        else:
            print("[merge] no hidden-state npz files found — skipping checkpoint_metrics plot")
    except Exception as e:
        print(f"[merge] WARNING: subspace/checkpoint plot failed: {e}")

    if args.experiment == "progressive":
        try:
            from src.poc.exp5.plots.dose_response import plot_dose_response
            progressive = [{**r, "x_value": len(r["layers"])} for r in all_scores if r.get("condition", "").startswith("skip_")]
            directional = [{**r, "x_value": r["alpha"]} for r in all_scores if r.get("condition", "").startswith("dir_alpha_")]
            if progressive:
                plot_dose_response(progressive, "x_value", plots_dir / "progressive_skip.png", "Exp5 — Progressive Skip")
            if directional:
                plot_dose_response(directional, "x_value", plots_dir / "directional_alpha.png", "Exp5 — Directional Alpha Sweep")
            print(f"[merge] wrote dose-response plots")
        except Exception as e:
            print(f"[merge] WARNING: dose-response plots failed: {e}")

    print(f"\n[merge] done — {merged_dir}")


if __name__ == "__main__":
    main()
