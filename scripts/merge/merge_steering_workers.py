#!/usr/bin/env python3
"""Merge exp6 worker outputs into a single results directory.

Usage:
    python scripts/merge_steering_workers.py --experiment A1 --variant it --n-workers 8
    python scripts/merge_steering_workers.py --experiment B1 --variant it --n-workers 8
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--n-workers", type=int, required=True)
    p.add_argument("--output-base", default="results/exp6")
    p.add_argument("--merged-name", default="",
                   help="Override merged dir name (default: merged_{experiment}_{variant})")
    p.add_argument("--source-dirs", nargs="*", default=None,
                   help="Explicit source dirs (default: auto-detect from {experiment}_{variant}_w*)")
    args = p.parse_args()

    base = Path(args.output_base)
    merged_name = args.merged_name or f"merged_{args.experiment}_{args.variant}"
    merged_dir = base / merged_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Find source directories
    if args.source_dirs:
        source_dirs = [Path(d) for d in args.source_dirs]
    else:
        source_dirs = []
        for i in range(args.n_workers):
            d = base / f"{args.experiment}_{args.variant}_w{i}"
            if d.exists():
                source_dirs.append(d)
            else:
                print(f"WARNING: worker dir not found: {d}", flush=True)

    print(f"Merging from {len(source_dirs)} source dirs into {merged_dir}...", flush=True)

    # Collect all scores, deduplicating within each worker first.
    # Workers split by CONDITIONS (not records): each worker processes all N records
    # for its assigned conditions. Preemption + restart can cause duplicate entries
    # within a single worker's scores.jsonl — take the last entry per (condition,
    # benchmark) to use the most recent (post-restart) score.
    all_scores: list[dict] = []
    all_samples: list[dict] = []

    for src_dir in source_dirs:
        raw_scores = _read_jsonl(src_dir / "scores.jsonl")
        samples = _read_jsonl(src_dir / "sample_outputs.jsonl")

        # Within-worker dedup: last entry wins for each (condition, benchmark)
        seen: dict[tuple[str, str], dict] = {}
        n_dupes = 0
        for row in raw_scores:
            key = (row["condition"], row["benchmark"])
            if key in seen:
                n_dupes += 1
            seen[key] = row  # last entry wins
        deduped = list(seen.values())
        if n_dupes:
            print(f"  {src_dir.name}: {len(raw_scores)} raw → {len(deduped)} after dedup "
                  f"({n_dupes} preemption duplicates removed)", flush=True)
        else:
            print(f"  {src_dir.name}: {len(deduped)} score rows, {len(samples)} samples", flush=True)

        all_scores.extend(deduped)
        all_samples.extend(samples)

    # Merge across workers: since conditions are worker-exclusive, each
    # (condition, benchmark) should appear exactly once after within-worker dedup.
    # If duplicates remain (e.g., overlapping worker assignments), take the entry
    # with the larger n.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in all_scores:
        groups[(row["condition"], row["benchmark"])].append(row)

    deduped_scores: list[dict] = []
    for (cond, bench), rows in sorted(groups.items()):
        if len(rows) == 1:
            deduped_scores.append(rows[0])
        else:
            # Unexpected cross-worker duplicate — take entry with largest n
            best = max(rows, key=lambda r: r.get("n", 0))
            print(f"  WARN: cross-worker duplicate ({cond}, {bench}) — "
                  f"{len(rows)} entries, keeping n={best.get('n')}", flush=True)
            deduped_scores.append(best)

    print(f"Total: {sum(len(_read_jsonl(d / 'scores.jsonl')) for d in source_dirs)} raw rows → "
          f"{len(deduped_scores)} conditions after dedup+merge across "
          f"{len(source_dirs)} workers", flush=True)

    # Deduplicate sample_outputs: preemption restarts can re-generate the same
    # (condition, record_id) records. Keep the last entry for each.
    seen_samples: dict[tuple[str, str], dict] = {}
    for s in all_samples:
        rid = s.get("record_id", s.get("id", ""))
        key = (s.get("condition", ""), rid)
        seen_samples[key] = s  # last entry wins
    deduped_samples = list(seen_samples.values())
    n_sample_dupes = len(all_samples) - len(deduped_samples)
    if n_sample_dupes:
        print(f"Sample outputs: {len(all_samples)} raw → {len(deduped_samples)} after dedup "
              f"({n_sample_dupes} duplicates removed)", flush=True)

    # Write merged files
    _write_jsonl(merged_dir / "scores.jsonl", deduped_scores)
    _write_jsonl(merged_dir / "sample_outputs.jsonl", deduped_samples)

    with open(merged_dir / "scores.json", "w") as f:
        json.dump(deduped_scores, f, indent=2)

    # Write CSV — use union of all keys to handle mixed A/B experiment schemas
    if deduped_scores:
        import csv
        import io
        seen_keys: dict[str, None] = {}
        for row in deduped_scores:
            seen_keys.update(dict.fromkeys(row.keys()))
        keys = list(seen_keys.keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{k: row.get(k, "") for k in keys} for row in deduped_scores])
        (merged_dir / "scores.csv").write_text(buf.getvalue())

    # Summary by condition
    cond_bench: dict[str, dict[str, float]] = defaultdict(dict)
    for row in deduped_scores:
        cond_bench[row["condition"]][row["benchmark"]] = row["value"]

    print(f"\nCondition summary ({len(cond_bench)} conditions):")
    for cond in sorted(cond_bench.keys()):
        vals = cond_bench[cond]
        bench_str = ", ".join(f"{b}={v:.4f}" if v is not None else f"{b}=None" for b, v in sorted(vals.items()))
        print(f"  {cond}: {bench_str}")

    # Generate dose-response plot
    try:
        _plot_dose_response(args.experiment, deduped_scores, merged_dir / "plots")
    except Exception as e:
        print(f"WARNING: plotting failed: {e}", flush=True)

    print(f"\nMerged results → {merged_dir}")


def _plot_dose_response(experiment: str, scores: list[dict], plots_dir: Path) -> None:
    """Generate dose-response plots for A experiments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict

    plots_dir.mkdir(parents=True, exist_ok=True)

    GOV_BENCHMARKS = {"structural_token_ratio", "turn_structure", "format_compliance"}
    CONTENT_BENCHMARKS = {"factual_em", "reasoning_em"}

    if experiment == "A1":
        x_key, x_label = "alpha", "α (correction strength)"
        conditions = [r for r in scores if r["condition"].startswith("A1_alpha_")]
    elif experiment == "A2":
        x_key, x_label = "beta", "β (injection magnitude)"
        conditions = [r for r in scores if r["condition"].startswith("A2_beta_")]
    else:
        return

    # Group by benchmark
    bench_data: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in conditions:
        x = row.get(x_key)
        if x is not None:
            bench_data[row["benchmark"]].append((float(x), row["value"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Governance metrics
    ax = axes[0]
    for bench, data in bench_data.items():
        if bench in GOV_BENCHMARKS:
            data.sort(key=lambda t: t[0])
            xs, ys = zip(*data)
            ax.plot(xs, ys, marker="o", label=bench)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Score")
    ax.set_title(f"{experiment}: Governance Metrics")
    ax.legend(fontsize=8)
    ax.axvline(x=1.0 if experiment == "A1" else 0.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # Content metrics
    ax = axes[1]
    for bench, data in bench_data.items():
        if bench in CONTENT_BENCHMARKS:
            data.sort(key=lambda t: t[0])
            xs, ys = zip(*data)
            ax.plot(xs, ys, marker="o", label=bench)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Score")
    ax.set_title(f"{experiment}: Content Metrics")
    ax.legend(fontsize=8)
    ax.axvline(x=1.0 if experiment == "A1" else 0.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = plots_dir / f"{experiment}_dose_response.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
