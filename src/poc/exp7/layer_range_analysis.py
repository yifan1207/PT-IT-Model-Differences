"""Layer range sensitivity analysis (Exp7 0F).

Parses results from 4 A1 runs with different corrective layer boundaries:
  18-33, 20-33 (canonical), 22-33, 20-31

Produces a sensitivity table showing whether the governance dose-response
is robust to ±2 layers on each boundary.

Usage:
  uv run python -m src.poc.exp7.layer_range_analysis \\
      --results-dir results/exp7/0F/ \\
      --output-dir results/exp7/0F/

Outputs:
  results/exp7/0F/layer_range_sensitivity_table.csv
  results/exp7/0F/plots/layer_range_sensitivity.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

LAYER_RANGES = ["18-33", "20-33", "22-33", "20-31"]
ALPHA_VALUES = [5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0, -0.5, -1.0, -2.0, -3.0, -5.0]

GOVERNANCE_BENCHMARKS = {"structural_token_ratio"}
CONTENT_BENCHMARKS = {"mmlu_forced_choice", "reasoning_em"}
SAFETY_BENCHMARKS = {"alignment_behavior"}


def _parse_alpha(condition: str) -> float | None:
    for prefix in ("A1_alpha_", "A1alpha_"):
        if condition.startswith(prefix):
            try:
                return float(condition[len(prefix):])
            except ValueError:
                pass
    return None


def load_scores(run_dir: Path) -> list[dict]:
    """Load scores.jsonl from a run directory."""
    scores_path = run_dir / "scores.jsonl"
    if not scores_path.exists():
        return []
    rows = []
    with open(scores_path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_all_layer_ranges(results_dir: Path) -> dict[str, list[dict]]:
    """Load results for all 4 layer range variants.

    Expects run directories named A1_it_layers_{range} under results_dir.
    """
    all_scores: dict[str, list[dict]] = {}
    for lr in LAYER_RANGES:
        run_dir = results_dir / f"A1_it_layers_{lr}"
        scores = load_scores(run_dir)
        if scores:
            all_scores[lr] = scores
            print(f"  {lr}: {len(scores)} score entries", flush=True)
        else:
            print(f"  {lr}: NOT FOUND (run pending?)", flush=True)
    return all_scores


def build_sensitivity_table(all_scores: dict[str, list[dict]]) -> list[dict]:
    """Build rows: {layer_range, alpha, benchmark, value}."""
    rows = []
    for layer_range, scores in all_scores.items():
        # Group by (condition, benchmark)
        by_cond_bench: dict[tuple[str, str], float] = {}
        for row in scores:
            key = (row["condition"], row["benchmark"])
            by_cond_bench[key] = row.get("value", float("nan"))

        for cond, bench in by_cond_bench:
            alpha = _parse_alpha(cond)
            if alpha is None:
                continue
            rows.append({
                "layer_range": layer_range,
                "alpha": alpha,
                "benchmark": bench,
                "value": by_cond_bench[(cond, bench)],
                "condition": cond,
            })
    return rows


def save_csv(rows: list[dict], output_path: Path) -> None:
    import csv
    if not rows:
        return
    fields = ["layer_range", "alpha", "benchmark", "value", "condition"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["benchmark"], r["layer_range"], r["alpha"])):
            w.writerow({k: row.get(k, "") for k in fields})


def plot_sensitivity(rows: list[dict], output_dir: Path) -> None:
    """3-panel sensitivity plot: governance / content / safety."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[0F] matplotlib not available — skipping plot", flush=True)
        return

    panels = [
        ("Governance (STR)", GOVERNANCE_BENCHMARKS),
        ("Content (MMLU/Reasoning)", CONTENT_BENCHMARKS),
        ("Safety (Alignment)", SAFETY_BENCHMARKS),
    ]

    colors = {"18-33": "#e74c3c", "20-33": "#2c3e50", "22-33": "#3498db", "20-31": "#27ae60"}
    linestyles = {"18-33": "--", "20-33": "-", "22-33": "-.", "20-31": ":"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, (title, benchmarks) in zip(axes, panels):
        for layer_range in LAYER_RANGES:
            layer_rows = [r for r in rows if r["layer_range"] == layer_range and r["benchmark"] in benchmarks]
            if not layer_rows:
                continue

            # Average over benchmarks in this panel
            from collections import defaultdict
            alpha_vals: dict[float, list[float]] = defaultdict(list)
            for r in layer_rows:
                alpha_vals[r["alpha"]].append(r["value"])

            alphas_sorted = sorted(alpha_vals.keys())
            means = [float(np.mean(alpha_vals[a])) for a in alphas_sorted]

            ax.plot(
                alphas_sorted, means,
                color=colors.get(layer_range, "grey"),
                linestyle=linestyles.get(layer_range, "-"),
                linewidth=2,
                label=f"layers {layer_range}",
                marker="o", markersize=4,
            )

        ax.axvline(x=1.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_xlabel("α (removal strength)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle("Layer Range Sensitivity (Exp7 0F)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out_path = output_dir / "plots" / "layer_range_sensitivity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[0F] Plot → {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Layer range sensitivity analysis (Exp7 0F)")
    p.add_argument("--results-dir", default="results/exp7/0F/",
                   help="Directory containing A1_it_layers_* subdirs")
    p.add_argument("--output-dir", default="results/exp7/0F/")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[0F] Loading layer range results from {results_dir}...", flush=True)
    all_scores = load_all_layer_ranges(results_dir)

    if not all_scores:
        print("[0F] No results found — check that 0F runs completed.", flush=True)
        return

    rows = build_sensitivity_table(all_scores)
    csv_path = output_dir / "layer_range_sensitivity_table.csv"
    save_csv(rows, csv_path)
    print(f"[0F] Sensitivity table ({len(rows)} rows) → {csv_path}", flush=True)

    plot_sensitivity(rows, output_dir)

    # Print summary
    canonical_rows = [r for r in rows if r["layer_range"] == "20-33"]
    for lr in LAYER_RANGES:
        lr_rows = [r for r in rows if r["layer_range"] == lr and r["benchmark"] == "structural_token_ratio"]
        if not lr_rows:
            continue
        alpha2 = next((r["value"] for r in lr_rows if abs(r["alpha"] - 2.0) < 0.01), float("nan"))
        print(f"  layers {lr}: STR at α=2.0 = {alpha2:.3f}", flush=True)


if __name__ == "__main__":
    main()
