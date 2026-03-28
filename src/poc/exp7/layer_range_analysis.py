"""Layer range sensitivity analysis (Exp7 0F).

Parses results from:
  Part 1: 4 A1 runs with different corrective layer boundaries
    18-33, 20-33 (canonical), 22-33, 20-31
  Part 2: Per-layer importance sweep (A1_single_layer)
    Single-layer removal at alpha=0 for layers 20-33

Produces sensitivity tables and per-layer importance ranking.
Connects per-layer importance to 0G commitment onset.

Usage:
  uv run python -m src.poc.exp7.layer_range_analysis \\
      --results-dir results/exp7/0F/ \\
      --output-dir results/exp7/0F/

Outputs:
  results/exp7/0F/layer_range_sensitivity_table.csv
  results/exp7/0F/single_layer_importance.json
  results/exp7/0F/plots/layer_range_sensitivity.png
  results/exp7/0F/plots/single_layer_importance.png
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


def _parse_single_layer(condition: str) -> int | None:
    """Parse layer index from A1single_layer_N condition name."""
    prefix = "A1single_layer_"
    if condition.startswith(prefix):
        try:
            return int(condition[len(prefix):])
        except ValueError:
            pass
    return None


def load_single_layer_results(results_dir: Path) -> list[dict] | None:
    """Load results from the A1_single_layer experiment."""
    # Try merged dir first, then raw dir
    for name in ["merged_A1_single_layer_it_v1", "A1_single_layer_it_v1"]:
        run_dir = results_dir / name
        scores = load_scores(run_dir)
        if scores:
            print(f"  single_layer: {len(scores)} score entries from {run_dir}", flush=True)
            return scores
    print("  single_layer: NOT FOUND (run pending?)", flush=True)
    return None


def analyze_single_layer_importance(scores: list[dict]) -> dict:
    """Compute per-layer importance from single-layer removal results.

    Importance = |baseline_STR - removed_STR| for each layer.
    Higher importance = more governance signal in that layer.
    """
    # Get baseline values per benchmark
    baseline_values: dict[str, float] = {}
    layer_scores: dict[int, dict[str, float]] = {}

    for row in scores:
        cond = row["condition"]
        bench = row["benchmark"]
        val = row.get("value", float("nan"))

        if cond == "A1single_baseline":
            baseline_values[bench] = val

        layer = _parse_single_layer(cond)
        if layer is not None:
            if layer not in layer_scores:
                layer_scores[layer] = {}
            layer_scores[layer][bench] = val

    if not baseline_values or not layer_scores:
        return {"error": "Missing baseline or layer scores"}

    ALL_BENCHMARKS = ["structural_token_ratio", "mmlu_forced_choice", "reasoning_em", "alignment_behavior"]

    # Compute importance per layer per benchmark
    importance: list[dict] = []
    for layer in sorted(layer_scores.keys()):
        entry: dict = {"layer": layer}
        for bench in ALL_BENCHMARKS:
            if bench in layer_scores[layer] and bench in baseline_values:
                delta = layer_scores[layer][bench] - baseline_values[bench]
                entry[f"{bench}_value"] = layer_scores[layer][bench]
                entry[f"{bench}_baseline"] = baseline_values[bench]
                entry[f"{bench}_delta"] = delta
                entry[f"{bench}_importance"] = abs(delta)
                # Keep STR-specific keys for backwards compatibility
                if bench == "structural_token_ratio":
                    entry["str_delta"] = delta
                    entry["str_importance"] = abs(delta)
        importance.append(entry)

    # Rank by STR importance
    importance.sort(key=lambda x: x.get("str_importance", 0), reverse=True)
    for rank, entry in enumerate(importance):
        entry["str_importance_rank"] = rank + 1

    # Re-sort by layer for output
    importance.sort(key=lambda x: x["layer"])

    return {
        "baseline_str": baseline_values.get("structural_token_ratio", float("nan")),
        "per_layer": importance,
        "most_important_layers": [
            e["layer"] for e in sorted(importance, key=lambda x: x.get("str_importance", 0), reverse=True)[:5]
        ],
        "interpretation": (
            "Layers with highest STR importance carry the most governance signal. "
            "Compare with 0G commitment onset: important layers should cluster "
            "near the commitment boundary (~layer 20-25)."
        ),
    }


def plot_single_layer_importance(importance_data: dict, output_dir: Path) -> None:
    """Bar chart of per-layer STR importance."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[0F] matplotlib not available -- skipping single-layer plot", flush=True)
        return

    per_layer = importance_data.get("per_layer", [])
    if not per_layer:
        return

    layers = [e["layer"] for e in per_layer]
    str_importance = [e.get("str_importance", 0) for e in per_layer]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c" if imp > np.mean(str_importance) else "#3498db" for imp in str_importance]
    ax.bar(layers, str_importance, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("|delta STR| (importance)")
    ax.set_title("Per-Layer Governance Importance (Single-Layer Removal at alpha=0)")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3, axis="y")

    # Mark top-5 layers
    top5 = importance_data.get("most_important_layers", [])[:5]
    for layer in top5:
        idx = layers.index(layer) if layer in layers else -1
        if idx >= 0:
            ax.annotate(
                f"#{per_layer[idx].get('str_importance_rank', '?')}",
                (layer, str_importance[idx]),
                textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=8, fontweight="bold",
            )

    out_path = output_dir / "plots" / "single_layer_importance.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[0F] Single-layer plot -> {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Layer range sensitivity analysis (Exp7 0F)")
    p.add_argument("--results-dir", default="results/exp7/0F/",
                   help="Directory containing A1_it_layers_* and A1_single_layer_* subdirs")
    p.add_argument("--output-dir", default="results/exp7/0F/")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Part 1: Layer range sensitivity ───────────────────────────────────────
    print(f"[0F] Loading layer range results from {results_dir}...", flush=True)
    all_scores = load_all_layer_ranges(results_dir)

    if all_scores:
        rows = build_sensitivity_table(all_scores)
        csv_path = output_dir / "layer_range_sensitivity_table.csv"
        save_csv(rows, csv_path)
        print(f"[0F] Sensitivity table ({len(rows)} rows) -> {csv_path}", flush=True)

        plot_sensitivity(rows, output_dir)

        # Print summary
        for lr in LAYER_RANGES:
            lr_rows = [r for r in rows if r["layer_range"] == lr and r["benchmark"] == "structural_token_ratio"]
            if not lr_rows:
                continue
            alpha2 = next((r["value"] for r in lr_rows if abs(r["alpha"] - 2.0) < 0.01), float("nan"))
            print(f"  layers {lr}: STR at alpha=2.0 = {alpha2:.3f}", flush=True)
    else:
        print("[0F] No layer range results found.", flush=True)

    # ── Part 2: Single-layer importance ───────────────────────────────────────
    print(f"\n[0F] Loading single-layer importance results...", flush=True)
    single_layer_scores = load_single_layer_results(results_dir)

    if single_layer_scores:
        importance = analyze_single_layer_importance(single_layer_scores)

        imp_path = output_dir / "single_layer_importance.json"
        with open(imp_path, "w") as f:
            json.dump(importance, f, indent=2)
        print(f"[0F] Single-layer importance -> {imp_path}", flush=True)

        top5 = importance.get("most_important_layers", [])
        print(f"  Top-5 most important layers: {top5}", flush=True)

        plot_single_layer_importance(importance, output_dir)
    else:
        print("[0F] No single-layer results found.", flush=True)


if __name__ == "__main__":
    main()
