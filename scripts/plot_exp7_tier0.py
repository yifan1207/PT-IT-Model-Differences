"""Tier 0 methodology validation figures (Exp7 0A–0I).

Generates all Tier 0 plots from precomputed result files.
All plots include statistical annotations (CIs, n-counts, effect sizes)
where supported by available data. Core data is saved to results/exp7/plots/data/.

Usage:
    uv run python scripts/plot_exp7_tier0.py
    uv run python scripts/plot_exp7_tier0.py --experiments 0A 0B 0D
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PLOTS_DIR = Path("results/exp7/plots")
DATA_DIR = PLOTS_DIR / "data"
ALL_EXPERIMENTS = ["0A", "0B", "0C", "0D", "0E", "0F", "0G", "0H", "0I", "0J"]


def _save_data(name: str, data: dict | list) -> None:
    """Save core data JSON to results/exp7/plots/data/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / f"{name}.json"
    with open(out, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  data → {out}", flush=True)


def _load_0D_ci() -> dict[str, dict[str, dict]]:
    """Load 0D bootstrap CI data.

    Returns {benchmark: {alpha_float: {ci_low, ci_high, mean}}}.
    Alpha keys are normalized floats for reliable lookup.
    """
    ci_path = Path("results/exp7/0D/ci_A1_programmatic.json")
    if not ci_path.exists():
        return {}
    with open(ci_path) as f:
        ci_raw = json.load(f)
    result: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in ci_raw:
        cond = row.get("condition", "")
        if not cond.startswith("A1_alpha_"):
            continue
        try:
            alpha = float(cond[len("A1_alpha_"):])
        except ValueError:
            continue
        bench = row.get("benchmark", "")
        result[bench][alpha] = {
            "ci_low": row.get("ci_low", row.get("ci_lo", float("nan"))),
            "ci_high": row.get("ci_high", row.get("ci_hi", float("nan"))),
            "mean": row.get("mean", float("nan")),
        }
    return dict(result)


# Benchmark name aliases — data may use exp3_ prefix or not
_BENCH_ALIASES = {
    "exp3_reasoning_em": "reasoning_em",
    "exp3_alignment_behavior": "alignment_behavior",
    "reasoning_em": "reasoning_em",
    "alignment_behavior": "alignment_behavior",
    "structural_token_ratio": "structural_token_ratio",
}


def _get_ci_band(ci_data: dict, bench: str, alphas: np.ndarray) -> tuple[list[float], list[float]] | None:
    """Get CI lo/hi arrays for a benchmark at given alpha values.

    Handles benchmark name aliases (exp3_ prefix vs without).
    Returns None if insufficient CI data available.
    """
    ci_bench_key = _BENCH_ALIASES.get(bench, bench)
    bench_ci = ci_data.get(ci_bench_key, {})
    if not bench_ci:
        return None
    lo, hi = [], []
    for a in alphas:
        entry = bench_ci.get(float(a))
        if entry:
            lo.append(entry["ci_low"])
            hi.append(entry["ci_high"])
        else:
            lo.append(float("nan"))
            hi.append(float("nan"))
    valid = sum(1 for v in lo if not np.isnan(v))
    if valid < 3:
        return None
    return lo, hi


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    })
    return plt


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ── 0A: Direction Calibration Stability ──────────────────────────────────────

def plot_0A_direction_stability(output_dir: Path = PLOTS_DIR) -> None:
    """Plots:
    1. Convergence curve: cos(d̂_S, d̂_canonical) vs subset size S, per layer group.
    2. Pairwise cosine heatmap at canonical layers (20-33).
    """
    plt = _setup_mpl()
    results_path = Path("results/exp7/0A/bootstrap_results.json")
    if not results_path.exists():
        print("[0A] bootstrap_results.json not found — skipping 0A plots", flush=True)
        return

    with open(results_path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Convergence curve
    ax = axes[0]
    convergence = data.get("convergence", {})
    layer_groups = {
        "early (1-11)":     list(range(1, 12)),
        "mid (12-19)":      list(range(12, 20)),
        "corrective (20-33)": list(range(20, 34)),
    }
    colors_conv = {"early (1-11)": "#3498db", "mid (12-19)": "#27ae60", "corrective (20-33)": "#e74c3c"}

    # Detect data format: {layer: {size: {cosine_to_canonical_mean, ...}}}
    # vs legacy {subset_sizes: [...], cosines_by_layer: {size: {layer: val}}}
    if "subset_sizes" in convergence:
        # Legacy format
        for grp_name, grp_layers in layer_groups.items():
            sizes = convergence["subset_sizes"]
            means, stds = [], []
            for s in sizes:
                key = str(s)
                if key not in convergence.get("cosines_by_layer", {}):
                    continue
                layer_cosines = convergence["cosines_by_layer"][key]
                grp_vals = [layer_cosines.get(str(l), float("nan")) for l in grp_layers]
                grp_vals = [v for v in grp_vals if not np.isnan(v)]
                means.append(np.mean(grp_vals) if grp_vals else float("nan"))
                stds.append(np.std(grp_vals) if grp_vals else 0.0)
            if means:
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.plot(sizes[:len(means)], means_arr,
                        color=colors_conv[grp_name], label=grp_name, linewidth=2, marker="o", markersize=4)
                ax.fill_between(sizes[:len(means)],
                                means_arr - stds_arr, means_arr + stds_arr,
                                color=colors_conv[grp_name], alpha=0.15)
    else:
        # Actual format: {layer_idx_str: {subset_size_str: {cosine_to_canonical_mean, ...}}}
        # Collect all subset sizes from first layer
        first_layer = next(iter(convergence.values()), {})
        sizes = sorted(int(s) for s in first_layer.keys())
        for grp_name, grp_layers in layer_groups.items():
            means, stds = [], []
            for s in sizes:
                grp_vals = []
                for l in grp_layers:
                    layer_data = convergence.get(str(l), {}).get(str(s), {})
                    val = layer_data.get("cosine_to_canonical_mean", float("nan"))
                    if not np.isnan(val):
                        grp_vals.append(val)
                means.append(np.mean(grp_vals) if grp_vals else float("nan"))
                stds.append(np.std(grp_vals) if grp_vals else 0.0)
            if means:
                means_arr = np.array(means)
                stds_arr = np.array(stds)
                ax.plot(sizes, means_arr,
                        color=colors_conv[grp_name], label=grp_name, linewidth=2, marker="o", markersize=4)
                ax.fill_between(sizes,
                                means_arr - stds_arr, means_arr + stds_arr,
                                color=colors_conv[grp_name], alpha=0.15)

    ax.axhline(0.95, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="threshold 0.95")
    ax.set_xlabel("Calibration subset size")
    ax.set_ylabel("cos(d̂_S, d̂_canonical)")
    ax.set_title("Direction Convergence Curve (0A)")
    ax.set_ylim(0.9, 1.005)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    # Add bootstrap n-count annotation
    n_bootstraps = data.get("n_bootstraps", data.get("n_splits", "?"))
    ax.annotate(f"n_boot={n_bootstraps}", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7, color="grey", fontstyle="italic")

    # Panel 2: Pairwise cosine at ALL layers (color-coded by group)
    ax2 = axes[1]
    per_layer = data.get("per_layer", {})
    all_layers = sorted(int(k) for k in per_layer.keys())
    layer_means = [per_layer[str(l)].get("pairwise_cosine_mean", float("nan")) for l in all_layers]
    layer_stds  = [per_layer[str(l)].get("pairwise_cosine_std",  0.0) for l in all_layers]

    # Color bars by layer group
    bar_colors = []
    for l in all_layers:
        if l < 12:
            bar_colors.append("#3498db")  # early
        elif l < 20:
            bar_colors.append("#27ae60")  # mid
        else:
            bar_colors.append("#e74c3c")  # corrective

    ax2.bar(all_layers, layer_means, color=bar_colors, alpha=0.8)
    ax2.errorbar(all_layers, layer_means, yerr=layer_stds, fmt="none", color="black", capsize=2)
    ax2.axhline(0.95, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="threshold 0.95")
    # Legend for groups
    from matplotlib.patches import Patch
    ax2.legend(handles=[
        Patch(color="#3498db", label="early (1-11)"),
        Patch(color="#27ae60", label="mid (12-19)"),
        Patch(color="#e74c3c", label="corrective (20-33)"),
    ], fontsize=7, loc="lower left")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Pairwise cosine similarity")
    ax2.set_title("Bootstrap Direction Stability (0A)")
    ax2.set_ylim(0.98, 1.002)
    ax2.grid(True, alpha=0.3, axis="y")
    # Annotate ±1σ shading
    ax2.annotate("error bars = ±1σ (bootstrap)", xy=(0.02, 0.02), xycoords="axes fraction",
                 fontsize=7, color="grey", fontstyle="italic")

    fig.suptitle("0A: Direction Calibration Sensitivity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0A_direction_stability.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0A] → {out}", flush=True)


# ── 0B: Matched-Token Direction ───────────────────────────────────────────────

def plot_0B_matched_cosine(output_dir: Path = PLOTS_DIR) -> None:
    """Cosine(d̂_matched, d̂_canonical) vs layer."""
    plt = _setup_mpl()

    # Support both legacy single file and per-label files
    base = Path("results/exp7/0B")
    cosine_files = sorted(base.glob("matched_cosines_*.json"))
    if not cosine_files:
        legacy = base / "matched_cosines.json"
        if legacy.exists():
            cosine_files = [legacy]
        else:
            print("[0B] No matched_cosines*.json found — skipping 0B plots", flush=True)
            return

    # Load 0A bootstrap direction stability for uncertainty band
    bootstrap_path = Path("results/exp7/0A/bootstrap_results.json")
    bootstrap_stds: dict[int, float] = {}
    if bootstrap_path.exists():
        with open(bootstrap_path) as f:
            bs_data = json.load(f)
        for layer_str, entry in bs_data.get("per_layer", {}).items():
            bootstrap_stds[int(layer_str)] = entry.get("pairwise_cosine_std", 0.0)

    colors = ["#2c3e50", "#e74c3c", "#27ae60", "#8e44ad"]
    fig, ax = plt.subplots(figsize=(9, 4))

    for i, fpath in enumerate(cosine_files):
        with open(fpath) as f:
            data = json.load(f)

        # Support both formats: top-level layer_N keys and nested per_layer dict
        if "per_layer" in data and isinstance(data["per_layer"], dict):
            pl = data["per_layer"]
            layers = sorted(int(k) for k in pl.keys())
            cosines = [pl[str(l)].get("cosine_matched_vs_canonical", float("nan")) for l in layers]
        else:
            layers = sorted(int(k.split("_")[1]) for k in data if k.startswith("layer_"))
            cosines = [data[f"layer_{l}"] for l in layers]
        label = fpath.stem.replace("matched_cosines_", "").replace("_", " ")
        ax.plot(layers, cosines, color=colors[i % len(colors)], linewidth=2,
                marker="o", markersize=4, label=label)

        # Add ±1σ uncertainty band from 0A bootstrap direction stability
        if bootstrap_stds and i == 0:  # only on first (governance) line
            stds = [bootstrap_stds.get(l, 0.0) for l in layers]
            cosines_arr = np.array(cosines)
            stds_arr = np.array(stds)
            ax.fill_between(layers,
                            cosines_arr - stds_arr, cosines_arr + stds_arr,
                            color=colors[i], alpha=0.15,
                            label="±1σ (direction stability)")

    ax.axhline(0.90, color="grey", linewidth=0.8, linestyle="--", alpha=0.7, label="threshold 0.90")
    ax.axvspan(20, 33, color="#e74c3c", alpha=0.08, label="corrective layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(d̂_matched, d̂_canonical)")
    ax.set_title("0B: Matched-Token Direction Validation")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0B_matched_token_cosine.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0B] → {out}", flush=True)


# ── 0C: Random-Matched Overlay ────────────────────────────────────────────────

def plot_0C_rand_matched(output_dir: Path = PLOTS_DIR) -> None:
    """Overlay: A1 (corrective direction) vs A1_rand_matched (magnitude-matched random).
    Adds multiseed ±1σ CI bands for random direction from 5-seed experiment."""
    plt = _setup_mpl()

    def _load_scores(scores_path: Path) -> dict[str, dict[str, float]]:
        """Returns {condition: {benchmark: value}}."""
        rows = _load_jsonl(scores_path)
        by_cond_bench: dict[str, dict[str, float]] = {}
        for r in rows:
            cond, bench, val = r.get("condition"), r.get("benchmark"), r.get("value", float("nan"))
            if cond and bench:
                by_cond_bench.setdefault(cond, {})[bench] = val
        return by_cond_bench

    def _parse_alpha(cond: str, prefix: str) -> float | None:
        if not cond.startswith(prefix):
            return None
        try:
            return float(cond[len(prefix):])
        except ValueError:
            return None

    a1_dir = Path("results/exp6/merged_A1_it_v4")
    rand_dir = Path("results/exp7/0C/merged_A1_rand_matched_it_v1")
    multiseed_dir = Path("results/exp7/0C/merged_A1_rand_multiseed_it_v1")

    if not a1_dir.exists() and not rand_dir.exists():
        print("[0C] Neither A1 nor 0C results found — skipping 0C plots", flush=True)
        return

    # Load multiseed data for CI bands: {bench: {alpha: [val_seed0, val_seed1, ...]}}
    multiseed_by_bench: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    n_seeds = 0
    if multiseed_dir.exists():
        ms_scores = _load_jsonl(multiseed_dir / "scores.jsonl")
        seeds_seen = set()
        for r in ms_scores:
            cond = r.get("condition", "")
            m = re.match(r"A1randms_s(\d+)_alpha_([-\d.]+)", cond)
            if m:
                seeds_seen.add(int(m.group(1)))
                alpha = float(m.group(2))
                bench = r.get("benchmark", "")
                val = r.get("value", float("nan"))
                if not np.isnan(val):
                    multiseed_by_bench[bench][alpha].append(val)
        n_seeds = len(seeds_seen)

    # Load 0D bootstrap CIs for corrective direction
    ci_data = _load_0D_ci()

    benchmarks = [
        ("Governance (STR)", "structural_token_ratio"),
        ("Content (Reasoning EM)", "exp3_reasoning_em"),
        ("Safety (Alignment)", "exp3_alignment_behavior"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    save_data: dict = {"benchmarks": {}, "n_seeds": n_seeds}

    for ax, (panel_title, bench) in zip(axes, benchmarks):
        bench_save: dict = {}

        for scores_path, prefix, color, label, ls in [
            (a1_dir / "scores.jsonl",   "A1_alpha_",          "#2c3e50", "corrective direction", "-"),
            (rand_dir / "scores.jsonl", "A1randmatched_alpha_", "#e67e22", "magnitude-matched random", "--"),
        ]:
            if not scores_path.exists():
                continue
            scores = _load_scores(scores_path)
            alphas, vals = [], []
            for cond, bench_vals in scores.items():
                alpha = _parse_alpha(cond, prefix)
                if alpha is not None and bench in bench_vals:
                    alphas.append(alpha)
                    vals.append(bench_vals[bench])
            if alphas:
                order = np.argsort(alphas)
                a_sorted = np.array(alphas)[order]
                v_sorted = np.array(vals)[order]
                ax.plot(a_sorted, v_sorted,
                        color=color, label=label, linewidth=2, linestyle=ls, marker="o", markersize=4)
                bench_save[label] = {
                    "alphas": [float(a) for a in a_sorted],
                    "values": [float(v) for v in v_sorted],
                }

                # Add 95% CI band for corrective direction from 0D bootstrap
                if label == "corrective direction" and ci_data:
                    ci_band = _get_ci_band(ci_data, bench, a_sorted)
                    if ci_band:
                        ax.fill_between(a_sorted, ci_band[0], ci_band[1],
                                        color=color, alpha=0.15, label="95% CI (bootstrap)")

        # Add multiseed CI band for random direction
        if bench in multiseed_by_bench:
            ms_data = multiseed_by_bench[bench]
            ms_alphas = sorted(ms_data.keys())
            ms_means = [float(np.mean(ms_data[a])) for a in ms_alphas]
            ms_stds = [float(np.std(ms_data[a], ddof=1)) for a in ms_alphas]

            ax.fill_between(
                ms_alphas,
                [m - s for m, s in zip(ms_means, ms_stds)],
                [m + s for m, s in zip(ms_means, ms_stds)],
                color="#e67e22", alpha=0.15,
                label=f"random ±1σ ({n_seeds} seeds)"
            )
            bench_save["multiseed"] = {
                "alphas": ms_alphas,
                "means": ms_means,
                "stds": ms_stds,
                "n_seeds": n_seeds,
            }

        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("α")
        ax.set_title(panel_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        save_data["benchmarks"][bench] = bench_save

    axes[0].set_ylabel("Score")
    fig.suptitle("0C: Magnitude-Matched Random Direction Control", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0C_rand_matched_overlay.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0C] → {out}", flush=True)

    _save_data("0C_rand_matched", save_data)


# ── 0D: Bootstrap CIs ─────────────────────────────────────────────────────────

def plot_0D_bootstrap_ci(output_dir: Path = PLOTS_DIR) -> None:
    """Dose-response with 95% CI shading for main benchmarks."""
    plt = _setup_mpl()

    ci_path = Path("results/exp7/0D/ci_A1_programmatic.json")
    if not ci_path.exists():
        print("[0D] ci_A1_programmatic.json not found — skipping 0D plots", flush=True)
        return

    with open(ci_path) as f:
        ci_data = json.load(f)

    # Handle both formats: list-of-dicts (actual) and nested dict (legacy)
    if isinstance(ci_data, list):
        # Convert flat list to nested {benchmark: {condition: stats}}
        ci_nested: dict = {}
        for row in ci_data:
            b = row.get("benchmark", "")
            c = row.get("condition", "")
            ci_nested.setdefault(b, {})[c] = row
        ci_data = ci_nested

    benchmarks = [
        ("Governance (STR)", ["structural_token_ratio"]),
        ("Content (Reasoning EM)", ["reasoning_em", "exp3_reasoning_em"]),
        ("Safety (Alignment)", ["alignment_behavior", "exp3_alignment_behavior"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (panel_title, bench_keys) in zip(axes, benchmarks):
        bench_data = {}
        for bk in bench_keys:
            bench_data = ci_data.get(bk, {})
            if bench_data:
                break
        if not bench_data:
            ax.set_title(f"{panel_title}\n(no data)")
            continue

        alphas, means, lo, hi = [], [], [], []
        for cond, stats in bench_data.items():
            if not cond.startswith("A1_alpha_"):
                continue
            try:
                alpha = float(cond[len("A1_alpha_"):])
            except ValueError:
                continue
            alphas.append(alpha)
            means.append(stats.get("mean", float("nan")))
            lo.append(stats.get("ci_low", stats.get("ci_lo", float("nan"))))
            hi.append(stats.get("ci_high", stats.get("ci_hi", float("nan"))))

        if alphas:
            order = np.argsort(alphas)
            alphas = np.array(alphas)[order]
            means = np.array(means)[order]
            lo = np.array(lo)[order]
            hi = np.array(hi)[order]
            ax.plot(alphas, means, color="#2c3e50", linewidth=2, marker="o", markersize=4)
            ax.fill_between(alphas, lo, hi, color="#2c3e50", alpha=0.20, label="95% CI")

        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("α")
        ax.set_title(panel_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle("0D: Bootstrap 95% CIs on A1 Dose-Response", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0D_bootstrap_ci.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0D] → {out}", flush=True)


# ── 0E: Classifier Robustness ─────────────────────────────────────────────────

def plot_0E_classifier_robustness(output_dir: Path = PLOTS_DIR) -> None:
    """Bar chart: Δ STR per perturbation type."""
    plt = _setup_mpl()

    rob_path = Path("results/exp7/0E/classifier_robustness.json")
    if not rob_path.exists():
        print("[0E] classifier_robustness.json not found — skipping 0E plots", flush=True)
        return

    with open(rob_path) as f:
        data = json.load(f)

    perturb = data.get("perturbation_tests", {})
    if not perturb:
        print("[0E] No perturbation results found", flush=True)
        return

    labels = list(perturb.keys())
    means = [perturb[k]["delta_str_mean"] for k in labels]
    stds = [perturb[k].get("delta_str_std", 0.0) for k in labels]
    maxabs = [perturb[k]["delta_str_max_abs"] for k in labels]
    n_conds = [perturb[k].get("n_conditions_tested", "?") for k in labels]

    short_labels = [l.replace("_to_", "→").replace("_", " ") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color="#3498db", alpha=0.8, label="mean Δ STR")
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black", capsize=3, linewidth=1.2,
                label="±1σ across conditions")
    ax.bar(x, maxabs, color="#e74c3c", alpha=0.2, label="max |Δ STR|")
    ax.axhline(0.01, color="grey", linewidth=0.8, linestyle="--", alpha=0.7, label="threshold 0.01")
    ax.axhline(-0.01, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right")
    ax.set_ylabel("Δ STR")
    ax.set_title("0E: Token Classifier Robustness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    # Annotate n
    ax.annotate(f"n_conditions={n_conds[0]}", xy=(0.02, 0.02), xycoords="axes fraction",
                fontsize=7, color="grey", fontstyle="italic")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0E_classifier_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0E] → {out}", flush=True)


# ── 0F: Layer Range Sensitivity ───────────────────────────────────────────────

def plot_0F_layer_range(output_dir: Path = PLOTS_DIR) -> None:
    """Layer range sensitivity with n-annotations. Saves data to plots/data/."""
    plt = _setup_mpl()

    csv_path = Path("results/exp7/0F/layer_range_sensitivity_table.csv")
    if not csv_path.exists():
        print("[0F] layer_range_sensitivity_table.csv not found — skipping 0F plots", flush=True)
        return

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append({
                    "layer_range": row["layer_range"],
                    "alpha": float(row["alpha"]),
                    "benchmark": row["benchmark"],
                    "value": float(row["value"]) if row["value"] else float("nan"),
                })
            except (ValueError, KeyError):
                continue

    if not rows:
        print("[0F] No rows in CSV", flush=True)
        return

    from src.poc.exp7.layer_range_analysis import plot_sensitivity
    plot_sensitivity(rows, Path("results/exp7/0F"))

    # Copy the output to the main plots dir
    src = Path("results/exp7/0F/plots/layer_range_sensitivity.png")
    if src.exists():
        import shutil
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, output_dir / "0F_layer_range_sensitivity.png")
        print(f"[0F] → {output_dir / '0F_layer_range_sensitivity.png'}", flush=True)

    # Save data
    layer_ranges_found = sorted(set(r["layer_range"] for r in rows))
    benchmarks_found = sorted(set(r["benchmark"] for r in rows))
    _save_data("0F_layer_range_sensitivity", {
        "n_rows": len(rows),
        "layer_ranges": layer_ranges_found,
        "benchmarks": benchmarks_found,
        "n_alpha_values": len(set(r["alpha"] for r in rows)),
        "rows": rows,
    })

    # Also save single-layer importance if available
    imp_path = Path("results/exp7/0F/single_layer_importance.json")
    if imp_path.exists():
        with open(imp_path) as f:
            imp_data = json.load(f)
        _save_data("0F_single_layer_importance", imp_data)


# ── 0G: Tuned-Lens vs Raw Commitment ─────────────────────────────────────────

def plot_0G_tuned_vs_raw(output_dir: Path = PLOTS_DIR) -> None:
    """Bar chart: raw vs tuned-lens commitment layers for PT and IT."""
    plt = _setup_mpl()

    results_path = Path("results/exp7/0G/tuned_lens_commitment.json")
    if not results_path.exists():
        print("[0G] tuned_lens_commitment.json not found — skipping 0G plots", flush=True)
        return

    with open(results_path) as f:
        data = json.load(f)

    variants = ["pt", "it"]
    methods = ["raw_logit_lens", "tuned_lens"]
    colors = {"raw_logit_lens": "#95a5a6", "tuned_lens": "#2c3e50"}
    labels = {"raw_logit_lens": "Raw logit-lens", "tuned_lens": "Tuned-lens"}

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(variants))
    w = 0.35
    for i, method in enumerate(methods):
        means = [data.get(v, {}).get(method, {}).get("mean_commitment_layer", float("nan"))
                 for v in variants]
        ax.bar(x + (i - 0.5) * w, means, w, color=colors[method], alpha=0.85, label=labels[method])

    ax.set_xticks(x)
    ax.set_xticklabels(["PT (base)", "IT (instruct)"])
    ax.set_ylabel("Mean commitment layer")
    ax.set_title("0G: Tuned-Lens vs Raw Logit-Lens Commitment Delay")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0G_tuned_vs_raw_commitment.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0G] → {out}", flush=True)


# ── 0H: Calibration Split ─────────────────────────────────────────────────────

def _compute_0H_ci(
    sample_outputs_path: Path,
    dataset_records: dict,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> dict[str, dict[float, dict]]:
    """Compute BCa bootstrap CIs per condition × benchmark from sample_outputs.

    Returns {benchmark: {alpha: {mean, ci_low, ci_high, n}}}.
    """
    from src.poc.exp7.bootstrap_ci import rescore_per_record, _bootstrap_ci_bca

    per_record = rescore_per_record(sample_outputs_path, dataset_records)

    result: dict[str, dict[float, dict]] = defaultdict(dict)
    for cond, benchmarks in per_record.items():
        if not cond.startswith("A1_alpha_"):
            continue
        try:
            alpha = float(cond[len("A1_alpha_"):])
        except ValueError:
            continue
        for bench, values in benchmarks.items():
            ci = _bootstrap_ci_bca(values, n_bootstrap, seed)
            result[bench][alpha] = ci

    return dict(result)


def _load_or_compute_0H_ci(
    series_label: str,
    sample_outputs_path: Path,
    dataset_records: dict,
    cache_dir: Path,
) -> dict[str, dict[float, dict]]:
    """Load cached CIs or compute + cache them."""
    safe_name = series_label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    cache_path = cache_dir / f"0H_ci_{safe_name}.json"

    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        # Convert string alpha keys back to float
        result: dict[str, dict[float, dict]] = {}
        for bench, alpha_dict in raw.items():
            result[bench] = {float(a): v for a, v in alpha_dict.items()}
        return result

    if not sample_outputs_path.exists():
        return {}

    print(f"  [0H] Computing BCa CIs for {series_label}...", flush=True)
    ci_data = _compute_0H_ci(sample_outputs_path, dataset_records)

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    serialisable = {
        bench: {str(a): v for a, v in alpha_dict.items()}
        for bench, alpha_dict in ci_data.items()
    }
    with open(cache_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  [0H] Cached → {cache_path}", flush=True)

    return ci_data


def plot_0H_calibration_split(output_dir: Path = PLOTS_DIR) -> None:
    """Multi-panel dose-response: governance-selected vs random-600 vs bottom-600
    direction across all benchmarks. Per-series 95% BCa CI bands computed from
    per-record re-scoring of sample_outputs.jsonl. Shows that the format-content
    dissociation is robust to the choice of calibration prompts."""
    plt = _setup_mpl()

    # Also check if merged bottom_dir exists
    bottom_dir = Path("results/exp7/0H/A1_it_bottom_dir")
    if not bottom_dir.exists():
        merged_bottom = Path("results/exp7/0H/merged_A1_it_bottom_dir")
        if merged_bottom.exists():
            bottom_dir = merged_bottom

    # (results_dir, color, label, linestyle, marker)
    series = [
        (Path("results/exp6/merged_A1_it_v4"),              "#2c3e50", "Canonical (top-600)", "-",  "o"),
        (Path("results/exp7/0H/A1_it_random_dir"),           "#2980b9", "Random-600",          "--", "s"),
        (bottom_dir,                                         "#c0392b", "Bottom-600",          ":",  "D"),
    ]

    # Panels: (display_title, ci_bench_keys, is_format_metric)
    PANELS = [
        ("Structural Token Ratio",  ["structural_token_ratio"],                    True),
        ("Format Compliance",       ["format_compliance_v2"],                      True),
        ("Alignment Behavior",      ["alignment_behavior"],                        True),
        ("Reasoning EM",            ["reasoning_em"],                              False),
        ("MMLU Accuracy",           ["mmlu_accuracy"],                             False),
    ]

    # Load dataset for re-scoring
    from src.poc.exp7.bootstrap_ci import load_dataset
    dataset_records = load_dataset("data/eval_dataset_v2.jsonl")

    # Load / compute CIs for each series
    series_ci: list[tuple[str, str, str, str, dict]] = []
    for results_dir, color, label, ls, marker in series:
        sample_path = results_dir / "sample_outputs.jsonl"
        ci_data = _load_or_compute_0H_ci(label, sample_path, dataset_records, DATA_DIR)
        series_ci.append((color, label, ls, marker, ci_data))

    if not any(cd for _, _, _, _, cd in series_ci):
        print("[0H] No 0H results found — skipping 0H plots", flush=True)
        return

    n_panels = len(PANELS)
    n_cols = 3
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows))
    axes_flat = axes.flatten()

    save_data: dict = {"panels": {}, "series_labels": [l for _, l, _, _, _ in series_ci]}

    for pi, (panel_title, bench_keys, is_format) in enumerate(PANELS):
        ax = axes_flat[pi]
        panel_save: dict = {}

        for color, label, ls, marker, ci_data in series_ci:
            # Find matching benchmark key
            bench_ci = {}
            for bk in bench_keys:
                if bk in ci_data:
                    bench_ci = ci_data[bk]
                    break
            if not bench_ci:
                continue

            alphas = sorted(bench_ci.keys())
            means = [bench_ci[a]["mean"] for a in alphas]
            lo = [bench_ci[a]["ci_low"] for a in alphas]
            hi = [bench_ci[a]["ci_high"] for a in alphas]
            ns = [bench_ci[a].get("n", 0) for a in alphas]

            a_arr = np.array(alphas)
            m_arr = np.array(means)
            lo_arr = np.array(lo)
            hi_arr = np.array(hi)

            # Label with n on first panel
            n_typical = int(np.median(ns)) if ns else 0
            plot_label = f"{label} (n≈{n_typical})" if pi == 0 and n_typical else label

            ax.plot(a_arr, m_arr, color=color, linewidth=2,
                    linestyle=ls, marker=marker, markersize=3.5,
                    markeredgewidth=0.5, markeredgecolor="white",
                    label=plot_label, zorder=3)

            # CI band
            valid = ~(np.isnan(lo_arr) | np.isnan(hi_arr))
            if valid.sum() >= 3:
                ax.fill_between(a_arr[valid], lo_arr[valid], hi_arr[valid],
                                color=color, alpha=0.10, zorder=1)

            panel_save[label] = {
                "alphas": [float(a) for a in alphas],
                "means": means,
                "ci_low": lo,
                "ci_high": hi,
                "n": ns,
            }

        # Reference line at α=1 (identity / no intervention)
        ax.axvline(1.0, color="#7f8c8d", linewidth=0.9, linestyle="--", alpha=0.6,
                   zorder=1)

        # Panel label: format vs content
        category = "Format" if is_format else "Content"
        ax.set_title(f"{panel_title}\n({category})", fontsize=10.5, fontweight="bold")
        ax.set_xlabel("α", fontsize=10)
        if pi % n_cols == 0:
            ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15, linewidth=0.5)

        # Legend on first panel only
        if pi == 0:
            leg = ax.legend(fontsize=7, loc="upper left", framealpha=0.9,
                            edgecolor="#cccccc")
            leg.get_frame().set_linewidth(0.5)

        save_data["panels"][panel_title] = panel_save

    # Hide unused axes
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        "0H: Calibration-Evaluation Split Validation\n"
        "95% BCa bootstrap CIs per direction variant",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0H_calibration_split.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[0H] → {out}", flush=True)

    _save_data("0H_calibration_split", save_data)


# ── 0I: Formula Comparison ────────────────────────────────────────────────────

def plot_0I_formula_comparison(output_dir: Path = PLOTS_DIR) -> None:
    """3-panel: 4 formula lines vs α for governance / content / safety."""
    plt = _setup_mpl()

    scores_path = Path("results/exp7/0I/merged_A1_formula_it_v1/scores.jsonl")
    if not scores_path.exists():
        print("[0I] merged A1_formula results not found — skipping 0I plots", flush=True)
        return

    rows = _load_jsonl(scores_path)

    METHOD_NAMES = ["mlp_proj_remove", "mlp_additive", "residual_proj_remove", "attn_proj_remove"]
    METHOD_LABELS = {
        "mlp_proj_remove":      "MLP proj-remove (canonical)",
        "mlp_additive":         "MLP additive",
        "residual_proj_remove": "Residual proj-remove",
        "attn_proj_remove":     "Attn proj-remove",
    }
    METHOD_COLORS = {
        "mlp_proj_remove":      "#2c3e50",
        "mlp_additive":         "#3498db",
        "residual_proj_remove": "#e74c3c",
        "attn_proj_remove":     "#27ae60",
    }

    # Load 0D bootstrap CIs — applies to mlp_proj_remove (canonical = A1)
    ci_data = _load_0D_ci()

    benchmarks = [
        ("Governance (STR)", "structural_token_ratio"),
        ("Content (Reasoning EM)", "exp3_reasoning_em"),
        ("Safety (Alignment)", "exp3_alignment_behavior"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (panel_title, bench) in zip(axes, benchmarks):
        for mname in METHOD_NAMES:
            prefix = f"A1formula_{mname}_alpha_"
            alphas, vals = [], []
            for r in rows:
                cond = r.get("condition", "")
                if r.get("benchmark") != bench or not cond.startswith(prefix):
                    continue
                try:
                    alpha = float(cond[len(prefix):])
                    alphas.append(alpha)
                    vals.append(r.get("value", float("nan")))
                except ValueError:
                    pass
            if alphas:
                order = np.argsort(alphas)
                a_sorted = np.array(alphas)[order]
                v_sorted = np.array(vals)[order]
                ax.plot(a_sorted, v_sorted,
                        color=METHOD_COLORS[mname],
                        label=METHOD_LABELS[mname],
                        linewidth=2, marker="o", markersize=3)

                # Add 95% CI band for canonical method (mlp_proj_remove = A1)
                if mname == "mlp_proj_remove" and ci_data:
                    ci_band = _get_ci_band(ci_data, bench, a_sorted)
                    if ci_band:
                        ax.fill_between(a_sorted, ci_band[0], ci_band[1],
                                        color=METHOD_COLORS[mname], alpha=0.15,
                                        label="95% CI (canonical)")

        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("α")
        ax.set_title(panel_title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle("0I: Intervention Formula Sensitivity", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0I_formula_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0I] → {out}", flush=True)

    # Save data
    save_rows: dict[str, dict[str, dict[str, list]]] = defaultdict(lambda: defaultdict(lambda: {"alphas": [], "values": []}))
    for r in rows:
        cond = r.get("condition", "")
        bench = r.get("benchmark", "")
        val = r.get("value", float("nan"))
        for mname in METHOD_NAMES:
            prefix = f"A1formula_{mname}_alpha_"
            if cond.startswith(prefix):
                try:
                    alpha = float(cond[len(prefix):])
                    save_rows[mname][bench]["alphas"].append(alpha)
                    save_rows[mname][bench]["values"].append(val)
                except ValueError:
                    pass
    _save_data("0I_formula_comparison", {
        "methods": METHOD_NAMES,
        "method_labels": METHOD_LABELS,
        "data": {m: dict(b) for m, b in save_rows.items()},
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def plot_0J_onset_sensitivity(output_dir: Path = PLOTS_DIR) -> None:
    """Delegate to onset_threshold_sensitivity.py plot function."""
    from src.poc.exp7.onset_threshold_sensitivity import (
        _load_profiles, build_onset_table, summarise_sensitivity, plot_sensitivity,
        CSV_PATH,
    )
    if not CSV_PATH.exists():
        print("[0J] L1_mean_delta_cosine.csv not found — skipping 0J plots", flush=True)
        return
    profiles = _load_profiles(CSV_PATH)
    rows = build_onset_table(profiles)
    summary = summarise_sensitivity(rows)
    plot_sensitivity(rows, summary, Path("results/exp7/0J"))

    import shutil
    src = Path("results/exp7/0J/plots/onset_sensitivity.png")
    if src.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, output_dir / "0J_onset_sensitivity.png")
        print(f"[0J] → {output_dir / '0J_onset_sensitivity.png'}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate all Tier 0 methodology figures (Exp7)")
    p.add_argument("--experiments", nargs="*", default=ALL_EXPERIMENTS,
                   choices=ALL_EXPERIMENTS,
                   help="Which experiments to plot (default: all)")
    p.add_argument("--output-dir", default=str(PLOTS_DIR))
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plotters = {
        "0A": plot_0A_direction_stability,
        "0B": plot_0B_matched_cosine,
        "0C": plot_0C_rand_matched,
        "0D": plot_0D_bootstrap_ci,
        "0E": plot_0E_classifier_robustness,
        "0F": plot_0F_layer_range,
        "0G": plot_0G_tuned_vs_raw,
        "0H": plot_0H_calibration_split,
        "0I": plot_0I_formula_comparison,
        "0J": plot_0J_onset_sensitivity,
    }

    for exp_id in args.experiments:
        print(f"--- Plotting {exp_id} ---", flush=True)
        try:
            plotters[exp_id](output_dir)
        except Exception as e:
            print(f"[{exp_id}] ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print(f"\nAll Tier 0 plots → {output_dir}", flush=True)


if __name__ == "__main__":
    main()
