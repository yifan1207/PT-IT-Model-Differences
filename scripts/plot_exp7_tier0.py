"""Tier 0 methodology validation figures (Exp7 0A–0I).

Generates all Tier 0 plots from precomputed result files.

Usage:
    uv run python scripts/plot_exp7_tier0.py
    uv run python scripts/plot_exp7_tier0.py --experiments 0A 0B 0D
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PLOTS_DIR = Path("results/exp7/plots")
ALL_EXPERIMENTS = ["0A", "0B", "0C", "0D", "0E", "0F", "0G", "0H", "0I", "0J"]


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

    for grp_name, grp_layers in layer_groups.items():
        if "subset_sizes" not in convergence:
            break
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

    ax.axhline(0.95, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="threshold 0.95")
    ax.set_xlabel("Calibration subset size")
    ax.set_ylabel("cos(d̂_S, d̂_canonical)")
    ax.set_title("Direction Convergence Curve (0A)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Pairwise cosine at corrective layers
    ax2 = axes[1]
    pairwise = data.get("pairwise_cosine", {})
    corr_layers = list(range(20, 34))
    layer_means = [pairwise.get(f"layer_{l}", {}).get("mean", float("nan")) for l in corr_layers]
    layer_stds  = [pairwise.get(f"layer_{l}", {}).get("std",  0.0) for l in corr_layers]

    ax2.bar(corr_layers, layer_means, color="#e74c3c", alpha=0.8, label="mean pairwise cosine")
    ax2.errorbar(corr_layers, layer_means, yerr=layer_stds, fmt="none", color="black", capsize=3)
    ax2.axhline(0.95, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="threshold 0.95")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Pairwise cosine similarity")
    ax2.set_title("Bootstrap Direction Stability (0A)")
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

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
    results_path = Path("results/exp7/0B/matched_cosines.json")
    if not results_path.exists():
        print("[0B] matched_cosines.json not found — skipping 0B plots", flush=True)
        return

    with open(results_path) as f:
        data = json.load(f)

    layers = sorted(int(k.split("_")[1]) for k in data if k.startswith("layer_"))
    cosines = [data[f"layer_{l}"] for l in layers]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(layers, cosines, color="#2c3e50", linewidth=2, marker="o", markersize=4)
    ax.axhline(0.90, color="grey", linewidth=0.8, linestyle="--", alpha=0.7, label="threshold 0.90")
    ax.axvspan(20, 33, color="#e74c3c", alpha=0.08, label="corrective layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(d̂_matched, d̂_canonical)")
    ax.set_title("0B: Matched-Token Direction Validation")
    ax.set_ylim(-0.1, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0B_matched_token_cosine.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0B] → {out}", flush=True)


# ── 0C: Random-Matched Overlay ────────────────────────────────────────────────

def plot_0C_rand_matched(output_dir: Path = PLOTS_DIR) -> None:
    """Overlay: A1 (corrective direction) vs A1_rand_matched (magnitude-matched random)."""
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

    if not a1_dir.exists() and not rand_dir.exists():
        print("[0C] Neither A1 nor 0C results found — skipping 0C plots", flush=True)
        return

    benchmarks = [
        ("Governance (STR)", "structural_token_ratio"),
        ("Content (Reasoning EM)", "exp3_reasoning_em"),
        ("Safety (Alignment)", "exp3_alignment_behavior"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (panel_title, bench) in zip(axes, benchmarks):
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
                ax.plot(np.array(alphas)[order], np.array(vals)[order],
                        color=color, label=label, linewidth=2, linestyle=ls, marker="o", markersize=4)

        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("α")
        ax.set_title(panel_title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle("0C: Magnitude-Matched Random Direction Control", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0C_rand_matched_overlay.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0C] → {out}", flush=True)


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

    benchmarks = [
        ("Governance (STR)", "structural_token_ratio"),
        ("Content (Reasoning EM)", "exp3_reasoning_em"),
        ("Safety (Alignment)", "exp3_alignment_behavior"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (panel_title, bench) in zip(axes, benchmarks):
        bench_data = ci_data.get(bench, {})
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
            lo.append(stats.get("ci_lo", float("nan")))
            hi.append(stats.get("ci_hi", float("nan")))

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
    maxabs = [perturb[k]["delta_str_max_abs"] for k in labels]

    short_labels = [l.replace("_to_", "→").replace("_", " ") for l in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, color="#3498db", alpha=0.8, label="mean Δ STR")
    ax.bar(x, maxabs, color="#e74c3c", alpha=0.3, label="max |Δ STR|")
    ax.axhline(0.01, color="grey", linewidth=0.8, linestyle="--", alpha=0.7, label="threshold 0.01")
    ax.axhline(-0.01, color="grey", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=20, ha="right")
    ax.set_ylabel("Δ STR")
    ax.set_title("0E: Token Classifier Robustness")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0E_classifier_robustness.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0E] → {out}", flush=True)


# ── 0F: Layer Range Sensitivity ───────────────────────────────────────────────

def plot_0F_layer_range(output_dir: Path = PLOTS_DIR) -> None:
    """Delegate to layer_range_analysis.py plot function."""
    plt = _setup_mpl()
    import csv

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

def plot_0H_calibration_split(output_dir: Path = PLOTS_DIR) -> None:
    """Dose-response overlay: governance-selected vs random vs bottom-600 direction."""
    plt = _setup_mpl()

    def _load_str_sweep(scores_path: Path, prefix: str) -> tuple[list[float], list[float]]:
        rows = _load_jsonl(scores_path)
        alphas, vals = [], []
        for r in rows:
            cond = r.get("condition", "")
            bench = r.get("benchmark", "")
            if bench != "structural_token_ratio" or not cond.startswith(prefix):
                continue
            try:
                alpha = float(cond[len(prefix):])
                alphas.append(alpha)
                vals.append(r.get("value", float("nan")))
            except ValueError:
                pass
        return alphas, vals

    series = [
        ("results/exp6/merged_A1_it_v4/scores.jsonl",            "A1_alpha_",          "#2c3e50", "governance-selected", "-"),
        ("results/exp7/0H/A1_it_random_dir/scores.jsonl",        "A1_alpha_",          "#3498db", "random-600", "--"),
        ("results/exp7/0H/A1_it_bottom_dir/scores.jsonl",        "A1_alpha_",          "#e74c3c", "bottom-600 (negative ctrl)", ":"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    found_any = False
    for path_str, prefix, color, label, ls in series:
        alphas, vals = _load_str_sweep(Path(path_str), prefix)
        if alphas:
            found_any = True
            order = np.argsort(alphas)
            ax.plot(np.array(alphas)[order], np.array(vals)[order],
                    color=color, label=label, linewidth=2, linestyle=ls, marker="o", markersize=4)

    if not found_any:
        print("[0H] No 0H results found — skipping 0H plots", flush=True)
        return

    ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_xlabel("α")
    ax.set_ylabel("Structural Token Ratio")
    ax.set_title("0H: Calibration-Evaluation Split Validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0H_calibration_split.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0H] → {out}", flush=True)


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
                ax.plot(np.array(alphas)[order], np.array(vals)[order],
                        color=METHOD_COLORS[mname],
                        label=METHOD_LABELS[mname],
                        linewidth=2, marker="o", markersize=3)

        ax.axvline(1.0, color="grey", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.set_xlabel("α")
        ax.set_title(panel_title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Score")
    fig.suptitle("0I: Intervention Formula Sensitivity", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / "0I_formula_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"[0I] → {out}", flush=True)


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
