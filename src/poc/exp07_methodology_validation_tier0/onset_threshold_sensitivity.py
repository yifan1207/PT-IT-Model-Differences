"""Corrective onset threshold sensitivity analysis (Exp7 0J).

Tests whether the ~59% corrective onset depth claim is robust to the choice
of threshold for detecting when IT first deviates from PT on the δ-cosine profile.

Input:
  results/cross_model/plots/data/L1_mean_delta_cosine.csv
    columns: model, model_label, variant, layer, n_layers, normalized_depth, mean_delta_cosine

Method:
  For each of 6 model families:
    1. Build IT and PT δ-cosine profiles per layer
    2. Compute difference signal: diff_ℓ = δ-cosine_IT(ℓ) - δ-cosine_PT(ℓ)
    3. Baseline: mean and std of diff over early layers (layers 1..BASELINE_LAYERS)
    4. σ-thresholds: onset = first ℓ where diff_ℓ > baseline_mean + k·baseline_std
       k ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
    5. Absolute thresholds: onset = first ℓ where diff_ℓ > τ
       τ ∈ {0.02, 0.05, 0.10, 0.15}
  Build 6×9 table: onset_layer and normalized_depth for each (family, threshold)
  Report range of onset layers per family across all 9 thresholds

Output:
  results/exp07_methodology_validation_tier0/0J/onset_table.json      — 6×9 table
  results/exp07_methodology_validation_tier0/0J/onset_table.csv       — same as CSV
  results/exp07_methodology_validation_tier0/0J/gemma_alt_ranges.json — Gemma onset at 0.5σ and 2σ for A1 reruns
  results/exp07_methodology_validation_tier0/0J/plots/onset_sensitivity.png

Usage:
  uv run python -m src.poc.exp07_methodology_validation_tier0.onset_threshold_sensitivity --output-dir results/exp07_methodology_validation_tier0/0J/
  uv run python -m src.poc.exp07_methodology_validation_tier0.onset_threshold_sensitivity --csv-path <custom_path>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

CSV_PATH = Path("results/cross_model/plots/data/L1_mean_delta_cosine.csv")
OUTPUT_DIR = Path("results/exp07_methodology_validation_tier0/0J")

# Early layers for baseline std estimate (exclude layer 0 which is embedding → always NaN)
BASELINE_LAYERS = 5   # use layers 1..5 to estimate background variation

# 6 model families in the paper
MODELS = [
    ("gemma3_4b",      "Gemma 3 4B",     34),
    ("llama31_8b",     "Llama 3.1 8B",   32),
    ("qwen3_4b",       "Qwen 3 4B",      36),
    ("mistral_7b",     "Mistral 7B v0.3",32),
    ("deepseek_v2_lite","DeepSeek-V2-Lite",27),
    ("olmo2_7b",       "OLMo 2 7B",      32),
]

# σ-thresholds and absolute thresholds
SIGMA_THRESHOLDS = [0.5, 0.75, 1.0, 1.5, 2.0]
ABS_THRESHOLDS   = [0.02, 0.05, 0.10, 0.15]
# Labels for columns (9 total)
THRESHOLD_LABELS = (
    [f"{k}σ" for k in SIGMA_THRESHOLDS] +
    [f"abs{t}" for t in ABS_THRESHOLDS]
)


def _load_profiles(csv_path: Path) -> dict[str, dict[str, dict[int, float]]]:
    """Load δ-cosine profiles from CSV.

    Returns: {model_name: {"it": {layer: value}, "pt": {layer: value}}}
    """
    profiles: dict[str, dict[str, dict[int, float]]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            variant = row["variant"]
            try:
                layer = int(row["layer"])
                value = float(row["mean_delta_cosine"])
            except (ValueError, KeyError):
                continue
            if np.isnan(value):
                continue
            profiles.setdefault(model, {}).setdefault(variant, {})[layer] = value
    return profiles


def _compute_onset(
    it_profile: dict[int, float],
    pt_profile: dict[int, float],
    n_layers: int,
    n_consecutive: int = 2,
) -> dict[str, int | None]:
    """Compute onset layer under all thresholds for one model.

    Onset is defined as the first layer where IT becomes SIGNIFICANTLY MORE NEGATIVE
    than PT on the δ-cosine profile (IT-PT drops below threshold), held for
    n_consecutive layers to avoid triggering on noise.

    The corrective stage is characterized by IT δ-cosine becoming more negative than PT,
    indicating stronger MLP opposition in the instruction-tuned model.

    σ-thresholds: onset = first sustained layer where
        diff_ℓ = IT_ℓ - PT_ℓ < baseline_mean - k · baseline_std
        (diff drops significantly below baseline, i.e. IT is noticeably below PT)

    Absolute thresholds: onset = first sustained layer where
        -(diff_ℓ) > τ  ↔  PT_ℓ - IT_ℓ > τ
        (IT is at least τ below PT)

    Returns dict mapping threshold_label → onset_layer (or None if not reached).
    """
    # Align layers
    common_layers = sorted(set(it_profile) & set(pt_profile))
    if not common_layers:
        return {t: None for t in THRESHOLD_LABELS}

    diff = np.array([it_profile[l] - pt_profile[l] for l in common_layers])
    layers_arr = np.array(common_layers)

    # Baseline: layers 6..BASELINE_END (skip early noisy layers and layer 0=NaN)
    # Use a stable mid-early region where both IT and PT behave similarly
    BASELINE_START = 6
    BASELINE_END   = 14
    baseline_mask  = (layers_arr >= BASELINE_START) & (layers_arr <= BASELINE_END)
    baseline_diff  = diff[baseline_mask]
    if len(baseline_diff) < 2:
        # Fallback: use first 8 non-NaN layers
        valid = ~np.isnan(diff)
        baseline_diff = diff[valid][:8]
    baseline_mean = float(np.nanmean(baseline_diff))
    baseline_std  = float(np.nanstd(baseline_diff) + 1e-6)

    results: dict[str, int | None] = {}

    # Search layers AFTER the baseline region
    search_mask   = layers_arr > BASELINE_END
    search_layers = layers_arr[search_mask]
    search_diff   = diff[search_mask]

    def _first_sustained(cond_arr: np.ndarray, lays: np.ndarray, n: int) -> int | None:
        """Return first layer where cond holds for n consecutive layers."""
        streak = 0
        streak_start = None
        for layer, c in zip(lays, cond_arr):
            if c:
                if streak == 0:
                    streak_start = int(layer)
                streak += 1
                if streak >= n:
                    return streak_start
            else:
                streak = 0
                streak_start = None
        return None

    # σ-thresholds: detect when IT-PT drops BELOW baseline - k*std
    for k in SIGMA_THRESHOLDS:
        neg_thresh = baseline_mean - k * baseline_std  # threshold to cross downward
        label = f"{k}σ"
        cond = search_diff < neg_thresh
        results[label] = _first_sustained(cond, search_layers, n_consecutive)

    # Absolute thresholds: PT - IT > τ  (IT is at least τ BELOW PT)
    for tau in ABS_THRESHOLDS:
        label = f"abs{tau}"
        cond = (-search_diff) > tau  # PT_ℓ - IT_ℓ > tau
        results[label] = _first_sustained(cond, search_layers, n_consecutive)

    return results


def build_onset_table(
    profiles: dict[str, dict[str, dict[int, float]]],
) -> list[dict]:
    """Build full 6×9 onset table.

    Returns list of dicts with keys:
      model, model_label, n_layers, threshold, onset_layer, normalized_depth
    """
    rows = []
    for model_name, model_label, n_layers in MODELS:
        if model_name not in profiles:
            print(f"  WARNING: {model_name} not in CSV — skipping", flush=True)
            continue

        it_profile = profiles[model_name].get("it", {})
        pt_profile = profiles[model_name].get("pt", {})

        if not it_profile or not pt_profile:
            print(f"  WARNING: {model_name} missing IT or PT profile", flush=True)
            continue

        onsets = _compute_onset(it_profile, pt_profile, n_layers)

        for thresh_label in THRESHOLD_LABELS:
            onset_layer = onsets.get(thresh_label)
            norm_depth  = round(onset_layer / n_layers, 3) if onset_layer is not None else None
            rows.append({
                "model":          model_name,
                "model_label":    model_label,
                "n_layers":       n_layers,
                "threshold":      thresh_label,
                "onset_layer":    onset_layer,
                "normalized_depth": norm_depth,
            })

    return rows


def summarise_sensitivity(rows: list[dict]) -> dict:
    """For each model, report onset range (min..max layer, min..max depth) across thresholds."""
    summary: dict[str, dict] = {}
    for model_name, model_label, n_layers in MODELS:
        model_rows = [r for r in rows if r["model"] == model_name]
        onsets = [r["onset_layer"] for r in model_rows if r["onset_layer"] is not None]
        depths = [r["normalized_depth"] for r in model_rows if r["normalized_depth"] is not None]

        if not onsets:
            continue

        # Current (1σ) onset
        sigma1_rows = [r for r in model_rows if r["threshold"] == "1.0σ"]
        current_onset = sigma1_rows[0]["onset_layer"] if sigma1_rows else None

        summary[model_name] = {
            "model_label":        model_label,
            "n_layers":           n_layers,
            "current_onset_1sigma": current_onset,
            "onset_min":          int(min(onsets)),
            "onset_max":          int(max(onsets)),
            "onset_range":        int(max(onsets) - min(onsets)),
            "depth_min":          float(min(depths)),
            "depth_max":          float(max(depths)),
            "depth_range_pp":     round((max(depths) - min(depths)) * 100, 1),
        }

    return summary


def get_gemma_alt_ranges(rows: list[dict]) -> dict:
    """Extract Gemma onset layers under 0.5σ and 2σ for A1 reruns (0J Gemma ablation)."""
    gemma_rows = {r["threshold"]: r for r in rows if r["model"] == "gemma3_4b"}
    n_layers = 34

    onset_05 = gemma_rows.get("0.5σ", {}).get("onset_layer")  # broader (earlier)
    onset_2  = gemma_rows.get("2.0σ", {}).get("onset_layer")  # narrower (later)
    onset_1  = gemma_rows.get("1.0σ", {}).get("onset_layer")  # canonical

    result = {
        "canonical":     {"onset_layer": onset_1, "n_layers": n_layers,
                          "layer_range": f"{onset_1}-{n_layers-1}" if onset_1 else None},
        "broader_0.5sig": {"onset_layer": onset_05, "n_layers": n_layers,
                           "layer_range": f"{onset_05}-{n_layers-1}" if onset_05 else None},
        "narrower_2sig": {"onset_layer": onset_2,  "n_layers": n_layers,
                          "layer_range": f"{onset_2}-{n_layers-1}" if onset_2 else None},
    }
    return result


def save_csv(rows: list[dict], output_path: Path) -> None:
    fields = ["model", "model_label", "n_layers", "threshold", "onset_layer", "normalized_depth"]
    with open(output_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["model"], r["threshold"])):
            w.writerow({k: row.get(k, "") for k in fields})


def plot_sensitivity(
    rows: list[dict],
    summary: dict,
    output_dir: Path,
) -> None:
    """Two-panel sensitivity plot:
    1. Onset layer vs threshold, per model (6 lines)
    2. Normalized depth range per model (bar chart)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[0J] matplotlib not available — skipping plots", flush=True)
        return

    model_colors = {
        "gemma3_4b":       "#e74c3c",
        "llama31_8b":      "#3498db",
        "qwen3_4b":        "#27ae60",
        "mistral_7b":      "#9b59b6",
        "deepseek_v2_lite":"#e67e22",
        "olmo2_7b":        "#1abc9c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Onset layer vs threshold (σ-based ONLY — these are comparable across models)
    ax = axes[0]
    sigma_labels = [t for t in THRESHOLD_LABELS if "σ" in t]
    for model_name, model_label, n_layers in MODELS:
        model_rows = {r["threshold"]: r["onset_layer"]
                      for r in rows if r["model"] == model_name}
        y_vals = [model_rows.get(t) for t in sigma_labels]
        y_arr = np.array([float("nan") if v is None else v for v in y_vals], dtype=float)
        x = np.arange(len(sigma_labels))
        n_found = int(np.sum(~np.isnan(y_arr)))
        marker = "o" if n_found > 0 else "x"
        ax.plot(x, y_arr, color=model_colors.get(model_name, "grey"),
                label=f"{model_label} ({n_found}/{len(sigma_labels)})",
                linewidth=2, marker=marker, markersize=5)

    ax.set_xticks(np.arange(len(sigma_labels)))
    ax.set_xticklabels(sigma_labels, rotation=30, ha="right", fontsize=9)
    ax.axvline(2.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6, label="current (1σ)")
    ax.set_xlabel("Threshold (σ-based, per-model normalized)")
    ax.set_ylabel("Onset layer (absolute)")
    ax.set_title("σ-Based Thresholds (Comparable)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: Detection rate heatmap — show which model×threshold combos found onset
    ax2 = axes[1]
    all_models = [(m, ml) for m, ml, _ in MODELS]
    table = np.full((len(all_models), len(THRESHOLD_LABELS)), np.nan)
    for i, (model_name, _) in enumerate(all_models):
        for j, t in enumerate(THRESHOLD_LABELS):
            onset = next((r["onset_layer"] for r in rows
                         if r["model"] == model_name and r["threshold"] == t), None)
            if onset is not None:
                n_layers = next(nl for mn, _, nl in MODELS if mn == model_name)
                table[i, j] = onset / n_layers  # normalized depth
            # else stays NaN → will show as red/missing

    # Draw cells: green = found (with depth label), red = not detected
    for i in range(len(all_models)):
        for j in range(len(THRESHOLD_LABELS)):
            if np.isnan(table[i, j]):
                ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor="#ffcccc", edgecolor="white", linewidth=1))
                ax2.text(j, i, "—", ha="center", va="center", fontsize=7, color="#cc0000")
            else:
                depth = table[i, j]
                # Color by how reasonable the depth is (green=55-65%, yellow=outside)
                if 0.45 <= depth <= 0.70:
                    fc = "#ccffcc"
                else:
                    fc = "#ffffcc"  # suspicious (too early or too late)
                ax2.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                              facecolor=fc, edgecolor="white", linewidth=1))
                ax2.text(j, i, f".{int(depth*100):02d}", ha="center", va="center",
                         fontsize=7, fontweight="bold")

    ax2.set_xticks(range(len(THRESHOLD_LABELS)))
    ax2.set_xticklabels(THRESHOLD_LABELS, rotation=30, ha="right", fontsize=8)
    ax2.set_yticks(range(len(all_models)))
    ax2.set_yticklabels([ml for _, ml in all_models], fontsize=8)
    ax2.set_xlim(-0.5, len(THRESHOLD_LABELS) - 0.5)
    ax2.set_ylim(-0.5, len(all_models) - 0.5)
    ax2.invert_yaxis()
    ax2.set_title("Detection Success\n(green=detected, red=no onset, yellow=suspicious depth)")

    # Panel 3: σ-only depth range (honest — only uses σ thresholds)
    ax3 = axes[2]
    all_model_names = [m for m, _, _ in MODELS]
    all_model_labels = [ml for _, ml, _ in MODELS]
    x = np.arange(len(all_model_names))

    for i, (m, ml) in enumerate(zip(all_model_names, all_model_labels)):
        # Only use σ-based thresholds for the range
        sigma_onsets = [r["onset_layer"] for r in rows
                        if r["model"] == m and "σ" in r["threshold"]
                        and r["onset_layer"] is not None]
        n_layers = next(nl for mn, _, nl in MODELS if mn == m)

        n_total = len(sigma_labels)
        n_found = len(sigma_onsets)
        color = model_colors.get(m, "grey")

        if sigma_onsets:
            lo = min(sigma_onsets) / n_layers
            hi = max(sigma_onsets) / n_layers
            sigma1 = next((r["onset_layer"] for r in rows
                          if r["model"] == m and r["threshold"] == "1.0σ"
                          and r["onset_layer"] is not None), None)

            ax3.plot([i, i], [lo, hi], color=color, linewidth=6, alpha=0.5)
            if sigma1 is not None:
                ax3.plot(i, sigma1 / n_layers, color=color, marker="D", markersize=8)
            ax3.text(i, hi + 0.02, f"{n_found}/{n_total}", ha="center", fontsize=7,
                     color=color, fontweight="bold")
        else:
            # No onset at any σ threshold — show explicitly
            ax3.plot(i, 0.5, marker="x", color=color, markersize=12, markeredgewidth=2)
            ax3.text(i, 0.53, f"0/{n_total}\nno onset", ha="center", fontsize=7,
                     color="#cc0000", fontweight="bold")

        ax3.text(i, 0.35, ml.replace(" ", "\n"), ha="center", fontsize=7, color=color)

    ax3.axhspan(0.45, 0.65, color="lightblue", alpha=0.25, label="45–65% zone")
    ax3.set_xticks([])
    ax3.set_ylabel("Normalized depth")
    ax3.set_ylim(0.30, 0.75)
    ax3.set_title("σ-Only Onset Depth\n(diamond=1σ, bar=range, fractions=detection rate)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle("0J: Corrective Onset Threshold Sensitivity (Honest View)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    out = output_dir / "plots" / "onset_sensitivity.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[0J] Plot → {out}", flush=True)


def print_summary_table(rows: list[dict], summary: dict) -> None:
    """Pretty-print the 6×9 onset table to stdout."""
    print("\n[0J] === Onset Layer Table (layer / normalized depth) ===", flush=True)
    header = f"{'Model':<22} | " + " | ".join(f"{t:>8}" for t in THRESHOLD_LABELS)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for model_name, model_label, n_layers in MODELS:
        model_rows = {r["threshold"]: r for r in rows if r["model"] == model_name}
        cells = []
        for t in THRESHOLD_LABELS:
            r = model_rows.get(t, {})
            ol = r.get("onset_layer")
            nd = r.get("normalized_depth")
            if ol is not None:
                cells.append(f"{ol:2d}/{nd:.2f}")
            else:
                cells.append("  N/A  ")
        print(f"{model_label:<22} | " + " | ".join(f"{c:>8}" for c in cells), flush=True)

    print("\n[0J] === Sensitivity Summary ===", flush=True)
    for model_name in summary:
        s = summary[model_name]
        print(
            f"  {s['model_label']:<22}: onset range {s['onset_min']}–{s['onset_max']} layers "
            f"(span={s['onset_range']}), depth {s['depth_min']:.2f}–{s['depth_max']:.2f} "
            f"({s['depth_range_pp']:.1f}pp)",
            flush=True,
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Corrective onset threshold sensitivity (Exp7 0J)")
    p.add_argument("--csv-path", default=str(CSV_PATH))
    p.add_argument("--output-dir", default=str(OUTPUT_DIR))
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    csv_path   = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[0J] ERROR: CSV not found at {csv_path}", flush=True)
        sys.exit(1)

    print(f"[0J] Loading δ-cosine profiles from {csv_path}...", flush=True)
    profiles = _load_profiles(csv_path)
    print(f"[0J] Loaded profiles for: {sorted(profiles.keys())}", flush=True)

    print("[0J] Computing onset layers under 9 threshold choices...", flush=True)
    rows = build_onset_table(profiles)

    # Save CSV
    csv_out = output_dir / "onset_table.csv"
    save_csv(rows, csv_out)
    print(f"[0J] Table ({len(rows)} rows) → {csv_out}", flush=True)

    # Save JSON
    json_out = output_dir / "onset_table.json"
    with open(json_out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[0J] Table → {json_out}", flush=True)

    # Summary
    summary = summarise_sensitivity(rows)
    summary_out = output_dir / "onset_summary.json"
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=2)

    # Gemma alt ranges for A1 reruns
    alt_ranges = get_gemma_alt_ranges(rows)
    alt_out = output_dir / "gemma_alt_ranges.json"
    with open(alt_out, "w") as f:
        json.dump(alt_ranges, f, indent=2)
    print(f"[0J] Gemma alt ranges → {alt_out}", flush=True)
    print(f"  canonical:     {alt_ranges['canonical']}", flush=True)
    print(f"  broader 0.5σ:  {alt_ranges['broader_0.5sig']}", flush=True)
    print(f"  narrower 2σ:   {alt_ranges['narrower_2sig']}", flush=True)

    print_summary_table(rows, summary)

    # Evaluate success criteria
    max_span = max((s["onset_range"] for s in summary.values()), default=0)
    max_depth_pp = max((s["depth_range_pp"] for s in summary.values()), default=0)
    print(f"\n[0J] Max onset span across families: {max_span} layers", flush=True)
    print(f"[0J] Max depth range across families: {max_depth_pp:.1f}pp", flush=True)
    if max_span <= 3 and max_depth_pp <= 15:
        print("[0J] ✓ PASS: onset stable (≤3 layer span, ≤15pp depth range)", flush=True)
    else:
        print("[0J] ⚠ WARNING: onset sensitivity exceeds criteria — "
              "consider reframing as 'gradual transition'", flush=True)

    if not args.no_plot:
        plot_sensitivity(rows, summary, output_dir)


if __name__ == "__main__":
    main()
