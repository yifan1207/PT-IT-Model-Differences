"""
Plot 2 (Exp3): Answer emergence and stabilisation — logit lens PT vs IT (Exp 1a).

For each generated token, tracks how the token's rank and probability evolve
across transformer layers — showing at what depth the model "commits" to its
prediction.

Four panels:
  A: Mean rank of final token per layer (lower rank = emerged earlier, rank 1 = top-1)
     Lines: primary model by category, PT dashed overlay if provided.
  B: Mean probability of final token per layer (higher = more committed).
  C: Mean KL(layer_i ∥ layer_33) per layer — 0 means fully committed at that layer.
  D: Heatmap [generation step × layer] of mean KL-to-final for primary results.
     Each cell = mean KL across all prompts that have reached that step.

REQUIRES: collect_emergence=True in Exp3Config.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.shared.constants import N_LAYERS

from src.poc.shared.plot_colors import SPLIT_COLORS as CATEGORY_COLORS, _DEFAULT_COLOR

_KL_HEATMAP_STEP_CAP = 80   # max generation steps shown in heatmap


def _layer_mean_sem(results: list[dict], metric_key: str,
                    n_layers: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-layer mean and SEM of a [step][layer] metric across all prompts×steps."""
    layer_vals: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        for step_vals in r.get(metric_key, []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    layer_vals[layer_i].append(float(v))
    means = np.array([np.mean(v) if v else float("nan") for v in layer_vals])
    sems  = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                      for v in layer_vals])
    return means, sems


def _layer_mean_by_cat(results: list[dict], metric_key: str,
                        n_layers: int = N_LAYERS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean/SEM split by category."""
    by_cat: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n_layers)])
    for r in results:
        cat = r.get("category", "unknown")
        for step_vals in r.get(metric_key, []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    by_cat[cat][layer_i].append(float(v))
    out = {}
    for cat, layer_lists in by_cat.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layer_lists])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                      for v in layer_lists])
        out[cat] = (m, s)
    return out


def _build_kl_heatmap(results: list[dict], step_cap: int = _KL_HEATMAP_STEP_CAP,
                       n_layers: int = N_LAYERS) -> np.ndarray:
    """Build [step_cap, n_layers] array of mean KL-to-final values.

    Cells with no data (step > prompt length) are NaN.
    """
    sums   = np.zeros((step_cap, n_layers), dtype=np.float64)
    counts = np.zeros((step_cap, n_layers), dtype=np.int32)
    for r in results:
        kl_data = r.get("kl_to_final", [])
        for step_i, step_vals in enumerate(kl_data):
            if step_i >= step_cap:
                break
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    sums[step_i, layer_i]   += float(v)
                    counts[step_i, layer_i] += 1
    with np.errstate(invalid="ignore"):
        heatmap = np.where(counts > 0, sums / counts, np.nan)
    return heatmap


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    """Render the emergence figure.

    Parameters
    ----------
    results     : primary results (IT or PT)
    output_dir  : output directory
    pt_results  : optional PT results for comparison overlay (dashed lines)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("next_token_rank"):
        print("  Plot 2 (Exp3) skipped — no emergence data (collect_emergence=False?)")
        return

    layers = np.arange(N_LAYERS)

    # ── aggregate primary results ─────────────────────────────────────────────
    rank_by_cat = _layer_mean_by_cat(results, "next_token_rank")
    prob_by_cat = _layer_mean_by_cat(results, "next_token_prob")
    kl_by_cat   = _layer_mean_by_cat(results, "kl_to_final")
    kl_heatmap  = _build_kl_heatmap(results)

    # ── aggregate PT overlay ──────────────────────────────────────────────────
    pt_rank_mean, pt_rank_sem = (None, None)
    pt_prob_mean, pt_prob_sem = (None, None)
    pt_kl_mean,   pt_kl_sem   = (None, None)
    if pt_results and pt_results[0].get("next_token_rank"):
        pt_rank_mean, pt_rank_sem = _layer_mean_sem(pt_results, "next_token_rank")
        pt_prob_mean, pt_prob_sem = _layer_mean_sem(pt_results, "next_token_prob")
        pt_kl_mean,   pt_kl_sem   = _layer_mean_sem(pt_results, "kl_to_final")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    primary_label = "IT" if any(r.get("category") for r in results) else "primary"

    # ── Panel A: rank per layer ───────────────────────────────────────────────
    ax_a.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
    ax_a.axvline(20, color="black", lw=0.8, ls=":", alpha=0.4, label="boundary (L20)")
    for cat, (m, s) in rank_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax_a.plot(layers, m, lw=1.8, color=c, label=f"{cat}")
        ax_a.fill_between(layers, m - s, m + s, alpha=0.12, color=c)
    if pt_rank_mean is not None:
        ax_a.plot(layers, pt_rank_mean, lw=1.8, color="#555", ls="--",
                  label="PT (all cats)")
        ax_a.fill_between(layers,
                          pt_rank_mean - pt_rank_sem,
                          pt_rank_mean + pt_rank_sem,
                          alpha=0.10, color="#555")
    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean rank of generated token\n(lower = emerged earlier)")
    ax_a.set_title("Panel A — Token rank emergence per layer")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: probability per layer ────────────────────────────────────────
    ax_b.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_b.axvline(20, color="black", lw=0.8, ls=":", alpha=0.4)
    for cat, (m, s) in prob_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax_b.plot(layers, m, lw=1.8, color=c, label=cat)
        ax_b.fill_between(layers, m - s, m + s, alpha=0.12, color=c)
    if pt_prob_mean is not None:
        ax_b.plot(layers, pt_prob_mean, lw=1.8, color="#555", ls="--", label="PT")
        ax_b.fill_between(layers,
                          pt_prob_mean - pt_prob_sem,
                          pt_prob_mean + pt_prob_sem,
                          alpha=0.10, color="#555")
    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("Mean probability of generated token")
    ax_b.set_title("Panel B — Token probability per layer")
    ax_b.legend(fontsize=8, ncol=2)
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel C: KL-to-final per layer ────────────────────────────────────────
    ax_c.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_c.axvline(20, color="black", lw=0.8, ls=":", alpha=0.4)
    for cat, (m, s) in kl_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax_c.plot(layers, m, lw=1.8, color=c, label=cat)
        ax_c.fill_between(layers, np.maximum(0, m - s), m + s, alpha=0.12, color=c)
    if pt_kl_mean is not None:
        ax_c.plot(layers, pt_kl_mean, lw=1.8, color="#555", ls="--", label="PT")
        ax_c.fill_between(layers,
                          np.maximum(0, pt_kl_mean - pt_kl_sem),
                          pt_kl_mean + pt_kl_sem,
                          alpha=0.10, color="#555")
    ax_c.set_xlabel("Transformer Layer")
    ax_c.set_ylabel("Mean KL(layer_i ∥ final) in nats")
    ax_c.set_title("Panel C — KL-to-final per layer  (0 = fully committed)")
    ax_c.legend(fontsize=8, ncol=2)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: KL heatmap [step × layer] ───────────────────────────────────
    # Mask all-NaN rows (no prompts reached that step)
    valid_rows = ~np.all(np.isnan(kl_heatmap), axis=1)
    hm_data    = kl_heatmap[valid_rows]
    n_valid    = hm_data.shape[0]

    vmax = np.nanquantile(hm_data, 0.95) if hm_data.size > 0 else 1.0
    im = ax_d.imshow(
        hm_data,
        aspect="auto",
        origin="upper",
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_d.axvline(11, color="white", lw=1.2, ls="--", alpha=0.7)
    ax_d.axvline(20, color="cyan",  lw=1.0, ls=":",  alpha=0.6)
    ax_d.set_xlabel("Transformer Layer")
    ax_d.set_ylabel("Generation step")
    ax_d.set_title(
        f"Panel D — KL heatmap [step × layer]  ({n_valid} steps shown)\n"
        "White dashed=dip(L11), cyan=boundary(L20)"
    )
    ax_d.set_xticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax_d, label="Mean KL-to-final (nats)")

    fig.suptitle(
        "Exp3 Plot 2 — Answer Emergence and Stabilisation (logit lens)\n"
        "How early does the model commit to its final prediction?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot2_emergence.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 2 (Exp3) saved → {out_path}")
