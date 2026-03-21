"""
Plot 6 (Exp3): Layer-to-final KL divergence trajectory PT vs IT (Experiment 2c).

KL(layer_i ∥ layer_33): how far is each intermediate layer's logit-lens
distribution from the model's final prediction?

  High KL at layer i = model's prediction at layer i differs from final output.
  Sharp KL drop at layer L = model committed to its answer at layer L.

If PT commits earlier (sharp KL drop at a lower layer) and IT keeps modifying
(KL stays high longer, drops later), that is clean corrective stage evidence.

Four panels:
  A: Mean KL-to-final per layer — primary (IT) by category with SEM bands,
     PT dashed overlay.
  B: KL heatmap [layer × generation step] for primary results (IT).
     Colour = mean KL across all prompts that have reached that step.
  C: Distribution of "commitment layer" — first layer where KL drops below
     threshold (0.1 nats). Histogram + median marker for IT vs PT.
  D: Mean KL-to-final for early (first 20%) vs late (last 20%) generation
     steps with SEM bands — does IT delay commitment in the corrective stage?

REQUIRES: collect_emergence=True in Exp3Config (provides kl_to_final field).
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.shared.constants import N_LAYERS

from src.poc.shared.plot_colors import SPLIT_COLORS as CATEGORY_COLORS, _DEFAULT_COLOR
_BOUNDARY         = 20
_KL_THRESHOLD     = 0.1   # nats — "committed" when KL drops below this
_HEATMAP_STEP_CAP = 80


def _kl_layer_by_cat(results: list[dict],
                      n_layers: int = N_LAYERS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-layer mean ± SEM of kl_to_final by category."""
    by_cat: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n_layers)])
    for r in results:
        cat = r.get("category", "unknown")
        for step_vals in r.get("kl_to_final", []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    by_cat[cat][layer_i].append(float(v))
    out = {}
    for cat, layers_list in by_cat.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layers_list])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                      for v in layers_list])
        out[cat] = (m, s)
    return out


def _build_kl_heatmap(results: list[dict],
                       step_cap:  int = _HEATMAP_STEP_CAP,
                       n_layers:  int = N_LAYERS) -> np.ndarray:
    """Build [n_layers, step_cap] heatmap of mean KL values.

    Rows = layers, columns = generation steps.
    """
    sums   = np.zeros((n_layers, step_cap), dtype=np.float64)
    counts = np.zeros((n_layers, step_cap), dtype=np.int32)
    for r in results:
        for step_i, step_vals in enumerate(r.get("kl_to_final", [])):
            if step_i >= step_cap:
                break
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    sums[layer_i, step_i]   += float(v)
                    counts[layer_i, step_i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _commitment_layers(results: list[dict],
                        threshold: float = _KL_THRESHOLD,
                        n_layers:  int   = N_LAYERS) -> list[int]:
    """For each prompt × step, find the first layer where KL < threshold.

    Returns list of commitment layer indices (one per prompt × step).
    """
    commit_layers = []
    for r in results:
        for step_vals in r.get("kl_to_final", []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)) and float(v) < threshold:
                    commit_layers.append(layer_i)
                    break
    return commit_layers


def _kl_by_generation_third(results: list[dict],
                              n_layers: int = N_LAYERS,
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± SEM KL per layer for early (first 20%) vs late (last 20%) generation steps.

    Returns (early_mean, early_sem, late_mean, late_sem) all [n_layers].
    """
    early_vals: list[list[float]] = [[] for _ in range(n_layers)]
    late_vals:  list[list[float]] = [[] for _ in range(n_layers)]

    for r in results:
        kl_data = r.get("kl_to_final", [])
        n_steps = len(kl_data)
        if n_steps < 3:
            continue
        cutoff_low  = max(1, int(0.20 * n_steps))
        cutoff_high = max(cutoff_low + 1, int(0.80 * n_steps))

        for step_i, step_vals in enumerate(kl_data):
            if step_i < cutoff_low:
                target = early_vals
            elif step_i >= cutoff_high:
                target = late_vals
            else:
                continue
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    target[layer_i].append(float(v))

    def _ms(buckets):
        m = np.array([np.mean(v) if v else float("nan") for v in buckets])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                      for v in buckets])
        return m, s

    early_m, early_s = _ms(early_vals)
    late_m,  late_s  = _ms(late_vals)
    return early_m, early_s, late_m, late_s


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("kl_to_final"):
        print("  Plot 6 (Exp3) skipped — no kl_to_final data (collect_emergence=False?)")
        return

    layers = np.arange(N_LAYERS)

    primary_by_cat = _kl_layer_by_cat(results)
    kl_heatmap     = _build_kl_heatmap(results)
    commit_it      = _commitment_layers(results)
    early_it_m, early_it_s, late_it_m, late_it_s = _kl_by_generation_third(results)

    pt_by_cat    = _kl_layer_by_cat(pt_results)   if pt_results else None
    commit_pt    = _commitment_layers(pt_results) if pt_results else None

    if pt_results:
        early_pt_m, early_pt_s, late_pt_m, late_pt_s = _kl_by_generation_third(pt_results)
    else:
        early_pt_m = early_pt_s = late_pt_m = late_pt_s = None

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: mean KL per layer with SEM bands ─────────────────────────────
    ax_a.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
    ax_a.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4,
                 label=f"boundary (L{_BOUNDARY})")
    ax_a.axhline(_KL_THRESHOLD, color="green", lw=0.8, ls="--", alpha=0.5,
                 label=f"commit threshold ({_KL_THRESHOLD} nat)")

    for cat, (m, s) in primary_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax_a.plot(layers, m, lw=1.8, color=c, label=cat)
        ax_a.fill_between(layers, np.maximum(0, m - s), m + s, alpha=0.12, color=c)

    if pt_by_cat:
        for cat, (m, s) in pt_by_cat.items():
            c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
            ax_a.plot(layers, m, lw=1.4, color=c, ls="--", alpha=0.55)
            ax_a.fill_between(layers, np.maximum(0, m - s), m + s,
                              alpha=0.06, color=c)

    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean KL(layer_i ∥ final) in nats  (±SEM)")
    ax_a.set_title("Panel A — KL-to-final per layer\n"
                   "IT solid / PT dashed  |  drop = commitment")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: KL heatmap [layer × step] ───────────────────────────────────
    valid_cols = ~np.all(np.isnan(kl_heatmap), axis=0)
    hm = kl_heatmap[:, valid_cols]
    n_valid_steps = hm.shape[1]
    vmax = np.nanquantile(hm, 0.95) if hm.size > 0 else 1.0

    im = ax_b.imshow(
        hm,
        aspect="auto",
        origin="upper",
        cmap="YlOrRd",
        vmin=0, vmax=vmax,
        interpolation="nearest",
    )
    ax_b.axhline(11, color="white", lw=1.2, ls="--", alpha=0.7)
    ax_b.axhline(_BOUNDARY, color="cyan",  lw=1.0, ls=":",  alpha=0.6)
    ax_b.set_ylabel("Transformer Layer")
    ax_b.set_xlabel("Generation Step")
    ax_b.set_title(
        f"Panel B — KL heatmap [layer × step]  ({n_valid_steps} steps)\n"
        "White dashed=dip(L11), cyan=boundary(L20)"
    )
    ax_b.set_yticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax_b, label="Mean KL-to-final (nats)")

    # ── Panel C: commitment layer distribution ────────────────────────────────
    bins = np.arange(0, N_LAYERS + 2) - 0.5
    if commit_it:
        ax_c.hist(commit_it, bins=bins, density=True, alpha=0.6,
                  color="#E65100", label=f"IT  (median={np.median(commit_it):.0f})")
        ax_c.axvline(np.median(commit_it), color="#E65100", lw=2, ls="--")
    if commit_pt:
        ax_c.hist(commit_pt, bins=bins, density=True, alpha=0.6,
                  color="#1565C0", label=f"PT  (median={np.median(commit_pt):.0f})")
        ax_c.axvline(np.median(commit_pt), color="#1565C0", lw=2, ls="--")

    ax_c.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6)
    ax_c.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_c.set_xlabel("First layer where KL < threshold (commitment layer)")
    ax_c.set_ylabel("Density (over prompts × steps)")
    ax_c.set_title(
        f"Panel C — Commitment layer distribution\n"
        f"(threshold = {_KL_THRESHOLD} nats; lower = commits earlier)"
    )
    ax_c.legend(fontsize=9)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: early vs late generation KL with SEM bands ──────────────────
    ax_d.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6)
    ax_d.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)

    ax_d.plot(layers, early_it_m, lw=2, color="#E65100",
              ls="-",  label="IT early steps (first 20%)")
    ax_d.fill_between(layers,
                      np.maximum(0, early_it_m - early_it_s),
                      early_it_m + early_it_s,
                      alpha=0.15, color="#E65100")

    ax_d.plot(layers, late_it_m, lw=2, color="#E65100",
              ls="--", label="IT late steps (last 20%)")
    ax_d.fill_between(layers,
                      np.maximum(0, late_it_m - late_it_s),
                      late_it_m + late_it_s,
                      alpha=0.10, color="#E65100")

    if early_pt_m is not None:
        ax_d.plot(layers, early_pt_m, lw=1.5, color="#1565C0",
                  ls="-",  alpha=0.6, label="PT early steps")
        ax_d.fill_between(layers,
                          np.maximum(0, early_pt_m - early_pt_s),
                          early_pt_m + early_pt_s,
                          alpha=0.07, color="#1565C0")
        ax_d.plot(layers, late_pt_m, lw=1.5, color="#1565C0",
                  ls="--", alpha=0.6, label="PT late steps")
        ax_d.fill_between(layers,
                          np.maximum(0, late_pt_m - late_pt_s),
                          late_pt_m + late_pt_s,
                          alpha=0.05, color="#1565C0")

    ax_d.set_xlabel("Transformer Layer")
    ax_d.set_ylabel("Mean KL-to-final (nats)  (±SEM)")
    ax_d.set_title("Panel D — KL for early vs late generation steps\n"
                   "Does commitment depth shift as the response progresses?")
    ax_d.legend(fontsize=8, ncol=2)
    ax_d.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Exp3 Plot 6 — KL-to-Final Trajectory: When Does the Model Commit?\n"
        "IT corrective stage: model revises commitment layer by layer",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot6_kl_trajectory.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 6 (Exp3) saved → {out_path}")
