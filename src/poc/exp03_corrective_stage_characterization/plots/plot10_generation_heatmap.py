"""
Plot 10 (Exp3): Generation-step resolved corrective stage (Experiment 2d).

The corrective stage (layers 20–33) operates on every generated token.
But is it equally active throughout the response, or does it "fade out" once
the format is established?

This plot maps the full (layer × generation_step) space of each metric,
showing how the model's internal dynamics evolve as it writes more tokens.

Four panels:
  A: Heatmap of mean layer_delta_cosine [layer × step] for IT.
     Red = layer pushes AGAINST the residual stream (suppressive).
     Blue = layer reinforces the residual stream.
     Key question: does the corrective region (rows 20–33) stay red throughout?
  B: Same heatmap for PT.
     If PT corrective region is much less red, post-training added suppression.
  C: Mean cosine in corrective layers (20–33) over generation steps with SEM band.
     Does IT's corrective suppression increase or decrease over time?
     PT dashed overlay.
  D: Standard deviation of cosine in corrective layers over generation steps.
     High variance = model is selectively suppressive (only some tokens).
     Low variance = uniform suppression regardless of what's being generated.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.poc.shared.constants import N_LAYERS

_BOUNDARY     = 20
_STEP_CAP     = 120


def _build_cosine_heatmap(results: list[dict],
                           step_cap:  int = _STEP_CAP,
                           n_layers:  int = N_LAYERS) -> np.ndarray:
    """[n_layers, step_cap] mean cosine heatmap."""
    sums   = np.zeros((n_layers, step_cap), dtype=np.float64)
    counts = np.zeros((n_layers, step_cap), dtype=np.int32)
    for r in results:
        for step_i, step_vals in enumerate(r.get("layer_delta_cosine", [])):
            if step_i >= step_cap:
                break
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    sums[layer_i, step_i]   += float(v)
                    counts[layer_i, step_i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _corrective_curve_with_sem(
        results: list[dict],
        boundary: int = _BOUNDARY,
        step_cap: int = _STEP_CAP,
        n_layers: int = N_LAYERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step mean, SEM, and std of corrective-layer cosine across prompts.

    For each step, we compute the mean cosine across corrective layers PER PROMPT,
    then take mean and SEM over prompts.  This gives prompt-level variance, which
    is the right unit of analysis (each prompt is an independent observation).

    Returns (mean, sem, std) each [step_cap].
    """
    # per-step, per-prompt mean cosine in corrective layers
    # step_means[step_i] = list of per-prompt means at that step
    step_means: list[list[float]] = [[] for _ in range(step_cap)]

    for r in results:
        for step_i, step_vals in enumerate(r.get("layer_delta_cosine", [])):
            if step_i >= step_cap:
                break
            corr_vals = [
                float(step_vals[li])
                for li in range(boundary, min(n_layers, len(step_vals)))
                if step_vals[li] is not None and not math.isnan(float(step_vals[li]))
            ]
            if corr_vals:
                step_means[step_i].append(float(np.mean(corr_vals)))

    mean_arr = np.full(step_cap, float("nan"))
    sem_arr  = np.full(step_cap, float("nan"))
    std_arr  = np.full(step_cap, float("nan"))

    for step_i, vals in enumerate(step_means):
        if len(vals) < 1:
            continue
        a = np.array(vals)
        mean_arr[step_i] = float(a.mean())
        std_arr[step_i]  = float(a.std()) if len(vals) > 1 else 0.0
        sem_arr[step_i]  = float(a.std() / math.sqrt(len(a))) if len(vals) > 1 else 0.0

    return mean_arr, sem_arr, std_arr


def _plot_heatmap(ax, hm: np.ndarray, title: str) -> None:
    valid_cols = ~np.all(np.isnan(hm), axis=0)
    hm_v       = hm[:, valid_cols]
    n_valid    = hm_v.shape[1]

    vabs = np.nanquantile(np.abs(hm_v), 0.95) if hm_v.size > 0 else 1.0

    im = ax.imshow(
        hm_v,
        aspect="auto",
        origin="upper",
        cmap="RdBu",
        vmin=-vabs, vmax=vabs,
        interpolation="nearest",
    )
    ax.axhline(_BOUNDARY - 0.5, color="lime", lw=1.5, ls="--", alpha=0.7,
               label=f"boundary (L{_BOUNDARY})")
    ax.axhline(11 - 0.5, color="white", lw=1.0, ls=":", alpha=0.6)
    ax.set_ylabel("Transformer Layer")
    ax.set_xlabel("Generation Step")
    ax.set_title(f"{title}  ({n_valid} steps shown)\n"
                 "Red = layer opposes residual | Blue = layer reinforces")
    ax.set_yticks(range(0, N_LAYERS, 4))
    plt.colorbar(im, ax=ax, label="Mean cos(δ_i, h_{i−1})")


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or "layer_delta_cosine" not in results[0]:
        print("  Plot 10 (Exp3) skipped — no layer_delta_cosine data")
        return

    it_hm                      = _build_cosine_heatmap(results)
    it_mean, it_sem, it_std    = _corrective_curve_with_sem(results)

    pt_hm                      = _build_cosine_heatmap(pt_results) if pt_results else None
    pt_mean, pt_sem, _pt_std   = (
        _corrective_curve_with_sem(pt_results) if pt_results else (None, None, None)
    )

    steps = np.arange(_STEP_CAP)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: IT cosine heatmap ─────────────────────────────────────────────
    _plot_heatmap(ax_a, it_hm, "Panel A — IT cosine heatmap [layer × step]")

    # ── Panel B: PT cosine heatmap ─────────────────────────────────────────────
    if pt_hm is not None:
        _plot_heatmap(ax_b, pt_hm, "Panel B — PT cosine heatmap [layer × step]")
    else:
        ax_b.text(0.5, 0.5, "PT results not provided",
                  ha="center", va="center", transform=ax_b.transAxes, fontsize=12)
        ax_b.set_title("Panel B — PT cosine heatmap (unavailable)")

    # ── Panel C: corrective-stage mean cosine over steps with SEM band ─────────
    ax_c.axhline(0, color="black", lw=0.7)
    valid_it = ~np.isnan(it_mean)
    ax_c.plot(steps[valid_it], it_mean[valid_it],
              lw=2, color="#E65100", label="IT corrective mean")
    ax_c.fill_between(steps[valid_it],
                      it_mean[valid_it] - it_sem[valid_it],
                      it_mean[valid_it] + it_sem[valid_it],
                      alpha=0.2, color="#E65100")

    if pt_mean is not None:
        valid_pt = ~np.isnan(pt_mean)
        ax_c.plot(steps[valid_pt], pt_mean[valid_pt],
                  lw=1.8, color="#1565C0", ls="--", alpha=0.7, label="PT corrective mean")
        ax_c.fill_between(steps[valid_pt],
                          pt_mean[valid_pt] - pt_sem[valid_pt],
                          pt_mean[valid_pt] + pt_sem[valid_pt],
                          alpha=0.12, color="#1565C0")

    ax_c.set_xlabel("Generation Step")
    ax_c.set_ylabel("Mean cos(δ, h_{prev}) in corrective layers  (±SEM)")
    ax_c.set_title("Panel C — Corrective suppression over generation steps\n"
                   "Sustained negative = consistently opposing; drift = fades out")
    ax_c.legend(fontsize=9)
    ax_c.grid(alpha=0.2)

    # ── Panel D: corrective-stage std cosine over generation steps ────────────
    valid_std = ~np.isnan(it_std)
    ax_d.plot(steps[valid_std], it_std[valid_std],
              lw=2, color="#E65100", label="IT corrective std (across prompts)")
    ax_d.set_xlabel("Generation Step")
    ax_d.set_ylabel("Std of per-prompt corrective cosine")
    ax_d.set_title("Panel D — Variance of corrective suppression across prompts per step\n"
                   "High std = selective (some prompts corrected more than others at this step)")
    ax_d.legend(fontsize=9)
    ax_d.grid(alpha=0.2)

    fig.suptitle(
        "Exp3 Plot 10 — Generation-Step Resolved Corrective Stage  (Exp 2d)\n"
        "Is the corrective stage active throughout generation, or only early on?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot10_generation_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 10 (Exp3) saved → {out_path}")
