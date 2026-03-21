"""
E3.9 — Step-to-Step Stability Analysis.

Uses step_to_step_kl[step][layer] to test whether the corrective stage behaves
like a fixed transform across generation steps or adapts to context.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.poc.shared.constants import N_LAYERS

_PHASES = {
    "content (0-11)": range(0, 12),
    "format (12-19)": range(12, 20),
    "corrective (20-33)": range(20, 34),
}


def _has_metric(results: list[dict], key: str) -> bool:
    return bool(results) and key in results[0] and bool(results[0].get(key))


def _layer_mean_sem(results: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    vals = [[] for _ in range(N_LAYERS)]
    for r in results:
        for step_i, step_vals in enumerate(r.get(key, [])):
            if step_i == 0:
                continue
            for layer_i, v in enumerate(step_vals[:N_LAYERS]):
                if v is not None and not math.isnan(float(v)):
                    vals[layer_i].append(float(v))
    means = np.array([np.mean(v) if v else float("nan") for v in vals], dtype=float)
    sems = np.array(
        [np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals],
        dtype=float,
    )
    return means, sems


def _phase_means(results: list[dict], key: str) -> dict[str, float]:
    out = {}
    for label, layers in _PHASES.items():
        vals = []
        for r in results:
            for step_i, step_vals in enumerate(r.get(key, [])):
                if step_i == 0:
                    continue
                for layer_i in layers:
                    if layer_i < len(step_vals):
                        v = step_vals[layer_i]
                        if v is not None and not math.isnan(float(v)):
                            vals.append(float(v))
        out[label] = float(np.mean(vals)) if vals else float("nan")
    return out


def _corrective_step_curve(results: list[dict], key: str) -> np.ndarray:
    max_steps = max((len(r.get(key, [])) for r in results), default=0)
    sums = np.zeros(max_steps, dtype=float)
    counts = np.zeros(max_steps, dtype=np.int32)
    for r in results:
        for step_i, step_vals in enumerate(r.get(key, [])):
            if step_i == 0:
                continue
            vals = []
            for layer_i in _PHASES["corrective (20-33)"]:
                if layer_i < len(step_vals):
                    v = step_vals[layer_i]
                    if v is not None and not math.isnan(float(v)):
                        vals.append(float(v))
            if vals:
                sums[step_i] += float(np.mean(vals))
                counts[step_i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def make_plot(results: list[dict], output_dir: str, pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not _has_metric(results, "step_to_step_kl"):
        print("  Plot E3.9 skipped — no step_to_step_kl data. Re-run collection with updated collector.")
        return

    layers = np.arange(N_LAYERS)
    it_mean, it_sem = _layer_mean_sem(results, "step_to_step_kl")
    pt_mean = pt_sem = None
    if pt_results and _has_metric(pt_results, "step_to_step_kl"):
        pt_mean, pt_sem = _layer_mean_sem(pt_results, "step_to_step_kl")

    it_phase = _phase_means(results, "step_to_step_kl")
    pt_phase = _phase_means(pt_results or [], "step_to_step_kl") if pt_results else {}
    it_curve = _corrective_step_curve(results, "step_to_step_kl")
    pt_curve = _corrective_step_curve(pt_results or [], "step_to_step_kl") if pt_results else np.array([])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    ax_a, ax_b, ax_c = axes

    ax_a.axvline(20, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_a.plot(layers, it_mean, color="#F57C00", lw=2, label="IT")
    ax_a.fill_between(layers, it_mean - it_sem, it_mean + it_sem, color="#F57C00", alpha=0.14)
    if pt_mean is not None:
        ax_a.plot(layers, pt_mean, color="#1976D2", lw=2, label="PT")
        ax_a.fill_between(layers, pt_mean - pt_sem, pt_mean + pt_sem, color="#1976D2", alpha=0.12)
    ax_a.set_title("Panel A — Mean step-to-step KL by layer")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("KL(step t || step t-1)")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend()

    phase_labels = list(_PHASES.keys())
    x = np.arange(len(phase_labels))
    w = 0.34
    ax_b.bar(x - w / 2, [it_phase[p] for p in phase_labels], width=w, color="#F57C00", label="IT")
    if pt_results:
        ax_b.bar(x + w / 2, [pt_phase.get(p, float("nan")) for p in phase_labels], width=w, color="#1976D2", label="PT")
    ax_b.set_xticks(x, phase_labels, rotation=15, ha="right")
    ax_b.set_ylabel("Mean KL(step t || step t-1)")
    ax_b.set_title("Panel B — Stability by phase")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend()

    step_x = np.arange(len(it_curve))
    ax_c.plot(step_x, it_curve, color="#F57C00", lw=2, label="IT")
    if pt_curve.size:
        ax_c.plot(np.arange(len(pt_curve)), pt_curve, color="#1976D2", lw=2, label="PT")
    ax_c.set_xlim(left=1 if len(step_x) > 1 else 0)
    ax_c.set_xlabel("Generation step")
    ax_c.set_ylabel("Mean corrective-stage step-to-step KL")
    ax_c.set_title("Panel C — Corrective-stage stability over generation")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend()

    fig.suptitle(
        "E3.9 — Step-to-Step Stability Analysis\n"
        "Lower corrective-stage KL implies a more fixed transform; higher KL implies context adaptation.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_9_step_stability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.9 saved → {out_path}")
