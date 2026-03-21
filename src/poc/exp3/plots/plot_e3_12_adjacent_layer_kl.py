"""
E3.12 — Adjacent-Layer KL Landscape.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.poc.shared.constants import N_LAYERS

_PHASES = {
    "content": range(1, 12),
    "format": range(12, 20),
    "corrective": range(20, 34),
}


def _has_metric(results: list[dict]) -> bool:
    return bool(results) and bool(results[0].get("kl_adjacent_layer"))


def _layer_mean_sem(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    vals = [[] for _ in range(N_LAYERS)]
    for r in results:
        for step_vals in r.get("kl_adjacent_layer", []):
            for layer_i, v in enumerate(step_vals[:N_LAYERS]):
                if v is not None and not math.isnan(float(v)):
                    vals[layer_i].append(float(v))
    means = np.array([np.mean(v) if v else float("nan") for v in vals], dtype=float)
    sems = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals], dtype=float)
    return means, sems


def _phase_means(results: list[dict]) -> dict[str, float]:
    out = {}
    for label, layers in _PHASES.items():
        vals = []
        for r in results:
            for step_vals in r.get("kl_adjacent_layer", []):
                for layer_i in layers:
                    if layer_i < len(step_vals):
                        v = step_vals[layer_i]
                        if v is not None and not math.isnan(float(v)):
                            vals.append(float(v))
        out[label] = float(np.mean(vals)) if vals else float("nan")
    return out


def make_plot(results: list[dict], output_dir: str, pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not _has_metric(results):
        print("  Plot E3.12 skipped — no kl_adjacent_layer data.")
        return

    layers = np.arange(N_LAYERS)
    it_mean, it_sem = _layer_mean_sem(results)
    pt_mean = pt_sem = None
    if pt_results and _has_metric(pt_results):
        pt_mean, pt_sem = _layer_mean_sem(pt_results)
    it_phase = _phase_means(results)
    pt_phase = _phase_means(pt_results or []) if pt_results else {}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    ax_a, ax_b = axes

    ax_a.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_a.axvline(20, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_a.plot(layers, it_mean, color="#F57C00", lw=2, label="IT")
    ax_a.fill_between(layers, it_mean - it_sem, it_mean + it_sem, color="#F57C00", alpha=0.14)
    if pt_mean is not None:
        ax_a.plot(layers, pt_mean, color="#1976D2", lw=2, label="PT")
        ax_a.fill_between(layers, pt_mean - pt_sem, pt_mean + pt_sem, color="#1976D2", alpha=0.12)
    peak_layer = int(np.nanargmax(it_mean))
    ax_a.annotate(f"IT peak L{peak_layer}", xy=(peak_layer, it_mean[peak_layer]),
                  xytext=(peak_layer + 1, it_mean[peak_layer] * 0.9),
                  arrowprops=dict(arrowstyle="->", color="black"), fontsize=8)
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Mean KL(layer_i || layer_i-1)")
    ax_a.set_title("Panel A — Adjacent-layer KL across the network")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend()

    labels = list(_PHASES.keys())
    x = np.arange(len(labels))
    w = 0.34
    ax_b.bar(x - w / 2, [it_phase[k] for k in labels], width=w, color="#F57C00", label="IT")
    if pt_results:
        ax_b.bar(x + w / 2, [pt_phase.get(k, float("nan")) for k in labels], width=w, color="#1976D2", label="PT")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("Mean adjacent-layer KL")
    ax_b.set_title("Panel B — Decision magnitude by phase")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend()

    fig.suptitle(
        "E3.12 — Adjacent-Layer KL Landscape\n"
        "Maps where the network changes its prediction distribution most strongly.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_12_adjacent_layer_kl.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.12 saved → {out_path}")
