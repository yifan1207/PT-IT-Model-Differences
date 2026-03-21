"""
E3.12 — Adjacent-Layer KL Landscape.

Maps where the model makes its biggest "mind changes" across layers by plotting
KL(lens_i || lens_{i-1}) per layer, split by phase and model variant.

Requires: kl_adjacent_layer field in results (collect_layer_extras=True, default).
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.poc.shared.constants import N_LAYERS
from src.poc.shared.plot_colors import SPLIT_COLORS, _DEFAULT_COLOR

_IT_COLOR = "#E65100"
_PT_COLOR = "#1565C0"

_PHASES = {
    "proposal\n(0-11)": range(0, 12),
    "format\n(12-19)": range(12, 20),
    "corrective\n(20-33)": range(20, 34),
}

_sanitise_nan = lambda v: None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def _mean_sem_per_layer(results: list[dict], n_layers: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_array, sem_array) of kl_adjacent_layer across all steps."""
    buckets: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        for step_kl in r.get("kl_adjacent_layer", []):
            for li, v in enumerate(step_kl[:n_layers]):
                sv = _sanitise_nan(v if not isinstance(v, float) else v)
                if v is not None and not (isinstance(v, float) and math.isnan(float(v))):
                    try:
                        buckets[li].append(float(v))
                    except (TypeError, ValueError):
                        pass
    m = np.array([np.mean(b) if b else float("nan") for b in buckets])
    s = np.array([np.std(b) / math.sqrt(len(b)) if len(b) > 1 else 0.0 for b in buckets])
    return m, s


def _mean_sem_per_layer_by_split(
    results: list[dict], n_layers: int = N_LAYERS
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns per-split (mean, sem) arrays."""
    split_buckets: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n_layers)])
    for r in results:
        split = r.get("split") or r.get("category", "?")
        for step_kl in r.get("kl_adjacent_layer", []):
            for li, v in enumerate(step_kl[:n_layers]):
                if v is not None and not (isinstance(v, float) and math.isnan(float(v))):
                    try:
                        split_buckets[split][li].append(float(v))
                    except (TypeError, ValueError):
                        pass
    out = {}
    for split, buckets in split_buckets.items():
        m = np.array([np.mean(b) if b else float("nan") for b in buckets])
        s = np.array([np.std(b) / math.sqrt(len(b)) if len(b) > 1 else 0.0 for b in buckets])
        out[split] = (m, s)
    return out


def _phase_mean_sem(
    results: list[dict], n_layers: int = N_LAYERS
) -> dict[str, tuple[float, float]]:
    """Returns per-phase (mean, sem) across all steps and records."""
    phase_buckets: dict[str, list[float]] = {label: [] for label in _PHASES}
    for r in results:
        for step_kl in r.get("kl_adjacent_layer", []):
            for label, layers in _PHASES.items():
                for li in layers:
                    if li < len(step_kl):
                        v = step_kl[li]
                        if v is not None and not (isinstance(v, float) and math.isnan(float(v))):
                            try:
                                phase_buckets[label].append(float(v))
                            except (TypeError, ValueError):
                                pass
    out = {}
    for label, vals in phase_buckets.items():
        if vals:
            out[label] = (float(np.mean(vals)), float(np.std(vals) / math.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        else:
            out[label] = (float("nan"), 0.0)
    return out


def make_plot(results: list[dict], output_dir: str, pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guard: check data availability
    if not results or not results[0].get("kl_adjacent_layer"):
        print("  Plot E3.12 skipped — no kl_adjacent_layer data. Re-run collection with collect_layer_extras=True.")
        return

    layers = np.arange(N_LAYERS)

    it_mean, it_sem = _mean_sem_per_layer(results)
    pt_mean: np.ndarray | None = None
    pt_sem: np.ndarray | None = None
    if pt_results and pt_results[0].get("kl_adjacent_layer"):
        pt_mean, pt_sem = _mean_sem_per_layer(pt_results)

    it_by_split = _mean_sem_per_layer_by_split(results)

    it_phase = _phase_mean_sem(results)
    pt_phase = _phase_mean_sem(pt_results) if pt_results and pt_results[0].get("kl_adjacent_layer") else {}

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: IT vs PT mean KL per layer ──────────────────────────────────
    ax_a.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5, label="Layer 11")
    ax_a.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5, label="Layer 20 (corrective)")
    ax_a.plot(layers, it_mean, color=_IT_COLOR, lw=2, label="IT")
    ax_a.fill_between(layers, it_mean - it_sem, it_mean + it_sem, color=_IT_COLOR, alpha=0.15)
    if pt_mean is not None:
        ax_a.plot(layers, pt_mean, color=_PT_COLOR, lw=2, ls="--", label="PT")
        ax_a.fill_between(layers, pt_mean - pt_sem, pt_mean + pt_sem, color=_PT_COLOR, alpha=0.12)
    ax_a.set_title("Panel A — Layer-to-Layer KL: IT vs PT")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Mean KL(lens_i ∥ lens_{i-1}) [nats]")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(fontsize=8)

    # ── Panel B: IT by split ──────────────────────────────────────────────────
    ax_b.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5)
    ax_b.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5)
    for split, (m, s) in sorted(it_by_split.items()):
        color = SPLIT_COLORS.get(split, _DEFAULT_COLOR)
        ax_b.plot(layers, m, color=color, lw=1.5, label=split)
        ax_b.fill_between(layers, m - s, m + s, color=color, alpha=0.10)
    ax_b.set_title("Panel B — Layer-to-Layer KL by Split (IT)")
    ax_b.set_xlabel("Layer")
    ax_b.set_ylabel("Mean KL(lens_i ∥ lens_{i-1}) [nats]")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(fontsize=8)

    # ── Panel C: IT - PT difference per layer ────────────────────────────────
    ax_c.axhline(0, color="black", lw=0.8, ls="--", alpha=0.6)
    ax_c.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5)
    ax_c.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5)
    if pt_mean is not None:
        diff = it_mean - pt_mean
        ax_c.plot(layers, diff, color=_IT_COLOR, lw=2)
        ax_c.fill_between(layers, diff, 0, where=diff >= 0, color=_IT_COLOR, alpha=0.15, label="IT > PT")
        ax_c.fill_between(layers, diff, 0, where=diff < 0, color=_PT_COLOR, alpha=0.15, label="PT > IT")
        ax_c.legend(fontsize=8)
    else:
        ax_c.text(0.5, 0.5, "No PT data available", transform=ax_c.transAxes, ha="center", va="center", color="grey")
    ax_c.set_title("Panel C — IT minus PT KL Difference Per Layer")
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel("KL difference (IT − PT) [nats]")
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: Phase-level grouped bar chart ────────────────────────────────
    phase_labels = list(_PHASES.keys())
    x = np.arange(len(phase_labels))
    w = 0.35
    it_vals = [it_phase[p][0] for p in phase_labels]
    it_errs = [it_phase[p][1] for p in phase_labels]
    ax_d.bar(x - w / 2, it_vals, width=w, color=_IT_COLOR, label="IT",
             yerr=it_errs, capsize=4, error_kw={"elinewidth": 1.2})
    if pt_phase:
        pt_vals = [pt_phase.get(p, (float("nan"), 0.0))[0] for p in phase_labels]
        pt_errs = [pt_phase.get(p, (float("nan"), 0.0))[1] for p in phase_labels]
        ax_d.bar(x + w / 2, pt_vals, width=w, color=_PT_COLOR, label="PT",
                 yerr=pt_errs, capsize=4, error_kw={"elinewidth": 1.2})
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(phase_labels, fontsize=9)
    ax_d.set_ylabel("Mean KL(lens_i ∥ lens_{i-1}) [nats]")
    ax_d.set_title("Panel D — Phase-Level Mean Adjacent KL")
    ax_d.grid(axis="y", alpha=0.25)
    ax_d.legend(fontsize=8)

    fig.suptitle(
        "E3.12 — Adjacent-Layer KL Landscape\n"
        "Maps where model makes biggest 'mind changes' across layers",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_12_kl_landscape.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.12 saved → {out_path}")
