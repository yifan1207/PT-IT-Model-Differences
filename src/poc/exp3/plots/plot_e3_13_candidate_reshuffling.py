"""
E3.13 — Candidate Set Reshuffling.
"""
from __future__ import annotations

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
    return bool(results) and bool(results[0].get("top5_token_ids_per_layer"))


def _jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return float("nan")
    return len(sa & sb) / len(sa | sb)


def _layer_overlap(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    vals = [[] for _ in range(N_LAYERS)]
    for r in results:
        for step in r.get("top5_token_ids_per_layer", []):
            for layer_i in range(1, min(len(step), N_LAYERS)):
                vals[layer_i].append(_jaccard(step[layer_i], step[layer_i - 1]))
    means = np.array([np.mean(v) if v else float("nan") for v in vals], dtype=float)
    sems = np.array([np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0.0 for v in vals], dtype=float)
    return means, sems


def _phase_overlap(results: list[dict]) -> dict[str, float]:
    out = {}
    for label, layers in _PHASES.items():
        vals = []
        for r in results:
            for step in r.get("top5_token_ids_per_layer", []):
                for layer_i in layers:
                    if layer_i < len(step):
                        vals.append(_jaccard(step[layer_i], step[layer_i - 1]))
        out[label] = float(np.mean(vals)) if vals else float("nan")
    return out


def _corrective_curve(results: list[dict]) -> np.ndarray:
    max_steps = max((len(r.get("top5_token_ids_per_layer", [])) for r in results), default=0)
    sums = np.zeros(max_steps, dtype=float)
    counts = np.zeros(max_steps, dtype=np.int32)
    for r in results:
        for step_i, step in enumerate(r.get("top5_token_ids_per_layer", [])):
            vals = []
            for layer_i in _PHASES["corrective"]:
                if layer_i < len(step):
                    vals.append(_jaccard(step[layer_i], step[layer_i - 1]))
            if vals:
                sums[step_i] += float(np.mean(vals))
                counts[step_i] += 1
    with np.errstate(invalid="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def make_plot(results: list[dict], output_dir: str, pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not _has_metric(results):
        print("  Plot E3.13 skipped — no top5_token_ids_per_layer data.")
        return

    layers = np.arange(N_LAYERS)
    it_mean, it_sem = _layer_overlap(results)
    pt_mean = pt_sem = None
    if pt_results and _has_metric(pt_results):
        pt_mean, pt_sem = _layer_overlap(pt_results)
    it_phase = _phase_overlap(results)
    pt_phase = _phase_overlap(pt_results or []) if pt_results else {}
    it_curve = _corrective_curve(results)
    pt_curve = _corrective_curve(pt_results or []) if pt_results else np.array([])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))
    ax_a, ax_b, ax_c = axes

    ax_a.axvline(20, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_a.plot(layers, it_mean, color="#F57C00", lw=2, label="IT")
    ax_a.fill_between(layers, it_mean - it_sem, it_mean + it_sem, color="#F57C00", alpha=0.14)
    if pt_mean is not None:
        ax_a.plot(layers, pt_mean, color="#1976D2", lw=2, label="PT")
        ax_a.fill_between(layers, pt_mean - pt_sem, pt_mean + pt_sem, color="#1976D2", alpha=0.12)
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Top-5 Jaccard(layer_i, layer_i-1)")
    ax_a.set_title("Panel A — Candidate-set overlap by layer")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend()

    labels = list(_PHASES.keys())
    x = np.arange(len(labels))
    w = 0.34
    ax_b.bar(x - w / 2, [it_phase[k] for k in labels], width=w, color="#F57C00", label="IT")
    if pt_results:
        ax_b.bar(x + w / 2, [pt_phase.get(k, float("nan")) for k in labels], width=w, color="#1976D2", label="PT")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("Mean top-5 overlap")
    ax_b.set_title("Panel B — Candidate reshuffling by phase")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend()

    ax_c.plot(np.arange(len(it_curve)), it_curve, color="#F57C00", lw=2, label="IT")
    if pt_curve.size:
        ax_c.plot(np.arange(len(pt_curve)), pt_curve, color="#1976D2", lw=2, label="PT")
    ax_c.set_xlabel("Generation step")
    ax_c.set_ylabel("Mean corrective-stage top-5 overlap")
    ax_c.set_title("Panel C — Corrective-stage candidate stability")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend()

    fig.suptitle(
        "E3.13 — Candidate Set Reshuffling\n"
        "Lower overlap means the model introduces new candidates rather than only reranking.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_13_candidate_reshuffling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.13 saved → {out_path}")
