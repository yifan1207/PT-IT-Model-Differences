"""
E3.11 — Feature Importance Ranking.

Uses compact feature-importance summaries collected during generation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_CORRECTIVE_START = 20


def _load_summary(path: str) -> dict[int, dict[str, np.ndarray]] | None:
    p = Path(path)
    if not p.exists():
        return None
    raw = np.load(p, allow_pickle=True)
    out: dict[int, dict[str, np.ndarray]] = {}
    for key in raw.files:
        prefix, _, layer_str = key.partition("_l")
        if not layer_str:
            continue
        layer = int(layer_str)
        out.setdefault(layer, {})[prefix] = raw[key]
    return out


def _rank_features(summary: dict[int, dict[str, np.ndarray]], top_k: int = 15) -> list[tuple[str, float, int, float]]:
    rows: list[tuple[str, float, int, float]] = []
    for layer_i in range(_CORRECTIVE_START, 34):
        if layer_i not in summary or "sum" not in summary[layer_i] or "count" not in summary[layer_i]:
            continue
        sums = summary[layer_i]["sum"].astype(float)
        counts = summary[layer_i]["count"].astype(int)
        active = np.flatnonzero(sums > 0)
        for feat_idx in active:
            total = float(sums[feat_idx])
            count = int(counts[feat_idx])
            mean = total / count if count > 0 else 0.0
            rows.append((f"L{layer_i}:F{feat_idx}", total, count, mean))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_k]


def _layer_mass(summary: dict[int, dict[str, np.ndarray]]) -> np.ndarray:
    vals = np.zeros(34, dtype=float)
    for layer_i in range(34):
        layer = summary.get(layer_i, {})
        sums = layer.get("sum")
        if sums is not None:
            vals[layer_i] = float(np.sum(sums))
    return vals


def _plot_rank_panel(ax, rows: list[tuple[str, float, int, float]], title: str, color: str) -> None:
    if not rows:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No feature summary data", ha="center", va="center")
        ax.axis("off")
        return
    labels = [r[0] for r in rows][::-1]
    scores = [r[1] for r in rows][::-1]
    ax.barh(np.arange(len(rows)), scores, color=color, alpha=0.85)
    ax.set_yticks(np.arange(len(rows)), labels)
    ax.set_xlabel("Importance score = activation_sum")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.2)


def make_plot(summary_path: str, output_dir: str, pt_summary_path: str | None = None, top_k: int = 15) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    it_summary = _load_summary(summary_path)
    if it_summary is None:
        print(f"  Plot E3.11 skipped — feature importance summary not found: {summary_path}")
        return
    pt_summary = _load_summary(pt_summary_path) if pt_summary_path else None

    it_rows = _rank_features(it_summary, top_k=top_k)
    pt_rows = _rank_features(pt_summary, top_k=top_k) if pt_summary else []
    it_mass = _layer_mass(it_summary)
    pt_mass = _layer_mass(pt_summary) if pt_summary else None

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_a, ax_b, ax_c = axes
    _plot_rank_panel(ax_a, it_rows, "Panel A — Top IT corrective-layer features", "#F57C00")
    _plot_rank_panel(ax_b, pt_rows, "Panel B — Top PT corrective-layer features", "#1976D2")

    layers = np.arange(34)
    ax_c.axvspan(_CORRECTIVE_START - 0.5, 33.5, color="#EEEEEE", alpha=0.7)
    ax_c.plot(layers, it_mass, color="#F57C00", lw=2, label="IT")
    if pt_mass is not None:
        ax_c.plot(layers, pt_mass, color="#1976D2", lw=2, label="PT")
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel("Total activation_sum")
    ax_c.set_title("Panel C — Layerwise importance mass")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend()

    fig.suptitle(
        "E3.11 — Feature Importance Ranking\n"
        "Ranking features by activation magnitude × frequency using compact run summaries.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_11_feature_importance.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.11 saved → {out_path}")
