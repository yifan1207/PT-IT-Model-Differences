"""
E3.13 — Candidate-Set Reshuffling Analysis.

Distinguishes "reranking within same candidates" from "bringing in entirely new
tokens" across the corrective stage.

Requires: top5_token_ids_per_layer field (collect_top5_tokens=True / --top5-tokens flag).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.poc.shared.constants import N_LAYERS

_IT_COLOR = "#E65100"
_PT_COLOR = "#1565C0"

_sanitise_nan = lambda v: None if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def _jaccard(set_a: set, set_b: set) -> float | None:
    """Jaccard similarity of two sets; None if both empty."""
    union = set_a | set_b
    if not union:
        return None
    return len(set_a & set_b) / len(union)


# ── Panel A helpers ────────────────────────────────────────────────────────────

def _panel_a_jaccard_vs_final(results: list[dict], n_layers: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """Per-layer mean Jaccard(top5[layer_i], top5[last_layer]) across all steps."""
    buckets: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        ids_per_layer_per_step = r.get("top5_token_ids_per_layer")
        if not ids_per_layer_per_step:
            continue
        for step_layers in ids_per_layer_per_step:
            if not step_layers:
                continue
            # last available layer's top5
            last_set = set(step_layers[-1]) if step_layers[-1] is not None else set()
            for li, top5 in enumerate(step_layers[:n_layers]):
                if top5 is None:
                    continue
                j = _jaccard(set(top5), last_set)
                if j is not None:
                    buckets[li].append(j)
    m = np.array([np.mean(b) if b else float("nan") for b in buckets])
    s = np.array([np.std(b) / math.sqrt(len(b)) if len(b) > 1 else 0.0 for b in buckets])
    return m, s


# ── Panel B helpers ────────────────────────────────────────────────────────────

def _panel_b_first_entry_distribution(results: list[dict], n_layers: int = N_LAYERS) -> np.ndarray:
    """
    Per-layer fraction of steps where the generated token first enters top-5.
    Returns array of length n_layers with counts (normalised to fractions).
    """
    counts = np.zeros(n_layers, dtype=float)
    total = 0
    for r in results:
        ids_per_layer_per_step = r.get("top5_token_ids_per_layer")
        generated = r.get("generated_tokens")
        if not ids_per_layer_per_step or not generated:
            continue
        for step_i, step_layers in enumerate(ids_per_layer_per_step):
            if not step_layers or step_i >= len(generated):
                continue
            gen_tok = generated[step_i]
            if gen_tok is None:
                continue
            # support both dict and raw int
            if isinstance(gen_tok, dict):
                next_token_id = gen_tok.get("token_id")
            else:
                next_token_id = int(gen_tok)
            if next_token_id is None:
                continue
            # Find first layer where token appears in top-5
            for li, top5 in enumerate(step_layers[:n_layers]):
                if top5 is not None and next_token_id in top5:
                    counts[li] += 1
                    total += 1
                    break
    if total > 0:
        return counts / total
    return counts


# ── Panel C helpers ────────────────────────────────────────────────────────────

def _panel_c_mean_rank(results: list[dict], n_layers: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-layer mean rank of generated token within top-5 (only when present).
    Rank 0 = highest probability position.
    """
    buckets: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        ids_per_layer_per_step = r.get("top5_token_ids_per_layer")
        generated = r.get("generated_tokens")
        if not ids_per_layer_per_step or not generated:
            continue
        for step_i, step_layers in enumerate(ids_per_layer_per_step):
            if not step_layers or step_i >= len(generated):
                continue
            gen_tok = generated[step_i]
            if gen_tok is None:
                continue
            if isinstance(gen_tok, dict):
                next_token_id = gen_tok.get("token_id")
            else:
                next_token_id = int(gen_tok)
            if next_token_id is None:
                continue
            for li, top5 in enumerate(step_layers[:n_layers]):
                if top5 is None:
                    continue
                if next_token_id in top5:
                    rank = list(top5).index(next_token_id)
                    buckets[li].append(float(rank))
    m = np.array([np.mean(b) if b else float("nan") for b in buckets])
    s = np.array([np.std(b) / math.sqrt(len(b)) if len(b) > 1 else 0.0 for b in buckets])
    return m, s


# ── Panel D helpers ────────────────────────────────────────────────────────────

def _panel_d_adjacent_jaccard(results: list[dict], n_layers: int = N_LAYERS) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-layer mean Jaccard(top5[layer_i], top5[layer_{i-1}]).
    Layer 0 is skipped (no predecessor). Returns array indexed by layer.
    """
    buckets: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        ids_per_layer_per_step = r.get("top5_token_ids_per_layer")
        if not ids_per_layer_per_step:
            continue
        for step_layers in ids_per_layer_per_step:
            if not step_layers or len(step_layers) < 2:
                continue
            for li in range(1, min(len(step_layers), n_layers)):
                prev = step_layers[li - 1]
                curr = step_layers[li]
                if prev is None or curr is None:
                    continue
                j = _jaccard(set(prev), set(curr))
                if j is not None:
                    buckets[li].append(j)
    m = np.array([np.mean(b) if b else float("nan") for b in buckets])
    s = np.array([np.std(b) / math.sqrt(len(b)) if len(b) > 1 else 0.0 for b in buckets])
    return m, s


# ── Main ───────────────────────────────────────────────────────────────────────

def make_plot(results: list[dict], output_dir: str, pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Guard: check data availability
    if not results or not results[0].get("top5_token_ids_per_layer"):
        print(
            "  Plot E3.13 skipped — no top5_token_ids_per_layer data. "
            "Re-run collection with --top5-tokens flag (collect_top5_tokens=True)."
        )
        return

    has_pt_top5 = bool(pt_results and pt_results[0].get("top5_token_ids_per_layer"))

    layers = np.arange(N_LAYERS)

    # Compute per-panel data
    it_jac_final, it_jac_final_sem = _panel_a_jaccard_vs_final(results)
    pt_jac_final = pt_jac_final_sem = None
    if has_pt_top5:
        pt_jac_final, pt_jac_final_sem = _panel_a_jaccard_vs_final(pt_results)

    it_entry = _panel_b_first_entry_distribution(results)
    pt_entry = _panel_b_first_entry_distribution(pt_results) if has_pt_top5 else None

    it_rank, it_rank_sem = _panel_c_mean_rank(results)
    pt_rank = pt_rank_sem = None
    if has_pt_top5:
        pt_rank, pt_rank_sem = _panel_c_mean_rank(pt_results)

    it_adj_jac, it_adj_jac_sem = _panel_d_adjacent_jaccard(results)
    pt_adj_jac = pt_adj_jac_sem = None
    if has_pt_top5:
        pt_adj_jac, pt_adj_jac_sem = _panel_d_adjacent_jaccard(pt_results)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A ───────────────────────────────────────────────────────────────
    ax_a.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5)
    ax_a.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5)
    ax_a.plot(layers, it_jac_final, color=_IT_COLOR, lw=2, label="IT")
    ax_a.fill_between(layers, it_jac_final - it_jac_final_sem, it_jac_final + it_jac_final_sem,
                      color=_IT_COLOR, alpha=0.15)
    if pt_jac_final is not None:
        ax_a.plot(layers, pt_jac_final, color=_PT_COLOR, lw=2, ls="--", label="PT")
        ax_a.fill_between(layers, pt_jac_final - pt_jac_final_sem, pt_jac_final + pt_jac_final_sem,
                          color=_PT_COLOR, alpha=0.12)
    ax_a.set_title("Panel A — Top-5 Set Jaccard vs Final Layer")
    ax_a.set_xlabel("Layer")
    ax_a.set_ylabel("Jaccard(top5[layer_i], top5[last_layer])")
    ax_a.set_ylim(0, 1.05)
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend(fontsize=8)

    # ── Panel B ───────────────────────────────────────────────────────────────
    ax_b.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5)
    ax_b.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5)
    ax_b.bar(layers, it_entry, color=_IT_COLOR, alpha=0.75, label="IT")
    if pt_entry is not None:
        ax_b.step(layers, pt_entry, color=_PT_COLOR, lw=2, where="mid", label="PT")
    ax_b.set_title("Panel B — Layer of First Top-5 Entry for Generated Token")
    ax_b.set_xlabel("Layer")
    ax_b.set_ylabel("Fraction of steps")
    ax_b.grid(axis="y", alpha=0.25)
    ax_b.legend(fontsize=8)

    # ── Panel C ───────────────────────────────────────────────────────────────
    ax_c.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5)
    ax_c.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5)
    ax_c.plot(layers, it_rank, color=_IT_COLOR, lw=2, label="IT")
    ax_c.fill_between(layers, it_rank - it_rank_sem, it_rank + it_rank_sem, color=_IT_COLOR, alpha=0.15)
    if pt_rank is not None:
        ax_c.plot(layers, pt_rank, color=_PT_COLOR, lw=2, ls="--", label="PT")
        ax_c.fill_between(layers, pt_rank - pt_rank_sem, pt_rank + pt_rank_sem, color=_PT_COLOR, alpha=0.12)
    ax_c.set_title("Panel C — Intra-Top-5 Rank of Generated Token Per Layer")
    ax_c.set_xlabel("Layer")
    ax_c.set_ylabel("Mean rank within top-5 (0 = highest prob)")
    ax_c.grid(axis="y", alpha=0.25)
    ax_c.legend(fontsize=8)

    # ── Panel D ───────────────────────────────────────────────────────────────
    ax_d.axvline(11, color="grey", lw=0.9, ls="--", alpha=0.5)
    ax_d.axvline(20, color="black", lw=0.9, ls="--", alpha=0.5)
    ax_d.plot(layers, it_adj_jac, color=_IT_COLOR, lw=2, label="IT")
    ax_d.fill_between(layers, it_adj_jac - it_adj_jac_sem, it_adj_jac + it_adj_jac_sem,
                      color=_IT_COLOR, alpha=0.15)
    if pt_adj_jac is not None:
        ax_d.plot(layers, pt_adj_jac, color=_PT_COLOR, lw=2, ls="--", label="PT")
        ax_d.fill_between(layers, pt_adj_jac - pt_adj_jac_sem, pt_adj_jac + pt_adj_jac_sem,
                          color=_PT_COLOR, alpha=0.12)
    ax_d.set_title("Panel D — Adjacent-Layer Top-5 Jaccard (Set Stability)")
    ax_d.set_xlabel("Layer")
    ax_d.set_ylabel("Jaccard(top5[i], top5[i-1])")
    ax_d.set_ylim(0, 1.05)
    ax_d.grid(axis="y", alpha=0.25)
    ax_d.legend(fontsize=8)

    fig.suptitle(
        "E3.13 — Candidate-Set Reshuffling Analysis\n"
        "Distinguishes reranking within candidates from introducing new tokens\n"
        "Requires --top5-tokens flag during collection",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_13_reshuffling.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.13 saved → {out_path}")
