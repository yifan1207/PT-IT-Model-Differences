"""
E3.10 — Mind Change Analysis.

Uses the first layer where the final generated token becomes top-1 under the
logit lens. This directly localises where the model commits to the token it
actually outputs.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_PHASE_BINS = {
    "content": range(0, 12),
    "format": range(12, 20),
    "corrective": range(20, 34),
}


def _has_required(results: list[dict]) -> bool:
    return (
        bool(results)
        and bool(results[0].get("top1_token_per_layer"))
        and bool(results[0].get("kl_adjacent_layer"))
        and bool(results[0].get("generated_tokens"))
    )


def _phase_for_layer(layer_i: int) -> str:
    for label, layers in _PHASE_BINS.items():
        if layer_i in layers:
            return label
    return "corrective"


def _mind_change_stats(results: list[dict]) -> tuple[dict[str, float], dict[str, float], dict[str, int]]:
    phase_counts = {k: 0 for k in _PHASE_BINS}
    phase_kl = {k: [] for k in _PHASE_BINS}
    corrected_tokens = Counter()
    total = 0

    for r in results:
        top1_steps = r.get("top1_token_per_layer", [])
        kl_steps = r.get("kl_adjacent_layer", [])
        generated = r.get("generated_tokens", [])
        for step_i, (step_top1, step_kl) in enumerate(zip(top1_steps, kl_steps)):
            if step_i >= len(generated):
                continue
            final_token_id = generated[step_i].get("token_id")
            final_token_str = generated[step_i].get("token_str", str(final_token_id))
            first_match_layer = None
            for layer_i, token_id in enumerate(step_top1):
                if token_id == final_token_id:
                    first_match_layer = layer_i
                    break
            if first_match_layer is None:
                continue
            phase = _phase_for_layer(first_match_layer)
            phase_counts[phase] += 1
            total += 1
            if first_match_layer < len(step_kl):
                v = step_kl[first_match_layer]
                if v is not None and not math.isnan(float(v)):
                    phase_kl[phase].append(float(v))
            if len(step_top1) > 20 and step_top1[20] != final_token_id and first_match_layer >= 20:
                corrected_tokens[final_token_str] += 1

    phase_frac = {k: (phase_counts[k] / total if total else 0.0) for k in _PHASE_BINS}
    phase_mean_kl = {
        k: (float(np.mean(phase_kl[k])) if phase_kl[k] else float("nan"))
        for k in _PHASE_BINS
    }
    return phase_frac, phase_mean_kl, dict(corrected_tokens.most_common(10))


def make_plot(results: list[dict], output_dir: str, pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if not _has_required(results):
        print("  Plot E3.10 skipped — no top1_token_per_layer / kl_adjacent_layer / generated_tokens data.")
        return

    it_frac, it_kl, it_corrected = _mind_change_stats(results)
    pt_frac = pt_kl = pt_corrected = None
    if pt_results and _has_required(pt_results):
        pt_frac, pt_kl, pt_corrected = _mind_change_stats(pt_results)

    labels = list(_PHASE_BINS.keys())
    x = np.arange(len(labels))
    w = 0.34

    fig, axes = plt.subplots(1, 4, figsize=(23, 5.4))
    ax_a, ax_b, ax_c, ax_d = axes

    ax_a.bar(x - w / 2, [it_frac[k] for k in labels], width=w, color="#F57C00", label="IT")
    if pt_frac is not None:
        ax_a.bar(x + w / 2, [pt_frac[k] for k in labels], width=w, color="#1976D2", label="PT")
    ax_a.set_xticks(x, labels)
    ax_a.set_ylabel("Fraction of generated tokens")
    ax_a.set_title("Panel A — Phase where final token first becomes top-1")
    ax_a.grid(axis="y", alpha=0.25)
    ax_a.legend()

    ax_b.bar(x - w / 2, [it_kl[k] for k in labels], width=w, color="#F57C00", label="IT")
    if pt_kl is not None:
        ax_b.bar(x + w / 2, [pt_kl[k] for k in labels], width=w, color="#1976D2", label="PT")
    ax_b.set_xticks(x, labels)
    ax_b.set_ylabel("Mean KL(layer_i || layer_i-1)")
    ax_b.set_title("Panel B — Adjacent-layer KL at mind-change layer")
    ax_b.grid(axis="y", alpha=0.25)

    corr_labels = list(it_corrected.keys())[:10][::-1]
    corr_vals = list(it_corrected.values())[:10][::-1]
    if corr_labels:
        ax_c.barh(np.arange(len(corr_labels)), corr_vals, color="#F57C00", alpha=0.85)
        ax_c.set_yticks(np.arange(len(corr_labels)), corr_labels)
        ax_c.set_xlabel("Count")
        ax_c.set_title("Panel C — IT tokens corrected after layer 20")
        ax_c.grid(axis="x", alpha=0.25)
    else:
        ax_c.axis("off")
        ax_c.text(0.5, 0.5, "No IT corrective-stage corrections found", ha="center", va="center")

    pt_corr_labels = list((pt_corrected or {}).keys())[:10][::-1]
    pt_corr_vals = list((pt_corrected or {}).values())[:10][::-1]
    if pt_corr_labels:
        ax_d.barh(np.arange(len(pt_corr_labels)), pt_corr_vals, color="#1976D2", alpha=0.85)
        ax_d.set_yticks(np.arange(len(pt_corr_labels)), pt_corr_labels)
        ax_d.set_xlabel("Count")
        ax_d.set_title("Panel D — PT tokens corrected after layer 20")
        ax_d.grid(axis="x", alpha=0.25)
    else:
        ax_d.axis("off")
        ax_d.text(0.5, 0.5, "No PT corrective-stage corrections found", ha="center", va="center")

    fig.suptitle(
        "E3.10 — Mind Change Analysis\n"
        "First layer where the final generated token becomes top-1 under the logit lens.",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = Path(output_dir) / "plot_e3_10_mind_change.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.10 saved → {out_path}")
