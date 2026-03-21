"""
Plot 8 (Exp3): Per-layer weight change PT → IT (Experiment 0c).

Uses pre-computed weight_shift.json which contains, for each parameter:
  frob_shift  : ||W_IT - W_PT||_F  (how much the weight matrix moved in absolute terms)
  cos_dist    : 1 - cosine(W_IT_flat, W_PT_flat)  (directional change, scale-independent)

Four panels:
  A: Frobenius norm shift per layer — MLP vs Attention split with std bands.
     Each point = mean over all sub-components in that layer/type.
     A spike at corrective layers (20–33) → post-training rewrote those weights.
  B: Cosine distance per layer — MLP vs Attention with std bands.
     Scale-independent version: detects direction change even when magnitude is similar.
  C: Component-type breakdown at corrective vs proposal layers with SEM error bars.
     Bar chart: mean frob_shift by component type, coloured by stage.
     Shows WHICH components (gate_proj? down_proj? q_proj?) changed most.
  D: Scatter frob_shift vs cos_dist per component, coloured by layer.
     Reveals whether changes were norm-scaling or genuine rotation of weight space.
"""
import re
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

_BOUNDARY    = 20
_N_LAYERS    = 34
_MLP_COMPS   = {"mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.fc1", "mlp.fc2"}
_ATTN_COMPS  = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                "self_attn.o_proj", "self_attn.out_proj"}


def _parse_weight_shift(data: list[dict]) -> dict:
    """Extract layer-indexed frob_shift and cos_dist by component type.

    Returns per-layer mean ± std for MLP and Attention groups,
    per-component stage statistics, and raw scatter points.
    """
    # per-layer, per-group lists of values
    mlp_frob_by_layer:  dict[int, list[float]] = defaultdict(list)
    mlp_cos_by_layer:   dict[int, list[float]] = defaultdict(list)
    attn_frob_by_layer: dict[int, list[float]] = defaultdict(list)
    attn_cos_by_layer:  dict[int, list[float]] = defaultdict(list)

    # per-component, per-stage lists
    by_comp_stage: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"prop": [], "corr": []})

    all_points: list[tuple[int, str, float, float]] = []

    for x in data:
        name  = x["name"]
        frob  = x.get("frob_shift")
        cos   = x.get("cos_dist")
        if frob is None or cos is None:
            continue
        m = re.search(r"layers\.(\d+)\.(.+)\.weight$", name)
        if not m:
            continue
        layer = int(m.group(1))
        comp  = m.group(2)
        stage = "corr" if layer >= _BOUNDARY else "prop"

        by_comp_stage[comp][stage].append(frob)
        all_points.append((layer, comp, frob, cos))

        if comp in _MLP_COMPS:
            mlp_frob_by_layer[layer].append(frob)
            mlp_cos_by_layer[layer].append(cos)
        elif comp in _ATTN_COMPS:
            attn_frob_by_layer[layer].append(frob)
            attn_cos_by_layer[layer].append(cos)

    def _mean_std(by_layer):
        m = np.array([np.mean(by_layer[l]) if by_layer[l] else float("nan")
                      for l in range(_N_LAYERS)])
        s = np.array([np.std(by_layer[l]) if len(by_layer[l]) > 1 else 0.0
                      for l in range(_N_LAYERS)])
        return m, s

    mlp_frob_m,  mlp_frob_s  = _mean_std(mlp_frob_by_layer)
    mlp_cos_m,   mlp_cos_s   = _mean_std(mlp_cos_by_layer)
    attn_frob_m, attn_frob_s = _mean_std(attn_frob_by_layer)
    attn_cos_m,  attn_cos_s  = _mean_std(attn_cos_by_layer)

    return {
        "mlp_frob_m": mlp_frob_m, "mlp_frob_s": mlp_frob_s,
        "mlp_cos_m":  mlp_cos_m,  "mlp_cos_s":  mlp_cos_s,
        "attn_frob_m": attn_frob_m, "attn_frob_s": attn_frob_s,
        "attn_cos_m":  attn_cos_m,  "attn_cos_s":  attn_cos_s,
        "by_comp_stage": dict(by_comp_stage),
        "all_points":    all_points,
    }


def make_plot(weight_shift_path: str, output_dir: str) -> None:
    path = Path(weight_shift_path)
    if not path.exists():
        print(f"  Plot 8 (Exp3) skipped — weight_shift.json not found: {path}")
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(path) as f:
        raw = json.load(f)

    stats  = _parse_weight_shift(raw)
    layers = np.arange(_N_LAYERS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    def _add_stage_lines(ax):
        ax.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
        ax.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4,
                   label=f"boundary (L{_BOUNDARY})")
        ax.axvspan(_BOUNDARY, _N_LAYERS - 1, alpha=0.05, color="red")

    # ── Panel A: Frobenius shift per layer with std bands ─────────────────────
    _add_stage_lines(ax_a)
    ax_a.plot(layers, stats["mlp_frob_m"],  lw=2, color="#E65100", label="MLP")
    ax_a.fill_between(layers,
                      np.maximum(0, stats["mlp_frob_m"] - stats["mlp_frob_s"]),
                      stats["mlp_frob_m"] + stats["mlp_frob_s"],
                      alpha=0.15, color="#E65100")
    ax_a.plot(layers, stats["attn_frob_m"], lw=2, color="#1565C0", label="Attention")
    ax_a.fill_between(layers,
                      np.maximum(0, stats["attn_frob_m"] - stats["attn_frob_s"]),
                      stats["attn_frob_m"] + stats["attn_frob_s"],
                      alpha=0.15, color="#1565C0")
    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean Frobenius shift  ||W_IT − W_PT||_F  (±std)")
    ax_a.set_title("Panel A — Weight magnitude change per layer\n"
                   "Spike at late layers → IT rewrote those weights most")
    ax_a.legend(fontsize=9)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: Cosine distance per layer with std bands ─────────────────────
    _add_stage_lines(ax_b)
    ax_b.plot(layers, stats["mlp_cos_m"],  lw=2, color="#E65100", label="MLP")
    ax_b.fill_between(layers,
                      np.maximum(0, stats["mlp_cos_m"] - stats["mlp_cos_s"]),
                      stats["mlp_cos_m"] + stats["mlp_cos_s"],
                      alpha=0.15, color="#E65100")
    ax_b.plot(layers, stats["attn_cos_m"], lw=2, color="#1565C0", label="Attention")
    ax_b.fill_between(layers,
                      np.maximum(0, stats["attn_cos_m"] - stats["attn_cos_s"]),
                      stats["attn_cos_m"] + stats["attn_cos_s"],
                      alpha=0.15, color="#1565C0")
    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("Mean cosine distance  (1 − cos)  (±std)")
    ax_b.set_title("Panel B — Weight direction change per layer\n"
                   "Scale-independent: did IT rotate the weight vectors?")
    ax_b.legend(fontsize=9)
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel C: Component breakdown with SEM error bars ─────────────────────
    bcs = stats["by_comp_stage"]
    comp_names_c = sorted(
        {c for c in bcs if c in _MLP_COMPS | _ATTN_COMPS},
        key=lambda k: -np.mean(bcs[k]["corr"] or [0])
    )

    def _ms(vals):
        if not vals:
            return 0.0, 0.0
        arr = np.array(vals)
        return float(arr.mean()), float(arr.std() / math.sqrt(len(arr)))

    x_c = np.arange(len(comp_names_c))
    bar_w = 0.35
    prop_means, prop_sems = zip(*[_ms(bcs[c]["prop"]) for c in comp_names_c])
    corr_means, corr_sems = zip(*[_ms(bcs[c]["corr"]) for c in comp_names_c])

    ax_c.bar(x_c - bar_w / 2, prop_means, bar_w,
             color="#1565C0", alpha=0.8, label="Proposal (L0–19)",
             yerr=prop_sems, capsize=3, error_kw={"elinewidth": 1.0})
    ax_c.bar(x_c + bar_w / 2, corr_means, bar_w,
             color="#E65100", alpha=0.8, label="Corrective (L20–33)",
             yerr=corr_sems, capsize=3, error_kw={"elinewidth": 1.0})
    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels(comp_names_c, rotation=30, ha="right", fontsize=8)
    ax_c.set_ylabel("Mean Frobenius shift  (±SEM)")
    ax_c.set_title("Panel C — Which components changed most?\n"
                   "Proposal vs corrective stage")
    ax_c.legend(fontsize=9)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: Scatter frob vs cos, coloured by layer ───────────────────────
    points = [(l, f, c) for l, comp, f, c in stats["all_points"]
              if comp in _MLP_COMPS | _ATTN_COMPS]
    if points:
        ls, fs, cs = zip(*points)
        sc = ax_d.scatter(fs, cs, c=ls, cmap="plasma", s=8, alpha=0.4)
        plt.colorbar(sc, ax=ax_d, label="Layer index")
    ax_d.set_xlabel("Frobenius shift  ||W_IT − W_PT||_F")
    ax_d.set_ylabel("Cosine distance  (1 − cos)")
    ax_d.set_title("Panel D — Frob vs cos distance per component\n"
                   "Top-right = big AND rotated  |  Top-left = rotated only")
    ax_d.grid(alpha=0.2)

    fig.suptitle(
        "Exp3 Plot 8 — Per-Layer Weight Change PT → IT  (Exp 0c)\n"
        "Which layers did post-training rewrite most?",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot8_weight_norm.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 8 (Exp3) saved → {out_path}")
