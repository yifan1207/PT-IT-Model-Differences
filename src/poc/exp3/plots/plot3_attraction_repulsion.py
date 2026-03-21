"""
Plot 3 (Exp3): Attraction vs repulsion decomposition per layer (Experiment 1b).

For each layer i, the logit-lens delta (logit_lens[i][token] - logit_lens[i-1][token])
tells us whether that layer pushed the model toward (+) or away from (-) the
generated token.

  Attraction at layer i : logit_delta_contrib[i] > 0
  Repulsion  at layer i : logit_delta_contrib[i] < 0

Key question: do IT late layers carry more repulsive mass (negative contributions)
than PT? This would show the corrective stage is suppressive, not merely directive.

Four panels:
  A: Mean logit delta contribution per layer — primary (IT) by category with SEM bands,
     PT dashed overlay if provided.
  B: Fraction of steps where layer i is repulsive (delta < 0), with SEM error bars.
  C: Mean absolute contribution per layer with SEM error bars.
  D: Mean contribution split into attraction (+) and repulsion (−) components
     stacked bar for corrective layers (20–33) vs proposal layers (0–19),
     with SEM error bars.

REQUIRES: collect_attribution=True in Exp3Config.
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.shared.constants import N_LAYERS

from src.poc.shared.plot_colors import SPLIT_COLORS as CATEGORY_COLORS, _DEFAULT_COLOR
_BOUNDARY      = 20


def _layer_contrib_stats(results: list[dict],
                          n_layers: int = N_LAYERS) -> dict:
    """Compute per-layer statistics on logit_delta_contrib.

    Returns dict with keys:
        mean_contrib   [n_layers]  mean logit delta (signed)
        sem_contrib    [n_layers]  SEM of mean logit delta
        mean_abs       [n_layers]  mean |delta|
        sem_abs        [n_layers]  SEM of mean |delta|
        repulsion_frac [n_layers]  fraction of steps where delta < 0
        sem_repulsion  [n_layers]  SEM of repulsion fraction (binomial)
    """
    vals_signed = [[] for _ in range(n_layers)]
    vals_abs    = [[] for _ in range(n_layers)]
    neg_c       = np.zeros(n_layers)
    counts      = np.zeros(n_layers)

    for r in results:
        for step_vals in r.get("logit_delta_contrib", []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is None or math.isnan(float(v)):
                    continue
                fv = float(v)
                vals_signed[layer_i].append(fv)
                vals_abs[layer_i].append(abs(fv))
                neg_c[layer_i]  += (1 if fv < 0 else 0)
                counts[layer_i] += 1

    mean_contrib   = np.array([np.mean(v) if v else float("nan") for v in vals_signed])
    sem_contrib    = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                               for v in vals_signed])
    mean_abs       = np.array([np.mean(v) if v else float("nan") for v in vals_abs])
    sem_abs        = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                               for v in vals_abs])
    with np.errstate(invalid="ignore", divide="ignore"):
        p              = np.where(counts > 0, neg_c / counts, np.nan)
        sem_repulsion  = np.where(counts > 1,
                                  np.sqrt(p * (1 - p) / counts),
                                  0.0)

    return {
        "mean_contrib":   mean_contrib,
        "sem_contrib":    sem_contrib,
        "mean_abs":       mean_abs,
        "sem_abs":        sem_abs,
        "repulsion_frac": p,
        "sem_repulsion":  sem_repulsion,
    }


def _layer_contrib_by_cat(results: list[dict],
                           n_layers: int = N_LAYERS) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean ± SEM of signed logit delta per layer by category."""
    by_cat: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n_layers)])
    for r in results:
        cat = r.get("category", "unknown")
        for step_vals in r.get("logit_delta_contrib", []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is None or math.isnan(float(v)):
                    continue
                by_cat[cat][layer_i].append(float(v))
    out = {}
    for cat, layer_lists in by_cat.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layer_lists])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0
                      for v in layer_lists])
        out[cat] = (m, s)
    return out


def _stage_attraction_repulsion(results: list[dict],
                                  boundary: int = _BOUNDARY,
                                  n_layers: int  = N_LAYERS) -> dict:
    """Mean ± SEM of total attraction and repulsion per stage (proposal vs corrective).

    Computes per-prompt totals, then mean/SEM across prompts.
    """
    prop_pos, prop_neg = [], []
    corr_pos, corr_neg = [], []

    for r in results:
        p_pos = p_neg = c_pos = c_neg = 0.0
        for step_vals in r.get("logit_delta_contrib", []):
            for layer_i, v in enumerate(step_vals[:n_layers]):
                if v is None or math.isnan(float(v)):
                    continue
                fv = float(v)
                if layer_i < boundary:
                    if fv >= 0:
                        p_pos += fv
                    else:
                        p_neg += fv
                else:
                    if fv >= 0:
                        c_pos += fv
                    else:
                        c_neg += fv
        prop_pos.append(p_pos)
        prop_neg.append(p_neg)
        corr_pos.append(c_pos)
        corr_neg.append(c_neg)

    def _ms(arr):
        a = np.array(arr)
        m = float(a.mean())
        s = float(a.std() / math.sqrt(len(a))) if len(a) > 1 else 0.0
        return m, s

    pp_m, pp_s = _ms(prop_pos)
    pn_m, pn_s = _ms(prop_neg)
    cp_m, cp_s = _ms(corr_pos)
    cn_m, cn_s = _ms(corr_neg)

    return {
        "prop_pos": pp_m, "prop_pos_sem": pp_s,
        "prop_neg": pn_m, "prop_neg_sem": pn_s,
        "corr_pos": cp_m, "corr_pos_sem": cp_s,
        "corr_neg": cn_m, "corr_neg_sem": cn_s,
    }


def make_plot(results: list[dict], output_dir: str,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("logit_delta_contrib"):
        print("  Plot 3 (Exp3) skipped — no logit_delta_contrib data")
        return

    layers = np.arange(N_LAYERS)

    primary_stats  = _layer_contrib_stats(results)
    primary_by_cat = _layer_contrib_by_cat(results)
    primary_stage  = _stage_attraction_repulsion(results)

    pt_stats  = _layer_contrib_stats(pt_results) if pt_results else None
    pt_stage  = _stage_attraction_repulsion(pt_results) if pt_results else None

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: signed mean delta per layer with SEM ─────────────────────────
    ax_a.axhline(0, color="black", lw=0.7)
    ax_a.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
    ax_a.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4,
                 label=f"boundary (L{_BOUNDARY})")
    ax_a.axvspan(_BOUNDARY, N_LAYERS - 1, alpha=0.06, color="red")

    for cat, (m, s) in primary_by_cat.items():
        c = CATEGORY_COLORS.get(cat, _DEFAULT_COLOR)
        ax_a.plot(layers, m, lw=1.8, color=c, label=cat, alpha=0.85)
        ax_a.fill_between(layers, m - s, m + s, alpha=0.15, color=c)

    if pt_stats is not None:
        ax_a.plot(layers, pt_stats["mean_contrib"], lw=1.8, color="#555", ls="--",
                  label="PT (all cats)", alpha=0.7)
        ax_a.fill_between(layers,
                          pt_stats["mean_contrib"] - pt_stats["sem_contrib"],
                          pt_stats["mean_contrib"] + pt_stats["sem_contrib"],
                          alpha=0.08, color="#555")

    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean logit delta (positive = attraction)")
    ax_a.set_title("Panel A — Signed logit delta per layer  (±SEM)\n"
                   "(positive = layer pushed toward generated token)")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: repulsion fraction per layer with SEM bars ───────────────────
    ax_b.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6)
    ax_b.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_b.axhline(0.5, color="red", lw=0.7, ls="--", alpha=0.5,
                 label="50% baseline")

    ax_b.bar(layers, primary_stats["repulsion_frac"],
             color="#E65100", alpha=0.7, width=0.7, label="IT repulsion fraction",
             yerr=primary_stats["sem_repulsion"], capsize=2,
             error_kw={"elinewidth": 0.8, "alpha": 0.6})
    if pt_stats is not None:
        ax_b.errorbar(layers, pt_stats["repulsion_frac"],
                      yerr=pt_stats["sem_repulsion"],
                      fmt="o--", color="#1565C0", ms=3, lw=1.2,
                      label="PT", alpha=0.8, capsize=2)

    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("Fraction of steps where delta < 0")
    ax_b.set_title("Panel B — Repulsion fraction per layer  (±SEM)\n"
                   "(>0.5 means layer more often pushes AWAY from generated token)")
    ax_b.set_ylim(0, 1)
    ax_b.legend(fontsize=8)
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel C: mean absolute contribution with SEM bars ─────────────────────
    ax_c.axvline(11, color="grey",  lw=0.8, ls=":", alpha=0.6)
    ax_c.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)

    ax_c.bar(layers, primary_stats["mean_abs"],
             color="#E65100", alpha=0.7, width=0.7, label="IT |delta|",
             yerr=primary_stats["sem_abs"], capsize=2,
             error_kw={"elinewidth": 0.8, "alpha": 0.6})
    if pt_stats is not None:
        ax_c.errorbar(layers, pt_stats["mean_abs"],
                      yerr=pt_stats["sem_abs"],
                      fmt="o--", color="#1565C0", ms=3, lw=1.2,
                      label="PT |delta|", alpha=0.8, capsize=2)

    ax_c.set_xlabel("Transformer Layer")
    ax_c.set_ylabel("Mean |logit delta|")
    ax_c.set_title("Panel C — Mean edit magnitude per layer  (±SEM)\n"
                   "(regardless of sign — total computational effort on this token)")
    ax_c.legend(fontsize=8)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: attraction vs repulsion by stage with SEM bars ───────────────
    stage_labels = ["Proposal\n(L0–19)", "Corrective\n(L20–33)"]
    x_pos = np.array([0, 1])
    bar_w = 0.35

    it_pos     = [primary_stage["prop_pos"],     primary_stage["corr_pos"]]
    it_neg     = [primary_stage["prop_neg"],     primary_stage["corr_neg"]]
    it_pos_sem = [primary_stage["prop_pos_sem"], primary_stage["corr_pos_sem"]]
    it_neg_sem = [primary_stage["prop_neg_sem"], primary_stage["corr_neg_sem"]]

    ax_d.bar(x_pos - bar_w / 2, it_pos, bar_w, label="IT attraction (+)",
             color="#E65100", alpha=0.85,
             yerr=it_pos_sem, capsize=4, error_kw={"elinewidth": 1.2})
    ax_d.bar(x_pos - bar_w / 2, it_neg, bar_w, label="IT repulsion (−)",
             color="#BF360C", alpha=0.85,
             yerr=it_neg_sem, capsize=4, error_kw={"elinewidth": 1.2})

    if pt_stage is not None:
        pt_pos     = [pt_stage["prop_pos"],     pt_stage["corr_pos"]]
        pt_neg     = [pt_stage["prop_neg"],     pt_stage["corr_neg"]]
        pt_pos_sem = [pt_stage["prop_pos_sem"], pt_stage["corr_pos_sem"]]
        pt_neg_sem = [pt_stage["prop_neg_sem"], pt_stage["corr_neg_sem"]]
        ax_d.bar(x_pos + bar_w / 2, pt_pos, bar_w, label="PT attraction (+)",
                 color="#1565C0", alpha=0.85,
                 yerr=pt_pos_sem, capsize=4, error_kw={"elinewidth": 1.2})
        ax_d.bar(x_pos + bar_w / 2, pt_neg, bar_w, label="PT repulsion (−)",
                 color="#0D47A1", alpha=0.85,
                 yerr=pt_neg_sem, capsize=4, error_kw={"elinewidth": 1.2})

    ax_d.axhline(0, color="black", lw=0.8)
    ax_d.set_xticks(x_pos)
    ax_d.set_xticklabels(stage_labels, fontsize=10)
    ax_d.set_ylabel("Mean total logit delta per prompt  (±SEM)")
    ax_d.set_title("Panel D — Attraction vs repulsion by stage\n"
                   "IT corrective stage: high repulsion = actively suppressing tokens")
    ax_d.legend(fontsize=8, ncol=2)
    ax_d.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Exp3 Plot 3 — Attraction vs Repulsion Decomposition per Layer\n"
        "Positive = layer pushes toward generated token; negative = pushes away",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot3_attraction_repulsion.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 3 (Exp3) saved → {out_path}")
