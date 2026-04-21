"""
E3.3 — Raw Completion Test (Category 5e).

Purpose
-------
Category 5e prompts are raw text continuations (Wikipedia snippets, no Q&A wrapper).
For IT, they are collected with apply_chat_template=False, so NO alignment-triggering
prompt structure is present.

Hypothesis:
  - If the corrective stage (layers 20-33) elevation appears in IT even on 5e prompts
    → the stage is WEIGHT-DRIVEN (baked into IT weights post-training).
  - If 5e in IT looks the same as PT → the stage is PROMPT-TRIGGERED.

Four panels:
  A: Mean layer_delta_norm per layer — IT 5e vs PT 5e vs IT all-A-mean (SEM bands).
     Corrective elevation at layers 20-33 in IT 5e = weight-driven evidence.
  B: Mean kl_to_final per layer — same grouping.
     Earlier KL drop in IT 5e vs PT 5e = commitment delay survives no-template.
  C: Mean corrective stage magnitude (mean delta_norm at layers 20–33) per A
     subcategory for IT vs PT.  Bar chart with error bars.
     Shows 5e in context of other subcategories — is it an outlier?
  D: Commitment layer distribution (histogram + median marker) for IT 5e vs PT 5e
     vs IT 5a (harmful — strongest alignment signal) for reference.
"""
from __future__ import annotations
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.shared.constants import N_LAYERS
from src.poc.shared.plot_colors import SPLIT_COLORS, _DEFAULT_COLOR

_BOUNDARY     = 20
_KL_THRESHOLD = 0.1   # nats — commitment

_SUBCATEGORY_COLORS = {
    "5a": "#D32F2F",
    "5b": "#FF7043",
    "5c": "#FBC02D",
    "5d": "#7B1FA2",
    "5e": "#78909C",
    # backward compat
    "4a": "#D32F2F",
    "4b": "#FF7043",
    "4c": "#FBC02D",
    "4d": "#7B1FA2",
    "4e": "#78909C",
}
_SUBCATS = ["5a", "5b", "5c", "5d", "5e"]
_SUBCAT_LABELS = {
    "5a": "5a Harmful",
    "5b": "5b Borderline",
    "5c": "5c Format",
    "5d": "5d Conversational",
    "5e": "5e Raw",
    "4a": "4a Harmful",
    "4b": "4b Borderline",
    "4c": "4c Format",
    "4d": "4d Conversational",
    "4e": "4e Raw",
}


def _get_subcat(r: dict) -> str:
    sc = r.get("alignment_subcategory") or r.get("metadata", {}).get("alignment_subcategory", "")
    # normalise old 4x → 5x labels for display
    return sc or ""


def _layer_metric_by_subcat(
    results: list[dict],
    metric_key: str,
    n_layers: int = N_LAYERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Mean ± SEM of metric_key per layer, grouped by alignment subcategory."""
    by_sc: dict[str, list[list[float]]] = defaultdict(lambda: [[] for _ in range(n_layers)])
    for r in results:
        if r.get("split") != "A":
            continue
        sc = _get_subcat(r)
        if not sc:
            continue
        for step_vals in r.get(metric_key, []):
            for li, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    by_sc[sc][li].append(float(v))
    out = {}
    for sc, layers_list in by_sc.items():
        m = np.array([np.mean(v) if v else float("nan") for v in layers_list])
        s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0 for v in layers_list])
        out[sc] = (m, s)
    return out


def _corrective_mean_by_subcat(
    results: list[dict],
    metric_key: str,
    boundary: int = _BOUNDARY,
    n_layers: int = N_LAYERS,
) -> dict[str, tuple[float, float]]:
    """Mean ± SEM of corrective-stage mean per prompt, grouped by subcategory."""
    by_sc: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r.get("split") != "A":
            continue
        sc = _get_subcat(r)
        if not sc:
            continue
        vals = []
        for step_vals in r.get(metric_key, []):
            for li, v in enumerate(step_vals[:n_layers]):
                if li >= boundary and v is not None and not math.isnan(float(v)):
                    vals.append(float(v))
        if vals:
            by_sc[sc].append(np.mean(vals))
    out = {}
    for sc, prompt_means in by_sc.items():
        m = float(np.mean(prompt_means))
        s = float(np.std(prompt_means) / math.sqrt(len(prompt_means))) if len(prompt_means) > 1 else 0.0
        out[sc] = (m, s)
    return out


def _commitment_layers(
    results: list[dict],
    subcat_filter: str | None = None,
    threshold: float = _KL_THRESHOLD,
    n_layers: int = N_LAYERS,
) -> list[int]:
    out = []
    for r in results:
        if subcat_filter and _get_subcat(r) not in (subcat_filter,):
            continue
        for step_vals in r.get("kl_to_final", []):
            for li, v in enumerate(step_vals[:n_layers]):
                if v is not None and not math.isnan(float(v)) and float(v) < threshold:
                    out.append(li)
                    break
    return out


def _mean_kl_by_subcat(
    results: list[dict],
    n_layers: int = N_LAYERS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return _layer_metric_by_subcat(results, "kl_to_final", n_layers)


def make_plot(
    results: list[dict],
    output_dir: str,
    pt_results: list[dict] | None = None,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    it_5e = [r for r in results if _get_subcat(r) in ("5e", "4e")]
    if not it_5e:
        print("  Plot E3.3 skipped — no 5e records found in results.")
        return

    layers = np.arange(N_LAYERS)

    # ── Data extraction ───────────────────────────────────────────────────────
    it_by_sc_delta  = _layer_metric_by_subcat(results, "layer_delta_norm")
    it_by_sc_kl     = _mean_kl_by_subcat(results)
    it_corrective   = _corrective_mean_by_subcat(results, "layer_delta_norm")

    pt_by_sc_delta:  dict = {}
    pt_by_sc_kl:     dict = {}
    pt_corrective:   dict = {}
    if pt_results:
        pt_by_sc_delta  = _layer_metric_by_subcat(pt_results, "layer_delta_norm")
        pt_by_sc_kl     = _mean_kl_by_subcat(pt_results)
        pt_corrective   = _corrective_mean_by_subcat(pt_results, "layer_delta_norm")

    # IT all-A mean (across all subcategories)
    all_a_it: list[list[float]] = [[] for _ in range(N_LAYERS)]
    for r in results:
        if r.get("split") != "A":
            continue
        for step_vals in r.get("layer_delta_norm", []):
            for li, v in enumerate(step_vals[:N_LAYERS]):
                if v is not None and not math.isnan(float(v)):
                    all_a_it[li].append(float(v))
    all_a_m = np.array([np.mean(v) if v else float("nan") for v in all_a_it])
    all_a_s = np.array([np.std(v)/math.sqrt(len(v)) if len(v) > 1 else 0. for v in all_a_it])

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: layer_delta_norm for IT 5e vs PT 5e vs IT all-A ──────────────
    ax_a.axvspan(_BOUNDARY, N_LAYERS, alpha=0.04, color="grey", label="corrective stage")
    ax_a.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_a.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)

    # IT all-A mean (grey reference)
    ax_a.plot(layers, all_a_m, lw=1.2, color="grey", ls="--", alpha=0.5, label="IT all-A mean")
    ax_a.fill_between(layers, np.maximum(0, all_a_m - all_a_s), all_a_m + all_a_s,
                      alpha=0.06, color="grey")

    for sc_key, sc_label_suffix in [("5e", "5e"), ("4e", "4e")]:
        if sc_key in it_by_sc_delta:
            m, s = it_by_sc_delta[sc_key]
            c = _SUBCATEGORY_COLORS[sc_key]
            ax_a.plot(layers, m, lw=2.2, color=c, label=f"IT {sc_label_suffix} raw")
            ax_a.fill_between(layers, np.maximum(0, m - s), m + s, alpha=0.15, color=c)
            if sc_key in (pt_by_sc_delta or {}):
                pm, ps = pt_by_sc_delta[sc_key]
                ax_a.plot(layers, pm, lw=1.5, color=c, ls="--", alpha=0.55,
                          label=f"PT {sc_label_suffix} raw")
                ax_a.fill_between(layers, np.maximum(0, pm - ps), pm + ps,
                                  alpha=0.07, color=c)
            break  # use whichever key found first

    ax_a.set_xlabel("Transformer Layer")
    ax_a.set_ylabel("Mean |Δresidual| (±SEM)")
    ax_a.set_title("Panel A — Layer delta norm: IT 5e raw vs PT 5e\n"
                   "Corrective elevation in IT 5e = weight-driven evidence")
    ax_a.legend(fontsize=8, ncol=2)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: kl_to_final for IT 5e vs PT 5e vs IT all-A ──────────────────
    ax_b.axvspan(_BOUNDARY, N_LAYERS, alpha=0.04, color="grey")
    ax_b.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_b.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_b.axhline(_KL_THRESHOLD, color="green", lw=0.8, ls="--", alpha=0.5,
                 label=f"commit threshold ({_KL_THRESHOLD} nat)")

    # IT all-A KL
    all_a_kl: list[list[float]] = [[] for _ in range(N_LAYERS)]
    for r in results:
        if r.get("split") != "A":
            continue
        for step_vals in r.get("kl_to_final", []):
            for li, v in enumerate(step_vals[:N_LAYERS]):
                if v is not None and not math.isnan(float(v)):
                    all_a_kl[li].append(float(v))
    all_a_kl_m = np.array([np.mean(v) if v else float("nan") for v in all_a_kl])
    ax_b.plot(layers, all_a_kl_m, lw=1.2, color="grey", ls="--", alpha=0.5,
              label="IT all-A KL mean")

    for sc_key in ("5e", "4e"):
        if sc_key in it_by_sc_kl:
            m, s = it_by_sc_kl[sc_key]
            c = _SUBCATEGORY_COLORS[sc_key]
            ax_b.plot(layers, m, lw=2.2, color=c, label=f"IT {sc_key}")
            ax_b.fill_between(layers, np.maximum(0, m - s), m + s, alpha=0.15, color=c)
            if pt_by_sc_kl and sc_key in pt_by_sc_kl:
                pm, ps = pt_by_sc_kl[sc_key]
                ax_b.plot(layers, pm, lw=1.5, color=c, ls="--", alpha=0.55,
                          label=f"PT {sc_key}")
            break

    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("Mean KL(layer_i ∥ final) nats (±SEM)")
    ax_b.set_title("Panel B — KL-to-final: IT 5e vs PT 5e\n"
                   "Earlier KL drop in IT 5e = commitment delay without chat template")
    ax_b.legend(fontsize=8, ncol=2)
    ax_b.grid(axis="y", alpha=0.25)

    # ── Panel C: corrective stage magnitude per subcategory bar chart ─────────
    present_scs: list[str] = []
    for sc in _SUBCATS:
        if sc in it_corrective or sc.replace("5", "4") in it_corrective:
            present_scs.append(sc)
    if not present_scs:
        # fallback: use old labels
        present_scs = [sc.replace("5", "4") for sc in _SUBCATS if sc.replace("5","4") in it_corrective]

    x = np.arange(len(present_scs))
    w = 0.35
    it_means = []
    it_sems  = []
    pt_means = []
    pt_sems  = []
    for sc in present_scs:
        sc_key = sc if sc in it_corrective else sc.replace("5", "4")
        m_it, s_it = it_corrective.get(sc_key, (float("nan"), 0.0))
        it_means.append(m_it)
        it_sems.append(s_it)
        m_pt, s_pt = pt_corrective.get(sc_key, (float("nan"), 0.0)) if pt_corrective else (float("nan"), 0.0)
        pt_means.append(m_pt)
        pt_sems.append(s_pt)

    bars_it = ax_c.bar(x - w/2, it_means, w, yerr=it_sems, capsize=4,
                        color=[_SUBCATEGORY_COLORS[sc] for sc in present_scs],
                        alpha=0.85, label="IT")
    if pt_results:
        ax_c.bar(x + w/2, pt_means, w, yerr=pt_sems, capsize=4,
                 color=[_SUBCATEGORY_COLORS[sc] for sc in present_scs],
                 alpha=0.45, hatch="//", label="PT")

    ax_c.set_xticks(x)
    ax_c.set_xticklabels([_SUBCAT_LABELS.get(sc, sc) for sc in present_scs],
                          rotation=20, ha="right", fontsize=8)
    ax_c.set_ylabel("Mean layer_delta_norm at layers 20–33")
    ax_c.set_title("Panel C — Corrective stage magnitude per A subcategory\n"
                   "IT solid / PT hatched  |  Is 5e an outlier?")
    ax_c.legend(fontsize=9)
    ax_c.grid(axis="y", alpha=0.25)

    # ── Panel D: commitment layer distribution — 5e vs 5a (reference) ─────────
    bins = np.arange(0, N_LAYERS + 2) - 0.5

    def _commit_for_sc(res, sc_key):
        alts = [sc_key, sc_key.replace("5", "4"), sc_key.replace("4", "5")]
        recs = [r for r in res if _get_subcat(r) in alts]
        return _commitment_layers(recs)

    it_5e_commit = _commit_for_sc(results, "5e")
    it_5a_commit = _commit_for_sc(results, "5a")
    pt_5e_commit = _commit_for_sc(pt_results, "5e") if pt_results else []

    for commits, label, color, alpha in [
        (it_5e_commit, f"IT 5e raw  (med={np.median(it_5e_commit):.0f})" if it_5e_commit else "IT 5e (empty)", "#78909C", 0.65),
        (pt_5e_commit, f"PT 5e raw  (med={np.median(pt_5e_commit):.0f})" if pt_5e_commit else None, "#78909C", 0.35),
        (it_5a_commit, f"IT 5a harmful (med={np.median(it_5a_commit):.0f})" if it_5a_commit else None, "#D32F2F", 0.45),
    ]:
        if commits and label:
            ax_d.hist(commits, bins=bins, density=True, alpha=alpha, color=color, label=label)
            ax_d.axvline(np.median(commits), color=color,
                         lw=2 if alpha > 0.5 else 1.2,
                         ls="-" if alpha > 0.5 else "--")

    ax_d.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6, label="dip (L11)")
    ax_d.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4, label=f"boundary (L{_BOUNDARY})")
    ax_d.set_xlabel("First layer where KL < threshold (commitment layer)")
    ax_d.set_ylabel("Density")
    ax_d.set_title(f"Panel D — Commitment layer distribution\n"
                   f"IT 5e vs PT 5e vs IT 5a  (threshold={_KL_THRESHOLD} nats)")
    ax_d.legend(fontsize=8)
    ax_d.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "E3.3 — Raw Completion Test (Category 5e)\n"
        "Does the corrective stage appear without chat-template triggering?\n"
        "Yes → weight-driven  |  No → prompt-triggered",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot_e3_3_raw_completion.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.3 saved → {out_path}")
