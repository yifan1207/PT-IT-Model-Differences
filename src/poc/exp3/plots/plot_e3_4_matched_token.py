"""
E3.4 — Matched-Token Commitment Analysis.

Purpose
-------
Controls for the "IT generates harder tokens" confound:

  - Find generation steps where PT and IT produced the SAME token on the same prompt.
  - For those "matched" steps, measure commitment layer (first layer where KL < threshold).
  - If IT still commits later than PT on matched tokens, the delay is NOT due to token
    difficulty — it is intrinsic to the IT corrective stage machinery.

Four panels:
  A: Match rate per split — fraction of steps where PT and IT generated the same token.
     Bar chart grouped by split (F, R, OOD, GEN, A).
     Low match on A-split → IT actively rewrites harmful/borderline tokens.

  B: KL-to-final trajectory — matched vs mismatched steps.
     IT matched (solid orange) vs PT matched (dashed orange)
     IT mismatched (solid blue) vs PT mismatched (dashed blue)
     Key: even on matched tokens, does IT have higher early-layer KL than PT?

  C: Commitment layer distribution — matched tokens only.
     IT (orange) vs PT (blue) histogram + median marker.
     This is the primary confound-control panel.

  D: Commitment layer distribution — mismatched tokens only.
     IT vs PT — expected to show larger IT delay (alignment rewriting).

REQUIRES: PT and IT results with kl_to_final + generated_tokens on the same prompts.
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
_KL_THRESHOLD = 0.1   # nats — "committed" when KL drops below this

_IT_COLOR  = "#E65100"   # deep orange — IT
_PT_COLOR  = "#1565C0"   # deep blue   — PT


def _build_token_id_map(results: list[dict]) -> dict[str, list[int]]:
    """Return {record_id: [token_id_step_0, token_id_step_1, ...]} for each result."""
    out: dict[str, list[int]] = {}
    for r in results:
        rid = r.get("record_id") or r.get("prompt_id", "")
        tokens = [t["token_id"] for t in r.get("generated_tokens", [])]
        if rid and tokens:
            out[rid] = tokens
    return out


def _compute_match_stats(
    it_results: list[dict],
    pt_token_map: dict[str, list[int]],
) -> tuple[
    dict[str, list[bool]],           # {record_id: [matched_step_0, ...]}
    dict[str, float],                # match rate per split
]:
    """Compute per-step match flag and per-split match rate."""
    match_by_id: dict[str, list[bool]] = {}
    split_matched: dict[str, int] = defaultdict(int)
    split_total:   dict[str, int] = defaultdict(int)

    for r in it_results:
        rid   = r.get("record_id") or r.get("prompt_id", "")
        split = r.get("split") or r.get("category", "?")
        it_tokens = [t["token_id"] for t in r.get("generated_tokens", [])]
        pt_tokens = pt_token_map.get(rid, [])

        n_common = min(len(it_tokens), len(pt_tokens))
        if n_common == 0:
            match_by_id[rid] = []
            continue

        flags = [it_tokens[i] == pt_tokens[i] for i in range(n_common)]
        match_by_id[rid] = flags
        split_matched[split] += sum(flags)
        split_total[split]   += len(flags)

    match_rate: dict[str, float] = {
        s: split_matched[s] / split_total[s] if split_total[s] > 0 else 0.0
        for s in split_total
    }
    return match_by_id, match_rate


def _kl_by_match(
    results: list[dict],
    match_by_id: dict[str, list[bool]],
    matched: bool,
    n_layers: int = N_LAYERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± SEM kl_to_final per layer for matched or mismatched steps."""
    buckets: list[list[float]] = [[] for _ in range(n_layers)]
    for r in results:
        rid = r.get("record_id") or r.get("prompt_id", "")
        flags = match_by_id.get(rid, [])
        for step_i, step_kl in enumerate(r.get("kl_to_final", [])):
            if step_i >= len(flags):
                break
            if flags[step_i] != matched:
                continue
            for li, v in enumerate(step_kl[:n_layers]):
                if v is not None and not math.isnan(float(v)):
                    buckets[li].append(float(v))
    m = np.array([np.mean(v) if v else float("nan") for v in buckets])
    s = np.array([np.std(v) / math.sqrt(len(v)) if len(v) > 1 else 0.0 for v in buckets])
    return m, s


def _commitment_from_kl(
    results: list[dict],
    match_by_id: dict[str, list[bool]] | None,
    matched: bool | None,
    threshold: float = _KL_THRESHOLD,
    n_layers: int = N_LAYERS,
) -> list[float]:
    """Commitment layers (fractional via linear interpolation), optionally filtered by match status.

    Uses the same interpolation as E3.5: linearly interpolates the crossing point
    between the last layer above threshold and first below, giving continuous values
    instead of snapping to integers.
    """
    out = []
    for r in results:
        rid = r.get("record_id") or r.get("prompt_id", "")
        flags = match_by_id.get(rid, []) if match_by_id else []
        for step_i, step_kl in enumerate(r.get("kl_to_final", [])):
            if match_by_id is not None and matched is not None:
                if step_i >= len(flags) or flags[step_i] != matched:
                    continue
            kl_vals = [float(v) if (v is not None and not math.isnan(float(v))) else float("nan")
                       for v in step_kl[:n_layers]]
            for li in range(len(kl_vals)):
                v = kl_vals[li]
                if math.isnan(v):
                    continue
                if v < threshold:
                    if li > 0 and not math.isnan(kl_vals[li - 1]) and kl_vals[li - 1] > threshold:
                        prev = kl_vals[li - 1]
                        t = (prev - threshold) / (prev - v)
                        out.append(li - 1 + t)
                    else:
                        out.append(float(li))
                    break
    return out


def make_plot(
    results: list[dict],
    output_dir: str,
    pt_results: list[dict] | None = None,
    threshold: float = _KL_THRESHOLD,
) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not pt_results:
        print("  Plot E3.4 skipped — PT results required for matched-token analysis.")
        return
    if not results[0].get("kl_to_final"):
        print("  Plot E3.4 skipped — no kl_to_final data (collect_emergence=False?).")
        return

    layers = np.arange(N_LAYERS)

    # ── Build match maps ──────────────────────────────────────────────────────
    pt_token_map = _build_token_id_map(pt_results)
    match_by_id, match_rate_by_split = _compute_match_stats(results, pt_token_map)

    # PT side: build match flags aligned with PT records
    pt_match_by_id: dict[str, list[bool]] = {}
    it_token_map = _build_token_id_map(results)
    for r in pt_results:
        rid   = r.get("record_id") or r.get("prompt_id", "")
        pt_tok = [t["token_id"] for t in r.get("generated_tokens", [])]
        it_tok = it_token_map.get(rid, [])
        n = min(len(pt_tok), len(it_tok))
        pt_match_by_id[rid] = [pt_tok[i] == it_tok[i] for i in range(n)] if n else []

    n_total_matched = sum(sum(v) for v in match_by_id.values())
    n_total_steps   = sum(len(v) for v in match_by_id.values())
    overall_rate    = n_total_matched / n_total_steps if n_total_steps > 0 else 0.0

    # ── KL trajectories ───────────────────────────────────────────────────────
    it_match_kl_m,   it_match_kl_s   = _kl_by_match(results,    match_by_id,    matched=True)
    it_mismatch_kl_m, it_mismatch_kl_s = _kl_by_match(results,  match_by_id,    matched=False)
    pt_match_kl_m,   pt_match_kl_s   = _kl_by_match(pt_results, pt_match_by_id, matched=True)
    pt_mismatch_kl_m, pt_mismatch_kl_s = _kl_by_match(pt_results, pt_match_by_id, matched=False)

    # ── Commitment layer distributions ────────────────────────────────────────
    it_commit_match    = _commitment_from_kl(results,    match_by_id,    matched=True,  threshold=threshold)
    it_commit_mismatch = _commitment_from_kl(results,    match_by_id,    matched=False, threshold=threshold)
    pt_commit_match    = _commitment_from_kl(pt_results, pt_match_by_id, matched=True,  threshold=threshold)
    pt_commit_mismatch = _commitment_from_kl(pt_results, pt_match_by_id, matched=False, threshold=threshold)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: match rate per split ─────────────────────────────────────────
    splits_sorted = sorted(match_rate_by_split.keys())
    rates = [match_rate_by_split[s] for s in splits_sorted]
    colors = [SPLIT_COLORS.get(s, _DEFAULT_COLOR) for s in splits_sorted]
    bars = ax_a.bar(splits_sorted, rates, color=colors, alpha=0.8)
    ax_a.axhline(overall_rate, color="black", lw=1.2, ls="--",
                 label=f"overall mean ({overall_rate:.1%})")
    for bar, rate in zip(bars, rates):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                  f"{rate:.1%}", ha="center", va="bottom", fontsize=8)
    ax_a.set_ylim(0, min(1.0, max(rates) * 1.3 + 0.05) if rates else 1.0)
    ax_a.set_ylabel("Fraction of steps where PT = IT token")
    ax_a.set_title(f"Panel A — Token match rate per split\n"
                   f"Overall: {n_total_matched:,}/{n_total_steps:,} steps matched "
                   f"({overall_rate:.1%})")
    ax_a.legend(fontsize=9)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: KL trajectory matched vs mismatched ──────────────────────────
    ax_b.axvspan(_BOUNDARY, N_LAYERS, alpha=0.04, color="grey")
    ax_b.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
    ax_b.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
    ax_b.axhline(threshold, color="green", lw=0.8, ls="--", alpha=0.5,
                 label=f"commit threshold ({threshold} nats)")

    for m, s, label, color, ls, alpha in [
        (it_match_kl_m,     it_match_kl_s,     "IT matched",    _IT_COLOR, "-",  1.0),
        (it_mismatch_kl_m,  it_mismatch_kl_s,  "IT mismatch",   _IT_COLOR, ":",  0.7),
        (pt_match_kl_m,     pt_match_kl_s,     "PT matched",    _PT_COLOR, "--", 0.8),
        (pt_mismatch_kl_m,  pt_mismatch_kl_s,  "PT mismatch",   _PT_COLOR, "-.", 0.6),
    ]:
        ax_b.plot(layers, m, lw=2, color=color, ls=ls, alpha=alpha, label=label)
        ax_b.fill_between(layers, np.maximum(0, m - s), m + s, alpha=0.06, color=color)

    ax_b.set_xlabel("Transformer Layer")
    ax_b.set_ylabel("Mean KL(layer_i ∥ final) nats (±SEM)")
    ax_b.set_title(f"Panel B — KL trajectory: matched vs mismatched tokens  (commit={threshold} nats)\n"
                   "IT matched still higher than PT matched → confound controlled")
    ax_b.legend(fontsize=8, ncol=2)
    ax_b.grid(axis="y", alpha=0.25)

    def _hist_panel(ax, it_c, pt_c, title, xlabel):
        """Draw histogram with both mean (solid) and median (dashed) lines for IT and PT."""
        bins = np.arange(0, N_LAYERS + 2) - 0.5
        if it_c:
            it_mean = np.mean(it_c); it_med = np.median(it_c)
            ax.hist(it_c, bins=bins, density=True, alpha=0.55, color=_IT_COLOR,
                    label=f"IT  mean={it_mean:.1f}  med={it_med:.1f}  n={len(it_c):,}")
            ax.axvline(it_mean, color=_IT_COLOR, lw=2.2, ls="-",  label="IT mean")
            ax.axvline(it_med,  color=_IT_COLOR, lw=2.2, ls="--", label="IT median")
        if pt_c:
            pt_mean = np.mean(pt_c); pt_med = np.median(pt_c)
            ax.hist(pt_c, bins=bins, density=True, alpha=0.40, color=_PT_COLOR,
                    label=f"PT  mean={pt_mean:.1f}  med={pt_med:.1f}  n={len(pt_c):,}")
            ax.axvline(pt_mean, color=_PT_COLOR, lw=2.2, ls="-",  label="PT mean")
            ax.axvline(pt_med,  color=_PT_COLOR, lw=2.2, ls="--", label="PT median")
        if it_c and pt_c:
            dm = np.mean(it_c) - np.mean(pt_c)
            dd = np.median(it_c) - np.median(pt_c)
            clr_m = "green" if dm > 0 else "red"; clr_d = "green" if dd > 0 else "red"
            ax.text(0.97, 0.97,
                    f"Δmean={dm:+.2f}L\nΔmed ={dd:+.2f}L",
                    transform=ax.transAxes, ha="right", va="top", fontsize=10,
                    fontweight="bold",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        ax.axvline(_BOUNDARY, color="black", lw=0.8, ls=":", alpha=0.4)
        ax.axvline(11, color="grey", lw=0.8, ls=":", alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=0.25)

    # ── Panel C: commitment layer — matched tokens (KEY PANEL) ────────────────
    _hist_panel(
        ax_c, it_commit_match, pt_commit_match,
        title=(f"Panel C — Commitment layer: MATCHED tokens only ★  (threshold={threshold} nats)\n"
               "Same token, same prompt → purely commitment depth diff"),
        xlabel=f"Commitment layer (interpolated, threshold={threshold} nats)",
    )

    # ── Panel D: commitment layer — mismatched tokens ─────────────────────────
    _hist_panel(
        ax_d, it_commit_mismatch, pt_commit_mismatch,
        title=(f"Panel D — Commitment layer: MISMATCHED tokens  (threshold={threshold} nats)\n"
               "⚠ post-divergence steps are in different contexts — comparison is approximate"),
        xlabel=f"Commitment layer (interpolated, threshold={threshold} nats)",
    )

    fig.suptitle(
        f"E3.4 — Matched-Token Commitment Analysis  (threshold={threshold} nats)\n"
        "Controls 'IT generates harder tokens' confound via same-token filtering\n"
        "Panel C is the key test: matched tokens, IT still commits later → intrinsic IT delay",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    thresh_tag = f"_t{int(threshold*10):02d}"  # e.g. 0.1→_t01, 0.2→_t02, 0.5→_t05
    out_path = Path(output_dir) / f"plot_e3_4_matched_token{thresh_tag}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot E3.4 saved → {out_path}")
