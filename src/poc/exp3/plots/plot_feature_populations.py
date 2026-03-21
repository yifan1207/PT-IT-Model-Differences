"""
Feature population analysis at the dip: pre-dip vs post-dip features (E1a).

Tests Prediction P5: Feature Category Transition.
"Pre-dip exclusive features are predominantly lexical/syntactic.
Post-dip exclusive features are predominantly semantic/entity-level.
In IT specifically, post-dip features include format/instruction features
absent from PT's post-dip population."

Since Neuronpedia does not yet index Gemma 3 4B Scope 2 transcoder features,
we characterise feature populations by their token-type co-activation — i.e.,
for each feature we ask: what types of tokens (CONTENT, DISCOURSE, STRUCTURAL,
PUNCTUATION, OTHER) was the model generating when this feature was active?

This directly tests P5 in our own terms:
- Pre-dip exclusive + active during CONTENT → lexical/surface features
- Post-dip exclusive + active during DISCOURSE/STRUCTURAL → format features
- Post-dip exclusive in IT only → IT-specific post-transition features

Data sources:
  exp3_results.json   → generated_tokens (for token type classification)
  exp3_results.npz    → active feature indices per step × layer

Four panels:
  A: Feature population sizes — pre-dip exclusive / surviving / post-dip exclusive
     for IT and PT. Confirms the dip is a real population transition, not suppression.
  B: Token-type fingerprint of each population (IT).
     For each population, what fraction of activation events occur during each
     token type?  If pre-dip ~ CONTENT and post-dip ~ DISCOURSE/STRUCTURAL, P5 holds.
  C: Same fingerprint for PT.
     Comparison: do IT's post-dip features have more DISCOURSE/STRUCTURAL mass
     than PT's post-dip features?
  D: IT-exclusive post-dip features — features active post-dip in IT but NOT PT.
     Their token-type fingerprint should be most strongly DISCOURSE/STRUCTURAL.
     This is the direct signature of IT-specific format features.

REQUIRES: exp3_results.json + exp3_results.npz for both PT and IT.
"""
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

from src.poc.exp3.analysis.token_types import classify_generated_tokens

_PRE_DIP_LAYER  = 10
_POST_DIP_LAYER = 12
_MIN_FREQUENCY  = 0.05   # feature must appear in >=5% of prompts to count
_MAX_FEATURES   = 300    # cap per population

_TOKEN_TYPES = ["CONTENT", "DISCOURSE", "STRUCTURAL", "PUNCTUATION", "OTHER"]
_TYPE_COLORS = {
    "CONTENT":     "#1565C0",
    "DISCOURSE":   "#E65100",
    "STRUCTURAL":  "#558B2F",
    "PUNCTUATION": "#6A1B9A",
    "OTHER":       "#78909C",
}


# ── data loading ──────────────────────────────────────────────────────────────

def _load(json_path: str, npz_path: str) -> tuple[dict, object]:
    """Load results JSON and features NPZ. Returns (results_by_id, npz)."""
    with open(json_path) as f:
        records = json.load(f)
    results_by_id = {r["prompt_id"]: r for r in records}
    npz = np.load(npz_path, allow_pickle=True)
    return results_by_id, npz


# ── feature population extraction (exp3 format: [n_steps, n_layers]) ──────────

def _extract_populations(npz, min_frequency: float = _MIN_FREQUENCY,
                          max_features: int = _MAX_FEATURES) -> dict:
    """Identify pre-dip exclusive, post-dip exclusive, and surviving features.

    For exp3 format: each npz key is a prompt_id; value is [n_steps, n_layers].
    A feature is counted as "present at layer L for this prompt" if it appears
    in ANY generation step for that prompt at layer L.
    """
    prompt_ids = npz.files
    n_prompts  = len(prompt_ids)

    count_pre  = defaultdict(int)   # how many prompts activate feature at L10
    count_post = defaultdict(int)   # how many prompts activate feature at L12

    for pid in prompt_ids:
        af = npz[pid]   # [n_steps, n_layers]
        n_layers = af.shape[1]

        # Union features across all steps at each layer
        pre_set  = set()
        post_set = set()
        if _PRE_DIP_LAYER < n_layers:
            for step in range(af.shape[0]):
                arr = af[step, _PRE_DIP_LAYER]
                if arr is not None and len(arr) > 0:
                    pre_set.update(arr.tolist())
        if _POST_DIP_LAYER < n_layers:
            for step in range(af.shape[0]):
                arr = af[step, _POST_DIP_LAYER]
                if arr is not None and len(arr) > 0:
                    post_set.update(arr.tolist())

        for f in pre_set:
            count_pre[f] += 1
        for f in post_set:
            count_post[f] += 1

    min_count = max(1, int(min_frequency * n_prompts))

    frequent_pre  = {f for f, c in count_pre.items()  if c >= min_count}
    frequent_post = {f for f, c in count_post.items() if c >= min_count}

    pre_excl = sorted(frequent_pre - frequent_post,
                      key=lambda f: -count_pre[f])[:max_features]
    post_excl = sorted(frequent_post - frequent_pre,
                       key=lambda f: -count_post[f])[:max_features]
    surviving = sorted(frequent_pre & frequent_post,
                       key=lambda f: -(count_pre[f] + count_post[f]))[:max_features]

    return {
        "pre_exclusive":  set(pre_excl),
        "post_exclusive": set(post_excl),
        "surviving":      set(surviving),
        "count_pre":      count_pre,
        "count_post":     count_post,
        "n_prompts":      n_prompts,
    }


# ── token-type fingerprinting ─────────────────────────────────────────────────

def _token_type_fingerprint(
        npz, results_by_id: dict,
        feature_sets: dict,
        layer: int) -> dict[str, dict[str, float]]:
    """For each feature population, compute token-type activation fractions.

    For each (prompt, step) where a population-member feature is active at `layer`,
    record the token type generated at that step.

    Returns dict: population_name → {token_type → fraction}.
    """
    # Accumulate activation counts: population → token_type → count
    pop_counts: dict[str, dict[str, int]] = {
        pop: defaultdict(int) for pop in feature_sets
    }

    for pid in npz.files:
        af = npz[pid]  # [n_steps, n_layers]
        r  = results_by_id.get(pid)
        if r is None:
            continue

        token_types = classify_generated_tokens(r.get("generated_tokens", []))
        n_steps     = min(af.shape[0], len(token_types))

        if layer >= af.shape[1]:
            continue

        for step in range(n_steps):
            tok_type = token_types[step]
            active   = set(af[step, layer].tolist()) if af[step, layer] is not None else set()

            for pop_name, pop_set in feature_sets.items():
                if active & pop_set:   # at least one population feature active
                    pop_counts[pop_name][tok_type] += 1

    # Normalise to fractions
    fingerprints = {}
    for pop_name, counts in pop_counts.items():
        total = sum(counts.values())
        if total == 0:
            fingerprints[pop_name] = {t: 0.0 for t in _TOKEN_TYPES}
        else:
            fingerprints[pop_name] = {t: counts[t] / total for t in _TOKEN_TYPES}
    return fingerprints


def _pop_sizes(pops: dict) -> tuple[int, int, int]:
    return len(pops["pre_exclusive"]), len(pops["surviving"]), len(pops["post_exclusive"])


# ── plotting ──────────────────────────────────────────────────────────────────

def _fingerprint_panel(ax, fingerprints: dict, title: str,
                        pops_order=("pre_exclusive", "surviving", "post_exclusive"),
                        pop_labels=("Pre-dip\nexclusive", "Surviving", "Post-dip\nexclusive")):
    """Stacked bar chart: each bar = one population, segments = token types."""
    x      = np.arange(len(pops_order))
    bottom = np.zeros(len(pops_order))

    for tok_type in _TOKEN_TYPES:
        vals = np.array([fingerprints.get(p, {}).get(tok_type, 0.0) for p in pops_order])
        ax.bar(x, vals, bottom=bottom, color=_TYPE_COLORS[tok_type],
               label=tok_type, alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pop_labels, fontsize=9)
    ax.set_ylabel("Fraction of activation events")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)


def make_plot(it_json: str, it_npz: str,
              pt_json: str, pt_npz: str,
              output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for p, label in [(it_json, "IT JSON"), (it_npz, "IT NPZ"),
                     (pt_json, "PT JSON"), (pt_npz, "PT NPZ")]:
        if not Path(p).exists():
            print(f"  plot_feature_populations skipped — {label} not found: {p}")
            return

    print("  Loading IT data ...")
    it_res, it_features = _load(it_json, it_npz)
    print("  Loading PT data ...")
    pt_res, pt_features = _load(pt_json, pt_npz)

    print("  Extracting IT feature populations ...")
    it_pops = _extract_populations(it_features)
    print("  Extracting PT feature populations ...")
    pt_pops = _extract_populations(pt_features)

    # IT-exclusive post-dip features: active post-dip in IT but not in PT's post-dip
    it_only_post = it_pops["post_exclusive"] - pt_pops["post_exclusive"]

    # Token-type fingerprints
    # For pre-dip populations: use pre_dip_layer
    # For post-dip populations: use post_dip_layer
    # Surviving: use pre_dip_layer (they exist at both)
    it_fp_pre = _token_type_fingerprint(
        it_features, it_res,
        {"pre_exclusive": it_pops["pre_exclusive"],
         "surviving":     it_pops["surviving"],
         "post_exclusive": it_pops["post_exclusive"]},
        layer=_PRE_DIP_LAYER,
    )
    # Re-check post-dip features at the post-dip layer
    it_fp_post = _token_type_fingerprint(
        it_features, it_res,
        {"pre_exclusive": it_pops["pre_exclusive"],
         "surviving":     it_pops["surviving"],
         "post_exclusive": it_pops["post_exclusive"]},
        layer=_POST_DIP_LAYER,
    )
    # Merge: pre_excl fingerprint from pre-dip layer; post_excl from post-dip layer;
    # surviving average both layers
    it_fp = {
        "pre_exclusive":  it_fp_pre["pre_exclusive"],
        "post_exclusive": it_fp_post["post_exclusive"],
        "surviving": {t: (it_fp_pre["surviving"][t] + it_fp_post["surviving"][t]) / 2
                      for t in _TOKEN_TYPES},
    }

    pt_fp_pre = _token_type_fingerprint(
        pt_features, pt_res,
        {"pre_exclusive": pt_pops["pre_exclusive"],
         "surviving":     pt_pops["surviving"],
         "post_exclusive": pt_pops["post_exclusive"]},
        layer=_PRE_DIP_LAYER,
    )
    pt_fp_post = _token_type_fingerprint(
        pt_features, pt_res,
        {"pre_exclusive": pt_pops["pre_exclusive"],
         "surviving":     pt_pops["surviving"],
         "post_exclusive": pt_pops["post_exclusive"]},
        layer=_POST_DIP_LAYER,
    )
    pt_fp = {
        "pre_exclusive":  pt_fp_pre["pre_exclusive"],
        "post_exclusive": pt_fp_post["post_exclusive"],
        "surviving": {t: (pt_fp_pre["surviving"][t] + pt_fp_post["surviving"][t]) / 2
                      for t in _TOKEN_TYPES},
    }

    it_only_fp = _token_type_fingerprint(
        it_features, it_res,
        {"it_only_post": it_only_post},
        layer=_POST_DIP_LAYER,
    )

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    # ── Panel A: population sizes IT vs PT ────────────────────────────────────
    it_sizes = _pop_sizes(it_pops)
    pt_sizes = _pop_sizes(pt_pops)
    pop_labels = ["Pre-dip\nexclusive", "Surviving", "Post-dip\nexclusive"]
    x_a   = np.arange(3)
    bar_w = 0.35

    ax_a.bar(x_a - bar_w / 2, it_sizes, bar_w, color="#E65100", alpha=0.85, label="IT")
    ax_a.bar(x_a + bar_w / 2, pt_sizes, bar_w, color="#1565C0", alpha=0.85, label="PT")

    for xi, v in zip(x_a - bar_w / 2, it_sizes):
        ax_a.text(xi, v + 1, str(v), ha="center", va="bottom", fontsize=8,
                  color="#E65100")
    for xi, v in zip(x_a + bar_w / 2, pt_sizes):
        ax_a.text(xi, v + 1, str(v), ha="center", va="bottom", fontsize=8,
                  color="#1565C0")

    ax_a.set_xticks(x_a)
    ax_a.set_xticklabels(pop_labels, fontsize=9)
    ax_a.set_ylabel(f"Feature count (≥{_MIN_FREQUENCY:.0%} of prompts)")
    ax_a.set_title(
        f"Panel A — Feature population sizes  IT vs PT\n"
        f"(L{_PRE_DIP_LAYER} exclusive vs surviving vs L{_POST_DIP_LAYER} exclusive)\n"
        f"IT-only post-dip features: {len(it_only_post)}"
    )
    ax_a.legend(fontsize=9)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: IT token-type fingerprint ────────────────────────────────────
    _fingerprint_panel(ax_b, it_fp,
                       f"Panel B — IT token-type fingerprint per population\n"
                       f"Pre-dip~CONTENT = lexical; Post-dip~DISCOURSE/STRUCTURAL = format")
    ax_b.legend(fontsize=7, ncol=3, loc="upper right")

    # ── Panel C: PT token-type fingerprint ────────────────────────────────────
    _fingerprint_panel(ax_c, pt_fp,
                       f"Panel C — PT token-type fingerprint per population\n"
                       f"PT post-dip should have less DISCOURSE/STRUCTURAL than IT")
    ax_c.legend(fontsize=7, ncol=3, loc="upper right")

    # ── Panel D: IT-only post-dip features fingerprint ────────────────────────
    n_it_only = len(it_only_post)
    if n_it_only > 0 and "it_only_post" in it_only_fp:
        _fingerprint_panel(
            ax_d, {"it_only_post": it_only_fp["it_only_post"]},
            f"Panel D — IT-only post-dip features (n={n_it_only})\n"
            f"Features active post-dip in IT but NOT in PT — should be most DISCOURSE/STRUCTURAL",
            pops_order=("it_only_post",),
            pop_labels=("IT-only\npost-dip",),
        )

        # Also overlay a text table of the top-5 IT-only post-dip features by frequency
        sorted_feats = sorted(it_only_post,
                              key=lambda f: -it_pops["count_post"].get(f, 0))[:5]
        lines = ["Top IT-only post-dip features:"]
        for f in sorted_feats:
            c = it_pops["count_post"].get(f, 0)
            lines.append(f"  feat {f:5d}  (active in {c}/{it_pops['n_prompts']} prompts)")
        ax_d.text(0.98, 0.60, "\n".join(lines), transform=ax_d.transAxes,
                  ha="right", va="top", fontsize=7.5,
                  bbox=dict(facecolor="white", alpha=0.85, edgecolor="grey"))
        ax_d.legend(fontsize=7, loc="upper left")
    else:
        ax_d.text(0.5, 0.5,
                  f"No IT-exclusive post-dip features found\n"
                  f"(all {len(it_pops['post_exclusive'])} IT post-dip features\n"
                  f"also appear in PT post-dip set)",
                  ha="center", va="center", transform=ax_d.transAxes, fontsize=11)
        ax_d.set_title("Panel D — IT-only post-dip features\n(none found at current threshold)")

    fig.suptitle(
        "Feature Population Analysis at the Dip  (E1a)\n"
        "Pre-dip exclusive vs surviving vs post-dip exclusive — characterised by token-type co-activation",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()

    out_path = Path(output_dir) / "plot_feature_populations.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature population plot saved → {out_path}")
