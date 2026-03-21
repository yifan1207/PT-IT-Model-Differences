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
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from src.poc.exp3.analysis.token_types import classify_generated_tokens
from src.poc.exp4.analysis.feature_alignment import (
    build_feature_event_sets,
    build_population_partition,
    mutual_nearest_feature_matches,
)

_PRE_DIP_LAYER  = 10
_POST_DIP_LAYER = 12
_MIN_FREQUENCY  = 0.05   # feature must appear in >=5% of prompts to count
_MATCH_TOP_K    = 512
_TOKEN_TYPES = ["CONTENT", "FUNCTION", "DISCOURSE", "STRUCTURAL", "PUNCTUATION", "OTHER"]
_TYPE_COLORS = {
    "CONTENT":     "#1565C0",
    "FUNCTION":    "#00897B",
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


def _npz_to_feature_dict(npz) -> dict[str, np.ndarray]:
    return {pid: npz[pid] for pid in npz.files}


def _build_prompt_cache(npz, results_by_id: dict, layers: tuple[int, ...]) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for pid in tqdm(npz.files, desc="  Caching prompt states", leave=False):
        af = npz[pid]
        r = results_by_id.get(pid)
        if r is None:
            continue
        token_types = classify_generated_tokens(r.get("generated_tokens", []))
        n_steps = min(af.shape[0], len(token_types))
        active_by_layer: dict[int, list[set[int]]] = {}
        for layer in layers:
            if layer >= af.shape[1]:
                active_by_layer[layer] = []
                continue
            step_sets: list[set[int]] = []
            for step in range(n_steps):
                arr = af[step, layer]
                step_sets.append(set() if arr is None else {int(x) for x in arr.tolist()})
            active_by_layer[layer] = step_sets
        cache[pid] = {
            "token_types": token_types[:n_steps],
            "active_by_layer": active_by_layer,
            "n_steps": n_steps,
        }
    return cache


def _aggregate_fingerprints(
    prompt_cache: dict[str, dict],
    feature_sets_by_layer: dict[int, dict[str, set[int]]],
) -> dict[str, dict[str, float]]:
    pop_counts: dict[str, dict[str, int]] = {}
    for populations in feature_sets_by_layer.values():
        for pop_name in populations:
            pop_counts.setdefault(pop_name, defaultdict(int))

    for prompt_data in tqdm(prompt_cache.values(), desc="  Aggregating fingerprints", leave=False):
        token_types = prompt_data["token_types"]
        active_by_layer = prompt_data["active_by_layer"]
        n_steps = prompt_data["n_steps"]
        for layer, populations in feature_sets_by_layer.items():
            step_sets = active_by_layer.get(layer, [])
            if not step_sets:
                continue
            for step in range(min(n_steps, len(step_sets))):
                tok_type = token_types[step]
                active = step_sets[step]
                for pop_name, pop_set in populations.items():
                    if active & pop_set:
                        pop_counts[pop_name][tok_type] += 1

    fingerprints = {}
    for pop_name, counts in pop_counts.items():
        total = sum(counts.values())
        fingerprints[pop_name] = {
            t: (counts[t] / total if total else 0.0) for t in _TOKEN_TYPES
        }
    return fingerprints


def _pop_sizes(pops: dict) -> tuple[int, int, int]:
    return len(pops["pre_exclusive"]), len(pops["surviving_pairs"]), len(pops["post_exclusive"])


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
    print("  Building per-prompt caches ...")
    it_cache = _build_prompt_cache(it_features, it_res, (_PRE_DIP_LAYER, _POST_DIP_LAYER))
    pt_cache = _build_prompt_cache(pt_features, pt_res, (_PRE_DIP_LAYER, _POST_DIP_LAYER))

    print("  Aligning IT feature populations ...")
    it_dict = _npz_to_feature_dict(it_features)
    it_pops = build_population_partition(
        it_dict,
        pre_layer=_PRE_DIP_LAYER,
        post_layer=_POST_DIP_LAYER,
        min_frequency=_MIN_FREQUENCY,
        event_mode="prompt",
        top_k_for_matching=_MATCH_TOP_K,
    )
    print("  Aligning PT feature populations ...")
    pt_dict = _npz_to_feature_dict(pt_features)
    pt_pops = build_population_partition(
        pt_dict,
        pre_layer=_PRE_DIP_LAYER,
        post_layer=_POST_DIP_LAYER,
        min_frequency=_MIN_FREQUENCY,
        event_mode="prompt",
        top_k_for_matching=_MATCH_TOP_K,
    )

    post_event_keys = [pid for pid in it_dict if pid in pt_dict]
    it_post_events, _ = build_feature_event_sets(it_dict, _POST_DIP_LAYER, event_mode="prompt")
    pt_post_events, _ = build_feature_event_sets(pt_dict, _POST_DIP_LAYER, event_mode="prompt")
    it_post_events = dict(sorted(
        it_post_events.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[:_MATCH_TOP_K])
    pt_post_events = dict(sorted(
        pt_post_events.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[:_MATCH_TOP_K])
    cross_matches = mutual_nearest_feature_matches(
        pt_post_events,
        it_post_events,
        post_event_keys,
        min_score=0.20,
    )
    matched_it_post = {m.feature_b for m in cross_matches}
    it_only_post = {
        feat for feat in it_pops["post_exclusive"]
        if feat not in matched_it_post
    }

    it_fps = _aggregate_fingerprints(
        it_cache,
        {
            _PRE_DIP_LAYER: {
                "pre_exclusive": it_pops["pre_exclusive"],
                "surviving_pre": it_pops["surviving_pre"],
            },
            _POST_DIP_LAYER: {
                "post_exclusive": it_pops["post_exclusive"],
                "surviving_post": it_pops["surviving_post"],
                "it_only_post": it_only_post,
            },
        },
    )
    it_fp = {
        "pre_exclusive":  it_fps["pre_exclusive"],
        "post_exclusive": it_fps["post_exclusive"],
        "surviving": {t: (it_fps["surviving_pre"][t] + it_fps["surviving_post"][t]) / 2
                      for t in _TOKEN_TYPES},
    }
    pt_fps = _aggregate_fingerprints(
        pt_cache,
        {
            _PRE_DIP_LAYER: {
                "pre_exclusive": pt_pops["pre_exclusive"],
                "surviving_pre": pt_pops["surviving_pre"],
            },
            _POST_DIP_LAYER: {
                "post_exclusive": pt_pops["post_exclusive"],
                "surviving_post": pt_pops["surviving_post"],
            },
        },
    )
    pt_fp = {
        "pre_exclusive":  pt_fps["pre_exclusive"],
        "post_exclusive": pt_fps["post_exclusive"],
        "surviving": {t: (pt_fps["surviving_pre"][t] + pt_fps["surviving_post"][t]) / 2
                      for t in _TOKEN_TYPES},
    }
    it_only_fp = {"it_only_post": it_fps.get("it_only_post", {t: 0.0 for t in _TOKEN_TYPES})}

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
        f"(alignment-based L{_PRE_DIP_LAYER} exclusive vs surviving vs L{_POST_DIP_LAYER} exclusive)\n"
        f"matching on top {_MATCH_TOP_K} recurrent features/layer; IT-only post-dip: {len(it_only_post)}"
    )
    ax_a.legend(fontsize=9)
    ax_a.grid(axis="y", alpha=0.25)

    # ── Panel B: IT token-type fingerprint ────────────────────────────────────
    _fingerprint_panel(ax_b, it_fp,
                       "Panel B — IT token-type fingerprint per population\n"
                       "Pre-dip~CONTENT/FUNCTION = lexical; Post-dip~DISCOURSE/STRUCTURAL = format")
    ax_b.legend(fontsize=7, ncol=3, loc="upper right")

    # ── Panel C: PT token-type fingerprint ────────────────────────────────────
    _fingerprint_panel(ax_c, pt_fp,
                       "Panel C — PT token-type fingerprint per population\n"
                       "PT post-dip should have less DISCOURSE/STRUCTURAL than IT")
    ax_c.legend(fontsize=7, ncol=3, loc="upper right")

    # ── Panel D: IT-only post-dip features fingerprint ────────────────────────
    n_it_only = len(it_only_post)
    if n_it_only > 0 and "it_only_post" in it_only_fp:
        _fingerprint_panel(
            ax_d, {"it_only_post": it_only_fp["it_only_post"]},
            f"Panel D — IT-only post-dip features (n={n_it_only})\n"
            "Features active post-dip in IT but unmatched to PT post-dip features",
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
