"""
Adjacent-layer feature continuity analysis (Experiment E0b).

Tests prediction P1: Feature Population Shift.
"The Jaccard similarity between active feature sets across the dip (layer 10→12)
is lower than at adjacent non-dip layers.  This shift is sharper in IT than PT."

Raw feature ids are layer-local, so direct set overlap is not meaningful across
layers. Instead, we align features by activation-profile similarity and compute
an alignment-based continuity score:
    C(A, B) = matched(A, B) / (|A| + |B| - matched(A, B))

These are computed per prompt (per-token position = last token of the prompt in
exp4's single-pass mode, or per generation step in exp3's autoregressive mode).
Final statistics are mean ± SEM across all prompts.

Input formats
-------------
Both exp3 and exp4 active_features .npz are supported:

  exp3 .npz: each key is a prompt_id; value is object-array [n_steps, n_layers]
             of int32 arrays.

  exp4 .npz: each key is a prompt_id; value is object-array [n_layers]
             of int32 arrays.

Use load_features_exp3 / load_features_exp4 to load each format, then
compute_jaccard_stats works on either.
"""
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from src.poc.exp4.analysis.feature_alignment import (
    EventMode,
    FeatureMatch,
    build_feature_event_sets,
    continuity_for_sets,
    mutual_nearest_feature_matches,
)


# ── loaders ───────────────────────────────────────────────────────────────────

def load_features_exp3(npz_path: str) -> dict[str, np.ndarray]:
    """Load exp3 active features .npz.

    Returns dict: prompt_id → object-array [n_steps, n_layers] of int32 arrays.
    """
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def load_features_exp4(npz_path: str) -> dict[str, np.ndarray]:
    """Load exp4 active features .npz.

    Returns dict: prompt_id → object-array [n_layers] of int32 arrays.
    Reshapes to [1, n_layers] so compute_jaccard_stats sees the same interface.
    """
    data = np.load(npz_path, allow_pickle=True)
    out  = {}
    for k in data.files:
        arr = data[k]         # [n_layers] object array
        out[k] = arr[None, :] # [1, n_layers]
    return out


# ── compatibility helper ──────────────────────────────────────────────────────

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    """Jaccard index between two sets of integer feature indices.

    a, b : 1-D int32 arrays (may be empty).
    Returns float in [0, 1].  Returns NaN if both sets are empty.
    """
    set_a = set(a.tolist())
    set_b = set(b.tolist())
    union = len(set_a | set_b)
    if union == 0:
        return float("nan")
    return len(set_a & set_b) / union


def continuity_curve_for_prompt(
    af: np.ndarray,
    layer_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], list[float]]:
    """Compute alignment-based continuity for each layer pair across all steps.

    af          : [n_steps, n_layers] object array of int32 arrays
    layer_pairs : list of (layer_a, layer_b) tuples to compare

    Returns dict: (layer_a, layer_b) → list[float] of length n_steps.
    """
    n_steps  = af.shape[0]
    n_layers = af.shape[1]
    results  = {pair: [] for pair in layer_pairs}
    features_dict = {"prompt": af}

    pair_matches: dict[tuple[int, int], list[FeatureMatch]] = {}
    for la, lb in layer_pairs:
        if la >= n_layers or lb >= n_layers:
            pair_matches[(la, lb)] = []
            continue
        events_a, event_keys = build_feature_event_sets(features_dict, la, event_mode="step")
        events_b, _ = build_feature_event_sets(features_dict, lb, event_mode="step")
        pair_matches[(la, lb)] = mutual_nearest_feature_matches(
            events_a, events_b, event_keys, min_score=0.0
        )

    for step in range(n_steps):
        for la, lb in layer_pairs:
            if la >= n_layers or lb >= n_layers:
                results[(la, lb)].append(float("nan"))
                continue
            arr_a = af[step, la]
            arr_b = af[step, lb]
            set_a = set() if arr_a is None else {int(x) for x in arr_a.tolist()}
            set_b = set() if arr_b is None else {int(x) for x in arr_b.tolist()}
            results[(la, lb)].append(continuity_for_sets(set_a, set_b, pair_matches[(la, lb)]))

    return results


def _pair_matches_for_prompt(
    af: np.ndarray,
    layer_pairs: list[tuple[int, int]],
    event_mode: EventMode = "step",
) -> dict[tuple[int, int], list[FeatureMatch]]:
    n_layers = af.shape[1]
    features_dict = {"prompt": af}
    needed_layers = sorted({layer for pair in layer_pairs for layer in pair if layer < n_layers})
    layer_event_sets: dict[int, tuple[dict[int, set], list]] = {}
    for layer in needed_layers:
        layer_event_sets[layer] = build_feature_event_sets(features_dict, layer, event_mode=event_mode)
    pair_matches: dict[tuple[int, int], list[FeatureMatch]] = {}
    for la, lb in layer_pairs:
        if la >= n_layers or lb >= n_layers:
            pair_matches[(la, lb)] = []
            continue
        events_a, event_keys = layer_event_sets[la]
        events_b, _ = layer_event_sets[lb]
        pair_matches[(la, lb)] = mutual_nearest_feature_matches(
            events_a, events_b, event_keys, min_score=0.0
        )
    return pair_matches


# ── aggregate statistics ──────────────────────────────────────────────────────

def compute_continuity_stats(
    features_dict: dict[str, np.ndarray],
    analysis_start: int = 8,
    analysis_end:   int = 15,
    dip_layer:      int = 11,
    event_mode: EventMode = "step",
    top_k_per_layer: int | None = None,
) -> dict:
    """Compute alignment-based continuity statistics across the analysis range.

    Parameters
    ----------
    features_dict  : output of load_features_exp3 or load_features_exp4
    analysis_start : first layer in analysis range (inclusive)
    analysis_end   : last layer exclusive (default 15 → layers 8–14)
    dip_layer      : the gate layer; we also compute cross-dip J(dip-1, dip+1)

    Returns
    -------
    stats : dict with keys:
        "adjacent_pairs"  : list of (la, lb) tuples (consecutive pairs)
        "cross_dip_pair"  : (dip_layer - 1, dip_layer + 1)
        "all_pairs"       : all_pairs = adjacent_pairs + [cross_dip_pair]
        "mean_continuity" : dict[(la,lb) → float]  mean over prompts × steps
        "sem_continuity"  : dict[(la,lb) → float]  SEM
        "per_prompt"      : dict[(la,lb) → list[float]]  one entry per prompt
                            (each entry = mean over steps for that prompt)
        "feature_death"   : dict[int → float]  mean unmatched features@L
        "feature_birth"   : dict[int → float]  mean unmatched features@L+1
    """
    # Build pair list: consecutive pairs in analysis range + cross-dip
    adjacent_pairs = [
        (la, la + 1)
        for la in range(analysis_start, analysis_end - 1)
    ]
    cross_dip_pair = (dip_layer - 1, dip_layer + 1)
    all_pairs      = adjacent_pairs + [cross_dip_pair]

    # Accumulate per-prompt means
    per_prompt: dict[tuple, list[float]] = {p: [] for p in all_pairs}
    # Feature death/birth per adjacent pair
    death_sums:  defaultdict[int, list] = defaultdict(list)
    birth_sums:  defaultdict[int, list] = defaultdict(list)

    matches_by_pair: dict[tuple[int, int], list[int]] = defaultdict(list)

    if event_mode == "prompt":
        prompt_ids = list(features_dict.keys())
        needed_layers = sorted({layer for pair in all_pairs for layer in pair})

        global_feat_events = {
            layer: build_feature_event_sets(features_dict, layer, event_mode="prompt")
            for layer in needed_layers
        }
        if top_k_per_layer is not None:
            filtered_global_feat_events = {}
            for layer, (feat_events, event_keys) in global_feat_events.items():
                ranked = sorted(
                    feat_events.items(),
                    key=lambda item: (-len(item[1]), item[0]),
                )[:top_k_per_layer]
                filtered_global_feat_events[layer] = ({feat: events for feat, events in ranked}, event_keys)
            global_feat_events = filtered_global_feat_events
        global_matches: dict[tuple[int, int], list[FeatureMatch]] = {}
        for la, lb in all_pairs:
            events_a, event_keys = global_feat_events.get(la, ({}, []))
            events_b, _ = global_feat_events.get(lb, ({}, []))
            global_matches[(la, lb)] = mutual_nearest_feature_matches(
                events_a, events_b, event_keys, min_score=0.0
            )

        prompt_layer_sets: dict[str, dict[int, set[int]]] = {}
        allowed_features = {
            layer: set(feat_events.keys()) for layer, (feat_events, _) in global_feat_events.items()
        }
        for prompt_id, af in tqdm(
            features_dict.items(),
            desc="  Continuity stats (prompt)",
            total=len(features_dict),
            leave=False,
        ):
            layer_sets: dict[int, set[int]] = {}
            for layer in needed_layers:
                if layer >= af.shape[1]:
                    layer_sets[layer] = set()
                    continue
                feats = set()
                for step in range(af.shape[0]):
                    arr = af[step, layer]
                    if arr is not None and len(arr) > 0:
                        feats.update(int(x) for x in arr.tolist())
                if top_k_per_layer is not None:
                    feats &= allowed_features.get(layer, set())
                layer_sets[layer] = feats
            prompt_layer_sets[prompt_id] = layer_sets

        for prompt_id in prompt_ids:
            layer_sets = prompt_layer_sets[prompt_id]
            for la, lb in all_pairs:
                set_a = layer_sets.get(la, set())
                set_b = layer_sets.get(lb, set())
                pair = (la, lb)
                matches = global_matches[pair]
                value = continuity_for_sets(set_a, set_b, matches)
                if not np.isnan(value):
                    per_prompt[pair].append(float(value))
                if pair in adjacent_pairs:
                    matched_a = {m.feature_a for m in matches if m.feature_a in set_a and m.feature_b in set_b}
                    matched_b = {m.feature_b for m in matches if m.feature_a in set_a and m.feature_b in set_b}
                    death_sums[la].append(float(len(set_a - matched_a)))
                    birth_sums[la].append(float(len(set_b - matched_b)))
                    matches_by_pair[pair].append(min(len(matched_a), len(matched_b)))

        mean_c = {}
        sem_c  = {}
        for pair in all_pairs:
            vals = np.array(per_prompt[pair])
            if len(vals) == 0:
                mean_c[pair] = float("nan")
                sem_c[pair]  = float("nan")
            else:
                mean_c[pair] = float(np.nanmean(vals))
                sem_c[pair]  = float(np.nanstd(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

        feature_death = {la: float(np.mean(v)) for la, v in death_sums.items()}
        feature_birth = {la: float(np.mean(v)) for la, v in birth_sums.items()}

        return {
            "adjacent_pairs": adjacent_pairs,
            "cross_dip_pair": cross_dip_pair,
            "all_pairs":      all_pairs,
            "mean_continuity": mean_c,
            "sem_continuity":  sem_c,
            "mean_jaccard":    mean_c,
            "sem_jaccard":     sem_c,
            "per_prompt":     per_prompt,
            "feature_death":  feature_death,
            "feature_birth":  feature_birth,
            "matches_by_pair": {pair: float(np.mean(vals)) for pair, vals in matches_by_pair.items()},
            "top_k_per_layer": top_k_per_layer,
        }

    for prompt_id, af in tqdm(
        features_dict.items(),
        desc=f"  Continuity stats ({event_mode})",
        total=len(features_dict),
        leave=False,
    ):
        n_layers = af.shape[1]
        pair_matches = _pair_matches_for_prompt(af, all_pairs, event_mode=event_mode)

        n_steps = af.shape[0]

        # Feature death / birth counts at adjacent pairs
        for la, lb in all_pairs:
            if la >= n_layers or lb >= n_layers:
                continue
            pair = (la, lb)
            matches = pair_matches[pair]
            matched_a_all = {m.feature_a for m in matches}
            matched_b_all = {m.feature_b for m in matches}
            continuity_values = []
            deaths_this_prompt = []
            births_this_prompt = []
            for step in range(n_steps):
                arr_step_a = af[step, la]
                arr_step_b = af[step, lb]
                set_a = set() if arr_step_a is None else {int(x) for x in arr_step_a.tolist()}
                set_b = set() if arr_step_b is None else {int(x) for x in arr_step_b.tolist()}
                continuity_values.append(continuity_for_sets(set_a, set_b, matches))
                step_matched_a = set_a & matched_a_all
                step_matched_b = set_b & matched_b_all
                matches_by_pair[pair].append(min(len(step_matched_a), len(step_matched_b)))
                deaths_this_prompt.append(len(set_a - step_matched_a))
                births_this_prompt.append(len(set_b - step_matched_b))
            finite = [v for v in continuity_values if not np.isnan(v)]
            if finite:
                per_prompt[pair].append(float(np.mean(finite)))
            if pair in adjacent_pairs:
                death_sums[la].append(float(np.mean(deaths_this_prompt)))
                birth_sums[la].append(float(np.mean(births_this_prompt)))

    mean_c = {}
    sem_c  = {}
    for pair in all_pairs:
        vals = np.array(per_prompt[pair])
        if len(vals) == 0:
            mean_c[pair] = float("nan")
            sem_c[pair]  = float("nan")
        else:
            mean_c[pair] = float(np.nanmean(vals))
            sem_c[pair]  = float(np.nanstd(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

    feature_death = {la: float(np.mean(v)) for la, v in death_sums.items()}
    feature_birth = {la: float(np.mean(v)) for la, v in birth_sums.items()}

    return {
        "adjacent_pairs": adjacent_pairs,
        "cross_dip_pair": cross_dip_pair,
        "all_pairs":      all_pairs,
        "mean_continuity": mean_c,
        "sem_continuity":  sem_c,
        "mean_jaccard":    mean_c,
        "sem_jaccard":     sem_c,
        "per_prompt":     per_prompt,
        "feature_death":  feature_death,
        "feature_birth":  feature_birth,
        "matches_by_pair": {pair: float(np.mean(vals)) for pair, vals in matches_by_pair.items()},
        "top_k_per_layer": top_k_per_layer,
    }


compute_jaccard_stats = compute_continuity_stats
jaccard_curve_for_prompt = continuity_curve_for_prompt


def dip_summary(stats: dict, dip_layer: int = 11) -> dict:
    """Extract key dip-specific statistics from compute_continuity_stats output.

    Returns dict with:
        j_across_dip      : J(dip-1, dip)   — entering the dip
        j_exiting_dip     : J(dip, dip+1)   — exiting the dip
        j_cross_dip       : J(dip-1, dip+1) — skipping the dip
        j_control_below   : mean J over pairs below dip-1 (baseline)
        j_control_above   : mean J over pairs above dip+1 (baseline)
        death_at_dip      : feature deaths at (dip-1 → dip)
        birth_after_dip   : feature births at (dip → dip+1)
    """
    mj = stats.get("mean_continuity")
    if mj is None:
        mj = stats.get("mean_jaccard", {})

    j_across  = mj.get((dip_layer - 1, dip_layer), float("nan"))
    j_exiting = mj.get((dip_layer, dip_layer + 1), float("nan"))
    j_cross   = mj.get((dip_layer - 1, dip_layer + 1), float("nan"))

    pairs    = stats["adjacent_pairs"]
    below    = [mj[p] for p in pairs if p[1] <= dip_layer - 1]
    above    = [mj[p] for p in pairs if p[0] >= dip_layer + 1]
    j_ctrl_b = float(np.nanmean(below)) if below else float("nan")
    j_ctrl_a = float(np.nanmean(above)) if above else float("nan")

    death = stats["feature_death"].get(dip_layer - 1, float("nan"))
    birth = stats["feature_birth"].get(dip_layer, float("nan"))

    return {
        "j_across_dip":    j_across,
        "j_exiting_dip":   j_exiting,
        "j_cross_dip":     j_cross,
        "j_control_below": j_ctrl_b,
        "j_control_above": j_ctrl_a,
        "death_at_dip":    death,
        "birth_after_dip": birth,
    }
