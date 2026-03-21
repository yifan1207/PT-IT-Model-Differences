"""
Adjacent-layer Jaccard similarity analysis (Experiment E0b).

Tests prediction P1: Feature Population Shift.
"The Jaccard similarity between active feature sets across the dip (layer 10→12)
is lower than at adjacent non-dip layers.  This shift is sharper in IT than PT."

The Jaccard index between two sets A, B is:
    J(A, B) = |A ∩ B| / |A ∪ B|

Value of 1.0 = identical feature sets.
Value of 0.0 = no overlap.

We compute:
  - J(L, L+1) for each consecutive pair of layers L in analysis_range
  - J(dip_layer - 1, dip_layer + 1) — cross-dip similarity (skipping dip layer)

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
from pathlib import Path


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


# ── Jaccard primitives ────────────────────────────────────────────────────────

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


def jaccard_curve_for_prompt(
    af: np.ndarray,
    layer_pairs: list[tuple[int, int]],
) -> dict[tuple[int, int], list[float]]:
    """Compute Jaccard for each layer pair across all steps.

    af          : [n_steps, n_layers] object array of int32 arrays
    layer_pairs : list of (layer_a, layer_b) tuples to compare

    Returns dict: (layer_a, layer_b) → list[float] of length n_steps.
    """
    n_steps  = af.shape[0]
    n_layers = af.shape[1]
    results  = {pair: [] for pair in layer_pairs}

    for step in range(n_steps):
        for la, lb in layer_pairs:
            if la >= n_layers or lb >= n_layers:
                results[(la, lb)].append(float("nan"))
                continue
            j = jaccard(af[step, la], af[step, lb])
            results[(la, lb)].append(j)

    return results


# ── aggregate statistics ──────────────────────────────────────────────────────

def compute_jaccard_stats(
    features_dict: dict[str, np.ndarray],
    analysis_start: int = 8,
    analysis_end:   int = 15,
    dip_layer:      int = 11,
) -> dict:
    """Compute Jaccard statistics for all prompts across the analysis range.

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
        "mean_jaccard"    : dict[(la,lb) → float]  mean over prompts × steps
        "sem_jaccard"     : dict[(la,lb) → float]  SEM
        "per_prompt"      : dict[(la,lb) → list[float]]  one entry per prompt
                            (each entry = mean over steps for that prompt)
        "feature_death"   : dict[int → float]  mean |features@L \ features@L+1|
        "feature_birth"   : dict[int → float]  mean |features@L+1 \ features@L|
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

    for prompt_id, af in features_dict.items():
        n_steps  = af.shape[0]
        n_layers = af.shape[1]

        pair_values = jaccard_curve_for_prompt(af, all_pairs)
        for pair, values in pair_values.items():
            finite = [v for v in values if not np.isnan(v)]
            if finite:
                per_prompt[pair].append(float(np.mean(finite)))

        # Feature death / birth counts at adjacent pairs
        for la, lb in adjacent_pairs:
            if la >= n_layers or lb >= n_layers:
                continue
            deaths_this_prompt = []
            births_this_prompt = []
            for step in range(n_steps):
                set_a = set(af[step, la].tolist())
                set_b = set(af[step, lb].tolist())
                deaths_this_prompt.append(len(set_a - set_b))
                births_this_prompt.append(len(set_b - set_a))
            death_sums[la].append(float(np.mean(deaths_this_prompt)))
            birth_sums[la].append(float(np.mean(births_this_prompt)))

    mean_j = {}
    sem_j  = {}
    for pair in all_pairs:
        vals = np.array(per_prompt[pair])
        if len(vals) == 0:
            mean_j[pair] = float("nan")
            sem_j[pair]  = float("nan")
        else:
            mean_j[pair] = float(np.nanmean(vals))
            sem_j[pair]  = float(np.nanstd(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

    feature_death = {la: float(np.mean(v)) for la, v in death_sums.items()}
    feature_birth = {la: float(np.mean(v)) for la, v in birth_sums.items()}

    return {
        "adjacent_pairs": adjacent_pairs,
        "cross_dip_pair": cross_dip_pair,
        "all_pairs":      all_pairs,
        "mean_jaccard":   mean_j,
        "sem_jaccard":    sem_j,
        "per_prompt":     per_prompt,
        "feature_death":  feature_death,
        "feature_birth":  feature_birth,
    }


def dip_summary(stats: dict, dip_layer: int = 11) -> dict:
    """Extract key dip-specific statistics from compute_jaccard_stats output.

    Returns dict with:
        j_across_dip      : J(dip-1, dip)   — entering the dip
        j_exiting_dip     : J(dip, dip+1)   — exiting the dip
        j_cross_dip       : J(dip-1, dip+1) — skipping the dip
        j_control_below   : mean J over pairs below dip-1 (baseline)
        j_control_above   : mean J over pairs above dip+1 (baseline)
        death_at_dip      : feature deaths at (dip-1 → dip)
        birth_after_dip   : feature births at (dip → dip+1)
    """
    mj = stats["mean_jaccard"]

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
