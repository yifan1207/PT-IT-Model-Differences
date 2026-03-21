"""
Attention pattern analysis for the dip layer comparison (Experiment E0a).

Tests prediction P2: Attention Divergence at the Dip.
"Layers 8–11 show the largest attention divergence between PT and IT despite
near-identical MLP weights.  IT attention entropy is lower (more focused) at
these layers.  Instruction attention mass is higher in IT at layers 8–11."

Three metrics (all computed from attention entropy stored in results JSON):
  1. Mean attention entropy per layer per head — H(layer, head)
     Shows whether IT focuses more sharply than PT at the dip layers.

  2. PT-IT attention divergence per layer
     Mean absolute entropy difference: |H_IT(layer) - H_PT(layer)|
     Large divergence at dip layers confirms attention-mediated sharpening.

  3. Attention concentration index
     Gini coefficient of the entropy distribution across heads per layer.
     Higher Gini = more heterogeneous focus across heads.

Input format (from exp4_results.json)
--------------------------------------
Each result record has:
    attn_entropy : list[n_layers] of list[n_heads] or None

If attn_entropy is None for all records, functions return NaN arrays.

Note on local vs global attention layers in Gemma 3 4B:
    Global layers (full context, i % 6 == 5): 5, 11, 17, 23, 29
    Local layers  (window=1024):               all others
    Entropy is lower for local layers (bounded by window context).
    We mark global vs local in outputs so plots can annotate accordingly.
"""
import numpy as np
from collections import defaultdict

# Gemma 3 4B: global attention at layers divisible by 6 after index 5
GLOBAL_ATTN_LAYERS = frozenset(i for i in range(34) if i % 6 == 5)


# ── loaders ───────────────────────────────────────────────────────────────────

def extract_attn_entropy(
    results: list[dict],
    n_layers: int = 34,
) -> dict:
    """Extract per-layer, per-head entropy from results JSON.

    Returns
    -------
    dict with keys:
        "by_layer"     : dict[int → list[list[float]]]
                         layer → list-of-records, each record = [n_heads] entropies
        "n_records"    : int — number of prompts with non-None entropy
        "n_heads"      : int — number of attention heads (inferred)
        "n_layers"     : int
    """
    by_layer: dict[int, list] = defaultdict(list)
    n_records = 0
    n_heads   = 8  # default for Gemma 3 4B

    for rec in results:
        attn_ent = rec.get("attn_entropy")
        if attn_ent is None:
            continue
        n_records += 1
        for layer_i, head_entropies in enumerate(attn_ent):
            if head_entropies is None:
                continue
            if len(head_entropies) > 0:
                n_heads = len(head_entropies)
            by_layer[layer_i].append(head_entropies)

    return {
        "by_layer":  dict(by_layer),
        "n_records": n_records,
        "n_heads":   n_heads,
        "n_layers":  n_layers,
    }


# ── metric computation ─────────────────────────────────────────────────────────

def compute_mean_entropy_profile(
    entropy_data: dict,
    n_layers: int = 34,
) -> dict:
    """Compute per-layer mean and SEM of attention entropy across prompts.

    Parameters
    ----------
    entropy_data : output of extract_attn_entropy

    Returns
    -------
    dict with keys:
        "layers"         : list[int] 0..n_layers-1
        "mean_entropy"   : [n_layers] float — mean across prompts and heads
        "sem_entropy"    : [n_layers] float — SEM
        "mean_per_head"  : [n_layers, n_heads] float — per-head mean
        "is_global"      : [n_layers] bool — True for global attention layers
    """
    by_layer = entropy_data["by_layer"]
    n_heads  = entropy_data["n_heads"]
    layers   = list(range(n_layers))

    mean_all  = []
    sem_all   = []
    mean_head = []

    for layer_i in layers:
        records = by_layer.get(layer_i, [])
        if not records:
            mean_all.append(float("nan"))
            sem_all.append(float("nan"))
            mean_head.append([float("nan")] * n_heads)
            continue

        arr = np.array(records, dtype=np.float64)  # [n_prompts, n_heads]
        mean_all.append(float(np.nanmean(arr)))
        sem_all.append(float(np.nanstd(arr) / np.sqrt(arr.size)) if arr.size > 1 else 0.0)
        per_head = [float(np.nanmean(arr[:, h])) for h in range(arr.shape[1])]
        mean_head.append(per_head)

    return {
        "layers":        layers,
        "mean_entropy":  mean_all,
        "sem_entropy":   sem_all,
        "mean_per_head": mean_head,
        "is_global":     [i in GLOBAL_ATTN_LAYERS for i in layers],
    }


def compute_entropy_divergence(
    pt_entropy: dict,
    it_entropy: dict,
    n_layers: int = 34,
) -> dict:
    """Compute per-layer absolute entropy divergence between PT and IT.

    Both inputs are outputs of compute_mean_entropy_profile.

    Returns
    -------
    dict with keys:
        "layers"         : list[int]
        "abs_diff"       : [n_layers] |mean_IT - mean_PT|
        "rel_diff"       : [n_layers] abs_diff / max(mean_PT, 1e-6)
        "direction"      : [n_layers] +1 if IT > PT (IT is less focused), -1 otherwise
    """
    layers  = list(range(n_layers))
    pt_mean = pt_entropy["mean_entropy"]
    it_mean = it_entropy["mean_entropy"]

    abs_diff  = []
    rel_diff  = []
    direction = []

    for layer_i in layers:
        pt_val = pt_mean[layer_i]
        it_val = it_mean[layer_i]
        if np.isnan(pt_val) or np.isnan(it_val):
            abs_diff.append(float("nan"))
            rel_diff.append(float("nan"))
            direction.append(0)
        else:
            diff = it_val - pt_val
            abs_diff.append(abs(diff))
            rel_diff.append(abs(diff) / max(abs(pt_val), 1e-6))
            direction.append(1 if diff > 0 else -1)

    return {
        "layers":    layers,
        "abs_diff":  abs_diff,
        "rel_diff":  rel_diff,
        "direction": direction,
    }


def compute_entropy_by_category(
    results: list[dict],
    entropy_data: dict,
) -> dict[str, dict]:
    """Split attention entropy statistics by prompt category.

    Returns dict: category → compute_mean_entropy_profile output.
    Useful for checking whether the dip attention pattern is task-independent.
    """
    # Group record indices by category
    cat_to_records: dict[str, list] = defaultdict(list)
    for rec in results:
        cat = rec.get("category", "unknown")
        attn_ent = rec.get("attn_entropy")
        if attn_ent is not None:
            cat_to_records[cat].append(attn_ent)

    out = {}
    for cat, rec_list in cat_to_records.items():
        # Build a synthetic results list for extract_attn_entropy
        fake_results = [{"attn_entropy": ae} for ae in rec_list]
        ent_data = extract_attn_entropy(fake_results, n_layers=entropy_data["n_layers"])
        out[cat] = compute_mean_entropy_profile(ent_data, n_layers=entropy_data["n_layers"])

    return out


def summarise_dip_region(profile: dict, dip_layer: int = 11, window: int = 3) -> dict:
    """Summarise attention entropy in the dip region vs. surrounding layers.

    Parameters
    ----------
    profile    : output of compute_mean_entropy_profile
    dip_layer  : the gate layer (default 11)
    window     : layers around dip to include in "dip region"

    Returns
    -------
    dict with keys:
        "dip_region_mean"  : mean entropy over [dip - window, dip + window]
        "pre_dip_mean"     : mean entropy over layers < dip - window
        "post_dip_mean"    : mean entropy over layers > dip + window
        "dip_min_layer"    : layer with minimum entropy around the dip
        "entropy_at_dip"   : entropy at exactly dip_layer
    """
    layers = profile["layers"]
    mean_e = profile["mean_entropy"]

    dip_start = max(0, dip_layer - window)
    dip_end   = min(len(layers) - 1, dip_layer + window)

    dip_vals  = [mean_e[l] for l in range(dip_start, dip_end + 1) if not np.isnan(mean_e[l])]
    pre_vals  = [mean_e[l] for l in range(0, dip_start) if not np.isnan(mean_e[l])]
    post_vals = [mean_e[l] for l in range(dip_end + 1, len(layers)) if not np.isnan(mean_e[l])]

    # Layer with minimum entropy in the dip region
    dip_region_layers = range(dip_start, dip_end + 1)
    dip_min_layer = min(
        (l for l in dip_region_layers if not np.isnan(mean_e[l])),
        key=lambda l: mean_e[l],
        default=dip_layer,
    )

    return {
        "dip_region_mean":  float(np.mean(dip_vals)) if dip_vals else float("nan"),
        "pre_dip_mean":     float(np.mean(pre_vals))  if pre_vals  else float("nan"),
        "post_dip_mean":    float(np.mean(post_vals)) if post_vals else float("nan"),
        "dip_min_layer":    dip_min_layer,
        "entropy_at_dip":   mean_e[dip_layer] if dip_layer < len(mean_e) else float("nan"),
    }
