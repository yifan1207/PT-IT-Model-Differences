"""
Intrinsic dimension (ID) estimation via TwoNN (Experiment E0c).

Tests prediction P6: ID Geometry Confirmation.
"PT shows smooth ID expansion→peak→contraction.  IT shows the same peak but
with sharper contraction at the dip layer, confirming the L0 finding with
an independent methodology that does not rely on transcoders."

Method
------
Uses the Two-Nearest-Neighbour (TwoNN) estimator (Facco et al. 2017):
  - For each point x_i, compute r1_i = distance to nearest neighbour,
                                r2_i = distance to second nearest neighbour.
  - The ratio mu_i = r2_i / r1_i follows a Pareto distribution whose exponent
    equals the intrinsic dimension.
  - Fit by MLE: plot -log(1-F(mu)) vs -log(mu) and read the slope.

Why TwoNN?
  - Only needs distance to the two nearest neighbours: O(n * d) with scikit-learn.
  - Does not require choosing a neighbourhood size k (unlike MLE / lPCA).
  - Works well with n ≈ 200 data points (the original paper used n ≥ 100).
  - Consistent estimator: unbiased as n → ∞.

Input
-----
Residual vectors saved by collect.py as {cfg.residuals_path} (.npz):
    Key: prompt_id
    Value: float32 [n_layers, d_model]

We stack across prompts to get [n_prompts, d_model] per layer, then estimate
ID independently for each layer.

Output
------
id_profile : dict with keys
    "layers"         : list[int]  — layer indices 0..33
    "id_twonn"       : list[float] — TwoNN estimate per layer
    "id_mle"         : list[float] — MLE (Levina-Bickel) estimate (fallback)
    "n_samples"      : int  — number of vectors used (= n_prompts)
    "model_variant"  : str  — "pt" or "it"
"""
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import skdim
    _SKDIM_AVAILABLE = True
except ImportError:
    _SKDIM_AVAILABLE = False
    print("[WARN] scikit-dimension not found; install with: uv add scikit-dimension")


# ── loaders ───────────────────────────────────────────────────────────────────

def load_residuals(npz_path: str) -> dict[str, np.ndarray]:
    """Load residuals .npz → dict: prompt_id → [n_layers, d_model]."""
    data = np.load(npz_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def stack_by_layer(
    residuals_dict: dict[str, np.ndarray],
    n_layers: int = 34,
) -> list[np.ndarray]:
    """Stack per-prompt residuals into per-layer matrices.

    Returns list of length n_layers; each element is [n_prompts, d_model].
    """
    prompt_ids = sorted(residuals_dict.keys())
    by_layer   = [[] for _ in range(n_layers)]

    for pid in prompt_ids:
        arr = residuals_dict[pid]   # [n_layers, d_model]
        for layer in range(min(n_layers, arr.shape[0])):
            by_layer[layer].append(arr[layer])  # [d_model]

    return [np.array(vecs, dtype=np.float32) for vecs in by_layer]


# ── TwoNN estimator ────────────────────────────────────────────────────────────

def estimate_id_twonn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """Estimate intrinsic dimension using TwoNN (Facco et al. 2017).

    X                 : [n_samples, n_features]
    discard_fraction  : fraction of largest mu values to discard (default 10%)

    Returns float ID estimate.  Returns NaN if scikit-dimension is unavailable
    or if n_samples < 3.
    """
    if not _SKDIM_AVAILABLE:
        return float("nan")
    if X.shape[0] < 3:
        return float("nan")

    estimator = skdim.id.TwoNN(discard_fraction=discard_fraction)
    try:
        estimator.fit(X.astype(np.float64))
        return float(estimator.dimension_)
    except Exception as e:
        print(f"    [WARN] TwoNN failed: {e}")
        return float("nan")


def estimate_id_mle(X: np.ndarray, k: int = 5) -> float:
    """Estimate ID using Maximum Likelihood (Levina-Bickel).

    k  : neighbourhood size for local density estimation.
    Fallback / cross-validation for TwoNN.
    Returns NaN if scikit-dimension is unavailable or n_samples < k+2.
    """
    if not _SKDIM_AVAILABLE:
        return float("nan")
    if X.shape[0] < k + 2:
        return float("nan")

    try:
        estimator = skdim.id.MLE(K=k)
        estimator.fit(X.astype(np.float64))
        return float(estimator.dimension_)
    except Exception as e:
        # skdim.id.MLE uses inspect.getargvalues which is broken on Python 3.13
        # (FrameLocalsProxy incompatibility).  Fall back silently.
        return float("nan")


# ── profile computation ────────────────────────────────────────────────────────

def compute_id_profile(
    npz_path: str,
    model_variant: str,
    n_layers: int = 34,
    discard_fraction: float = 0.1,
    run_mle: bool = True,
) -> dict:
    """Compute TwoNN (and optionally MLE) ID at every layer.

    Parameters
    ----------
    npz_path        : path to exp4_residuals.npz
    model_variant   : "pt" or "it"
    n_layers        : number of transformer layers (default 34)
    discard_fraction: TwoNN hyperparameter (see estimate_id_twonn)
    run_mle         : also compute MLE estimates (slower; good cross-check)

    Returns
    -------
    id_profile dict as described in module docstring.
    """
    print(f"  Loading residuals from {npz_path} ...")
    residuals_dict = load_residuals(npz_path)
    n_prompts = len(residuals_dict)
    print(f"  Stacking {n_prompts} prompts × {n_layers} layers ...")
    by_layer = stack_by_layer(residuals_dict, n_layers=n_layers)

    id_twonn = []
    id_mle   = []
    for layer_i, X in enumerate(by_layer):
        id_val = estimate_id_twonn(X, discard_fraction=discard_fraction)
        id_twonn.append(id_val)
        if run_mle:
            id_mle.append(estimate_id_mle(X))
        print(f"    Layer {layer_i:2d}: TwoNN={id_val:.2f}"
              + (f"  MLE={id_mle[-1]:.2f}" if run_mle else ""))

    return {
        "layers":        list(range(n_layers)),
        "id_twonn":      id_twonn,
        "id_mle":        id_mle,
        "n_samples":     n_prompts,
        "model_variant": model_variant,
    }


def compare_profiles(pt_profile: dict, it_profile: dict) -> dict:
    """Compute comparison statistics between PT and IT ID profiles.

    Returns dict with:
        "dip_sharpness_pt"  : |ID[dip-1] - ID[dip+1]| / ID[dip-1]  for PT
        "dip_sharpness_it"  : same for IT
        "sharpening_ratio"  : dip_sharpness_it / dip_sharpness_pt
        "id_peak_layer_pt"  : layer with max ID in PT
        "id_peak_layer_it"  : layer with max ID in IT
        "id_at_dip_pt"      : ID at dip_layer for PT
        "id_at_dip_it"      : ID at dip_layer for IT
    """
    def _dip_sharpness(profile: dict, dip: int) -> float:
        ids = profile["id_twonn"]
        if dip <= 0 or dip + 1 >= len(ids):
            return float("nan")
        pre  = ids[dip - 1]
        post = ids[dip + 1]
        dip_val = ids[dip]
        baseline = max(pre, post, 1e-6)
        return abs(pre - post) / baseline

    layers_pt = pt_profile["id_twonn"]
    layers_it = it_profile["id_twonn"]

    # Find the dip layer by looking for a local minimum between layers 8-16
    # (rough range based on the hypothesis)
    dip_search_range = range(8, min(17, len(layers_it)))
    if dip_search_range:
        dip_layer_it = min(dip_search_range,
                          key=lambda l: layers_it[l] if not np.isnan(layers_it[l]) else np.inf)
        dip_layer_pt = min(dip_search_range,
                          key=lambda l: layers_pt[l] if not np.isnan(layers_pt[l]) else np.inf)
    else:
        dip_layer_it = dip_layer_pt = 11

    # Use the IT dip layer as the canonical reference for sharpness comparison
    canonical_dip = dip_layer_it

    sharp_pt = _dip_sharpness(pt_profile, canonical_dip)
    sharp_it = _dip_sharpness(it_profile, canonical_dip)

    peak_pt = int(np.nanargmax(layers_pt)) if any(not np.isnan(v) for v in layers_pt) else -1
    peak_it = int(np.nanargmax(layers_it)) if any(not np.isnan(v) for v in layers_it) else -1

    return {
        "canonical_dip_layer":  canonical_dip,
        "dip_sharpness_pt":     sharp_pt,
        "dip_sharpness_it":     sharp_it,
        "sharpening_ratio":     sharp_it / sharp_pt if sharp_pt > 0 else float("nan"),
        "id_peak_layer_pt":     peak_pt,
        "id_peak_layer_it":     peak_it,
        "id_at_dip_pt":         layers_pt[canonical_dip],
        "id_at_dip_it":         layers_it[canonical_dip],
    }
