"""
Alignment tax quantification (Experiment 2b).

Works directly on existing exp2 or exp3 results JSON — no new inference needed.

The "alignment tax" is the fraction of total MLP computation (measured by
sum of layer delta norms) that falls in the corrective stage (layers >= boundary)
vs the proposal stage (layers < boundary).

For each prompt result r:
    proposal_work   = Σ_{i=0}^{boundary-1}  layer_delta_norm[step][i]   (summed over steps)
    corrective_work = Σ_{i=boundary}^{33}   layer_delta_norm[step][i]   (summed over steps)
    tax             = corrective_work / (proposal_work + corrective_work)

Aggregated across all prompts in a category.

The key comparison is PT tax vs IT tax.  If IT has a significantly higher
corrective fraction (more work done in late layers), that is direct quantitative
evidence for the corrective stage hypothesis.
"""
import math
import numpy as np
from collections import defaultdict


PROPOSAL_END_LAYER = 20   # default; override by passing boundary= to functions


def compute_per_prompt_tax(
    result: dict,
    boundary: int = PROPOSAL_END_LAYER,
) -> dict:
    """Compute alignment tax for a single prompt result.

    Returns a dict with:
        proposal_norm_total    float  sum of layer_delta_norm over proposal layers and all steps
        corrective_norm_total  float  sum over corrective layers and all steps
        tax                    float  corrective / (proposal + corrective), in [0, 1]
        n_steps                int    number of generation steps recorded
    """
    layer_delta_norm = result.get("layer_delta_norm", [])
    if not layer_delta_norm:
        return {"proposal_norm_total": float("nan"), "corrective_norm_total": float("nan"),
                "tax": float("nan"), "n_steps": 0}

    proposal_total   = 0.0
    corrective_total = 0.0

    for step_norms in layer_delta_norm:
        for layer, norm_val in enumerate(step_norms):
            if math.isnan(norm_val):
                continue
            if layer < boundary:
                proposal_total += norm_val
            else:
                corrective_total += norm_val

    total = proposal_total + corrective_total
    tax = corrective_total / total if total > 0 else float("nan")

    return {
        "proposal_norm_total":   proposal_total,
        "corrective_norm_total": corrective_total,
        "tax": tax,
        "n_steps": len(layer_delta_norm),
    }


def compute_tax_by_category(
    results: list[dict],
    boundary: int = PROPOSAL_END_LAYER,
) -> dict[str, dict]:
    """Compute alignment tax statistics grouped by category.

    Returns dict: category → {
        mean_tax, std_tax, median_tax, micro_tax, per_prompt_taxes: list[float]
    }
    """
    per_cat: dict[str, list[float]] = defaultdict(list)
    sums: dict[str, dict[str, float]] = defaultdict(lambda: {"proposal": 0.0, "corrective": 0.0, "prompts": 0})

    for r in results:
        cat = r.get("category", "unknown")
        t = compute_per_prompt_tax(r, boundary=boundary)
        if not math.isnan(t["tax"]):
            per_cat[cat].append(t["tax"])
            sums[cat]["proposal"] += t["proposal_norm_total"]
            sums[cat]["corrective"] += t["corrective_norm_total"]
            sums[cat]["prompts"] += 1

    summary = {}
    for cat, data in sums.items():
        taxes = per_cat[cat]
        arr = np.array(taxes)
        total = data["proposal"] + data["corrective"]
        micro_tax = data["corrective"] / total if total > 0 else float("nan")
        summary[cat] = {
            "mean_tax":         float(arr.mean()),
            "std_tax":          float(arr.std()),
            "median_tax":       float(np.median(arr)),
            "micro_tax":        float(micro_tax),
            "per_prompt_taxes": taxes,
            "n_prompts":        data["prompts"],
        }
    return summary


def compute_per_layer_contribution_fraction(
    results: list[dict],
) -> dict[str, np.ndarray]:
    """Per-layer mean delta norm as a fraction of total, by category.

    Returns dict: category → np.ndarray [N_LAYERS] of mean fractional contribution.
    Each element: mean over all prompts × steps of
        layer_delta_norm[step][layer] / sum_i(layer_delta_norm[step][i])
    """
    from src.poc.shared.constants import N_LAYERS

    sums:   dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_LAYERS))
    counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(N_LAYERS))

    for r in results:
        cat = r.get("category", "unknown")
        for step_norms in r.get("layer_delta_norm", []):
            total = sum(v for v in step_norms if not math.isnan(v))
            if total == 0:
                continue
            for layer, norm_val in enumerate(step_norms):
                if not math.isnan(norm_val):
                    sums[cat][layer]   += norm_val / total
                    counts[cat][layer] += 1

    result = {}
    for cat in sums:
        with np.errstate(invalid="ignore"):
            result[cat] = np.where(counts[cat] > 0,
                                   sums[cat] / counts[cat], np.nan)
    return result
