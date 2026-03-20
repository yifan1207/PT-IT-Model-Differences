"""
Pre-correction layer analysis (Experiment 1c).

Tests whether IC / OOC / R task-type universality holds in layers 0–19
(the proposal stage), or whether it is a correction-stage artifact.

Works directly on existing exp2 or exp3 results — no new inference needed.
Simply filters all per-layer metrics to layers [0, boundary).
"""
import numpy as np
from collections import defaultdict


def filter_to_proposal_stage(
    results: list[dict],
    boundary: int = 20,
    keys: list[str] | None = None,
) -> list[dict]:
    """Return a copy of results with all per-layer metrics truncated to layers 0..boundary-1.

    Parameters
    ----------
    results : list[dict]
        Full exp2 or exp3 results.
    boundary : int
        First corrective-stage layer (exclusive upper bound for proposal stage).
    keys : list[str] or None
        Per-layer keys to truncate.  Defaults to all known [step][layer] keys.

    Returns
    -------
    list[dict] with the same structure but layer dimension sliced to [:boundary].
    """
    if keys is None:
        keys = [
            "residual_norm", "layer_delta_norm", "layer_delta_cosine",
            "l0", "logit_lens_entropy",
            "next_token_rank", "next_token_prob", "kl_to_final",
            "logit_delta_contrib", "transcoder_mse",
        ]

    filtered = []
    for r in results:
        r_copy = dict(r)
        for key in keys:
            if key in r_copy and r_copy[key]:
                r_copy[key] = [
                    step_vals[:boundary] for step_vals in r_copy[key]
                ]
        filtered.append(r_copy)
    return filtered
