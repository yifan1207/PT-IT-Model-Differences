"""
Plot 2 (Exp3): Answer emergence and stabilisation — logit lens PT vs IT (Exp 1a).

For each generated token, tracks how the token's rank and probability evolve
across transformer layers — showing at what depth the model "commits" to its
prediction.

Four panels:
  A: Mean rank of final token per layer (lower = emerged earlier)
  B: Mean probability of final token per layer
  C: Mean KL(layer_i ∥ layer_33) per layer  (0 = fully committed)
  D: Heatmap [layer × generation step] of KL-to-final for one category

X-axis (A/B/C): transformer layer
Color (D): KL to final
Lines: PT vs IT (same prompt set), optionally split by IC/OOC/R

REQUIRES: collect_emergence=True in Exp3Config.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


VARIANT_STYLE = {
    "pt": {"color": "#1565C0", "label": "PT (pretrained)",   "ls": "-"},
    "it": {"color": "#E65100", "label": "IT (instruction)",  "ls": "--"},
}

CATEGORY_STYLE = {
    "in_context":     {"color": "#2196F3", "label": "IC"},
    "out_of_context": {"color": "#FF9800", "label": "OOC"},
    "reasoning":      {"color": "#4CAF50", "label": "R"},
}


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("next_token_rank"):
        print("  Plot 2 (Exp3) skipped — no emergence data (collect_emergence=False?)")
        return

    # TODO: implement full 4-panel figure.
    # Key aggregations needed:
    #   mean_rank[layer]  = mean over all prompts × steps of next_token_rank[step][layer]
    #   mean_prob[layer]  = mean over all prompts × steps of next_token_prob[step][layer]
    #   mean_kl[layer]    = mean over all prompts × steps of kl_to_final[step][layer]
    # These should be computed separately for PT and IT results (loaded from two files),
    # or from a single results file with a 'model_variant' field added per result.
    print("  Plot 2 (Exp3) — emergence: TODO implement after data collection.")
