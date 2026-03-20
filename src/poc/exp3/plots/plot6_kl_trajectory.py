"""
Plot 6 (Exp3): Layer-to-final KL divergence trajectory PT vs IT (Experiment 2c).

KL(layer_i ∥ layer_33): how far is each intermediate layer's logit-lens
distribution from the model's final prediction?

  High KL at layer i = model's prediction at layer i is very different from final
  Sharp KL drop at layer L = model committed to its answer at layer L

This subsumes several entropy ideas: not just "how uncertain is the model" but
"when did the model form the prediction it ultimately produces?"

If PT commits earlier (sharp KL drop at a lower layer) and IT keeps modifying
(KL stays high longer, drops later), that is clean corrective stage evidence.

Panels:
  A: Mean KL-to-final per layer — PT vs IT (line plot)
  B: Mean KL-to-final heatmap [layer × generation step] for IT
  C: Layer of commitment (where KL first drops below threshold) — distribution PT vs IT
  D: KL trajectory for early vs late generation thirds (does IT commitment depth change?)

REQUIRES: collect_emergence=True in Exp3Config (provides kl_to_final field).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("kl_to_final"):
        print("  Plot 6 (Exp3) skipped — no kl_to_final data (collect_emergence=False?)")
        return

    # TODO: implement full 4-panel figure.
    print("  Plot 6 (Exp3) — KL trajectory: TODO implement after data collection.")
