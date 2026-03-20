"""
Plot 3 (Exp3): Attraction vs repulsion decomposition per layer (Experiment 1b).

For each layer i, the logit-lens delta (logit_lens[i][token] - logit_lens[i-1][token])
tells us whether that layer pushed the model toward (+) or away from (-) the
generated token.

  Attraction at layer i = logit_delta_contrib[i] when > 0
  Repulsion  at layer i = logit_delta_contrib[i] when < 0

Key question: do IT late layers carry more repulsive mass (negative contributions)
than PT?  This would show the corrective stage is suppressive, not merely directive.

Panels:
  A: Mean logit delta contribution per layer — PT vs IT
     (positive = pushing toward generated token, negative = pushing away)
  B: Fraction of steps where layer i is repulsive (delta < 0) — PT vs IT
  C: Mean |repulsion| per layer for corrective-stage layers (20+) — PT vs IT
  D: Scatter of attraction vs repulsion balance per layer and token type

REQUIRES: collect_attribution=True in Exp3Config.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("logit_delta_contrib"):
        print("  Plot 3 (Exp3) skipped — no logit_delta_contrib data")
        return

    # TODO: implement full 4-panel figure.
    print("  Plot 3 (Exp3) — attraction/repulsion: TODO implement after data collection.")
