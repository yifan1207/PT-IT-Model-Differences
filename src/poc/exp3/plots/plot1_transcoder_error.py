"""
Plot 1 (Exp3): Transcoder reconstruction MSE per layer, PT vs IT (Experiment 0a).

For each layer i, the transcoder is supposed to approximate the MLP output.
Reconstruction error = ||transcoder(x_i) - mlp_actual(x_i)||² / d_model.

If IT's late-layer error is 2× PT's, the L0 increase in late layers (from exp2)
is partially artifactual — the transcoder is less accurate there, so L0 may
reflect encoding noise rather than genuine increased feature activity.

  X-axis : transformer layer (0–33)
  Y-axis : mean MSE across all prompts × generation steps
  Lines  : PT (dark) vs IT (orange), ±1 SEM

REQUIRES: collect_transcoder_mse=True in Exp3Config and results from exp3 run.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or not results[0].get("transcoder_mse"):
        print("  Plot 1 (Exp3) skipped — no transcoder_mse data")
        return

    # TODO: implement once transcoder_mse data is available.
    # Structure mirrors exp2's plot1_l0_per_layer.py.
    # cat_arrays = aggregate by category from results["transcoder_mse"]
    # Plot mean ± SEM per layer for each category / model variant.
    print("  Plot 1 (Exp3) — transcoder MSE: TODO implement after data collection.")
