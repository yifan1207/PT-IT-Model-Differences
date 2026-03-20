"""
Plot 7 (Exp3): Ablated vs normal generation comparison (Experiments 3a / 3b).

Compares IT generation with and without the corrective stage (layers 25–33
MLP zeroed) to causally demonstrate what the corrective stage contributes.

Expected result:
  - Ablated output: semantically similar to normal IT (content preserved)
    but format degraded (less structured, less polite, more like PT)
  - Metric: cosine similarity of residual streams between ablated and normal runs

Panels:
  A: Residual stream cosine similarity (ablated vs normal) per layer — how much
     does zeroing layers 25–33 propagate backward?
  B: Output token distribution comparison — do ablated outputs resemble PT outputs?
  C: Generation length distribution — does ablation affect response length?
  D: Example side-by-side generated text (PT / IT normal / IT ablated)

REQUIRES: results from interventions/ablation.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def make_plot(results: list[dict], output_dir: str,
              ablated_results: list[dict] | None = None,
              pt_results: list[dict] | None = None) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if ablated_results is None:
        print("  Plot 7 (Exp3) skipped — no ablated_results provided")
        return

    # TODO: implement once ablation data is available.
    print("  Plot 7 (Exp3) — ablation comparison: TODO implement after ablation runs.")
