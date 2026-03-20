"""
Plot 4 (Exp3): Corrective stage magnitude stratified by token type (Experiment 2a).

Splits generated tokens into types (CONTENT, PUNCTUATION, DISCOURSE,
STRUCTURAL, OTHER) using analysis/token_types.py and compares the corrective
stage metric (late-layer delta norm, cosine, logit delta) across token types
for PT vs IT.

Hypothesis: IT's corrective stage will be larger for DISCOURSE and STRUCTURAL
tokens than for CONTENT tokens, while PT shows no such stratification.

Panels:
  A: Box plot — corrective-stage delta norm by token type, PT vs IT
  B: Mean cos(delta_i, h_{i-1}) for layers 20–33 by token type, PT vs IT
  C: Mean logit delta contribution for layers 20–33 by token type
  D: Token type frequency distribution in IT vs PT responses

REQUIRES: generated_tokens field (available in exp2 results).
Token classification uses analysis/token_types.py — no new inference needed.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.poc.exp3.analysis.token_types import classify_generated_tokens, stratify_by_token_type


def make_plot(results: list[dict], output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if not results or "generated_tokens" not in results[0]:
        print("  Plot 4 (Exp3) skipped — no generated_tokens data")
        return

    # TODO: implement full 4-panel figure using stratify_by_token_type().
    # This plot can run on exp2 results (layer_delta_cosine is available).
    print("  Plot 4 (Exp3) — token stratification: TODO implement.")
