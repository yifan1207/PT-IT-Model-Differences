#!/usr/bin/env bash
# Rerun exp2 IT δ-cosine collection without chat template.
#
# Ablation: tests whether the IT model's negative δ-cosine at corrective layers
# is driven by the chat template (template-gated) or by model weights alone
# (weight-intrinsic). The δ-cosine was originally measured WITH chat template.
#
# Results written to: results/exp2/it_*_notmpl/
# Then generates side-by-side comparison plots (plot10 and plot11) in:
#   results/exp2/plots_notmpl_comparison/
#
# Usage:
#   bash scripts/run_exp2_notmpl.sh
set -euo pipefail

echo "=== exp2 IT no-template ablation ==="

TMPL_RESULTS="results/exp2/it_16k_l0_big_affine_t512/exp2_results.json"
if [ ! -f "$TMPL_RESULTS" ]; then
    echo "[$(date +%T)] With-template results missing. Running collection (--variant it)..."
    uv run python -m src.poc.exp02_ic_ooc_reasoning_mechanistic_comparison.run --variant it
    echo "[$(date +%T)] With-template collection done."
fi

echo "[$(date +%T)] Starting collection (--variant it --no-chat-template)..."
uv run python -m src.poc.exp02_ic_ooc_reasoning_mechanistic_comparison.run --variant it --no-chat-template

echo "[$(date +%T)] Collection done. Running comparison plots..."

uv run python scripts/plot_exp2_notmpl_comparison.py

echo "[$(date +%T)] Done."
echo "  results/exp2/it_*_notmpl/"
echo "  results/exp2/plots_notmpl_comparison/"
