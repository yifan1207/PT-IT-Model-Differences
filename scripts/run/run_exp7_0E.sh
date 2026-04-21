#!/usr/bin/env bash
# Exp7 0E: Token Classifier Robustness
# Tests sensitivity of the STR metric to boundary-adjacent token reclassifications.
# CPU-only, ~10 min.
#
# Usage:
#   bash scripts/run_exp7_0E.sh
#   bash scripts/run_exp7_0E.sh --n-outputs 200   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

SAMPLE_OUTPUTS="results/exp6/merged_A1_it_v4/sample_outputs.jsonl"
if [[ ! -f "$SAMPLE_OUTPUTS" ]]; then
    echo "[0E] WARNING: $SAMPLE_OUTPUTS not found. Searching for alternative..."
    SAMPLE_OUTPUTS=$(ls results/exp6/merged_A1_it*/sample_outputs.jsonl 2>/dev/null | head -1 || echo "")
    if [[ -z "$SAMPLE_OUTPUTS" ]]; then
        echo "[0E] ERROR: No sample_outputs.jsonl found in merged A1 IT dirs."
        exit 1
    fi
    echo "[0E] Using: $SAMPLE_OUTPUTS"
fi

mkdir -p logs/exp7 results/exp7/0E

echo "=== Exp7 0E: Token classifier robustness (CPU, ~10 min) ==="
echo "[0E] sample-outputs: $SAMPLE_OUTPUTS"

uv run python -m src.poc.exp07_methodology_validation_tier0.token_classifier_robustness \
    --sample-outputs "$SAMPLE_OUTPUTS" \
    --output-dir results/exp7/0E/ \
    --n-perturb 500 \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee logs/exp7/0E.log

echo "=== [0E] Done. Results in results/exp7/0E/ ==="
echo "Check: max |Δ STR| < 0.01 in classifier_robustness.json"
