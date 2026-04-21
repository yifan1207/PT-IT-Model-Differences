#!/usr/bin/env bash
# Exp7 0D: Bootstrap CIs on All Main Figures
# Re-scores per-record outputs from A1 experiment and computes 95% CI
# via 10,000 bootstrap resamples. CPU-only, ~30 min.
#
# Usage:
#   bash scripts/run_exp7_0D.sh
#   bash scripts/run_exp7_0D.sh --n-bootstrap 1000   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

MERGED_DIR="results/exp06_corrective_direction_steering/merged_A1_it_v4"
if [[ ! -d "$MERGED_DIR" ]]; then
    echo "[0D] WARNING: $MERGED_DIR not found. Using first available merged A1 dir."
    MERGED_DIR=$(ls -d results/exp06_corrective_direction_steering/merged_A1_it* 2>/dev/null | head -1 || echo "")
    if [[ -z "$MERGED_DIR" ]]; then
        echo "[0D] ERROR: No merged A1 IT results found."
        exit 1
    fi
    echo "[0D] Using: $MERGED_DIR"
fi

mkdir -p logs/exp7 results/exp07_methodology_validation_tier0/0D

echo "=== Exp7 0D: Bootstrap CIs (CPU, ~30 min) ==="
echo "[0D] merged-dir: $MERGED_DIR"

uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_ci \
    --merged-dir "$MERGED_DIR" \
    --n-bootstrap 10000 \
    --seed 42 \
    --output-dir results/exp07_methodology_validation_tier0/0D/ \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee logs/exp7/0D.log

echo "=== [0D] Done. Results in results/exp07_methodology_validation_tier0/0D/ ==="
echo "Check: CI widths < 0.03 on governance metrics"
