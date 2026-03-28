#!/usr/bin/env bash
# Exp7 0F: Corrective Layer Range Justification
# Runs 4 A1 experiments with different corrective layer boundaries:
#   18-33 (wider start),  20-33 (canonical),
#   22-33 (narrower),     20-31 (shorter end)
# Tests whether the governance dose-response is robust to ±2 layers on each boundary.
# Each run uses 8 GPU workers. Runs are sequential (~2 hrs total).
#
# Usage:
#   bash scripts/run_exp7_0F.sh
#   bash scripts/run_exp7_0F.sh --n-eval-examples 100   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
if [[ ! -f "$CORR_DIR" ]]; then
    echo "[0F] ERROR: corrective_directions.npz not found at $CORR_DIR"
    exit 1
fi

OUTPUT_BASE="results/exp7/0F"
NW=8

mkdir -p logs/exp7 "$OUTPUT_BASE"

run_layer_range() {
    local RANGE=$1
    local START="${RANGE%-*}"
    local END="${RANGE#*-}"
    local N_LAYERS_VAL=$((END + 1))   # e.g. end=33 → n_layers=34; end=31 → n_layers=32
    local RUN_NAME="A1_it_layers_${RANGE}"

    echo ""
    echo "=== [0F] Layer range ${RANGE} (proposal_boundary=${START}, n_layers=${N_LAYERS_VAL}) ==="

    pids=()
    for i in $(seq 0 $((NW-1))); do
        uv run python -m src.poc.exp6.run \
            --experiment A1 \
            --variant it \
            --worker-index "$i" --n-workers "$NW" \
            --device "cuda:${i}" \
            --run-name "${RUN_NAME}_w${i}" \
            --output-base "$OUTPUT_BASE" \
            --corrective-direction-path "$CORR_DIR" \
            --proposal-boundary "$START" \
            --n-layers "$N_LAYERS_VAL" \
            "${EXTRA_ARGS[@]}" \
            > logs/exp7/0F_${RANGE}_w${i}.log 2>&1 &
        pids+=($!)
    done

    echo "[0F] waiting for ${RANGE} workers..."
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[0F] ERROR: worker failed for range ${RANGE}"
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[0F] Some workers failed for range ${RANGE}. Check logs/exp7/0F_${RANGE}_w*.log"
        exit 1
    fi

    echo "[0F] Merging ${RANGE}..."
    src_dirs=()
    for i in $(seq 0 $((NW-1))); do src_dirs+=("${OUTPUT_BASE}/${RUN_NAME}_w${i}"); done

    uv run python scripts/merge_exp6_workers.py \
        --experiment A1 \
        --variant it \
        --n-workers "$NW" \
        --merged-name "${RUN_NAME}" \
        --output-base "$OUTPUT_BASE" \
        --source-dirs "${src_dirs[@]}"

    echo "[0F] Done: ${OUTPUT_BASE}/${RUN_NAME}/"
}

echo "[0F] Running 4 layer-range A1 experiments (sequential)..."
for RANGE in "18-33" "20-33" "22-33" "20-31"; do
    run_layer_range "$RANGE"
done

echo ""
echo "=== [0F] All layer-range runs complete. Analyzing... ==="
uv run python -m src.poc.exp7.layer_range_analysis \
    --results-dir "$OUTPUT_BASE/" \
    --output-dir "$OUTPUT_BASE/"

echo "=== [0F] Done. Results in ${OUTPUT_BASE}/ ==="
echo "Check: ${OUTPUT_BASE}/layer_range_sensitivity_table.csv"
echo "Check: ${OUTPUT_BASE}/plots/layer_range_sensitivity.png"
