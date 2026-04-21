#!/usr/bin/env bash
# Exp7 0F: Corrective Layer Range Justification + Per-Layer Importance Sweep
#
# Part 1: Layer range sensitivity (4 runs)
#   Runs A1 experiments with different corrective layer boundaries:
#     18-33 (wider start),  20-33 (canonical),
#     22-33 (narrower),     20-31 (shorter end)
#   Tests whether the governance dose-response is robust to +/-2 layers on each boundary.
#
# Part 2: Per-layer importance sweep (14 single-layer runs)
#   Runs A1_single_layer: removes one corrective layer at a time (alpha=0) for layers 20-33.
#   Identifies which specific layers carry the most governance signal.
#   Connects to 0G (tuned-lens commitment onset): most important layers should cluster
#   near the commitment boundary.
#
# Each run uses 8 GPU workers.
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

run_experiment() {
    local EXPERIMENT=$1
    local RUN_NAME=$2
    shift 2
    local EXTRA_RUN_ARGS=("$@")

    echo ""
    echo "=== [0F] ${EXPERIMENT}: ${RUN_NAME} (${NW} GPUs) ==="

    pids=()
    for i in $(seq 0 $((NW-1))); do
        uv run python -m src.poc.exp06_corrective_direction_steering.run \
            --experiment "$EXPERIMENT" \
            --variant it \
            --worker-index "$i" --n-workers "$NW" \
            --device "cuda:${i}" \
            --run-name "${RUN_NAME}" \
            --output-base "$OUTPUT_BASE" \
            --corrective-direction-path "$CORR_DIR" \
            "${EXTRA_RUN_ARGS[@]}" \
            "${EXTRA_ARGS[@]}" \
            > logs/exp7/0F_${RUN_NAME}_w${i}.log 2>&1 &
        pids+=($!)
    done

    echo "[0F] waiting for ${RUN_NAME} workers..."
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[0F] ERROR: worker failed for ${RUN_NAME}"
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[0F] Some workers failed. Check logs/exp7/0F_${RUN_NAME}_w*.log"
        exit 1
    fi

    echo "[0F] Merging ${RUN_NAME}..."
    # exp6/run.py appends _w{worker_index} to run_name → dir is ${RUN_NAME}_w${i}
    src_dirs=()
    for i in $(seq 0 $((NW-1))); do src_dirs+=("${OUTPUT_BASE}/${RUN_NAME}_w${i}"); done

    uv run python scripts/merge_steering_workers.py \
        --experiment "$EXPERIMENT" \
        --variant it \
        --n-workers "$NW" \
        --merged-name "merged_${RUN_NAME}" \
        --output-base "$OUTPUT_BASE" \
        --source-dirs "${src_dirs[@]}"

    echo "[0F] Done: ${OUTPUT_BASE}/merged_${RUN_NAME}/"
}

# ── Part 1: Layer range sensitivity (4 runs) ─────────────────────────────────
echo "[0F] === Part 1: Layer range sensitivity (4 runs, sequential) ==="

for RANGE in "18-33" "20-33" "22-33" "20-31"; do
    START="${RANGE%-*}"
    END="${RANGE#*-}"
    N_LAYERS_VAL=$((END + 1))

    run_experiment A1 "A1_it_layers_${RANGE}" \
        --proposal-boundary "$START" \
        --n-layers "$N_LAYERS_VAL"
done

# ── Part 2: Per-layer importance sweep (single-layer removals) ───────────────
echo ""
echo "[0F] === Part 2: Per-layer importance sweep (A1_single_layer) ==="

run_experiment A1_single_layer "A1_single_layer_it_v1"

# ── Analysis ─────────────────────────────────────────────────────────────────
echo ""
echo "=== [0F] All runs complete. Analyzing... ==="
uv run python -m src.poc.exp07_methodology_validation_tier0.layer_range_analysis \
    --results-dir "$OUTPUT_BASE/" \
    --output-dir "$OUTPUT_BASE/"

echo ""
echo "=== [0F] Done ==="
echo "Results:"
echo "  Layer range sensitivity: ${OUTPUT_BASE}/merged_A1_it_layers_*/"
echo "  Per-layer importance:    ${OUTPUT_BASE}/merged_A1_single_layer_it_v1/"
echo "  Analysis:                ${OUTPUT_BASE}/layer_range_sensitivity_table.csv"
echo "  Plots:                   ${OUTPUT_BASE}/plots/"
