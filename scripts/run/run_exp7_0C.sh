#!/usr/bin/env bash
# Exp7 0C: Projection-Magnitude-Matched Random Direction Control
# Runs two experiments:
#   1. A1_rand_matched: single seed (42), full alpha sweep (15 conditions)
#   2. A1_rand_matched_multiseed: 5 seeds, reduced alpha sweep (26 conditions)
#
# Expected: governance metrics flat across all alpha -> effect is direction-specific.
# Multi-seed: mean +/- std across 5 seeds should show tight CIs around flat lines.
#
# Usage:
#   bash scripts/run_exp7_0C.sh
#   bash scripts/run_exp7_0C.sh --n-eval-examples 100   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
if [[ ! -f "$CORR_DIR" ]]; then
    echo "[0C] ERROR: corrective_directions.npz not found at $CORR_DIR"
    exit 1
fi

OUTPUT_BASE="results/exp7/0C"
NW=8

mkdir -p logs/exp7 "$OUTPUT_BASE"

run_0c_experiment() {
    local EXPERIMENT=$1
    local RUN_NAME=$2

    echo ""
    echo "=== [0C] ${EXPERIMENT}: ${RUN_NAME} (${NW} GPUs) ==="
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
            "${EXTRA_ARGS[@]}" \
            > logs/exp7/0C_${RUN_NAME}_w${i}.log 2>&1 &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then echo "[0C] worker failed for $RUN_NAME"; failed=1; fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[0C] Check logs/exp7/0C_${RUN_NAME}_w*.log"
        exit 1
    fi

    # exp6/run.py appends _w{worker_index} to run_name → dir is ${RUN_NAME}_w${i}
    src_dirs=()
    for i in $(seq 0 $((NW-1))); do src_dirs+=("${OUTPUT_BASE}/${RUN_NAME}_w${i}"); done
    uv run python scripts/merge_steering_workers.py \
        --experiment "$EXPERIMENT" --variant it --n-workers "$NW" \
        --merged-name "merged_${RUN_NAME}" --output-base "$OUTPUT_BASE" \
        --source-dirs "${src_dirs[@]}"
}

# Step 1: Single-seed full sweep
run_0c_experiment A1_rand_matched A1_rand_matched_it_v1

# Step 2: Multi-seed reduced sweep
run_0c_experiment A1_rand_matched_multiseed A1_rand_multiseed_it_v1

# Step 3: Report scale factor distribution
echo ""
echo "=== [0C] Generating scale factor distribution report ==="
uv run python -c "
import json, numpy as np
from pathlib import Path

# Collect scale factor stats from worker logs
scale_factors = []
for log_path in sorted(Path('logs/exp7').glob('0C_A1_rand_matched_it_v1_w*.log')):
    text = log_path.read_text()
    # Scale factors are logged by the interventions module
    # Parse from any diagnostics output

# If we have the module-level accumulator, use it
try:
    from src.poc.exp06_corrective_direction_steering.interventions import get_and_clear_scale_factor_log
    factors = get_and_clear_scale_factor_log()
    if factors:
        arr = np.array(factors)
        report = {
            'n_samples': len(arr),
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'p5': float(np.percentile(arr, 5)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
            'max': float(arr.max()),
            'pct_above_100': float((arr > 100).mean() * 100),
            'expected_sqrt_d': float(np.sqrt(2560)),
        }
        out_path = Path('results/exp7/0C/scale_factor_distribution.json')
        out_path.write_text(json.dumps(report, indent=2))
        print(f'Scale factor distribution:')
        print(f'  mean={report[\"mean\"]:.1f}, median={report[\"median\"]:.1f}')
        print(f'  p5={report[\"p5\"]:.1f}, p95={report[\"p95\"]:.1f}, max={report[\"max\"]:.1f}')
        print(f'  % above 100x: {report[\"pct_above_100\"]:.1f}%')
        print(f'  Expected sqrt(d): {report[\"expected_sqrt_d\"]:.1f}')
except Exception as e:
    print(f'Scale factor report skipped: {e}')
" 2>&1

echo ""
echo "=== [0C] Done ==="
echo "Results:"
echo "  ${OUTPUT_BASE}/merged_A1_rand_matched_it_v1/   — single seed"
echo "  ${OUTPUT_BASE}/merged_A1_rand_multiseed_it_v1/ — 5 seeds"
echo "  ${OUTPUT_BASE}/scale_factor_distribution.json  — per-token scale stats"
