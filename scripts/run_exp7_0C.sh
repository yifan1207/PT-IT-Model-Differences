#!/usr/bin/env bash
# Exp7 0C: Projection-Magnitude-Matched Random Direction Control
# Runs A1_rand_matched experiment: removes a random direction from IT activations
# at corrective layers (20-33), but scales the perturbation to match the per-token
# projection magnitude of the corrective direction. This is a fair like-for-like
# comparison that eliminates the projection-magnitude confound in A1_rand.
#
# Expected: governance metrics flat across all α → effect is direction-specific.
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
RUN_NAME="A1_rand_matched_it_v1"
NW=8

mkdir -p logs/exp7 "$OUTPUT_BASE"

echo "=== Exp7 0C: Magnitude-matched random direction control (${NW} GPUs) ==="
pids=()
for i in $(seq 0 $((NW-1))); do
    echo "[0C] launching worker $i on cuda:$i"
    uv run python -m src.poc.exp6.run \
        --experiment A1_rand_matched \
        --variant it \
        --worker-index "$i" --n-workers "$NW" \
        --device "cuda:${i}" \
        --run-name "${RUN_NAME}_w${i}" \
        --output-base "$OUTPUT_BASE" \
        --corrective-direction-path "$CORR_DIR" \
        "${EXTRA_ARGS[@]}" \
        > logs/exp7/0C_w${i}.log 2>&1 &
    pids+=($!)
done

echo "[0C] waiting for workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[0C] ERROR: worker $pid failed"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[0C] Some workers failed. Check logs/exp7/0C_w*.log"
    exit 1
fi

echo "=== [0C] Merging worker results... ==="
src_dirs=()
for i in $(seq 0 $((NW-1))); do src_dirs+=("${OUTPUT_BASE}/${RUN_NAME}_w${i}"); done

uv run python scripts/merge_exp6_workers.py \
    --experiment A1_rand_matched \
    --variant it \
    --n-workers "$NW" \
    --merged-name "merged_${RUN_NAME}" \
    --output-base "$OUTPUT_BASE" \
    --source-dirs "${src_dirs[@]}"

echo "=== [0C] Done. Results in ${OUTPUT_BASE}/merged_${RUN_NAME}/ ==="
echo "Check: governance metrics flat across α values in scores.jsonl"
