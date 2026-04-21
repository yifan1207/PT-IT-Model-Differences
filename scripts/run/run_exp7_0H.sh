#!/usr/bin/env bash
# Exp7 0H: Calibration-Evaluation Split Validation
# Tests whether top-600 governance-based record selection is necessary, or whether
# a random 600/800 split produces comparable governance dose-response results.
#
# Steps:
#   1. Collect MLP activations for random-600 and bottom-600 records (8 GPUs)
#   2. Compute directions from random-600 and bottom-600
#   3. Run A1 α-sweep on held-out 800 using each direction
#
# Usage:
#   bash scripts/run_exp7_0H.sh
#   bash scripts/run_exp7_0H.sh --n-records 20   # quick test

set -euo pipefail

N_RECORDS_ARG=()
EXTRA_EVAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --n-records)
            shift
            N_RECORDS_ARG=(--n-records "$1")
            shift ;;
        *)
            EXTRA_EVAL_ARGS+=("$1")
            shift ;;
    esac
done

GEN_MERGED="results/precompute_v2_work/gen_merged.jsonl"
if [[ ! -f "$GEN_MERGED" ]]; then
    echo "[0H] ERROR: gen_merged.jsonl not found at $GEN_MERGED"
    exit 1
fi

mkdir -p logs/exp7 results/exp07_methodology_validation_tier0/0H

# ── Step 1: Create split and collect activations ──────────────────────────────
echo "=== [0H] Step 1: Activation collection (random-600 + bottom-600, 8 GPUs) ==="
pids=()
for i in {0..7}; do
    echo "[0H] launching worker $i on cuda:$i"
    uv run python -m src.poc.exp07_methodology_validation_tier0.precompute_random_split \
        --worker-index "$i" --n-workers 8 \
        --device "cuda:${i}" \
        "${N_RECORDS_ARG[@]}" \
        > logs/exp7/0H_w${i}.log 2>&1 &
    pids+=($!)
done

echo "[0H] waiting for workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[0H] ERROR: worker $pid failed"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[0H] Some workers failed. Check logs/exp7/0H_w*.log"
    exit 1
fi

# ── Step 2: Compute directions ────────────────────────────────────────────────
echo "=== [0H] Step 2: Computing random and bottom directions... ==="
uv run python -m src.poc.exp07_methodology_validation_tier0.precompute_random_split --compute-directions

# ── Step 3: A1 on held-out 800 with random direction ─────────────────────────
echo "=== [0H] Step 3a: A1 on held-out 800 (random direction, 8 GPUs) ==="
RAND_DIR="results/exp07_methodology_validation_tier0/0H/random_directions.npz"
if [[ ! -f "$RAND_DIR" ]]; then
    echo "[0H] ERROR: random_directions.npz not found. Step 2 may have failed."
    exit 1
fi

# Load held-out IDs and compute n-eval
HELD_OUT_N=$(python3 -c "import json; print(len(json.load(open('results/exp07_methodology_validation_tier0/0H/held_out_800_ids.json'))))")
echo "[0H] held-out records: $HELD_OUT_N"

HELD_OUT_IDS="results/exp07_methodology_validation_tier0/0H/held_out_800_ids.json"

run_a1_held_out() {
    local RUN_NAME=$1
    local DIR_PATH=$2
    local NW=8

    pids=()
    for i in $(seq 0 $((NW-1))); do
        uv run python -m src.poc.exp06_corrective_direction_steering.run \
            --experiment A1 \
            --variant it \
            --worker-index "$i" --n-workers "$NW" \
            --device "cuda:${i}" \
            --run-name "${RUN_NAME}_w${i}" \
            --output-base "results/exp07_methodology_validation_tier0/0H" \
            --corrective-direction-path "$DIR_PATH" \
            --eval-record-ids "$HELD_OUT_IDS" \
            "${EXTRA_EVAL_ARGS[@]}" \
            > logs/exp7/0H_${RUN_NAME}_w${i}.log 2>&1 &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[0H] ERROR: worker failed for ${RUN_NAME}"
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then exit 1; fi

    src_dirs=()
    for i in $(seq 0 $((NW-1))); do src_dirs+=("results/exp07_methodology_validation_tier0/0H/${RUN_NAME}_w${i}"); done
    uv run python scripts/merge_steering_workers.py \
        --experiment A1 --variant it --n-workers "$NW" \
        --merged-name "$RUN_NAME" \
        --output-base "results/exp07_methodology_validation_tier0/0H" \
        --source-dirs "${src_dirs[@]}"
    echo "[0H] Done: results/exp07_methodology_validation_tier0/0H/${RUN_NAME}/"
}

echo "=== [0H] Running A1 with random-600 direction on held-out 800 ==="
run_a1_held_out "A1_it_random_dir" "$RAND_DIR"

echo "=== [0H] Running A1 with bottom-600 direction on held-out 800 (negative control) ==="
BOTTOM_DIR="results/exp07_methodology_validation_tier0/0H/bottom_directions.npz"
if [[ -f "$BOTTOM_DIR" ]]; then
    run_a1_held_out "A1_it_bottom_dir" "$BOTTOM_DIR"
else
    echo "[0H] WARNING: bottom_directions.npz not found, skipping negative control."
fi

echo "=== [0H] Done. Results in results/exp07_methodology_validation_tier0/0H/ ==="
echo "Check: A1 dose-response with random direction comparable to governance-selected direction"
