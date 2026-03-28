#!/usr/bin/env bash
# Exp7 0B: Matched-Token Direction Validation
# Force-decodes PT on IT token sequences to compute the matched-token direction,
# then computes cosine similarity to the canonical corrective direction.
#
# Prerequisites: run_exp7_0A.sh must complete first (reuses IT free-running acts)
#
# Usage:
#   bash scripts/run_exp7_0B.sh
#   bash scripts/run_exp7_0B.sh --n-records 20   # quick test

set -euo pipefail

N_RECORDS_ARG=""
for arg in "$@"; do
    if [[ "$arg" == "--n-records" ]]; then
        shift
        N_RECORDS_ARG="--n-records $1"
        shift
    fi
done

if [[ ! -d results/exp7/0A/acts/ ]]; then
    echo "[0B] ERROR: results/exp7/0A/acts/ not found. Run run_exp7_0A.sh first."
    exit 1
fi

mkdir -p logs/exp7

echo "=== Exp7 0B: PT force-decode activation collection (8 GPUs) ==="
pids=()
for i in {0..7}; do
    echo "[0B] launching force-decode worker $i on cuda:$i"
    uv run python -m src.poc.exp7.force_decode_acts \
        --worker-index "$i" --n-workers 8 --device "cuda:$i" \
        --output-dir results/exp7/0B/acts/ \
        $N_RECORDS_ARG \
        > logs/exp7/0B_w${i}.log 2>&1 &
    pids+=($!)
done

echo "[0B] waiting for workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[0B] ERROR: worker $pid failed"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[0B] Some workers failed. Check logs/exp7/0B_w*.log"
    exit 1
fi

echo "=== [0B] Merging force-decode workers... ==="
uv run python -m src.poc.exp7.force_decode_acts \
    --merge-only --n-workers 8 --output-dir results/exp7/0B/acts/

echo "=== [0B] Computing matched-token direction comparison... ==="
uv run python -m src.poc.exp7.bootstrap_directions \
    --it-acts-dir results/exp7/0A/acts/ \
    --pt-forced-acts-dir results/exp7/0B/acts/ \
    --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \
    --matched-mode \
    --output-dir results/exp7/0B/

echo "=== [0B] Done. Results in results/exp7/0B/ ==="
echo "Check: results/exp7/0B/matched_cosines.json — cosine layers 20-33 > 0.90"
