#!/usr/bin/env bash
# Exp7 0A: Direction Calibration Sensitivity
# Collects per-record MLP activations for 600 selected records across 8 GPUs,
# then runs bootstrap stability analysis.
#
# Usage:
#   bash scripts/run_exp7_0A.sh
#   bash scripts/run_exp7_0A.sh --n-records 20   # quick test

set -euo pipefail

N_RECORDS_ARG=""
for arg in "$@"; do
    if [[ "$arg" == "--n-records" ]]; then
        shift
        N_RECORDS_ARG="--n-records $1"
        shift
    fi
done

mkdir -p logs/exp7

echo "=== Exp7 0A: Per-record activation collection (8 GPUs) ==="
pids=()
for i in {0..7}; do
    echo "[0A] launching worker $i on cuda:$i"
    uv run python -m src.poc.exp7.collect_per_record_acts \
        --worker-index "$i" --n-workers 8 --device "cuda:$i" \
        --output-dir results/exp7/0A/acts/ \
        $N_RECORDS_ARG \
        > logs/exp7/0A_w${i}.log 2>&1 &
    pids+=($!)
done

echo "[0A] waiting for workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[0A] ERROR: worker $pid failed"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[0A] Some workers failed. Check logs/exp7/0A_w*.log"
    exit 1
fi

echo "=== [0A] All workers done. Merging... ==="
uv run python -m src.poc.exp7.collect_per_record_acts \
    --merge-only --n-workers 8 --output-dir results/exp7/0A/acts/

echo "=== [0A] Running bootstrap direction stability analysis... ==="
uv run python -m src.poc.exp7.bootstrap_directions \
    --acts-dir results/exp7/0A/acts/ \
    --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \
    --n-bootstrap 50 --seed 42 \
    --output-dir results/exp7/0A/

echo "=== [0A] Done. Results in results/exp7/0A/ ==="
echo "Check: results/exp7/0A/bootstrap_results.json — pairwise cosine layers 20-33 > 0.95"
