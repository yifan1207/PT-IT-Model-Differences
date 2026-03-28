#!/usr/bin/env bash
# Exp7 0A: Direction Calibration Sensitivity
# Collects per-record MLP activations for selected records across 8 GPUs,
# then runs bootstrap stability analysis.
#
# Three phases:
#   1. Collect per-record activations (8 GPUs)
#   2. Bootstrap direction stability analysis
#   3. OOD direction test: collect activations from TriviaQA/ARC prompts (8 GPUs)
#      and compare direction to canonical
#
# Usage:
#   bash scripts/run_exp7_0A.sh
#   bash scripts/run_exp7_0A.sh --n-records 20   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

NW=8
mkdir -p logs/exp7

# ── Phase 1: Per-record activation collection ────────────────────────────────
echo "=== Exp7 0A Phase 1: Per-record activation collection (${NW} GPUs) ==="
pids=()
for i in $(seq 0 $((NW-1))); do
    echo "[0A] launching worker $i on cuda:$i"
    uv run python -m src.poc.exp7.collect_per_record_acts \
        --worker-index "$i" --n-workers "$NW" --device "cuda:$i" \
        --output-dir results/exp7/0A/acts/ \
        "${EXTRA_ARGS[@]}" \
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

echo "=== [0A] Phase 1 done. Merging... ==="
uv run python -m src.poc.exp7.collect_per_record_acts \
    --merge-only --n-workers "$NW" --output-dir results/exp7/0A/acts/

# ── Phase 2: Bootstrap direction stability analysis ──────────────────────────
echo ""
echo "=== [0A] Phase 2: Bootstrap direction stability analysis ==="
uv run python -m src.poc.exp7.bootstrap_directions \
    --acts-dir results/exp7/0A/acts/ \
    --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \
    --n-bootstrap 50 --seed 42 \
    --output-dir results/exp7/0A/

# ── Phase 3: OOD direction test ──────────────────────────────────────────────
echo ""
echo "=== [0A] Phase 3: OOD activation collection (TriviaQA + ARC) ==="
pids=()
for i in $(seq 0 $((NW-1))); do
    echo "[0A-OOD] launching worker $i on cuda:$i"
    uv run python -m src.poc.exp7.collect_ood_acts \
        --worker-index "$i" --n-workers "$NW" --device "cuda:$i" \
        --output-dir results/exp7/0A/ood_acts/ \
        "${EXTRA_ARGS[@]}" \
        > logs/exp7/0A_ood_w${i}.log 2>&1 &
    pids+=($!)
done

echo "[0A-OOD] waiting for workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[0A-OOD] ERROR: worker $pid failed"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[0A-OOD] Some OOD workers failed. Check logs/exp7/0A_ood_w*.log"
    echo "[0A-OOD] OOD test skipped — continuing with main results."
else
    echo "[0A-OOD] Merging OOD acts..."
    uv run python -m src.poc.exp7.collect_ood_acts \
        --merge-only --n-workers "$NW" --output-dir results/exp7/0A/ood_acts/

    echo "[0A-OOD] Running OOD direction comparison..."
    uv run python -m src.poc.exp7.bootstrap_directions \
        --acts-dir results/exp7/0A/acts/ \
        --ood-acts-dir results/exp7/0A/ood_acts/ \
        --canonical-npz results/exp5/precompute_v2/precompute/corrective_directions.npz \
        --output-dir results/exp7/0A/
fi

echo ""
echo "=== [0A] Done ==="
echo "Results:"
echo "  results/exp7/0A/bootstrap_results.json  — pairwise cosine (expect > 0.95 at layers 20-33)"
echo "  results/exp7/0A/ood_direction_test.json  — OOD cosine (expect > 0.90 at corrective layers)"
