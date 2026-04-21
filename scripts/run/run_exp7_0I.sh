#!/usr/bin/env bash
# Exp7 0I: Intervention Formula Sensitivity
# Tests 4 intervention formula variants at 14 α values (matching A1 sweep,
# including negatives) to confirm the governance-content dissociation is robust:
#   1. mlp_proj_remove      — canonical (directional_remove on MLP output)
#   2. mlp_additive         — additive injection (directional_add)
#   3. residual_proj_remove — projection-removal on full residual stream
#   4. attn_proj_remove     — projection-removal on self_attn output only
# 60 conditions total (4 methods × 14 alphas + 4 baselines). ~3.5 hrs on 8 GPUs.
# Re-running will reuse existing results and only compute new alpha conditions.
#
# Usage:
#   bash scripts/run_exp7_0I.sh
#   bash scripts/run_exp7_0I.sh --n-eval-examples 100   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
if [[ ! -f "$CORR_DIR" ]]; then
    echo "[0I] ERROR: corrective_directions.npz not found at $CORR_DIR"
    exit 1
fi

OUTPUT_BASE="results/exp7/0I"
RUN_NAME="A1_formula_it_v1"
NW=4  # must match original run (4 workers) so resume finds existing results

mkdir -p logs/exp7 "$OUTPUT_BASE"

echo "=== Exp7 0I: Intervention formula sensitivity (${NW} GPUs, 60 conditions) ==="
pids=()
for i in $(seq 0 $((NW-1))); do
    echo "[0I] launching worker $i on cuda:$i"
    uv run python -m src.poc.exp06_corrective_direction_steering.run \
        --experiment A1_formula \
        --variant it \
        --worker-index "$i" --n-workers "$NW" \
        --device "cuda:${i}" \
        --run-name "${RUN_NAME}" \
        --output-base "$OUTPUT_BASE" \
        --corrective-direction-path "$CORR_DIR" \
        "${EXTRA_ARGS[@]}" \
        > logs/exp7/0I_w${i}.log 2>&1 &
    pids+=($!)
done

echo "[0I] waiting for workers..."
failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        echo "[0I] ERROR: worker $pid failed"
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "[0I] Some workers failed. Check logs/exp7/0I_w*.log"
    exit 1
fi

echo "=== [0I] Merging worker results... ==="
# exp6/run.py appends _w{worker_index} to run_name → dir is ${RUN_NAME}_w${i}
src_dirs=()
for i in $(seq 0 $((NW-1))); do src_dirs+=("${OUTPUT_BASE}/${RUN_NAME}_w${i}"); done

uv run python scripts/merge_steering_workers.py \
    --experiment A1_formula \
    --variant it \
    --n-workers "$NW" \
    --merged-name "merged_${RUN_NAME}" \
    --output-base "$OUTPUT_BASE" \
    --source-dirs "${src_dirs[@]}"

echo "=== [0I] Done. Results in ${OUTPUT_BASE}/merged_${RUN_NAME}/ ==="
echo "Check: governance-content dissociation present in ≥2 of 4 formula variants"
