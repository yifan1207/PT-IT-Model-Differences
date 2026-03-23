#!/usr/bin/env bash
# Run the full exp5 pipeline across N GPUs in parallel.
#
# Precompute (mean MLP outputs + corrective directions) is also parallelised:
# each GPU processes 1/N of the reference records, then partials are merged.
#
# Usage:
#   bash scripts/run_exp5_parallel.sh [experiment] [variant] [n_gpus]
#
# Examples:
#   bash scripts/run_exp5_parallel.sh phase it 8
#   bash scripts/run_exp5_parallel.sh cartography it 8
#   bash scripts/run_exp5_parallel.sh progressive it 4
#
# Resume: re-run the same command. Finished (condition, benchmark) pairs are skipped.
# Logs:   logs/exp5_<experiment>_<variant>_w<N>.log

set -euo pipefail

EXPERIMENT="${1:-phase}"
VARIANT="${2:-it}"
N_GPUS="${3:-8}"
DATASET="data/exp3_dataset.jsonl"
PRECOMPUTE_RUN="precompute_${VARIANT}"
PRECOMPUTE_DIR="results/exp5/${PRECOMPUTE_RUN}/precompute"
MEAN_PATH="${PRECOMPUTE_DIR}/mean_mlp_outputs.npz"
DIR_PATH="${PRECOMPUTE_DIR}/corrective_directions.npz"
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo " exp5 parallel pipeline"
echo " experiment : ${EXPERIMENT}"
echo " variant    : ${VARIANT}"
echo " gpus       : ${N_GPUS}"
echo " $(date)"
echo "========================================"

FREE_GB=$(df -BG . | awk 'NR==2 {gsub("G",""); print $4}')
echo "[disk] ${FREE_GB} GB free"
if [ "${FREE_GB}" -lt 5 ]; then
    echo "[disk] ERROR: less than 5 GB free — aborting"; exit 1
fi

needs_mean() {
    [[ "${EXPERIMENT}" == "phase" || "${EXPERIMENT}" == "cartography" || "${EXPERIMENT}" == "progressive" ]]
}
needs_directions() {
    [[ "${EXPERIMENT}" == "phase" || "${EXPERIMENT}" == "progressive" ]]
}

# ── Step 1: Parallel precompute — mean MLP outputs ───────────────────────────
if needs_mean && [ ! -f "${MEAN_PATH}" ]; then
    echo ""
    echo "[1a] Computing mean MLP outputs across ${N_GPUS} GPUs ..."
    PIDS=()
    for GPU in $(seq 0 $((N_GPUS - 1))); do
        LOG="${LOG_DIR}/exp5_precompute_mean_${VARIANT}_w${GPU}.log"
        CUDA_VISIBLE_DEVICES="${GPU}" uv run python -m src.poc.exp5.precompute \
            --variant "${VARIANT}" --dataset "${DATASET}" \
            --compute mean \
            --run-name "${PRECOMPUTE_RUN}" \
            --device cuda \
            --worker-index "${GPU}" --n-workers "${N_GPUS}" \
            > "${LOG}" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "${PID}" || { echo "Precompute worker failed"; exit 1; }; done

    echo "[1b] Merging mean MLP partials ..."
    uv run python -m src.poc.exp5.precompute \
        --variant "${VARIANT}" --dataset "${DATASET}" \
        --compute merge-mean \
        --run-name "${PRECOMPUTE_RUN}" \
        --n-workers "${N_GPUS}" \
        2>&1 | tee "${LOG_DIR}/exp5_precompute_mean_${VARIANT}_merge.log"
    echo "[1] Done — ${MEAN_PATH}"
else
    echo "[1] Mean MLP outputs: $([ -f "${MEAN_PATH}" ] && echo 'already done' || echo 'not needed')"
fi

# ── Step 2: Parallel precompute — corrective directions ──────────────────────
if needs_directions && [ ! -f "${DIR_PATH}" ]; then
    echo ""
    echo "[2a] Computing corrective directions across ${N_GPUS} GPUs ..."
    echo "     Each GPU loads PT then IT sequentially (~17 GB peak per GPU)"
    PIDS=()
    for GPU in $(seq 0 $((N_GPUS - 1))); do
        LOG="${LOG_DIR}/exp5_precompute_dir_${VARIANT}_w${GPU}.log"
        CUDA_VISIBLE_DEVICES="${GPU}" uv run python -m src.poc.exp5.precompute \
            --variant "${VARIANT}" --dataset "${DATASET}" \
            --compute directions \
            --run-name "${PRECOMPUTE_RUN}" \
            --device cuda \
            --worker-index "${GPU}" --n-workers "${N_GPUS}" \
            > "${LOG}" 2>&1 &
        PIDS+=($!)
    done
    for PID in "${PIDS[@]}"; do wait "${PID}" || { echo "Precompute worker failed"; exit 1; }; done

    echo "[2b] Merging direction partials ..."
    uv run python -m src.poc.exp5.precompute \
        --variant "${VARIANT}" --dataset "${DATASET}" \
        --compute merge-directions \
        --run-name "${PRECOMPUTE_RUN}" \
        --n-workers "${N_GPUS}" \
        2>&1 | tee "${LOG_DIR}/exp5_precompute_dir_${VARIANT}_merge.log"
    echo "[2] Done — ${DIR_PATH}"
else
    echo "[2] Corrective directions: $([ -f "${DIR_PATH}" ] && echo 'already done' || echo 'not needed')"
fi

# ── Step 3: Parallel evaluation ───────────────────────────────────────────────
echo ""
echo "[3] Launching ${N_GPUS} evaluation workers ..."

EXTRA_ARGS=()
[ -f "${MEAN_PATH}" ] && EXTRA_ARGS+=(--mean-acts-path "${MEAN_PATH}")
[ -f "${DIR_PATH}" ]  && EXTRA_ARGS+=(--corrective-direction-path "${DIR_PATH}")

PIDS=()
for GPU in $(seq 0 $((N_GPUS - 1))); do
    LOG="${LOG_DIR}/exp5_${EXPERIMENT}_${VARIANT}_w${GPU}.log"
    echo "  GPU ${GPU} → ${LOG}"
    CUDA_VISIBLE_DEVICES="${GPU}" uv run python -m src.poc.exp5.run \
        --experiment "${EXPERIMENT}" \
        --variant "${VARIANT}" \
        --device cuda \
        --worker-index "${GPU}" \
        --n-workers "${N_GPUS}" \
        "${EXTRA_ARGS[@]}" \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "PIDs: ${PIDS[*]}"
echo "Watch: tail -f ${LOG_DIR}/exp5_${EXPERIMENT}_${VARIANT}_w0.log"
echo ""
echo "Waiting for all workers ..."

FAILED=0
for PID in "${PIDS[@]}"; do
    wait "${PID}" || { echo "Worker PID ${PID} failed"; FAILED=1; }
done
[ "${FAILED}" -ne 0 ] && { echo "ERROR: workers failed — check logs before merging"; exit 1; }

# ── Step 4: Merge ─────────────────────────────────────────────────────────────
echo ""
echo "[4] Merging worker outputs ..."
uv run python scripts/merge_exp5_workers.py \
    --experiment "${EXPERIMENT}" \
    --variant "${VARIANT}" \
    --n-workers "${N_GPUS}"

echo ""
echo "========================================"
echo " Done — $(date)"
echo " Results: results/exp5/merged_${EXPERIMENT}_${VARIANT}/"
echo "========================================"
