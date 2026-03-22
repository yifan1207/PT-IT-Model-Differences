#!/usr/bin/env bash
# Run the full exp5 pipeline sequentially with resume support.
#
# Usage:
#   bash scripts/run_exp5.sh [experiment] [variant]
#
#   experiment : baseline | phase | progressive | cartography | subspace  (default: phase)
#   variant    : it | pt                                                   (default: it)
#
# Examples:
#   bash scripts/run_exp5.sh               # phase experiment, IT model
#   bash scripts/run_exp5.sh baseline      # baseline only
#   bash scripts/run_exp5.sh progressive   # progressive skip + directional sweep
#   bash scripts/run_exp5.sh phase pt      # phase experiment, PT model
#
# Resume: just re-run the same command. Finished conditions are skipped.
# Disk:   checked every 200 records and before each stage. Stops at 2 GB free.

set -euo pipefail

EXPERIMENT="${1:-phase}"
VARIANT="${2:-it}"
DATASET="data/exp3_dataset.jsonl"
PRECOMPUTE_RUN="precompute_${VARIANT}"
PRECOMPUTE_DIR="results/exp5/${PRECOMPUTE_RUN}/precompute"
MEAN_PATH="${PRECOMPUTE_DIR}/mean_mlp_outputs.npz"
DIR_PATH="${PRECOMPUTE_DIR}/corrective_directions.npz"

echo "========================================"
echo " exp5 pipeline"
echo " experiment : ${EXPERIMENT}"
echo " variant    : ${VARIANT}"
echo " $(date)"
echo "========================================"

# ── Disk check ───────────────────────────────────────────────────────────────
FREE_GB=$(df -BG . | awk 'NR==2 {gsub("G",""); print $4}')
echo "[disk] ${FREE_GB} GB free"
if [ "${FREE_GB}" -lt 3 ]; then
    echo "[disk] ERROR: less than 3 GB free — aborting to prevent data loss"
    exit 1
fi

# ── Precompute mean MLP outputs (needed for mean ablation) ───────────────────
needs_mean() {
    [[ "${EXPERIMENT}" == "phase" || "${EXPERIMENT}" == "cartography" || "${EXPERIMENT}" == "progressive" ]]
}

if needs_mean && [ ! -f "${MEAN_PATH}" ]; then
    echo ""
    echo "[1/3] Computing mean MLP outputs (1000 records) ..."
    uv run python -m src.poc.exp5.precompute \
        --variant "${VARIANT}" \
        --dataset "${DATASET}" \
        --compute mean \
        --run-name "${PRECOMPUTE_RUN}"
    echo "[1/3] Done — ${MEAN_PATH}"
else
    echo "[1/3] Mean MLP outputs: $([ -f "${MEAN_PATH}" ] && echo 'already done' || echo 'not needed for this experiment')"
fi

# ── Precompute corrective directions (needed for directional ablation) ────────
needs_directions() {
    [[ "${EXPERIMENT}" == "phase" || "${EXPERIMENT}" == "progressive" ]]
}

if needs_directions && [ ! -f "${DIR_PATH}" ]; then
    echo ""
    echo "[2/3] Computing corrective directions (1000 records, loads PT + IT sequentially) ..."
    uv run python -m src.poc.exp5.precompute \
        --variant "${VARIANT}" \
        --dataset "${DATASET}" \
        --compute directions \
        --run-name "${PRECOMPUTE_RUN}"
    echo "[2/3] Done — ${DIR_PATH}"
else
    echo "[2/3] Corrective directions: $([ -f "${DIR_PATH}" ] && echo 'already done' || echo 'not needed for this experiment')"
fi

# ── Run experiment ────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Running experiment: ${EXPERIMENT} (variant=${VARIANT})"
echo "      Completed conditions are skipped automatically on resume."
echo ""

# Build the argument list — only pass precompute paths if the files exist.
EXTRA_ARGS=()
if [ -f "${MEAN_PATH}" ]; then
    EXTRA_ARGS+=(--mean-acts-path "${MEAN_PATH}")
fi
if [ -f "${DIR_PATH}" ]; then
    EXTRA_ARGS+=(--corrective-direction-path "${DIR_PATH}")
fi

uv run python -m src.poc.exp5.run \
    --experiment "${EXPERIMENT}" \
    --variant "${VARIANT}" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "========================================"
echo " Done — $(date)"
echo "========================================"
