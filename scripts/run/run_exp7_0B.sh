#!/usr/bin/env bash
# Exp7 0B: Matched-Token Direction Validation
# Force-decodes models on each other's token sequences to control for the
# KV-cache / token-distribution confound.
#
# Runs 4 conditions:
#   1. Forward + governance: PT forced on IT tokens (governance-selected 600)
#   2. Forward + random:     PT forced on IT tokens (random 600 from 0H)
#   3. Reverse + governance: IT forced on PT tokens (governance-selected 600)
#   4. Reverse + governance: IT forced on PT tokens (governance-selected 600)
#
# Condition 1 isolates the token confound.
# Condition 2 tests token confound + prompt generalization.
# Conditions 3-4 add the reverse direction (IT on PT tokens).
#
# Prerequisites: run_exp7_0A.sh must complete first (reuses IT free-running acts)
#
# Usage:
#   bash scripts/run_exp7_0B.sh
#   bash scripts/run_exp7_0B.sh --n-records 20   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

CANONICAL="results/exp05_corrective_direction_ablation_cartography/precompute_v2/precompute/corrective_directions.npz"
NW=8

mkdir -p logs/exp7

run_force_decode() {
    local DIRECTION=$1
    local RECORD_SET=$2
    local LABEL="${DIRECTION}_${RECORD_SET}"

    echo ""
    echo "=== [0B] Force-decode: ${DIRECTION} direction, ${RECORD_SET} records (${NW} GPUs) ==="

    pids=()
    for i in $(seq 0 $((NW-1))); do
        uv run python -m src.poc.exp07_methodology_validation_tier0.force_decode_acts \
            --worker-index "$i" --n-workers "$NW" --device "cuda:$i" \
            --direction "$DIRECTION" --record-set "$RECORD_SET" \
            "${EXTRA_ARGS[@]}" \
            > logs/exp7/0B_${LABEL}_w${i}.log 2>&1 &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then echo "[0B] worker failed for $LABEL"; failed=1; fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[0B] Check logs/exp7/0B_${LABEL}_w*.log"
        exit 1
    fi

    echo "[0B] Merging ${LABEL}..."
    uv run python -m src.poc.exp07_methodology_validation_tier0.force_decode_acts \
        --merge-only --n-workers "$NW" \
        --direction "$DIRECTION" --record-set "$RECORD_SET"
}

# ── Step 1: Collect force-decode activations (4 conditions) ──────────────────
run_force_decode forward governance
run_force_decode forward random
run_force_decode reverse governance

# ── Step 2: Compare matched-token directions to canonical ────────────────────
echo ""
echo "=== [0B] Computing matched-token direction comparisons ==="

# (a) Forward + governance: isolates token confound only
echo "[0B] Forward governance (PT on IT tokens, selected 600)..."
uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_directions \
    --it-acts-dir results/exp07_methodology_validation_tier0/0A/acts/ \
    --pt-forced-acts-dir results/exp07_methodology_validation_tier0/0B/forward_governance/acts/ \
    --canonical-npz "$CANONICAL" \
    --matched-mode --matched-label governance_selected \
    --output-dir results/exp07_methodology_validation_tier0/0B/

# (b) Forward + random: tests token confound + prompt generalization
echo "[0B] Forward random (PT on IT tokens, random 600)..."
uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_directions \
    --it-acts-dir results/exp07_methodology_validation_tier0/0A/acts/ \
    --pt-forced-acts-dir results/exp07_methodology_validation_tier0/0B/forward_random/acts/ \
    --canonical-npz "$CANONICAL" \
    --matched-mode --matched-label random_600 \
    --output-dir results/exp07_methodology_validation_tier0/0B/

# (c) Reverse + governance: IT forced on PT tokens
echo "[0B] Reverse governance (IT on PT tokens, selected 600)..."
uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_directions \
    --it-acts-dir results/exp07_methodology_validation_tier0/0B/reverse_governance/acts/ \
    --pt-forced-acts-dir results/exp07_methodology_validation_tier0/0A/acts/ \
    --canonical-npz "$CANONICAL" \
    --matched-mode --matched-label reverse_governance \
    --output-dir results/exp07_methodology_validation_tier0/0B/

echo ""
echo "=== [0B] Done. Results in results/exp07_methodology_validation_tier0/0B/ ==="
echo "Files:"
echo "  matched_cosines_governance_selected.json  — forward, governance 600"
echo "  matched_cosines_random_600.json           — forward, random 600"
echo "  matched_cosines_reverse_governance.json   — reverse, governance 600"
echo ""
echo "Check: corrective layer cosine > 0.90 in all conditions"
echo "Check: corrective_vs_early_drop — if > 0.05, token confound is layer-specific"
