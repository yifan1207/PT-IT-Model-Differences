#!/usr/bin/env bash
# Exp7 0J: Corrective Onset Threshold Sensitivity Analysis
# Tests whether the ~59% corrective onset depth claim is robust to the threshold
# used to define "onset" on the δ-cosine profile.
#
# Steps:
#   1. CPU analysis: compute onset under 9 threshold choices (5 σ-based, 4 absolute)
#      for all 6 model families → 6×9 table
#   2. Extract Gemma onset layers under 0.5σ (broader) and 2σ (narrower)
#   3. Re-run Gemma A1 α-sweep with those alternative layer ranges
#      to verify dose-response is qualitatively unchanged
#
# Prerequisites: results/cross_model/plots/data/L1_mean_delta_cosine.csv (already exists)
#
# Usage:
#   bash scripts/run_exp7_0J.sh
#   bash scripts/run_exp7_0J.sh --skip-a1-reruns   # analysis only, no GPU needed
#   bash scripts/run_exp7_0J.sh --n-eval-examples 100   # quick test

set -euo pipefail

SKIP_A1=0
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-a1-reruns) SKIP_A1=1; shift ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

CSV_PATH="results/cross_model/plots/data/L1_mean_delta_cosine.csv"
if [[ ! -f "$CSV_PATH" ]]; then
    echo "[0J] ERROR: L1_mean_delta_cosine.csv not found at $CSV_PATH"
    echo "[0J] This file is produced by the cross-model L1 collection."
    exit 1
fi

CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
OUTPUT_DIR="results/exp7/0J"
NW=8

mkdir -p logs/exp7 "$OUTPUT_DIR"

# ── Step 1: CPU analysis ──────────────────────────────────────────────────────
echo "=== [0J] Step 1: Onset threshold sensitivity analysis (CPU, ~1 min) ==="
uv run python -m src.poc.exp7.onset_threshold_sensitivity \
    --csv-path "$CSV_PATH" \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee logs/exp7/0J_analysis.log

ALT_RANGES="$OUTPUT_DIR/gemma_alt_ranges.json"
if [[ ! -f "$ALT_RANGES" ]]; then
    echo "[0J] ERROR: gemma_alt_ranges.json not produced. Check the analysis step."
    exit 1
fi

echo "[0J] Gemma alternative layer ranges:"
cat "$ALT_RANGES"

if [[ "$SKIP_A1" -eq 1 ]]; then
    echo "[0J] --skip-a1-reruns set. Skipping GPU steps."
    echo "=== [0J] Done (analysis only). Results in $OUTPUT_DIR/ ==="
    exit 0
fi

# ── Step 2: Extract Gemma alt onset layers ────────────────────────────────────
echo ""
echo "=== [0J] Step 2: Extracting Gemma onset layers for A1 reruns ==="

# Parse onset layers from JSON
ONSET_05=$(uv run python -c "
import json
d = json.load(open('$ALT_RANGES'))
v = d['broader_0.5sig']['onset_layer']
print(v if v is not None else 17)
")
ONSET_2=$(uv run python -c "
import json
d = json.load(open('$ALT_RANGES'))
v = d['narrower_2sig']['onset_layer']
print(v if v is not None else 23)
")
NLAYERS=34   # Gemma always 34 layers

echo "[0J] Broader onset (0.5σ): layer ${ONSET_05} → A1 range ${ONSET_05}-$((NLAYERS-1))"
echo "[0J] Narrower onset (2σ):  layer ${ONSET_2} → A1 range ${ONSET_2}-$((NLAYERS-1))"

if [[ ! -f "$CORR_DIR" ]]; then
    echo "[0J] ERROR: corrective_directions.npz not found at $CORR_DIR"
    exit 1
fi

run_gemma_a1() {
    local LABEL=$1
    local BOUNDARY=$2
    local RUN_NAME="A1_gemma_onset_${LABEL}"

    echo ""
    echo "=== [0J] Running Gemma A1 with onset=${LABEL} (proposal_boundary=${BOUNDARY}, range=${BOUNDARY}-$((NLAYERS-1))) ==="

    pids=()
    for i in $(seq 0 $((NW-1))); do
        uv run python -m src.poc.exp6.run \
            --experiment A1 \
            --variant it \
            --worker-index "$i" --n-workers "$NW" \
            --device "cuda:${i}" \
            --run-name "${RUN_NAME}_w${i}" \
            --output-base "$OUTPUT_DIR" \
            --corrective-direction-path "$CORR_DIR" \
            --proposal-boundary "$BOUNDARY" \
            --n-layers "$NLAYERS" \
            "${EXTRA_ARGS[@]}" \
            > logs/exp7/0J_${LABEL}_w${i}.log 2>&1 &
        pids+=($!)
    done

    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[0J] ERROR: worker failed for onset=${LABEL}"
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[0J] Check logs/exp7/0J_${LABEL}_w*.log"
        exit 1
    fi

    src_dirs=()
    for i in $(seq 0 $((NW-1))); do src_dirs+=("${OUTPUT_DIR}/${RUN_NAME}_w${i}"); done
    uv run python scripts/merge_exp6_workers.py \
        --experiment A1 --variant it --n-workers "$NW" \
        --merged-name "${RUN_NAME}" \
        --output-base "$OUTPUT_DIR" \
        --source-dirs "${src_dirs[@]}"

    echo "[0J] Done: ${OUTPUT_DIR}/${RUN_NAME}/"
}

# ── Step 3: A1 reruns with alternative layer ranges ───────────────────────────
echo ""
echo "=== [0J] Step 3: Gemma A1 reruns with alternative onset thresholds ==="

run_gemma_a1 "broader_05sig"  "$ONSET_05"
run_gemma_a1 "narrower_2sig"  "$ONSET_2"

# ── Step 4: Plot all results ──────────────────────────────────────────────────
echo ""
echo "=== [0J] Step 4: Generating plots ==="
uv run python scripts/plot_exp7_tier0.py --experiments 0J --output-dir results/exp7/plots/

echo ""
echo "=== [0J] Done. Results in $OUTPUT_DIR/ ==="
echo "Check: $OUTPUT_DIR/onset_table.csv"
echo "Check: $OUTPUT_DIR/onset_summary.json (onset_range ≤ 3 layers per family)"
echo "Check: $OUTPUT_DIR/plots/onset_sensitivity.png"
echo "Check: Gemma A1 dose-response shape similar across onset choices"
