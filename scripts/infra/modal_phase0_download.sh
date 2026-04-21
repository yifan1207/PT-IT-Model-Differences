#!/usr/bin/env bash
# Download Phase 0 results from Modal Volume and sync to GCS.
#
# Usage:
#   bash scripts/modal_phase0_download.sh              # download all
#   bash scripts/modal_phase0_download.sh --gcs        # download + sync to GCS
#   bash scripts/modal_phase0_download.sh --model llama31_8b  # single model
set -euo pipefail

VOLUME="phase0-results"
LOCAL_BASE="results/cross_model"
GCS_BUCKET="gs://pt-vs-it-results"
SYNC_GCS=false
MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gcs) SYNC_GCS=true; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
if [[ -n "$MODEL" ]]; then
    MODELS=("$MODEL")
fi

echo "=== Phase 0 Results Download ==="
echo "Volume: $VOLUME"
echo "Models: ${MODELS[*]}"
echo ""

for model in "${MODELS[@]}"; do
    echo "── $model ──"

    # Directions
    local_dir="${LOCAL_BASE}/${model}/directions"
    mkdir -p "$local_dir"
    echo "  Downloading directions..."
    modal volume get "$VOLUME" \
        "cross_model/${model}/directions" \
        "$local_dir/" 2>/dev/null || echo "  (no directions yet)"

    # Steering results
    local_dir="${LOCAL_BASE}/${model}/exp6"
    mkdir -p "$local_dir"
    echo "  Downloading exp6 results..."
    modal volume get "$VOLUME" \
        "cross_model/${model}/exp6" \
        "$local_dir/" 2>/dev/null || echo "  (no exp6 results yet)"

    # Show what we got
    if [[ -d "${LOCAL_BASE}/${model}/directions" ]]; then
        n_npz=$(find "${LOCAL_BASE}/${model}/directions" -name "*.npz" 2>/dev/null | wc -l)
        echo "  Directions: ${n_npz} .npz files"
    fi
    if [[ -d "${LOCAL_BASE}/${model}/exp6" ]]; then
        n_scores=$(find "${LOCAL_BASE}/${model}/exp6" -name "scores.jsonl" 2>/dev/null | wc -l)
        echo "  Exp6: ${n_scores} scores.jsonl files"
    fi
    echo ""
done

echo "=== Download complete ==="

if $SYNC_GCS; then
    echo ""
    echo "=== Syncing to GCS ==="
    gsutil -m rsync -r "$LOCAL_BASE/" "${GCS_BUCKET}/cross_model/"
    echo "=== GCS sync complete ==="
fi
