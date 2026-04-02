#!/bin/bash
# Download 0G eval results from Modal Volume to local server.
# Run after Modal eval completes (or to check intermediate progress).
set -euo pipefail

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
VARIANTS=(pt it)
VOLUME_NAME="0g-results"

echo "=== Downloading results from Modal Volume: $VOLUME_NAME ==="

for model in "${MODELS[@]}"; do
    local_dir="results/cross_model/$model/tuned_lens/commitment"
    mkdir -p "$local_dir"

    for variant in "${VARIANTS[@]}"; do
        echo "[download] $model/$variant"

        # Download JSONL
        modal volume get "$VOLUME_NAME" \
            "/$model/tuned_lens/commitment/tuned_lens_commitment_${variant}.jsonl" \
            "$local_dir/" 2>/dev/null || echo "  (no JSONL yet)"

        # Download summary
        modal volume get "$VOLUME_NAME" \
            "/$model/tuned_lens/commitment/summary_${variant}.json" \
            "$local_dir/" 2>/dev/null || echo "  (no summary yet)"
    done
done

echo ""
echo "=== Download complete ==="
echo "Check results:"
for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        jsonl="results/cross_model/$model/tuned_lens/commitment/tuned_lens_commitment_${variant}.jsonl"
        if [[ -f "$jsonl" ]]; then
            n=$(wc -l < "$jsonl")
            echo "  $model/$variant: $n prompts"
        else
            echo "  $model/$variant: not yet available"
        fi
    done
done

echo ""
echo "To sync to GCS:"
echo "  gsutil -m rsync -r results/cross_model/ gs://pt-vs-it-results/cross_model/"
