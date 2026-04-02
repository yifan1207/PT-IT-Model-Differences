#!/bin/bash
# Upload trained tuned-lens probes to Modal Volume for cloud eval.
# Run this ONCE after training completes on the local server.
#
# Prerequisites:
#   modal token set ...   (already done)
#   All 12 training_summary.json files exist
#
# Uploads only probe_layer_*.pt files (not checkpoints/commitment data)
# Total size: ~17 GB across 6 models × 2 variants
set -euo pipefail

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
VARIANTS=(pt it)
VOLUME_NAME="0g-probes"

echo "=== Uploading probes to Modal Volume: $VOLUME_NAME ==="

# Verify all training is complete
missing=0
for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        summary="results/cross_model/$model/tuned_lens/$variant/training_summary.json"
        if [[ ! -f "$summary" ]]; then
            echo "ERROR: Missing $summary — training not complete!"
            missing=1
        fi
    done
done
if ((missing)); then
    echo "Aborting: not all training is complete."
    exit 1
fi
echo "All 12 training summaries verified."

# Create volume if needed (idempotent)
modal volume create "$VOLUME_NAME" 2>/dev/null || true

# Upload probes for each model/variant
for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        src_dir="results/cross_model/$model/tuned_lens/$variant"
        n_probes=$(ls "$src_dir"/probe_layer_*.pt 2>/dev/null | wc -l)
        size=$(du -sm "$src_dir"/probe_layer_*.pt 2>/dev/null | tail -1 | cut -f1)

        echo "[upload] $model/$variant: $n_probes probes (~${size} MB)"
        modal volume put "$VOLUME_NAME" \
            "$src_dir"/probe_layer_*.pt \
            "/$model/$variant/"
    done
done

echo ""
echo "=== Upload complete ==="
echo "Verify with: modal volume ls $VOLUME_NAME"
