#!/usr/bin/env bash
# L3: Weight change localization — per-layer RMS diff for all 6 model pairs.
#
# No GPU required. Loads models on CPU (~16GB RAM per pair for 8B models).
# Run sequentially; each model pair takes ~1-3 min.
#
# Output: results/cross_model/{model}/weight_diff.json
#
# Usage:
#   bash scripts/run_cross_model_L3.sh
#   bash scripts/run_cross_model_L3.sh gemma3_4b llama31_8b   # specific models only

set -euo pipefail

MODELS=("gemma3_4b" "llama31_8b" "qwen3_4b" "mistral_7b" "deepseek_v2_lite" "olmo2_7b")

# Allow overriding via positional args
if [[ $# -gt 0 ]]; then
    MODELS=("$@")
fi

echo "=== L3 weight diff: ${MODELS[*]} ==="

for model in "${MODELS[@]}"; do
    echo "[$(date +%T)] $model ..."
    if uv run python -m src.poc.cross_model.collect_L3 --model "$model"; then
        echo "[$(date +%T)] Done $model → results/cross_model/$model/weight_diff.json"
    else
        echo "WARN: $model FAILED (model inaccessible?) — skipping"
    fi
done

echo "=== L3 complete ==="
