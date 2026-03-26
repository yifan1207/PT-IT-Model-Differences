#!/usr/bin/env bash
# L8: Intrinsic dimensionality profile — single forward pass per prompt.
#
# Runs PT and IT sequentially per model. All 6 models run on GPU 0 by default
# (one model at a time). Each model pair takes ~20-40 min on a single GPU.
#
# Output: results/cross_model/{model}/{variant}/L8_id_profile.json
#
# Usage:
#   bash scripts/run_cross_model_L8.sh
#   bash scripts/run_cross_model_L8.sh --dry-run

set -euo pipefail

MODELS=("gemma3_4b" "llama31_8b" "qwen3_4b" "mistral_7b" "deepseek_v2_lite" "olmo2_7b")
DATASET="data/eval_dataset_v2.jsonl"
DEVICE="cuda:0"
DRY_RUN=0

for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done
[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" || echo "=== L8 intrinsic dimensionality ==="

for model in "${MODELS[@]}"; do
    for variant in pt it; do
        cmd=(
            uv run python -m src.poc.cross_model.collect_L8
            --model "$model"
            --variant "$variant"
            --dataset "$DATASET"
            --device "$DEVICE"
        )
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[dry] ${cmd[*]}"
        else
            echo "[$(date +%T)] $model $variant ..."
            mkdir -p "logs/cross_model"
            "${cmd[@]}" 2>&1 | tee "logs/cross_model/L8_${model}_${variant}.log"
            echo "[$(date +%T)] Done $model $variant"
        fi
    done
done

echo "=== L8 complete ==="
