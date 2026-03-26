#!/usr/bin/env bash
# L9: Attention entropy divergence — single forward pass with eager attention.
#
# REQUIRES attn_implementation="eager" — model loaded with full attention matrices.
# This is handled inside collect_L9.py automatically.
#
# Runs PT and IT sequentially per model on GPU 0. Each model pair takes ~30-60 min.
#
# Output: results/cross_model/{model}/{variant}/L9_attn_entropy.json
#
# Usage:
#   bash scripts/run_cross_model_L9.sh
#   bash scripts/run_cross_model_L9.sh --dry-run

set -euo pipefail

MODELS=("gemma3_4b" "llama31_8b" "qwen3_4b" "mistral_7b" "deepseek_v2_lite" "olmo2_7b")
DATASET="data/eval_dataset_v2.jsonl"
DEVICE="cuda:0"
DRY_RUN=0

for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done
[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" || echo "=== L9 attention entropy ==="

for model in "${MODELS[@]}"; do
    for variant in pt it; do
        cmd=(
            uv run python -m src.poc.cross_model.collect_L9
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
            "${cmd[@]}" 2>&1 | tee "logs/cross_model/L9_${model}_${variant}.log"
            echo "[$(date +%T)] Done $model $variant"
        fi
    done
done

echo "=== L9 complete ==="
