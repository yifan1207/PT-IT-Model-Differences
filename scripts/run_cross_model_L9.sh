#!/usr/bin/env bash
# L9: Attention entropy divergence — single forward pass with eager attention.
#
# REQUIRES attn_implementation="eager" — handled inside collect_L9.py automatically.
# Flash/SDPA attention does NOT return attention weight matrices.
#
# Runs 8 parallel workers per model/variant combination on GPUs 0-7.
# Workers collect per-prompt attention entropy; merge step aggregates mean/SEM.
# Models run sequentially (PT then IT per model, models in order).
#
# Output: results/cross_model/{model}/{variant}/
#   L9_w{0..7}.jsonl       — per-worker per-prompt JSONL
#   L9_results.jsonl        — merged per-prompt results
#   L9_summary.json         — aggregated mean/SEM entropy per layer/head
#
# Usage:
#   bash scripts/run_cross_model_L9.sh                      # all 6 models
#   bash scripts/run_cross_model_L9.sh --models gemma3_4b   # single model
#   bash scripts/run_cross_model_L9.sh --dry-run            # print only

set -euo pipefail

# ── Model cache on large disk (/mnt/storage, 13 TB free) ──────────────────────
export HF_HOME=/mnt/storage/yifan/hf_cache
export HF_TOKEN=${HF_TOKEN:-$(cat /mnt/storage/yifan/hf_cache/token 2>/dev/null || true)}

MODELS=("gemma3_4b" "llama31_8b" "qwen3_4b" "mistral_7b" "deepseek_v2_lite" "olmo2_7b")
DATASET="data/eval_dataset_v2.jsonl"
NW=8
DRY_RUN=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --models)    IFS=',' read -ra MODELS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" || echo "=== L9 attention entropy divergence ==="
echo "Models: ${MODELS[*]}"

run_variant() {
    local model=$1 variant=$2
    local log_dir="logs/cross_model/L9_${model}_${variant}"
    mkdir -p "$log_dir"

    echo "=== $model $variant: $NW workers on GPU 0-$((NW-1)) ==="

    if [[ "$DRY_RUN" == "1" ]]; then
        for ((i=0; i<NW; i++)); do
            echo "[dry] uv run python -m src.poc.cross_model.collect_L9 --model $model --variant $variant --device cuda:$i --worker-index $i --n-workers $NW"
        done
        echo "[dry] merge: uv run python -m src.poc.cross_model.collect_L9 --model $model --variant $variant --merge-only --n-workers $NW"
        return
    fi

    # Launch parallel workers — each saves L9_w{i}.jsonl (per-prompt entropy)
    local pids=()
    for ((i=0; i<NW; i++)); do
        uv run python -m src.poc.cross_model.collect_L9 \
            --model "$model" \
            --variant "$variant" \
            --dataset "$DATASET" \
            --device "cuda:$i" \
            --worker-index "$i" \
            --n-workers "$NW" \
            > "$log_dir/w${i}.log" 2>&1 &
        pids+=($!)
    done

    # Wait for all workers
    local failed=0
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || {
            echo "ERROR: $model $variant worker $i failed (see $log_dir/w${i}.log)"
            failed=1
        }
    done
    [[ "$failed" == "1" ]] && return 1

    echo "[$(date +%T)] Workers done. Merging and aggregating entropy for $model $variant ..."
    uv run python -m src.poc.cross_model.collect_L9 \
        --model "$model" --variant "$variant" \
        --merge-only --n-workers "$NW" \
        >> "$log_dir/merge.log" 2>&1
    echo "[$(date +%T)] Done $model $variant"
}

for model in "${MODELS[@]}"; do
    run_variant "$model" pt || echo "WARN: $model pt FAILED (model inaccessible?) — skipping"
    run_variant "$model" it || echo "WARN: $model it FAILED (model inaccessible?) — skipping"
done

echo "=== L9 complete ==="
