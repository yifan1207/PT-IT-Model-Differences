#!/usr/bin/env bash
# L1+L2: δ-cosine heatmap and commitment delay — autoregressive generation.
#
# Runs 8 parallel workers per model/variant combination on GPUs 0-7.
# Models run sequentially (PT then IT per model, models in order).
# Each model pair takes ~1-3h depending on model size and generation length.
#
# Output: results/cross_model/{model}/{variant}/
#   L1L2_w{0..7}.jsonl    — per-worker results
#   L1L2_results.jsonl    — merged results
#   L1L2_mean_cosine.npy  — mean δ-cosine [n_layers]
#
# Usage:
#   bash scripts/run_cross_model_L1L2.sh                      # all 6 models
#   bash scripts/run_cross_model_L1L2.sh --models gemma3_4b   # single model
#   bash scripts/run_cross_model_L1L2.sh --dry-run            # print only

set -euo pipefail

# ── Model cache on large disk (/mnt/storage, 13 TB free) ──────────────────────
# Prevents HF model downloads from filling the 439 GB boot disk.
export HF_HOME=/mnt/storage/yifan/hf_cache
export HF_TOKEN=${HF_TOKEN:-$(cat /mnt/storage/yifan/hf_cache/token 2>/dev/null || true)}

MODELS=("gemma3_4b" "llama31_8b" "qwen3_4b" "mistral_7b" "deepseek_v2_lite" "olmo2_7b")
DATASET="data/eval_dataset_v2.jsonl"
NW=8
MAX_NEW_TOKENS=512
DRY_RUN=0

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)   DRY_RUN=1; shift ;;
        --models)    IFS=',' read -ra MODELS <<< "$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" || echo "=== L1+L2 δ-cosine + commitment delay ==="
echo "Models: ${MODELS[*]}"

run_variant() {
    local model=$1 variant=$2
    local log_dir="logs/cross_model/L1L2_${model}_${variant}"
    mkdir -p "$log_dir"

    echo "=== $model $variant: $NW workers on GPU 0-$((NW-1)) ==="

    if [[ "$DRY_RUN" == "1" ]]; then
        for ((i=0; i<NW; i++)); do
            echo "[dry] uv run python -m src.poc.cross_model.collect_L1L2 --model $model --variant $variant --device cuda:$i --worker-index $i --n-workers $NW"
        done
        echo "[dry] merge: uv run python -m src.poc.cross_model.collect_L1L2 --model $model --variant $variant --merge-only --n-workers $NW"
        return
    fi

    # Launch parallel workers
    local pids=()
    for ((i=0; i<NW; i++)); do
        uv run python -m src.poc.cross_model.collect_L1L2 \
            --model "$model" \
            --variant "$variant" \
            --dataset "$DATASET" \
            --device "cuda:$i" \
            --worker-index "$i" \
            --n-workers "$NW" \
            --max-new-tokens "$MAX_NEW_TOKENS" \
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

    echo "[$(date +%T)] Workers done. Merging $model $variant ..."
    uv run python -m src.poc.cross_model.collect_L1L2 \
        --model "$model" --variant "$variant" \
        --merge-only --n-workers "$NW" \
        >> "$log_dir/merge.log" 2>&1
    echo "[$(date +%T)] Done $model $variant"
}

for model in "${MODELS[@]}"; do
    run_variant "$model" pt || echo "WARN: $model pt FAILED (model inaccessible?) — skipping"
    run_variant "$model" it || echo "WARN: $model it FAILED (model inaccessible?) — skipping"
done

echo "=== L1+L2 complete ==="
