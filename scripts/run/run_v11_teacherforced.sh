#!/bin/bash
# exp11 v11.1 teacher-forced multi-branch run on Modal.
#
# Pipelines:
#   C            = IT free-run teacher, chat-template prompt
#   A'           = PT teacher-forced to C, raw prompt
#   B            = PT+IT graft teacher-forced to C, raw prompt (B1 branch)
#   A'_tmpl      = PT teacher-forced to C, chat-template prompt
#   B2           = PT+IT graft teacher-forced to C, chat-template prompt
#
# Deploys the app, triggers the v11 orchestrator, then downloads and plots results.
#
# Usage:
#   bash scripts/run_v11_teacherforced.sh [--deploy] [--n-prompts N] [--seed S]
#
# Defaults: n_prompts=400, prompt_seed=0, chunk_size=64.
#
# Output: results/exp11_matched_prefix_mlp_graft/data/exp11_exp3_400rand_v11_teacherforced/{model}/
#         results/exp11_matched_prefix_mlp_graft/plots/exp11_exp3_400rand_v11_teacherforced/

set -euo pipefail

DEPLOY=false
N_PROMPTS=400
PROMPT_SEED=0
CHUNK_SIZE=64
RUN_PREFIX="exp11_exp3_400rand_v11_teacherforced"
DATASET="exp3"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --deploy) DEPLOY=true; shift ;;
        --n-prompts) N_PROMPTS="$2"; shift 2 ;;
        --seed) PROMPT_SEED="$2"; shift 2 ;;
        --chunk-size) CHUNK_SIZE="$2"; shift 2 ;;
        --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 2 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ "$DEPLOY" == "true" ]]; then
    echo "=== Deploying Modal app ==="
    uv run modal deploy src/poc/exp11_matched_prefix_mlp_graft/modal_exp11.py
fi

echo "=== Triggering v11 teacher-forced orchestrator ==="
echo "run_prefix=$RUN_PREFIX dataset=$DATASET n_prompts=$N_PROMPTS seed=$PROMPT_SEED chunk_size=$CHUNK_SIZE"
uv run modal run src/poc/exp11_matched_prefix_mlp_graft/modal_exp11.py \
    --mode balanced-10gpu-v11-teacherforced \
    --run-name "$RUN_PREFIX" \
    --dataset "$DATASET" \
    --n-prompts "$N_PROMPTS" \
    --prompt-seed "$PROMPT_SEED" \
    --chunk-size "$CHUNK_SIZE"

LOCAL_DATA_ROOT="results/exp11_matched_prefix_mlp_graft/data/$RUN_PREFIX"
LOCAL_PLOT_ROOT="results/exp11_matched_prefix_mlp_graft/plots/$RUN_PREFIX"
mkdir -p "$LOCAL_DATA_ROOT" "$LOCAL_PLOT_ROOT"

echo "=== Downloading merged per-model outputs from Modal volume ==="
MODELS=(gemma3_4b llama31_8b mistral_7b olmo2_7b qwen3_4b deepseek_v2_lite)
for model in "${MODELS[@]}"; do
    remote="/${RUN_PREFIX}_${model}"
    local="$LOCAL_DATA_ROOT/$model"
    mkdir -p "$local"
    echo "[download] $model"
    uv run modal volume get exp11-runs "$remote" "$local" --force 2>/dev/null \
        || echo "  (no merged dir yet for $model)"
done

echo "=== Running analyze on each model ==="
for model in "${MODELS[@]}"; do
    run_dir="$LOCAL_DATA_ROOT/$model"
    if [[ -f "$run_dir/config.json" ]]; then
        uv run python -m src.poc.exp11_matched_prefix_mlp_graft.analyze \
            --run-dir "$run_dir" \
            --plot-dir "$run_dir/plots"
    else
        echo "  (skip $model — no config.json)"
    fi
done

echo "=== Generating overview plots ==="
uv run python scripts/plot_exp11_overview.py \
    --run-root "$LOCAL_DATA_ROOT" \
    --out-dir  "$LOCAL_PLOT_ROOT"

echo "=== DONE ==="
echo "Data: $LOCAL_DATA_ROOT"
echo "Plots: $LOCAL_PLOT_ROOT"
