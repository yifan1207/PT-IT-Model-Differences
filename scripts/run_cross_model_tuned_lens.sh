#!/usr/bin/env bash
# Cross-model tuned-lens probe training, commitment evaluation, and transfer test.
#
# Three phases:
#   Phase 1: Train probes for all 6 models × 2 variants (PT + IT)
#   Phase 2: Evaluate commitment for all 12 variants.
#   Phase 3: Transfer tests for all 6 models (PT-probes-on-IT).
#
# GPU scheduling (8 GPUs available):
#   - Each model uses 2 GPUs (PT on even, IT on odd) run in parallel
#   - Up to 4 models run simultaneously (8 GPUs / 2 = 4 concurrent)
#   - DeepSeek-V2-Lite: multi-GPU (uses all GPUs, runs alone)
#
# Usage:
#   bash scripts/run_cross_model_tuned_lens.sh
#   bash scripts/run_cross_model_tuned_lens.sh --n-tokens 10000 --n-steps 1000  # quick test

set -euo pipefail

# Limit CPU threads per process to avoid contention with 8 concurrent models
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=("$@")

# Models grouped by size for scheduling.
# Batch 1: 4 smaller/medium models (8 GPUs, 2 per model)
# Batch 2: 1 remaining model (2 GPUs)
# Batch 3: DeepSeek (multi-GPU, runs alone)
BATCH1_MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b)
BATCH2_MODELS=(olmo2_7b)
DEEPSEEK=deepseek_v2_lite
ALL_MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b deepseek_v2_lite)

mkdir -p logs/cross_model/tuned_lens

echo "=== Cross-Model Tuned-Lens Pipeline ==="
echo "Models: ${ALL_MODELS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

# ── Helper: run one model's PT+IT on two GPUs ──────────────────────────────
run_model_pair() {
    local model="$1"
    local gpu_pt="$2"
    local gpu_it="$3"
    local phase="$4"  # "train" or "eval"

    local extra_flags=()
    if [[ "$phase" == "eval" ]]; then
        extra_flags+=(--eval-only)
    fi

    local pids=()
    for variant_gpu in "pt:${gpu_pt}" "it:${gpu_it}"; do
        local variant="${variant_gpu%%:*}"
        local gpu="${variant_gpu##*:}"
        echo "[${phase}] ${model}/${variant} on cuda:${gpu}"
        uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" --device "cuda:${gpu}" \
            "${extra_flags[@]}" \
            "${EXTRA_ARGS[@]}" \
            > "logs/cross_model/tuned_lens/${phase}_${model}_${variant}.log" 2>&1 &
        pids+=($!)
    done

    # Return PIDs via global variable (bash limitation)
    MODEL_PIDS=("${pids[@]}")
}

# ── Helper: wait for a set of PIDs ─────────────────────────────────────────
wait_pids() {
    local label="$1"
    shift
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[${label}] ERROR: PID $pid failed"
            failed=1
        fi
    done
    return $failed
}

# ── Phase 1: Train probes ──────────────────────────────────────────────────
echo "=== Phase 1: Probe Training ==="

# Batch 1: 4 models on GPUs 0-7 (2 GPUs each, all in parallel)
echo ""
echo "--- Batch 1: ${BATCH1_MODELS[*]} (parallel, 8 GPUs) ---"
all_pids=()
for i in "${!BATCH1_MODELS[@]}"; do
    model="${BATCH1_MODELS[$i]}"
    gpu_pt=$((i * 2))
    gpu_it=$((i * 2 + 1))
    run_model_pair "$model" "$gpu_pt" "$gpu_it" "train"
    all_pids+=("${MODEL_PIDS[@]}")
done

if ! wait_pids "train-batch1" "${all_pids[@]}"; then
    echo "[train] WARNING: Some batch 1 training jobs failed. Check logs."
fi
echo "[train] Batch 1 complete."

# Batch 2: remaining single-GPU models
echo ""
echo "--- Batch 2: ${BATCH2_MODELS[*]} ---"
all_pids=()
for model in "${BATCH2_MODELS[@]}"; do
    run_model_pair "$model" 0 1 "train"
    all_pids+=("${MODEL_PIDS[@]}")
done

if ! wait_pids "train-batch2" "${all_pids[@]}"; then
    echo "[train] WARNING: Some batch 2 training jobs failed. Check logs."
fi
echo "[train] Batch 2 complete."

# DeepSeek: multi-GPU (runs alone, sequentially PT then IT)
echo ""
echo "--- DeepSeek-V2-Lite (multi-GPU, sequential) ---"
for variant in pt it; do
    echo "[train] ${DEEPSEEK}/${variant} (multi-GPU)"
    uv run python -m src.poc.cross_model.tuned_lens \
        --model "$DEEPSEEK" --variant "$variant" --device cuda:0 \
        "${EXTRA_ARGS[@]}" \
        > "logs/cross_model/tuned_lens/train_${DEEPSEEK}_${variant}.log" 2>&1 || {
            echo "[train] WARNING: ${DEEPSEEK}/${variant} failed. Check log."
        }
    echo "[train] ${DEEPSEEK}/${variant} done."
done

echo ""
echo "=== Phase 1 complete. ==="

# ── Phase 2: Evaluate commitment ───────────────────────────────────────────
echo ""
echo "=== Phase 2: Commitment Evaluation ==="

# Same batching as Phase 1
echo ""
echo "--- Batch 1: ${BATCH1_MODELS[*]} (parallel, 8 GPUs) ---"
all_pids=()
for i in "${!BATCH1_MODELS[@]}"; do
    model="${BATCH1_MODELS[$i]}"
    gpu_pt=$((i * 2))
    gpu_it=$((i * 2 + 1))
    run_model_pair "$model" "$gpu_pt" "$gpu_it" "eval"
    all_pids+=("${MODEL_PIDS[@]}")
done

if ! wait_pids "eval-batch1" "${all_pids[@]}"; then
    echo "[eval] WARNING: Some batch 1 eval jobs failed. Check logs."
fi
echo "[eval] Batch 1 complete."

echo ""
echo "--- Batch 2: ${BATCH2_MODELS[*]} ---"
all_pids=()
for model in "${BATCH2_MODELS[@]}"; do
    run_model_pair "$model" 0 1 "eval"
    all_pids+=("${MODEL_PIDS[@]}")
done

if ! wait_pids "eval-batch2" "${all_pids[@]}"; then
    echo "[eval] WARNING: Some batch 2 eval jobs failed. Check logs."
fi
echo "[eval] Batch 2 complete."

# DeepSeek eval (multi-GPU, sequential)
echo ""
echo "--- DeepSeek-V2-Lite eval (multi-GPU, sequential) ---"
for variant in pt it; do
    echo "[eval] ${DEEPSEEK}/${variant} (multi-GPU)"
    uv run python -m src.poc.cross_model.tuned_lens \
        --model "$DEEPSEEK" --variant "$variant" --device cuda:0 \
        --eval-only \
        "${EXTRA_ARGS[@]}" \
        > "logs/cross_model/tuned_lens/eval_${DEEPSEEK}_${variant}.log" 2>&1 || {
            echo "[eval] WARNING: ${DEEPSEEK}/${variant} eval failed. Check log."
        }
    echo "[eval] ${DEEPSEEK}/${variant} done."
done

echo ""
echo "=== Phase 2 complete. ==="

# ── Phase 3: Transfer tests ───────────────────────────────────────────────
echo ""
echo "=== Phase 3: Transfer Tests ==="

# Transfer tests are lightweight (1 model each, ~2 min) — run all 5 non-DeepSeek in parallel
# Each loads IT model only, so fits on 1 GPU each
NON_DS_MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b)
all_pids=()
for i in "${!NON_DS_MODELS[@]}"; do
    model="${NON_DS_MODELS[$i]}"
    echo "[transfer] ${model} on cuda:${i}"
    uv run python -m src.poc.cross_model.tuned_lens \
        --model "$model" --transfer-test --device "cuda:${i}" \
        > "logs/cross_model/tuned_lens/transfer_${model}.log" 2>&1 &
    all_pids+=($!)
done

if ! wait_pids "transfer-all" "${all_pids[@]}"; then
    echo "[transfer] WARNING: Some transfer tests failed. Check logs."
fi

# DeepSeek: multi-GPU, runs alone
echo "[transfer] ${DEEPSEEK} (multi-GPU)"
uv run python -m src.poc.cross_model.tuned_lens \
    --model "$DEEPSEEK" --transfer-test --device cuda:0 \
    > "logs/cross_model/tuned_lens/transfer_${DEEPSEEK}.log" 2>&1 || {
        echo "[transfer] WARNING: ${DEEPSEEK} transfer test failed."
    }

echo ""
echo "=== Phase 3 complete. ==="

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=== All phases complete. ==="
echo "Results:"
echo "  Probes:     results/cross_model/{model}/tuned_lens/{pt,it}/probe_layer_*.pt"
echo "  Commitment: results/cross_model/{model}/tuned_lens/commitment/"
echo "  Transfer:   results/cross_model/{model}/tuned_lens/commitment/transfer_test.json"
echo ""
echo "Next step: uv run python scripts/plot_tuned_lens_commitment.py"
