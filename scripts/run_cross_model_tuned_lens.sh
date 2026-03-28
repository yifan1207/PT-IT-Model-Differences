#!/usr/bin/env bash
# Cross-model tuned-lens probe training, commitment evaluation, and transfer test.
#
# Three phases:
#   Phase 1: Train probes for all 6 models × 2 variants (PT + IT)
#            Models run sequentially (GPU memory), PT/IT in parallel on 2 GPUs.
#   Phase 2: Evaluate commitment for all 12 variants.
#   Phase 3: Transfer tests for all 6 models (PT-probes-on-IT).
#
# Usage:
#   bash scripts/run_cross_model_tuned_lens.sh
#   bash scripts/run_cross_model_tuned_lens.sh --n-tokens 10000 --n-steps 1000  # quick test
#
# GPU requirements:
#   - Most models: 1 GPU per variant (2 GPUs for parallel PT+IT training)
#   - DeepSeek-V2-Lite: multi-GPU (uses all visible GPUs per variant, runs sequentially)

set -euo pipefail

EXTRA_ARGS=("$@")

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
VARIANTS=(pt it)

mkdir -p logs/cross_model/tuned_lens

echo "=== Cross-Model Tuned-Lens Pipeline ==="
echo "Models: ${MODELS[*]}"
echo "Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

# ── Phase 1: Train probes ──────────────────────────────────────────────────
echo "=== Phase 1: Probe Training ==="

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Training probes for ${model} ---"

    if [[ "$model" == "deepseek_v2_lite" ]]; then
        # DeepSeek uses multi-GPU — run PT and IT sequentially
        for variant in "${VARIANTS[@]}"; do
            echo "[train] ${model}/${variant} (multi-GPU, sequential)"
            uv run python -m src.poc.cross_model.tuned_lens \
                --model "$model" --variant "$variant" --device cuda:0 \
                "${EXTRA_ARGS[@]}" \
                > "logs/cross_model/tuned_lens/train_${model}_${variant}.log" 2>&1
            echo "[train] ${model}/${variant} done."
        done
    else
        # Single-GPU models: run PT on cuda:0, IT on cuda:1 in parallel
        pids=()
        for i in "${!VARIANTS[@]}"; do
            variant="${VARIANTS[$i]}"
            gpu="cuda:${i}"
            echo "[train] ${model}/${variant} on ${gpu}"
            uv run python -m src.poc.cross_model.tuned_lens \
                --model "$model" --variant "$variant" --device "$gpu" \
                "${EXTRA_ARGS[@]}" \
                > "logs/cross_model/tuned_lens/train_${model}_${variant}.log" 2>&1 &
            pids+=($!)
        done

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                echo "[train] ERROR: PID $pid failed for ${model}"
                failed=1
            fi
        done
        if [[ "$failed" -ne 0 ]]; then
            echo "[train] WARNING: Some training jobs failed for ${model}. Check logs."
            echo "[train] Continuing with next model..."
        else
            echo "[train] ${model} PT+IT training complete."
        fi
    fi
done

echo ""
echo "=== Phase 1 complete. ==="

# ── Phase 2: Evaluate commitment ───────────────────────────────────────────
echo ""
echo "=== Phase 2: Commitment Evaluation ==="

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- Evaluating ${model} ---"

    if [[ "$model" == "deepseek_v2_lite" ]]; then
        for variant in "${VARIANTS[@]}"; do
            echo "[eval] ${model}/${variant} (multi-GPU, sequential)"
            uv run python -m src.poc.cross_model.tuned_lens \
                --model "$model" --variant "$variant" --device cuda:0 \
                --eval-only \
                > "logs/cross_model/tuned_lens/eval_${model}_${variant}.log" 2>&1
            echo "[eval] ${model}/${variant} done."
        done
    else
        pids=()
        for i in "${!VARIANTS[@]}"; do
            variant="${VARIANTS[$i]}"
            gpu="cuda:${i}"
            echo "[eval] ${model}/${variant} on ${gpu}"
            uv run python -m src.poc.cross_model.tuned_lens \
                --model "$model" --variant "$variant" --device "$gpu" \
                --eval-only \
                > "logs/cross_model/tuned_lens/eval_${model}_${variant}.log" 2>&1 &
            pids+=($!)
        done

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then
                echo "[eval] ERROR: PID $pid failed for ${model}"
                failed=1
            fi
        done
        if [[ "$failed" -ne 0 ]]; then
            echo "[eval] WARNING: Some eval jobs failed for ${model}. Check logs."
        else
            echo "[eval] ${model} PT+IT evaluation complete."
        fi
    fi
done

echo ""
echo "=== Phase 2 complete. ==="

# ── Phase 3: Transfer tests ───────────────────────────────────────────────
echo ""
echo "=== Phase 3: Transfer Tests ==="

for model in "${MODELS[@]}"; do
    echo "[transfer] ${model}"
    uv run python -m src.poc.cross_model.tuned_lens \
        --model "$model" --transfer-test --device cuda:0 \
        > "logs/cross_model/tuned_lens/transfer_${model}.log" 2>&1 || {
            echo "[transfer] WARNING: ${model} transfer test failed. Check log."
            continue
        }
    echo "[transfer] ${model} done."
done

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
echo "Next step: run scripts/plot_tuned_lens_commitment.py to generate figures."
