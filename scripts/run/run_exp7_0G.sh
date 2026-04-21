#!/bin/bash
# 0G: Train tuned-lens probes for ALL 6 models × 2 variants = 12 jobs
# Uses all 8 GPUs in fixed batches (8 parallel, then 4 parallel)
# Joint training mode (~34× faster than per-layer sequential)
#
# Belrose et al. 2023 exact recipe:
#   SGD Nesterov, lr=1.0, momentum=0.9, wd=0, 250 steps, 262K tokens/step
#   70M unique tokens from C4 (no recycling), linear LR decay, identity init
#   KL divergence loss, gradient clipping norm 1.0
#
# Estimated: ~2-3h per 4B model, ~3-5h per 7-8B model
# Wall time with 8 GPUs: ~5-6h total (batch 1: 8 jobs, batch 2: 4 jobs)
set -euo pipefail

export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/exp7/0G

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b deepseek_v2_lite olmo2_7b)
VARIANTS=(pt it)

# Build list of all (model, variant) pairs
declare -a JOBS=()
for m in "${MODELS[@]}"; do
    for v in "${VARIANTS[@]}"; do
        JOBS+=("${m}:${v}")
    done
done

echo "[$(date '+%H:%M')] === 0G Tuned Lens Training: ${#JOBS[@]} jobs on 8 GPUs ==="

# ── Phase 1: Training (batch-based, 8 at a time) ────────────────────────
NGPUS=8
batch=0
for ((start=0; start<${#JOBS[@]}; start+=NGPUS)); do
    batch=$((batch + 1))
    end=$((start + NGPUS))
    if ((end > ${#JOBS[@]})); then end=${#JOBS[@]}; fi

    echo "[$(date '+%H:%M')] --- Training batch $batch: jobs $start-$((end-1)) ---"

    pids=()
    for ((i=start; i<end; i++)); do
        gpu=$((i - start))
        IFS=: read -r model variant <<< "${JOBS[$i]}"
        logfile="logs/exp7/0G/${model}_${variant}.log"

        echo "[$(date '+%H:%M')] Training ${model}/${variant} on cuda:${gpu}"
        uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" \
            --device "cuda:${gpu}" \
            > "$logfile" 2>&1 &
        pids+=($!)
    done

    # Wait for ALL jobs in this batch
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[$(date '+%H:%M')] PID $pid FAILED"
            failed=1
        fi
    done
    if ((failed)); then
        echo "[$(date '+%H:%M')] WARNING: batch $batch had failures — check logs"
    fi
    echo "[$(date '+%H:%M')] --- Training batch $batch complete ---"
done

echo "[$(date '+%H:%M')] === Training COMPLETE ==="

# ── Phase 2: Eval (batch-based, 8 at a time) ────────────────────────────
echo "[$(date '+%H:%M')] === Eval phase ==="
batch=0
for ((start=0; start<${#JOBS[@]}; start+=NGPUS)); do
    batch=$((batch + 1))
    end=$((start + NGPUS))
    if ((end > ${#JOBS[@]})); then end=${#JOBS[@]}; fi

    echo "[$(date '+%H:%M')] --- Eval batch $batch ---"

    pids=()
    for ((i=start; i<end; i++)); do
        gpu=$((i - start))
        IFS=: read -r model variant <<< "${JOBS[$i]}"
        logfile="logs/exp7/0G/${model}_${variant}_eval.log"

        echo "[$(date '+%H:%M')] Eval ${model}/${variant} on cuda:${gpu} (2936 prompts)"
        uv run python -m src.poc.cross_model.tuned_lens \
            --model "$model" --variant "$variant" \
            --device "cuda:${gpu}" --eval-only \
            --dataset data/exp3_dataset.jsonl \
            > "$logfile" 2>&1 &
        pids+=($!)
    done

    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    echo "[$(date '+%H:%M')] --- Eval batch $batch complete ---"
done

echo "[$(date '+%H:%M')] === Eval COMPLETE ==="

# ── Phase 3: Cross-model CIs + plots ────────────────────────────────────
echo "[$(date '+%H:%M')] === Regenerating cross-model CIs (per-token method) ==="
uv run python -m src.poc.exp07_methodology_validation_tier0.bootstrap_ci \
    --merged-dir results/exp6/merged_A1_it_v4 \
    --cross-model-dir results/cross_model \
    > logs/exp7/0G/cross_model_ci.log 2>&1 || echo "[0G] cross-model CI had errors"

cp results/exp7/0D/ci_cross_model.json results/exp7/plots/data/ci_cross_model.json 2>/dev/null || true
echo "[$(date '+%H:%M')] === CIs regenerated ==="

echo "[$(date '+%H:%M')] === Generating commitment plots ==="
uv run python scripts/plot_commitment_delay.py \
    > logs/exp7/0G/plot_commitment.log 2>&1 || echo "[0G] plot had errors"

echo "[$(date '+%H:%M')] === Regenerating all Tier 0 plots ==="
uv run python scripts/plot_validation_tier0.py \
    > logs/exp7/0G/plot_tier0.log 2>&1 || echo "[0G] tier0 plots had errors"

echo "[$(date '+%H:%M')] === 0G ALL DONE ==="
