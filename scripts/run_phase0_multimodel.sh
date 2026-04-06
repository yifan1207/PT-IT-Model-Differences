#!/usr/bin/env bash
# Phase 0: Multi-model steering — precompute directions + A1 alpha-sweep + evaluation.
#
# Extends the causal steering story from Gemma-only to all 6 model families.
# IT models use chat template; PT models use raw format B.
#
# Steps:
#   1. precompute  — Extract corrective directions for 5 non-Gemma models
#   2. validate    — Bootstrap stability + 3-point sanity steering check
#   3. steer       — Full A1 alpha-sweep on all 6 models (+logit lens)
#   4. judge       — Post-hoc LLM judge (G1, G2, S1, S2)
#   5. pca         — 1A: PCA of IT-PT direction (rank-1 justification)
#   6. id-steering — 1C: TwoNN ID profiles at alpha=1.0, 0.0, -1.0
#   7. commitment  — 1B: Commitment delay vs alpha (post-processing, CPU only)
#   8. plots       — Generate all multi-model figures
#
# Usage:
#   bash scripts/run_phase0_multimodel.sh --step precompute
#   bash scripts/run_phase0_multimodel.sh --step validate
#   bash scripts/run_phase0_multimodel.sh --step steer
#   bash scripts/run_phase0_multimodel.sh --step steer --model llama31_8b  # single model
#   bash scripts/run_phase0_multimodel.sh --step judge
#   bash scripts/run_phase0_multimodel.sh --step pca
#   bash scripts/run_phase0_multimodel.sh --step id-steering
#   bash scripts/run_phase0_multimodel.sh --step commitment
#   bash scripts/run_phase0_multimodel.sh --step plots
#
# GPU Layout (8x H100 80GB):
#   GPUs 0-1: Reserved for other users (check nvidia-smi)
#   GPUs 2-7: Available for Phase 0 (6 GPUs)
#
# Estimated wall time (GPUs 2-7):
#   precompute:  ~6h  (5 models, 2 workers each)
#   validate:    ~2h
#   steer:       ~17h (6 models, 2 workers each, +logit lens 20% overhead)
#   judge:       ~4h  (API-bound, overlaps with below)
#   pca (1A):    ~1.5h (6 models, IT+PT acts collection)
#   id-steer(1C):~0.5h (6 models, 3 alpha values, 200 prompts)
#   commit (1B): ~0.5h (CPU post-processing only)
#   Total:       ~28h wall clock

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

ALL_MODELS=(gemma3_4b llama31_8b olmo2_7b mistral_7b deepseek_v2_lite qwen3_4b)
NEW_MODELS=(llama31_8b olmo2_7b mistral_7b deepseek_v2_lite qwen3_4b)

DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
N_WORKERS=2            # workers per model during steering
LOG_DIR="logs/phase0"
JUDGE_MODEL="google/gemini-2.5-flash"

# Available GPUs (adjust if GPU 0/1 are free)
AVAIL_GPUS=(2 3 4 5 6 7)
N_GPUS=${#AVAIL_GPUS[@]}

# Model-specific configs derived from MODEL_REGISTRY
declare -A N_LAYERS=(
    [gemma3_4b]=34     [llama31_8b]=32   [olmo2_7b]=32
    [mistral_7b]=32    [deepseek_v2_lite]=27 [qwen3_4b]=36
)
declare -A PROP_BOUND=(
    [gemma3_4b]=20     [llama31_8b]=19   [olmo2_7b]=19
    [mistral_7b]=19    [deepseek_v2_lite]=16 [qwen3_4b]=22
)
declare -A MAX_GEN=(
    [gemma3_4b]=200    [llama31_8b]=200  [olmo2_7b]=200
    [mistral_7b]=200   [deepseek_v2_lite]=64 [qwen3_4b]=200
)

mkdir -p "$LOG_DIR"

# ── Parse args ────────────────────────────────────────────────────────────────

STEP=""
SINGLE_MODEL=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)   STEP="$2"; shift 2 ;;
        --model)  SINGLE_MODEL="$2"; shift 2 ;;
        *)        echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$STEP" ]]; then
    echo "Usage: $0 --step {precompute|validate|steer|judge|plots} [--model MODEL]"
    exit 1
fi

# ── Helper: assign GPU to a job ───────────────────────────────────────────────

gpu_idx=0
next_gpu() {
    local g=${AVAIL_GPUS[$gpu_idx]}
    gpu_idx=$(( (gpu_idx + 1) % N_GPUS ))
    echo "$g"
}


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: PRECOMPUTE corrective directions
# ══════════════════════════════════════════════════════════════════════════════

precompute_model() {
    local model=$1

    # Check if already done
    local dir_path="results/cross_model/${model}/directions/corrective_directions.npz"
    if [[ -f "$dir_path" ]]; then
        echo "[precompute] $model: directions already exist, skipping."
        return 0
    fi

    echo "=== Precompute directions: $model ==="

    # Phase 1: gen (2 workers)
    local pids=()
    for ((wi=0; wi<N_WORKERS; wi++)); do
        local gpu=$(next_gpu)
        echo "  [gen] worker $wi on cuda:$gpu"
        uv run python scripts/precompute_directions_multimodel.py \
            --model-name "$model" --phase gen \
            --worker-index "$wi" --n-workers "$N_WORKERS" \
            --device "cuda:${gpu}" \
            > "$LOG_DIR/${model}_precompute_gen_w${wi}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[precompute] ERROR: $model gen worker failed. Check $LOG_DIR/${model}_precompute_gen_w*.log"
            return 1
        fi
    done

    # Phase 2: score (CPU)
    echo "  [score] computing contrast scores..."
    uv run python scripts/precompute_directions_multimodel.py \
        --model-name "$model" --phase score \
        > "$LOG_DIR/${model}_precompute_score.log" 2>&1

    # Phase 3: acts (2 workers)
    pids=()
    for ((wi=0; wi<N_WORKERS; wi++)); do
        local gpu=$(next_gpu)
        echo "  [acts] worker $wi on cuda:$gpu"
        uv run python scripts/precompute_directions_multimodel.py \
            --model-name "$model" --phase acts \
            --worker-index "$wi" --n-workers "$N_WORKERS" \
            --device "cuda:${gpu}" \
            > "$LOG_DIR/${model}_precompute_acts_w${wi}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[precompute] ERROR: $model acts worker failed. Check $LOG_DIR/${model}_precompute_acts_w*.log"
            return 1
        fi
    done

    # Phase 4: merge (CPU)
    echo "  [merge] computing directions..."
    uv run python scripts/precompute_directions_multimodel.py \
        --model-name "$model" --phase merge \
        > "$LOG_DIR/${model}_precompute_merge.log" 2>&1

    echo "[precompute] $model DONE -> results/cross_model/${model}/directions/"
}

if [[ "$STEP" == "precompute" ]]; then
    # Copy existing Gemma directions
    GEMMA_SRC="results/exp5/precompute_v2/precompute/corrective_directions.npz"
    GEMMA_DST="results/cross_model/gemma3_4b/directions"
    if [[ -f "$GEMMA_SRC" ]] && [[ ! -f "$GEMMA_DST/corrective_directions.npz" ]]; then
        mkdir -p "$GEMMA_DST"
        cp "$GEMMA_SRC" "$GEMMA_DST/"
        cp "${GEMMA_SRC%.npz}.meta.json" "$GEMMA_DST/" 2>/dev/null || true
        echo "[precompute] Copied existing Gemma directions to $GEMMA_DST/"
    fi

    models_to_run=("${NEW_MODELS[@]}")
    if [[ -n "$SINGLE_MODEL" ]]; then
        models_to_run=("$SINGLE_MODEL")
    fi

    # Run models sequentially (each model uses 2 GPUs internally via workers)
    # Two models can run in parallel if we have 4+ GPUs available
    if [[ ${#models_to_run[@]} -gt 1 ]] && [[ $N_GPUS -ge 4 ]]; then
        # Batch models in pairs for parallel precompute
        for ((i=0; i<${#models_to_run[@]}; i+=2)); do
            pids=()
            for ((j=i; j<i+2 && j<${#models_to_run[@]}; j++)); do
                precompute_model "${models_to_run[$j]}" &
                pids+=($!)
            done
            for pid in "${pids[@]}"; do
                wait "$pid" || echo "WARNING: a precompute job failed"
            done
        done
    else
        for model in "${models_to_run[@]}"; do
            precompute_model "$model"
        done
    fi

    echo "=== Precompute complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: VALIDATE (bootstrap stability + 3-point sanity check)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$STEP" == "validate" ]]; then
    models_to_run=("${ALL_MODELS[@]}")
    if [[ -n "$SINGLE_MODEL" ]]; then
        models_to_run=("$SINGLE_MODEL")
    fi

    for model in "${models_to_run[@]}"; do
        dir_path="results/cross_model/${model}/directions/corrective_directions.npz"
        if [[ ! -f "$dir_path" ]]; then
            echo "[validate] SKIP $model: no directions found at $dir_path"
            continue
        fi

        echo "=== Validate: $model ==="

        # 3-point sanity check: alpha=1.0 (baseline), 0.0 (full removal), -1.0 (amplify)
        # Quick run: 100 prompts, IT with chat template
        local_gpu=$(next_gpu)
        echo "  [sanity] 3-point check on cuda:$local_gpu (100 prompts)..."
        uv run python -m src.poc.exp6.run \
            --experiment A1 \
            --model-name "$model" \
            --variant it \
            --dataset "$DATASET" \
            --n-eval-examples 100 \
            --device "cuda:${local_gpu}" \
            --run-name "validate_${model}_3pt" \
            --corrective-direction-path "$dir_path" \
            --output-base "results/cross_model/${model}/validation" \
            > "$LOG_DIR/${model}_validate_3pt.log" 2>&1

        echo "  [sanity] $model done. Check: $LOG_DIR/${model}_validate_3pt.log"
    done

    echo "=== Validation complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: STEER (full A1 alpha-sweep, all 6 models, IT with chat template)
# ══════════════════════════════════════════════════════════════════════════════

steer_model() {
    local model=$1
    local gpu_start=$2
    local dir_path="results/cross_model/${model}/directions/corrective_directions.npz"
    local out_base="results/cross_model/${model}/exp6"
    local run_name="A1_${model}_it_v1"
    local mg=${MAX_GEN[$model]}

    echo "=== A1 alpha-sweep: $model (GPUs ${gpu_start}-$((gpu_start+N_WORKERS-1))) ==="

    local pids=()
    for ((wi=0; wi<N_WORKERS; wi++)); do
        local gpu=$((gpu_start + wi))
        uv run python -m src.poc.exp6.run \
            --experiment A1 \
            --model-name "$model" \
            --variant it \
            --collect-logit-lens \
            --dataset "$DATASET" \
            --n-eval-examples "$N_EVAL" \
            --device "cuda:${gpu}" \
            --worker-index "$wi" \
            --n-workers "$N_WORKERS" \
            --run-name "$run_name" \
            --corrective-direction-path "$dir_path" \
            --output-base "$out_base" \
            --max-gen-tokens "$mg" \
            > "$LOG_DIR/${model}_A1_w${wi}.log" 2>&1 &
        pids+=($!)
    done

    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "[steer] ERROR: $model worker failed"
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "[steer] $model: some workers failed. Check $LOG_DIR/${model}_A1_w*.log"
        return 1
    fi

    # Merge workers
    local src_dirs=()
    for ((wi=0; wi<N_WORKERS; wi++)); do
        src_dirs+=("${out_base}/${run_name}_w${wi}")
    done

    echo "  [merge] $model..."
    uv run python scripts/merge_exp6_workers.py \
        --experiment A1 --variant it --n-workers "$N_WORKERS" \
        --merged-name "merged_${run_name}" \
        --output-base "$out_base" \
        --source-dirs "${src_dirs[@]}" \
        > "$LOG_DIR/${model}_merge.log" 2>&1

    echo "[steer] $model DONE -> ${out_base}/merged_${run_name}/"
}

if [[ "$STEP" == "steer" ]]; then
    models_to_run=("${ALL_MODELS[@]}")
    if [[ -n "$SINGLE_MODEL" ]]; then
        models_to_run=("$SINGLE_MODEL")
    fi

    # Verify directions exist
    for model in "${models_to_run[@]}"; do
        dir_path="results/cross_model/${model}/directions/corrective_directions.npz"
        if [[ ! -f "$dir_path" ]]; then
            echo "ERROR: No directions for $model at $dir_path. Run --step precompute first."
            exit 1
        fi
    done

    # Run models in batches. Each model needs 2 GPUs. With 6 GPUs, run 3 at a time.
    batch_size=$(( N_GPUS / N_WORKERS ))
    for ((i=0; i<${#models_to_run[@]}; i+=batch_size)); do
        pids=()
        for ((j=0; j<batch_size && i+j<${#models_to_run[@]}; j++)); do
            local_model="${models_to_run[$((i+j))]}"
            local_gpu_start=${AVAIL_GPUS[$((j*N_WORKERS))]}
            steer_model "$local_model" "$local_gpu_start" &
            pids+=($!)
        done
        echo "[steer] batch $((i/batch_size + 1)): waiting for ${#pids[@]} models..."
        for pid in "${pids[@]}"; do
            wait "$pid" || echo "WARNING: a steer job failed"
        done
    done

    echo "=== Steering complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 4: JUDGE (post-hoc LLM judge)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$STEP" == "judge" ]]; then
    models_to_run=("${ALL_MODELS[@]}")
    if [[ -n "$SINGLE_MODEL" ]]; then
        models_to_run=("$SINGLE_MODEL")
    fi

    for model in "${models_to_run[@]}"; do
        merged="results/cross_model/${model}/exp6/merged_A1_${model}_it_v1"
        if [[ ! -d "$merged" ]]; then
            echo "[judge] SKIP $model: merged dir not found at $merged"
            continue
        fi
        echo "=== LLM Judge: $model ==="
        uv run python scripts/llm_judge_exp6.py \
            --merged-dir "$merged" \
            --model "$JUDGE_MODEL" --workers 16 --tasks g1 g2 s1 s2 \
            > "$LOG_DIR/${model}_judge.log" 2>&1 &
    done
    wait
    echo "=== Judge complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 5: PCA of corrective direction (1A)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$STEP" == "pca" ]]; then
    models_to_run=("${ALL_MODELS[@]}")
    if [[ -n "$SINGLE_MODEL" ]]; then
        models_to_run=("$SINGLE_MODEL")
    fi

    echo "=== 1A: PCA of IT-PT corrective direction ==="
    # Each model needs 1 GPU. Run 6 at a time.
    gpu_idx=0
    pids=()
    for model in "${models_to_run[@]}"; do
        work_dir="results/cross_model/${model}/directions_work"
        if [[ ! -f "$work_dir/selected.json" ]]; then
            echo "[pca] SKIP $model: no precompute work dir"
            continue
        fi
        gpu=${AVAIL_GPUS[$gpu_idx]}
        gpu_idx=$(( (gpu_idx + 1) % N_GPUS ))
        echo "  [pca] $model on cuda:$gpu"
        uv run python -m src.poc.exp8.pca_rank1 \
            --model-name "$model" --device "cuda:${gpu}" \
            > "$LOG_DIR/${model}_pca.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "WARNING: a PCA job failed"
    done
    echo "=== PCA complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 6: ID under steering (1C)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$STEP" == "id-steering" ]]; then
    models_to_run=("${ALL_MODELS[@]}")
    if [[ -n "$SINGLE_MODEL" ]]; then
        models_to_run=("$SINGLE_MODEL")
    fi

    echo "=== 1C: ID under steering (TwoNN at alpha=1.0, 0.0, -1.0) ==="
    gpu_idx=0
    pids=()
    for model in "${models_to_run[@]}"; do
        dir_path="results/cross_model/${model}/directions/corrective_directions.npz"
        if [[ ! -f "$dir_path" ]]; then
            echo "[id-steering] SKIP $model: no directions"
            continue
        fi
        gpu=${AVAIL_GPUS[$gpu_idx]}
        gpu_idx=$(( (gpu_idx + 1) % N_GPUS ))
        echo "  [id-steering] $model on cuda:$gpu"
        uv run python -m src.poc.exp8.id_under_steering \
            --model-name "$model" --device "cuda:${gpu}" \
            > "$LOG_DIR/${model}_id_steering.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || echo "WARNING: an ID-steering job failed"
    done
    echo "=== ID under steering complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 7: Commitment delay under steering (1B post-processing)
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$STEP" == "commitment" ]]; then
    echo "=== 1B: Commitment delay under steering (post-processing) ==="
    uv run python -m src.poc.exp8.commitment_vs_alpha
    echo "=== Commitment analysis complete ==="
    exit 0
fi


# ══════════════════════════════════════════════════════════════════════════════
# Step 8: PLOTS
# ══════════════════════════════════════════════════════════════════════════════

if [[ "$STEP" == "plots" ]]; then
    echo "=== Generating Phase 0 multi-model plots ==="
    uv run python scripts/plot_phase0_multimodel_dose_response.py
    echo "=== Plots complete ==="
    exit 0
fi

echo "ERROR: Unknown step '$STEP'"
echo "Valid steps: precompute, validate, steer, judge, pca, id-steering, commitment, plots"
exit 1
