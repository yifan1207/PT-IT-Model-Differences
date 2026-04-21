#!/usr/bin/env bash
# A1_rand: Random-direction α-sweep control for A1 specificity claim.
#
# Runs the identical α-sweep as A1 (14 values + baseline, layers 20-33,
# directional_remove) but replaces the corrective IT-PT direction with a
# fixed random unit vector (seed=42, one per layer).
#
# Expected result: flat metrics across all α.
# If flat  → governance effect is direction-specific (supports mechanistic claim).
# If not flat → any perturbation at corrective layers causes the effect
#               (weakens specificity claim, needs reframing).
#
# GPU layout: 8 workers on GPU 0-7 (15 conditions split round-robin)
#
# Usage:
#   bash scripts/run_exp6_A1_rand.sh            # full run
#   bash scripts/run_exp6_A1_rand.sh --dry-run  # print commands only

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ── Config ────────────────────────────────────────────────────────────────────
DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
RAND_DIR="results/exp06_corrective_direction_steering/precompute/random_directions.npz"
CONTENT_DIR="results/exp06_corrective_direction_steering/precompute/content_direction_aggregate.npz"
JUDGE_MODEL="google/gemini-2.5-flash"
LOG_DIR="logs/exp6_A1_rand"
RUN_NAME="A1_rand_it_v1"
NW=8   # 15 conditions split round-robin across 8 GPUs

mkdir -p "$LOG_DIR"

[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" \
    || echo "=== A1_rand: random-direction specificity control ==="

# ── Step 1: generate random directions (CPU, <1s) ─────────────────────────────
echo "[$(date +%T)] Generating random directions (seed=42)..."
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] python scripts/generate_random_directions.py --out $RAND_DIR --seed 42"
else
    uv run python scripts/generate_random_directions.py \
        --out "$RAND_DIR" --seed 42 \
        > "$LOG_DIR/generate_rand.log" 2>&1
    echo "[$(date +%T)] Done → $RAND_DIR"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
run_worker() {
    local widx=$1 gpu=$2
    local log="$LOG_DIR/${RUN_NAME}_w${widx}.log"
    local cmd=(
        uv run python src/poc/exp06_corrective_direction_steering/run.py
        --experiment   A1_rand
        --variant      it
        --dataset      "$DATASET"
        --n-eval-examples "$N_EVAL"
        --device       "cuda:${gpu}"
        --worker-index "$widx"
        --n-workers    "$NW"
        --run-name     "$RUN_NAME"
        --corrective-direction-path "$RAND_DIR"
        --content-direction-path    "$CONTENT_DIR"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] ${cmd[*]}"
    else
        echo "[$(date +%T)] START $RUN_NAME w${widx} gpu${gpu}"
        "${cmd[@]}" > "$log" 2>&1
        echo "[$(date +%T)] DONE  $RUN_NAME w${widx}"
    fi
}

# ── Step 2: run workers ───────────────────────────────────────────────────────
echo "=== $RUN_NAME: $NW workers on GPU 0-$((NW-1)) ==="
if [[ "$DRY_RUN" == "1" ]]; then
    for ((i=0; i<NW; i++)); do run_worker "$i" "$i"; done
else
    pids=()
    for ((i=0; i<NW; i++)); do
        run_worker "$i" "$i" &
        pids+=($!)
    done
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}" || {
            echo "ERROR: $RUN_NAME w${i} failed (see $LOG_DIR/${RUN_NAME}_w${i}.log)"
            exit 1
        }
    done
fi

# ── Step 3: merge + judge ─────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] merge → results/exp06_corrective_direction_steering/merged_${RUN_NAME}"
    echo "[dry] judge → results/exp06_corrective_direction_steering/merged_${RUN_NAME}/llm_judge_v2_scores.jsonl"
else
    src_dirs=()
    for ((i=0; i<NW; i++)); do src_dirs+=("results/exp06_corrective_direction_steering/${RUN_NAME}_w${i}"); done

    echo "[$(date +%T)] MERGE $RUN_NAME"
    uv run python scripts/merge_steering_workers.py \
        --experiment A1_rand --variant it --n-workers "$NW" \
        --merged-name "merged_${RUN_NAME}" \
        --source-dirs "${src_dirs[@]}" \
        > "$LOG_DIR/merge_${RUN_NAME}.log" 2>&1

    echo "[$(date +%T)] JUDGE $RUN_NAME"
    uv run python scripts/llm_judge.py \
        --merged-dir "results/exp06_corrective_direction_steering/merged_${RUN_NAME}" \
        --model "$JUDGE_MODEL" --workers 16 --tasks g1 g2 s1 s2 \
        > "$LOG_DIR/judge_${RUN_NAME}.log" 2>&1

    echo "[$(date +%T)] JUDGE DONE $RUN_NAME"
fi

echo "=== A1_rand complete ==="
echo "  results/exp06_corrective_direction_steering/merged_${RUN_NAME}/"
