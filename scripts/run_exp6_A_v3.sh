#!/usr/bin/env bash
# Run exp6 A-experiments (A1, A1_early, A1_mid, A2) on eval_dataset_v2.jsonl.
#
# GPU layout (8 × H100):
#   GPU 0,1 → A1       IT  (corrective layers 20-33, 14 conditions)
#   GPU 2,3 → A1_early IT  (early layers 0-7,        14 conditions)
#   GPU 4,5 → A1_mid   IT  (mid layers 8-19,         14 conditions)
#   GPU 6,7 → A2       PT  (inject into PT, 20-33,   25 conditions)
#
# Each experiment runs 2 parallel workers. After both workers finish,
# the script merges and immediately runs the LLM judge (Gemini Flash).
#
# Usage:
#   bash scripts/run_exp6_A_v3.sh            # full run
#   bash scripts/run_exp6_A_v3.sh --dry-run  # print commands only

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ── Config ────────────────────────────────────────────────────────────────────
DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
CORR_DIR="results/exp5/precompute_it/precompute/corrective_directions.npz"
CONTENT_DIR="results/exp6/precompute/content_direction_aggregate.npz"
JUDGE_MODEL="google/gemini-2.5-flash-preview"
LOG_DIR="logs/exp6_A_v3"

mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_worker() {
    local exp=$1 variant=$2 gpu=$3 widx=$4 nw=$5 run_name=$6
    local log="$LOG_DIR/${run_name}_w${widx}.log"
    local cmd=(
        uv run python src/poc/exp6/run.py
        --experiment   "$exp"
        --variant      "$variant"
        --dataset      "$DATASET"
        --n-eval-examples "$N_EVAL"
        --device       "cuda:${gpu}"
        --worker-index "$widx"
        --n-workers    "$nw"
        --run-name     "$run_name"
        --corrective-direction-path "$CORR_DIR"
        --content-direction-path    "$CONTENT_DIR"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] ${cmd[*]}"
    else
        echo "[$(date +%T)] START $run_name w$widx gpu$gpu"
        "${cmd[@]}" > "$log" 2>&1
        echo "[$(date +%T)] DONE  $run_name w$widx"
    fi
}

merge_and_judge() {
    local run_name=$1 exp=$2 variant=$3 nw=$4
    local src_dirs=()
    for ((i=0; i<nw; i++)); do
        src_dirs+=("results/exp6/${run_name}_w${i}")
    done
    local merge_log="$LOG_DIR/merge_${run_name}.log"
    local judge_log="$LOG_DIR/judge_${run_name}.log"

    echo "[$(date +%T)] MERGE $run_name"
    if [[ "$DRY_RUN" == "0" ]]; then
        uv run python scripts/merge_exp6_workers.py \
            --experiment "$exp" \
            --variant    "$variant" \
            --n-workers  "$nw" \
            --merged-name "merged_${run_name}" \
            --source-dirs "${src_dirs[@]}" \
            > "$merge_log" 2>&1
        echo "[$(date +%T)] JUDGE $run_name (Gemini Flash)"
        uv run python scripts/llm_judge_exp6.py \
            --merged-dir "results/exp6/merged_${run_name}" \
            --model      "$JUDGE_MODEL" \
            --workers    16 \
            --tasks      g1 g2 s1 s2 \
            > "$judge_log" 2>&1
        echo "[$(date +%T)] JUDGE DONE $run_name"
    else
        echo "[dry] merge → results/exp6/merged_${run_name}"
        echo "[dry] judge → results/exp6/merged_${run_name}/llm_judge_v2_scores.jsonl"
    fi
}

# ── Experiment runner: 2 workers → wait → merge+judge ─────────────────────────
run_experiment() {
    local exp=$1 variant=$2 gpu0=$3 gpu1=$4 run_name=$5
    echo "=== $run_name: launching 2 workers on GPU $gpu0,$gpu1 ==="

    if [[ "$DRY_RUN" == "1" ]]; then
        run_worker "$exp" "$variant" "$gpu0" 0 2 "$run_name"
        run_worker "$exp" "$variant" "$gpu1" 1 2 "$run_name"
        merge_and_judge "$run_name" "$exp" "$variant" 2
        return
    fi

    # Launch both workers in background, capture PIDs
    run_worker "$exp" "$variant" "$gpu0" 0 2 "$run_name" &
    PID0=$!
    run_worker "$exp" "$variant" "$gpu1" 1 2 "$run_name" &
    PID1=$!

    # Wait for both to finish
    wait "$PID0" || { echo "ERROR: $run_name w0 failed (see $LOG_DIR/${run_name}_w0.log)"; exit 1; }
    wait "$PID1" || { echo "ERROR: $run_name w1 failed (see $LOG_DIR/${run_name}_w1.log)"; exit 1; }

    merge_and_judge "$run_name" "$exp" "$variant" 2
}

# ═════════════════════════════════════════════════════════════════════════════
# Main: launch all 4 experiments in parallel (each experiment uses 2 GPUs)
# ═════════════════════════════════════════════════════════════════════════════

[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN MODE ===" || echo "=== Starting exp6 A-experiments (v3, eval_dataset_v2) ==="

# All 4 experiments run concurrently across 8 GPUs.
# Each run_experiment call blocks until its 2 workers AND judge finish —
# but since they run in subshells (&), all 4 proceed in parallel.

run_experiment A1       it 0 1 "A1_it_v3"       &
run_experiment A1_early it 2 3 "A1_early_it_v3" &
run_experiment A1_mid   it 4 5 "A1_mid_it_v3"   &
run_experiment A2       pt 6 7 "A2_pt_v3"        &

# Wait for all 4 experiments (including their merge+judge) to complete
wait
echo "=== ALL DONE ==="
echo "Results:"
for n in A1_it_v3 A1_early_it_v3 A1_mid_it_v3 A2_pt_v3; do
    echo "  results/exp6/merged_${n}/"
done
