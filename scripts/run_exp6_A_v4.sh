#!/usr/bin/env bash
# Run exp6 A-experiments (A1, A1_early, A1_mid, A2) with v2 direction vectors.
#
# All experiments use the SAME direction file (layers 1-33, unified precompute).
# Each experiment applies the direction to a different layer subset:
#   A1        → layers 20-33  (corrective group)
#   A1_early  → layers  1-11  (early group, from same direction file)
#   A1_mid    → layers 12-19  (mid group, from same direction file)
#   A2        → content direction injected into PT at layers 20-33
#
# GPU layout (8 × H100):
#   GPU 0,1 → A1       IT  (corrective layers 20-33)
#   GPU 2,3 → A1_early IT  (early layers  1-11)
#   GPU 4,5 → A1_mid   IT  (mid layers 12-19)
#   GPU 6,7 → A2       PT  (content direction, layers 20-33)
#
# Usage:
#   bash scripts/run_exp6_A_v4.sh            # full run
#   bash scripts/run_exp6_A_v4.sh --dry-run  # print commands only

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ── Config ────────────────────────────────────────────────────────────────────
DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
# All layer groups share a single direction file (layers 1-33).
# A1/A1_early/A1_mid apply to different layer subsets via their ablation_layers config.
CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
EARLY_DIR="$CORR_DIR"
MID_DIR="$CORR_DIR"
CONTENT_DIR="results/exp6/precompute/content_direction_aggregate.npz"
JUDGE_MODEL="google/gemini-2.5-flash-preview"
LOG_DIR="logs/exp6_A_v4"

mkdir -p "$LOG_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
run_worker() {
    local exp=$1 variant=$2 gpu=$3 widx=$4 nw=$5 run_name=$6 corr_path=$7
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
        --corrective-direction-path "$corr_path"
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
    for ((i=0; i<nw; i++)); do src_dirs+=("results/exp6/${run_name}_w${i}"); done
    local merge_log="$LOG_DIR/merge_${run_name}.log"
    local judge_log="$LOG_DIR/judge_${run_name}.log"

    echo "[$(date +%T)] MERGE $run_name"
    if [[ "$DRY_RUN" == "0" ]]; then
        uv run python scripts/merge_exp6_workers.py \
            --experiment "$exp" --variant "$variant" --n-workers "$nw" \
            --merged-name "merged_${run_name}" \
            --source-dirs "${src_dirs[@]}" > "$merge_log" 2>&1
        echo "[$(date +%T)] JUDGE $run_name"
        uv run python scripts/llm_judge_exp6.py \
            --merged-dir "results/exp6/merged_${run_name}" \
            --model "$JUDGE_MODEL" --workers 16 --tasks g1 g2 s1 s2 \
            > "$judge_log" 2>&1
        echo "[$(date +%T)] JUDGE DONE $run_name"
    else
        echo "[dry] merge → results/exp6/merged_${run_name}"
        echo "[dry] judge → results/exp6/merged_${run_name}/llm_judge_v2_scores.jsonl"
    fi
}

run_experiment() {
    local exp=$1 variant=$2 gpu_start=$3 nw=$4 run_name=$5 corr_path=$6
    echo "=== $run_name: $nw workers on GPU ${gpu_start}-$((gpu_start+nw-1)) ==="

    local pids=()
    for ((i=0; i<nw; i++)); do
        local gpu=$((gpu_start + i))
        if [[ "$DRY_RUN" == "1" ]]; then
            run_worker "$exp" "$variant" "$gpu" "$i" "$nw" "$run_name" "$corr_path"
        else
            run_worker "$exp" "$variant" "$gpu" "$i" "$nw" "$run_name" "$corr_path" &
            pids+=($!)
        fi
    done

    if [[ "$DRY_RUN" == "0" ]]; then
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" || {
                echo "ERROR: $run_name w${i} failed (see $LOG_DIR/${run_name}_w${i}.log)"
                exit 1
            }
        done
    fi

    merge_and_judge "$run_name" "$exp" "$variant" "$nw"
}

# ═════════════════════════════════════════════════════════════════════════════
[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" || echo "=== Starting exp6 A-experiments v4 ==="

# Each experiment uses its own direction vector for the correct layer group.
#                     exp        var  gpu_start  nw  run_name            direction_path
run_experiment        A1         it   0          2   "A1_it_v4"          "$CORR_DIR"  &
run_experiment        A1_early   it   2          2   "A1_early_it_v4"    "$EARLY_DIR" &
run_experiment        A1_mid     it   4          2   "A1_mid_it_v4"      "$MID_DIR"   &
run_experiment        A2         pt   6          2   "A2_pt_v4"          "$CORR_DIR"  &

wait
echo "=== ALL DONE ==="
for n in A1_it_v4 A1_early_it_v4 A1_mid_it_v4 A2_pt_v4; do
    echo "  results/exp6/merged_${n}/"
done
