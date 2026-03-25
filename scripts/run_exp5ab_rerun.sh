#!/usr/bin/env bash
# Rerun Exp5A (progressive skip) and Exp5B (α-sweep) with eval_dataset_v2 and v2
# precomputed directions.  Implemented as Exp6 A5a/A5b experiments.
#
# Exp5A (A5a) — Progressive skip:
#   Zero MLP+attention at layers [start..33], start ∈ {33,32,...,20}.
#   No direction needed.  IT model.  9 conditions (baseline + 8 skip depths).
#
# Exp5B (A5b) — α-sweep at layers 20-33:
#   α ∈ {-1,-0.5,0,0.5,1.0,1.5,2.0} using v2 corrective direction.
#   IT model.  8 conditions (baseline + 7 α values).
#
# GPU layout (8 × H100, all GPUs used):
#   GPU 0,1,2,3 → A5a workers 0,1,2,3  (350 records each, 1400 total)
#   GPU 4,5,6,7 → A5b workers 0,1,2,3  (350 records each, 1400 total)
# Both experiments run in parallel; 4 workers each halves per-condition time vs 2 workers.
#
# Scores are merged with weighted average across workers (merge_exp6_workers.py),
# then judged with G1/G2/S1/S2 LLM rubrics.
#
# Usage:
#   bash scripts/run_exp5ab_rerun.sh
#   bash scripts/run_exp5ab_rerun.sh --dry-run

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
JUDGE_MODEL="google/gemini-2.5-flash"
LOG_DIR="logs/exp5ab_rerun"
NW=4   # workers per experiment — uses all 8 GPUs (4 per experiment)

mkdir -p "$LOG_DIR"

# ── Worker launcher ────────────────────────────────────────────────────────────
run_worker() {
    local exp=$1 variant=$2 gpu=$3 widx=$4 nw=$5 run_name=$6
    local log="$LOG_DIR/${run_name}_w${widx}.log"
    local cmd=(
        uv run python src/poc/exp6/run.py
        --experiment      "$exp"
        --variant         "$variant"
        --dataset         "$DATASET"
        --n-eval-examples "$N_EVAL"
        --device          "cuda:${gpu}"
        --worker-index    "$widx"
        --n-workers       "$nw"
        --run-name        "$run_name"
        --corrective-direction-path "$CORR_DIR"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] ${cmd[*]}"
    else
        echo "[$(date +%T)] START $run_name w${widx} gpu${gpu}"
        "${cmd[@]}" > "$log" 2>&1
        echo "[$(date +%T)] DONE  $run_name w${widx}"
    fi
}

# ── Merge + judge ──────────────────────────────────────────────────────────────
merge_and_judge() {
    local run_name=$1 exp=$2 variant=$3 nw=$4
    local src_dirs=()
    for ((i=0; i<nw; i++)); do src_dirs+=("results/exp6/${run_name}_w${i}"); done
    local merge_log="$LOG_DIR/merge_${run_name}.log"
    local judge_log="$LOG_DIR/judge_${run_name}.log"

    echo "[$(date +%T)] MERGE $run_name (weighted avg across $nw workers)"
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
        echo "[dry] merge  → results/exp6/merged_${run_name}"
        echo "[dry] judge  → results/exp6/merged_${run_name}/llm_judge_v2_scores.jsonl"
    fi
}

# ── Experiment runner (N workers, configurable GPU range) ──────────────────────
run_experiment() {
    local exp=$1 variant=$2 gpu_start=$3 nw=$4 run_name=$5
    echo "=== $run_name: $nw workers on GPU ${gpu_start}-$((gpu_start+nw-1)) ==="

    local pids=()
    for ((i=0; i<nw; i++)); do
        local gpu=$((gpu_start + i))
        if [[ "$DRY_RUN" == "1" ]]; then
            run_worker "$exp" "$variant" "$gpu" "$i" "$nw" "$run_name"
        else
            run_worker "$exp" "$variant" "$gpu" "$i" "$nw" "$run_name" &
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

# ══════════════════════════════════════════════════════════════════════════════
[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" || echo "=== Starting Exp5A/5B rerun ==="

#              exp   var  gpu_start  nw   run_name
run_experiment  A5a  it   0          $NW  "A5a_it_v1"  &
run_experiment  A5b  it   4          $NW  "A5b_it_v1"  &

wait
echo "=== ALL DONE ==="
for n in A5a_it_v1 A5b_it_v1; do
    echo "  results/exp6/merged_${n}/"
done
