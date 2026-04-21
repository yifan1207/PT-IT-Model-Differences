#!/usr/bin/env bash
# A5a_layerspec: Progressive-skip layer specificity control.
#
# Runs progressive skip at two new layer ranges to complement A5a (corrective):
#   A5a_early: skip from end of early layers  (1–11),  12 conditions
#   A5a_mid:   skip from end of mid layers   (12–19),   9 conditions
#
# Together with existing A5a (corrective, 20–33), these three sweeps answer:
#   "Does skipping ANY consecutive layers degrade governance,
#    or is it specific to the corrective stage (20–33)?"
#
# Expected: Early/mid skipping degrades content/reasoning but NOT governance.
# Only corrective skipping shows the format-before-coherence dissociation.
#
# GPU layout: 8 workers per experiment, run sequentially (early then mid).
# Plot: all three ranges overlaid on shared axes (n_layers_skipped x-axis).
#
# Usage:
#   bash scripts/run_exp6_A5a_layerspec.sh            # full run
#   bash scripts/run_exp6_A5a_layerspec.sh --dry-run  # print commands only

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ── Config ────────────────────────────────────────────────────────────────────
DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
JUDGE_MODEL="google/gemini-2.5-flash"
NW=8

[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" \
    || echo "=== A5a_layerspec: progressive-skip layer specificity ==="

# ── Helpers ───────────────────────────────────────────────────────────────────
run_worker() {
    local exp=$1 run_name=$2 widx=$3 gpu=$4
    local log_dir="logs/exp6_${run_name}"
    mkdir -p "$log_dir"
    local log="$log_dir/${run_name}_w${widx}.log"
    local cmd=(
        uv run python src/poc/exp06_corrective_direction_steering/run.py
        --experiment   "$exp"
        --variant      it
        --dataset      "$DATASET"
        --n-eval-examples "$N_EVAL"
        --device       "cuda:${gpu}"
        --worker-index "$widx"
        --n-workers    "$NW"
        --run-name     "$run_name"
    )
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] ${cmd[*]}"
    else
        echo "[$(date +%T)] START $run_name w${widx} gpu${gpu}"
        "${cmd[@]}" > "$log" 2>&1
        echo "[$(date +%T)] DONE  $run_name w${widx}"
    fi
}

run_experiment() {
    local exp=$1 run_name=$2
    local log_dir="logs/exp6_${run_name}"
    mkdir -p "$log_dir"

    echo "=== $run_name: $NW workers on GPU 0-$((NW-1)) ==="
    if [[ "$DRY_RUN" == "1" ]]; then
        for ((i=0; i<NW; i++)); do run_worker "$exp" "$run_name" "$i" "$i"; done
    else
        local pids=()
        for ((i=0; i<NW; i++)); do
            run_worker "$exp" "$run_name" "$i" "$i" &
            pids+=($!)
        done
        for i in "${!pids[@]}"; do
            wait "${pids[$i]}" || {
                echo "ERROR: $run_name w${i} failed (see $log_dir/${run_name}_w${i}.log)"
                exit 1
            }
        done
    fi

    # Merge
    local src_dirs=()
    for ((i=0; i<NW; i++)); do src_dirs+=("results/exp6/${run_name}_w${i}"); done

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] merge → results/exp6/merged_${run_name}"
        echo "[dry] judge → results/exp6/merged_${run_name}/llm_judge_v2_scores.jsonl"
    else
        echo "[$(date +%T)] MERGE $run_name"
        uv run python scripts/merge_steering_workers.py \
            --experiment "$exp" --variant it --n-workers "$NW" \
            --merged-name "merged_${run_name}" \
            --source-dirs "${src_dirs[@]}" \
            > "$log_dir/merge_${run_name}.log" 2>&1

        echo "[$(date +%T)] JUDGE $run_name"
        uv run python scripts/llm_judge.py \
            --merged-dir "results/exp6/merged_${run_name}" \
            --model "$JUDGE_MODEL" --workers 16 --tasks g1 g2 s1 s2 \
            > "$log_dir/judge_${run_name}.log" 2>&1

        echo "[$(date +%T)] DONE  $run_name"
    fi
}

# ── Step 1: A5a_early (layers 1–11, 12 conditions) ───────────────────────────
run_experiment A5a_early A5a_early_it_v1

# ── Step 2: A5a_mid (layers 12–19, 9 conditions) ─────────────────────────────
run_experiment A5a_mid A5a_mid_it_v1

# ── Step 3: Layerspec comparison plot ────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] plot → results/exp6/plots_layerspec/A5a_layer_specificity.png"
else
    echo "[$(date +%T)] PLOT A5a_layerspec"
    mkdir -p results/exp6/plots_layerspec
    uv run python scripts/plot_steering_dose_response.py \
        --experiment A5a_layerspec \
        --a5a-early-dir results/exp6/merged_A5a_early_it_v1 \
        --a5a-mid-dir   results/exp6/merged_A5a_mid_it_v1 \
        --a5a-dir       results/exp6/merged_A5a_it_v1 \
        >> logs/exp6_A5a_early_it_v1/plot_layerspec.log 2>&1
    echo "[$(date +%T)] PLOT DONE"
fi

echo "=== A5a_layerspec complete ==="
echo "  results/exp6/plots_layerspec/A5a_layer_specificity.png"
