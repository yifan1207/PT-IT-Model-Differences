#!/usr/bin/env bash
# A1_notmpl + A5a_notmpl: No-chat-template ablation suite.
#
# Tests whether exp6 governance effects are weight-intrinsic or template-gated.
# Runs two experiments sequentially, both with apply_chat_template=False on IT model:
#
#   A1_notmpl:  Identical α-sweep as A1 (14 values + baseline, layers 20-33,
#               directional_remove) but without chat template.
#               Expected: similar dose-response to A1 → weight-intrinsic ✓
#               If flat → governance is template-gated (needs reframing).
#
#   A5a_notmpl: Progressive skip at corrective layers (20-33), no template.
#               Paired with A5a: tests whether layer specialization persists
#               without template. Format-before-coherence dissociation without
#               template → weight-intrinsic layer specialization ✓
#
# GPU layout: 8 workers on GPU 0-7 per experiment (sequential).
#
# Usage:
#   bash scripts/run_exp6_A1_notmpl.sh            # full run
#   bash scripts/run_exp6_A1_notmpl.sh --dry-run  # print commands only

set -euo pipefail
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ── Config ────────────────────────────────────────────────────────────────────
DATASET="data/eval_dataset_v2.jsonl"
N_EVAL=1400
CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
CONTENT_DIR="results/exp6/precompute/content_direction_aggregate.npz"
JUDGE_MODEL="google/gemini-2.5-flash"
NW=8

[[ "$DRY_RUN" == "1" ]] && echo "=== DRY RUN ===" \
    || echo "=== A1_notmpl + A5a_notmpl: no-template ablation suite ==="

# ── Helpers ───────────────────────────────────────────────────────────────────
run_worker() {
    local exp=$1 run_name=$2 widx=$3 gpu=$4
    local log_dir="logs/exp6_${run_name}"
    mkdir -p "$log_dir"
    local log="$log_dir/${run_name}_w${widx}.log"
    local cmd=(
        uv run python src/poc/exp6/run.py
        --experiment   "$exp"
        --variant      it
        --dataset      "$DATASET"
        --n-eval-examples "$N_EVAL"
        --device       "cuda:${gpu}"
        --worker-index "$widx"
        --n-workers    "$NW"
        --run-name     "$run_name"
        --corrective-direction-path "$CORR_DIR"
        --content-direction-path    "$CONTENT_DIR"
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

    local src_dirs=()
    for ((i=0; i<NW; i++)); do src_dirs+=("results/exp6/${run_name}_w${i}"); done

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry] merge → results/exp6/merged_${run_name}"
        echo "[dry] judge → results/exp6/merged_${run_name}/llm_judge_v2_scores.jsonl"
    else
        echo "[$(date +%T)] MERGE $run_name"
        uv run python scripts/merge_exp6_workers.py \
            --experiment "$exp" --variant it --n-workers "$NW" \
            --merged-name "merged_${run_name}" \
            --source-dirs "${src_dirs[@]}" \
            > "$log_dir/merge_${run_name}.log" 2>&1

        echo "[$(date +%T)] JUDGE $run_name"
        uv run python scripts/llm_judge_exp6.py \
            --merged-dir "results/exp6/merged_${run_name}" \
            --model "$JUDGE_MODEL" --workers 16 --tasks g1 g2 s1 s2 \
            > "$log_dir/judge_${run_name}.log" 2>&1

        echo "[$(date +%T)] DONE  $run_name"
    fi
}

# ── Step 1: A1_notmpl (α-sweep, directional_remove, no template) ──────────────
run_experiment A1_notmpl A1_notmpl_it_v1

# ── Step 2: A5a_notmpl (progressive skip, corrective layers, no template) ─────
run_experiment A5a_notmpl A5a_notmpl_it_v1

# ── Step 3: Generate plots ─────────────────────────────────────────────────────
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry] plot A1_notmpl_vs_pt → results/exp6/merged_A1_notmpl_it_v1/plots/"
    echo "[dry] plot A5a_notmpl_vs_template → results/exp6/plots_notmpl/"
    echo "[dry] plot A1 updated (with notmpl overlay) → results/exp6/merged_A1_it_v4/plots/"
else
    echo "[$(date +%T)] PLOT A1_notmpl_vs_pt"
    mkdir -p results/exp6/merged_A1_notmpl_it_v1/plots results/exp6/plots_notmpl
    uv run python scripts/plot_exp6_dose_response.py \
        --experiment A1_notmpl \
        --a1-notmpl-dir results/exp6/merged_A1_notmpl_it_v1 \
        --a2-dir results/exp6/merged_A2_pt_v4 \
        >> logs/exp6_A1_notmpl_it_v1/plot.log 2>&1

    echo "[$(date +%T)] PLOT A5a_notmpl_vs_template"
    uv run python scripts/plot_exp6_dose_response.py \
        --experiment A5a_notmpl \
        --a5a-dir        results/exp6/merged_A5a_it_v1 \
        --a5a-notmpl-dir results/exp6/merged_A5a_notmpl_it_v1 \
        >> logs/exp6_A1_notmpl_it_v1/plot.log 2>&1

    echo "[$(date +%T)] PLOT A1 with notmpl overlay"
    uv run python scripts/plot_exp6_dose_response.py \
        --experiment A1 \
        --a1-dir results/exp6/merged_A1_it_v4 \
        --a2-dir results/exp6/merged_A2_pt_v4 \
        --a1-notmpl-dir results/exp6/merged_A1_notmpl_it_v1 \
        >> logs/exp6_A1_notmpl_it_v1/plot.log 2>&1

    echo "[$(date +%T)] PLOTS DONE"
fi

echo "=== A1_notmpl + A5a_notmpl complete ==="
echo "  results/exp6/merged_A1_notmpl_it_v1/"
echo "  results/exp6/merged_A5a_notmpl_it_v1/"
echo "  results/exp6/plots_notmpl/"
