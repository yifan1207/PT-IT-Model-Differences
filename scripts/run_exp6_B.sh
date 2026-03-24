#!/usr/bin/env bash
# Launch exp6 Approach B (feature steering) experiments.
# Run ONLY after:
#   1. A1/A2 complete and results are satisfactory
#   2. B0 classification complete (logs/exp6_B0_classify.log shows completion)
#   3. merge_governance_feature_sets.py has been run
#   4. precompute.py --task mean-feature-acts has been run
#   5. precompute.py --task governance-direction has been run
#
# Usage:
#   nohup bash scripts/run_exp6_B.sh > logs/exp6_B_pipeline.log 2>&1 &

set -euo pipefail
PYTHONPATH=/home/yifan/structral-semantic-features
export PYTHONPATH
cd /home/yifan/structral-semantic-features

LOG() { echo "[exp6B $(date '+%H:%M:%S')] $*" | tee -a logs/exp6_B_pipeline.log; }
mkdir -p logs results/exp6

CORRECTIVE_DIR="results/exp5/precompute_it/precompute/corrective_directions.npz"
GOVERNANCE_FEATURES="results/exp6/governance_feature_sets.json"
MEAN_FEATURE_ACTS="results/exp6/precompute/mean_feature_acts_it"
GOVERNANCE_DIRECTION="results/exp6/precompute/governance_directions.npz"

# ─── B1: Feature clamping (IT, γ sweep × feature set sweep) ──────────────────
LOG "Launching B1 (feature clamp, IT model) on all 8 GPUs..."
B1_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment B1 \
        --variant it \
        --device "cuda:$i" \
        --governance-features-path "$GOVERNANCE_FEATURES" \
        --mean-feature-acts-path "$MEAN_FEATURE_ACTS" \
        --worker-index "$i" \
        --n-workers 8 \
        --run-name "B1_it" \
        >> "logs/exp6_B1_w${i}.log" 2>&1 &
    B1_PIDS+=($!)
done
LOG "B1 workers: ${B1_PIDS[*]}"

for pid in "${B1_PIDS[@]}"; do
    wait "$pid" || LOG "  WARNING: B1 worker $pid exited non-zero"
done
LOG "B1 complete."

PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment B1 --variant it --n-workers 8 \
    2>&1 | tee -a logs/exp6_B_pipeline.log

# ─── B2: W_dec governance direction injection ─────────────────────────────────
# B2 has two sub-groups: B2a (inject into PT model) and B2b (subtract from IT model).
# run.py's variant filter skips conditions that don't match the loaded model variant,
# so we launch two separate sets of workers — one per variant.
LOG "Launching B2 (wdec inject, IT variant — B2b subtract conditions) on all 8 GPUs..."
B2_IT_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment B2 \
        --variant it \
        --device "cuda:$i" \
        --corrective-direction-path "$CORRECTIVE_DIR" \
        --governance-features-path "$GOVERNANCE_FEATURES" \
        --governance-direction-path "$GOVERNANCE_DIRECTION" \
        --worker-index "$i" \
        --n-workers 8 \
        --run-name "B2_it" \
        >> "logs/exp6_B2_it_w${i}.log" 2>&1 &
    B2_IT_PIDS+=($!)
done
LOG "B2 IT workers: ${B2_IT_PIDS[*]}"

LOG "Launching B2 (wdec inject, PT variant — B2a inject conditions) on all 8 GPUs..."
B2_PT_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment B2 \
        --variant pt \
        --device "cuda:$i" \
        --corrective-direction-path "$CORRECTIVE_DIR" \
        --governance-features-path "$GOVERNANCE_FEATURES" \
        --governance-direction-path "$GOVERNANCE_DIRECTION" \
        --worker-index "$i" \
        --n-workers 8 \
        --run-name "B2_pt" \
        >> "logs/exp6_B2_pt_w${i}.log" 2>&1 &
    B2_PT_PIDS+=($!)
done
LOG "B2 PT workers: ${B2_PT_PIDS[*]}"

for pid in "${B2_IT_PIDS[@]}" "${B2_PT_PIDS[@]}"; do
    wait "$pid" || LOG "  WARNING: B2 worker $pid exited non-zero"
done
LOG "B2 complete."

# Merge both B2 variants together
PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment B2 --variant it --n-workers 8 \
    2>&1 | tee -a logs/exp6_B_pipeline.log
PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment B2 --variant pt --n-workers 8 \
    2>&1 | tee -a logs/exp6_B_pipeline.log

# ─── B3: Control features ─────────────────────────────────────────────────────
LOG "Launching B3 (control features) on all 8 GPUs..."
B3_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment B3 \
        --variant it \
        --device "cuda:$i" \
        --governance-features-path "$GOVERNANCE_FEATURES" \
        --mean-feature-acts-path "$MEAN_FEATURE_ACTS" \
        --worker-index "$i" \
        --n-workers 8 \
        --run-name "B3_it" \
        >> "logs/exp6_B3_w${i}.log" 2>&1 &
    B3_PIDS+=($!)
done

for pid in "${B3_PIDS[@]}"; do
    wait "$pid" || LOG "  WARNING: B3 worker $pid exited non-zero"
done
LOG "B3 complete."

PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment B3 --variant it --n-workers 8 \
    2>&1 | tee -a logs/exp6_B_pipeline.log

# ─── B4: Layer specificity ────────────────────────────────────────────────────
LOG "Launching B4 (layer specificity) on 3 GPUs (one per layer range)..."
for gpu in 0 1 2; do
    ranges=("early_20_25" "mid_26_29" "late_30_33")
    lr="${ranges[$gpu]}"
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment B4 \
        --variant it \
        --device "cuda:$gpu" \
        --governance-features-path "$GOVERNANCE_FEATURES" \
        --mean-feature-acts-path "$MEAN_FEATURE_ACTS" \
        --feature-layer-range "$lr" \
        --run-name "B4_it_${lr}" \
        >> "logs/exp6_B4_${lr}.log" 2>&1 &
done
wait
LOG "B4 complete."

PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment B4 --variant it --n-workers 3 \
    2>&1 | tee -a logs/exp6_B_pipeline.log

LOG "=== EXP6 APPROACH B COMPLETE ==="
