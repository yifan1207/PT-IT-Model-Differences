#!/usr/bin/env bash
# Exp6 pipeline: waits for exp5 completion_pipeline.sh, then runs exp6 Approach A.
# Also launches B0 feature classification immediately (no GPU needed).
#
# Usage:
#   nohup bash scripts/exp6_pipeline.sh > logs/exp6_pipeline.log 2>&1 &
#
# Monitor:
#   tail -f logs/exp6_pipeline.log

set -euo pipefail
PYTHONPATH=/home/yifan/structral-semantic-features
export PYTHONPATH
cd /home/yifan/structral-semantic-features

LOG() { echo "[exp6 $(date '+%H:%M:%S')] $*" | tee -a logs/exp6_pipeline.log; }
EXP5_PIPELINE_PID=564179   # PID of completion_pipeline.sh

# ─── Step 0: Build exp6 dataset (fast, can run now) ──────────────────────────
LOG "Building exp6 dataset..."
PYTHONPATH=$PYTHONPATH uv run python scripts/build_exp6_dataset.py \
    2>&1 | tee -a logs/exp6_pipeline.log
LOG "exp6 dataset ready."

# ─── Step 0b: Launch B0 feature classification in background ─────────────────
# OPENROUTER_API_KEY is loaded from .env by classify_governance_features.py itself.
LOG "Launching B0 feature classification (background, no GPU needed)..."
mkdir -p logs
nohup bash -c "PYTHONPATH=$PYTHONPATH \
    uv run python scripts/classify_governance_features.py \
    >> logs/exp6_B0_classify.log 2>&1" &
B0_CLASSIFY_PID=$!
LOG "B0 classification running (PID $B0_CLASSIFY_PID)"

# ─── Step 0c: B0 structural enrichment and diff scores (no GPU) ──────────────
LOG "Computing structural enrichment and IT-PT diff scores..."
PYTHONPATH=$PYTHONPATH uv run python scripts/compute_structural_enrichment.py \
    2>&1 | tee -a logs/exp6_pipeline.log || LOG "WARNING: structural enrichment failed (non-fatal)"
PYTHONPATH=$PYTHONPATH uv run python scripts/compute_feature_diff_scores.py \
    2>&1 | tee -a logs/exp6_pipeline.log || LOG "WARNING: feature diff scores failed (non-fatal)"
LOG "B0 enrichment and diff scores done."

# ─── Step 1: Wait for exp5 pipeline to complete ──────────────────────────────
LOG "Waiting for exp5 completion_pipeline (PID $EXP5_PIPELINE_PID)..."
while kill -0 "$EXP5_PIPELINE_PID" 2>/dev/null; do
    LOG "  exp5 still running, sleeping 120s..."
    sleep 120
done
LOG "exp5 pipeline complete. All 8 GPUs now available."

# ─── Step 2: Precompute content-layer IT-PT directions (layers 0-11) ─────────
LOG "Precomputing content-layer corrective directions (layers 0-11) on GPU 7..."
PYTHONPATH=$PYTHONPATH uv run python scripts/precompute_extra_directions.py \
    --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
    --out-npz results/exp6/precompute/content_directions.npz \
    --device cuda:7 \
    2>&1 | tee -a logs/exp6_pipeline.log

PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/precompute.py \
    --task aggregate-content-direction \
    --content-directions-npz results/exp6/precompute/content_directions.npz \
    --out-path results/exp6/precompute/content_direction_aggregate.npz \
    2>&1 | tee -a logs/exp6_pipeline.log
LOG "Content direction precompute done."

# ─── Step 3: Launch A1 (IT model, directional_remove, α sweep) ───────────────
LOG "Launching A1 (IT, α sweep) on all 8 GPUs..."
mkdir -p logs
A1_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment A1 \
        --variant it \
        --device "cuda:$i" \
        --corrective-direction-path results/exp5/precompute_it/precompute/corrective_directions.npz \
        --content-direction-path results/exp6/precompute/content_direction_aggregate.npz \
        --worker-index "$i" \
        --n-workers 8 \
        --run-name "A1_it" \
        >> "logs/exp6_A1_w${i}.log" 2>&1 &
    A1_PIDS+=($!)
done
LOG "A1 workers launched: ${A1_PIDS[*]}"

# Wait for A1 to complete
LOG "Waiting for A1 workers..."
for pid in "${A1_PIDS[@]}"; do
    wait "$pid" || LOG "  WARNING: A1 worker PID $pid exited non-zero"
done
LOG "A1 complete."

# ─── Step 4: Merge A1 results ────────────────────────────────────────────────
LOG "Merging A1 results..."
PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment A1 --variant it --n-workers 8 \
    2>&1 | tee -a logs/exp6_pipeline.log
LOG "A1 merge done."

# ─── Step 5: Launch A2 (PT model, directional_add, β sweep + controls) ───────
LOG "Launching A2 (PT, β sweep) on all 8 GPUs..."
A2_PIDS=()
for i in 0 1 2 3 4 5 6 7; do
    PYTHONPATH=$PYTHONPATH uv run python src/poc/exp6/run.py \
        --experiment A2 \
        --variant pt \
        --device "cuda:$i" \
        --corrective-direction-path results/exp5/precompute_it/precompute/corrective_directions.npz \
        --content-direction-path results/exp6/precompute/content_direction_aggregate.npz \
        --worker-index "$i" \
        --n-workers 8 \
        --run-name "A2_pt" \
        >> "logs/exp6_A2_w${i}.log" 2>&1 &
    A2_PIDS+=($!)
done
LOG "A2 workers launched: ${A2_PIDS[*]}"

for pid in "${A2_PIDS[@]}"; do
    wait "$pid" || LOG "  WARNING: A2 worker PID $pid exited non-zero"
done
LOG "A2 complete."

# ─── Step 6: Merge A2 results ────────────────────────────────────────────────
LOG "Merging A2 results..."
PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp6_workers.py \
    --experiment A2 --variant pt --n-workers 8 \
    2>&1 | tee -a logs/exp6_pipeline.log
LOG "A2 merge done."

LOG "=== EXP6 APPROACH A COMPLETE ==="
LOG "A1 results: results/exp6/merged_A1_it/"
LOG "A2 results: results/exp6/merged_A2_pt/"
LOG ""
LOG "Next steps (run manually after reviewing A results):"
LOG "  1. Wait for B0 classification to finish: tail -f logs/exp6_B0_classify.log"
LOG "  2. Merge feature sets: uv run python scripts/merge_governance_feature_sets.py"
LOG "  3. Precompute mean feature acts: uv run python src/poc/exp6/precompute.py --task mean-feature-acts"
LOG "  4. Compute governance directions: uv run python src/poc/exp6/precompute.py --task governance-direction"
LOG "  5. Launch B1: bash scripts/run_exp6_B.sh"
