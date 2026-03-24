#!/bin/bash
# Post-B1/B3/B4 completion pipeline:
# 1. Wait for B4 to finish, then launch B2_it
# 2. Wait for B1 to finish, then merge + post-hoc score
# 3. Wait for B2+B3 to finish, then merge + post-hoc score
# 4. Wait for B4 to finish, merge + post-hoc score

set -e
cd /home/yifan/structral-semantic-features
export PYTHONPATH=/home/yifan/structral-semantic-features

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helper: wait for PIDs ─────────────────────────────────────────────────────
wait_for_pids() {
    local label="$1"; shift
    log "Waiting for $label..."
    for pid in "$@"; do
        while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    done
    log "$label done."
}

# ── Phase 1: Wait for B4, then launch B2_it ──────────────────────────────────
B4_PIDS=($(ps aux | grep "exp6/run.py.*B4" | grep python3 | grep -v grep | awk '{print $2}'))
if [ ${#B4_PIDS[@]} -gt 0 ]; then
    wait_for_pids "B4 workers" "${B4_PIDS[@]}"
fi

log "B4 finished. Launching B2_it workers on GPUs 1,3,5,7..."
for i in 0 1 2 3; do
    gpu=$((i * 2 + 1))
    nohup env PYTHONPATH=$PYTHONPATH \
        uv run python src/poc/exp6/run.py \
        --experiment B2 --variant it \
        --device "cuda:$gpu" \
        --governance-direction-path results/exp6/precompute/governance_directions.npz \
        --governance-features-path results/exp6/governance_feature_sets.json \
        --mean-feature-acts-path results/exp6/precompute/mean_feature_acts_it \
        --worker-index "$i" --n-workers 4 \
        --run-name "B2_it" \
        >> "logs/exp6_B2_it_w${i}.log" 2>&1 &
    log "Launched B2_it worker $i on cuda:$gpu (PID $!)"
done

# ── Phase 2: Merge B4 ─────────────────────────────────────────────────────────
log "Merging B4..."
uv run python scripts/merge_exp6_workers.py --experiment B4 --variant it --n-workers 4
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_B4_it
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_B4_it
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_B4_it

# ── Phase 3: Wait for B1, merge, post-hoc score ───────────────────────────────
B1_PIDS=($(ps aux | grep "exp6/run.py.*B1" | grep python3 | grep -v grep | awk '{print $2}'))
if [ ${#B1_PIDS[@]} -gt 0 ]; then
    wait_for_pids "B1 workers" "${B1_PIDS[@]}"
fi

log "Merging B1..."
uv run python scripts/merge_exp6_workers.py --experiment B1 --variant it --n-workers 8
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_B1_it
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_B1_it
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_B1_it

# ── Phase 4: Wait for B3, merge, post-hoc score ───────────────────────────────
B3_PIDS=($(ps aux | grep "exp6/run.py.*B3" | grep python3 | grep -v grep | awk '{print $2}'))
if [ ${#B3_PIDS[@]} -gt 0 ]; then
    wait_for_pids "B3 workers" "${B3_PIDS[@]}"
fi

log "Merging B3..."
uv run python scripts/merge_exp6_workers.py --experiment B3 --variant it --n-workers 4
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_B3_it
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_B3_it
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_B3_it

# ── Phase 5: Wait for B2 (both pt and it), merge, post-hoc score ──────────────
B2_PIDS=($(ps aux | grep "exp6/run.py.*B2" | grep python3 | grep -v grep | awk '{print $2}'))
if [ ${#B2_PIDS[@]} -gt 0 ]; then
    wait_for_pids "B2 workers" "${B2_PIDS[@]}"
fi

log "Merging B2..."
uv run python scripts/merge_exp6_workers.py --experiment B2 --variant pt --n-workers 7
uv run python scripts/merge_exp6_workers.py --experiment B2 --variant it --n-workers 4
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_B2_pt
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_B2_pt
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_B2_pt
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_B2_it
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_B2_it
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_B2_it

log "=== EXP6 B-EXPERIMENTS COMPLETE ==="

# ── Phase 6: A1 layer-specificity ablations ───────────────────────────────────
# Precompute IT-PT directions for early (0-7) and mid (8-19) layers on GPU 0
log "Precomputing ablation directions for layers 0-19 (using GPU 0)..."
uv run python scripts/precompute_ablation_directions.py --device cuda:0 --n-records 500
log "Precompute done. Launching A1_early + A1_mid workers..."

# A1_early: IT model, layers 0-7, α sweep (14 conditions), 8 workers
for i in 0 1 2 3 4 5 6 7; do
    nohup env PYTHONPATH=$PYTHONPATH \
        uv run python src/poc/exp6/run.py \
        --experiment A1_early --variant it \
        --device "cuda:$i" \
        --corrective-direction-path results/exp6/precompute/ablation_directions_early.npz \
        --worker-index "$i" --n-workers 8 \
        --run-name "A1_early_it" \
        >> "logs/exp6_A1_early_w${i}.log" 2>&1 &
    log "Launched A1_early worker $i on cuda:$i (PID $!)"
done

# Wait for A1_early to finish (it's fast — only 14 conditions / 8 workers ≈ 2 per worker)
A1_EARLY_PIDS=($(ps aux | grep "exp6/run.py.*A1_early" | grep python3 | grep -v grep | awk '{print $2}'))
if [ ${#A1_EARLY_PIDS[@]} -gt 0 ]; then
    wait_for_pids "A1_early workers" "${A1_EARLY_PIDS[@]}"
fi

# A1_mid: IT model, layers 8-19, α sweep (14 conditions), 8 workers
for i in 0 1 2 3 4 5 6 7; do
    nohup env PYTHONPATH=$PYTHONPATH \
        uv run python src/poc/exp6/run.py \
        --experiment A1_mid --variant it \
        --device "cuda:$i" \
        --corrective-direction-path results/exp6/precompute/ablation_directions_mid.npz \
        --worker-index "$i" --n-workers 8 \
        --run-name "A1_mid_it" \
        >> "logs/exp6_A1_mid_w${i}.log" 2>&1 &
    log "Launched A1_mid worker $i on cuda:$i (PID $!)"
done

A1_MID_PIDS=($(ps aux | grep "exp6/run.py.*A1_mid" | grep python3 | grep -v grep | awk '{print $2}'))
if [ ${#A1_MID_PIDS[@]} -gt 0 ]; then
    wait_for_pids "A1_mid workers" "${A1_MID_PIDS[@]}"
fi

# Merge + post-hoc score A1 ablations
log "Merging A1_early and A1_mid..."
uv run python scripts/merge_exp6_workers.py --experiment A1_early --variant it --n-workers 8
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_A1_early_it
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_A1_early_it
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_A1_early_it

uv run python scripts/merge_exp6_workers.py --experiment A1_mid --variant it --n-workers 8
uv run python scripts/rescore_format_compliance.py --merged-dir results/exp6/merged_A1_mid_it
uv run python scripts/rescore_alignment_behavior.py --merged-dir results/exp6/merged_A1_mid_it
uv run python scripts/score_coherent_assistant_rate.py --merged-dir results/exp6/merged_A1_mid_it

log "=== ALL EXP6 EXPERIMENTS COMPLETE (including A1 ablations) ==="
log "Run: uv run python scripts/plot_exp6_B.py for B plots"
log "Run: uv run python scripts/plot_exp6_dose_response.py for ablation comparison plots"
