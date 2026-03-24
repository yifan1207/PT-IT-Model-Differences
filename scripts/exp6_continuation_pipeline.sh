#!/bin/bash
# Final pipeline phase: wait for A1_early, then run A1_mid
# B1/B2/B3/B4 all done+merged. A1_early running.

cd /home/yifan/structral-semantic-features
export PYTHONPATH=/home/yifan/structral-semantic-features

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_experiment() {
    local exp="$1"
    log "Waiting for $exp workers..."
    while ps aux | grep -- "--experiment $exp" | grep python3 | grep -qv grep 2>/dev/null; do
        sleep 60
    done
    log "$exp done."
}

merge_and_score() {
    local exp="$1" variant="$2" nworkers="$3"
    log "Merging ${exp}_${variant}..."
    uv run python scripts/merge_exp6_workers.py --experiment "$exp" --variant "$variant" --n-workers "$nworkers" 2>&1 | tail -3
    uv run python scripts/rescore_format_compliance.py --merged-dir "results/exp6/merged_${exp}_${variant}" 2>&1 | tail -2 || true
    uv run python scripts/rescore_alignment_behavior.py --merged-dir "results/exp6/merged_${exp}_${variant}" 2>&1 | tail -2 || true
    uv run python scripts/score_coherent_assistant_rate.py --merged-dir "results/exp6/merged_${exp}_${variant}" 2>&1 | tail -2 || true
}

# ── Wait for A1_early ─────────────────────────────────────────────────────────
wait_for_experiment "A1_early"
merge_and_score A1_early it 8

# ── Launch A1_mid ─────────────────────────────────────────────────────────────
log "Launching A1_mid (layers 8-19) on all 8 GPUs..."
for i in 0 1 2 3 4 5 6 7; do
    nohup env PYTHONPATH=$PYTHONPATH \
        uv run python src/poc/exp6/run.py \
        --experiment A1_mid --variant it \
        --device "cuda:$i" \
        --corrective-direction-path results/exp6/precompute/ablation_directions_mid.npz \
        --worker-index "$i" --n-workers 8 \
        --run-name "A1_mid_it" \
        >> "logs/exp6_A1_mid_w${i}.log" 2>&1 &
    log "A1_mid worker $i on cuda:$i (PID $!)"
done

wait_for_experiment "A1_mid"
merge_and_score A1_mid it 8

log "=== ALL EXP6 EXPERIMENTS COMPLETE ==="
