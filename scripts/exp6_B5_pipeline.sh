#!/bin/bash
# Launch B5 after A1_mid finishes, then merge and plot.
# B5: full γ sweep × 4 layer ranges using method12_top100 (IT model, 41 conditions)

cd /home/yifan/structral-semantic-features
export PYTHONPATH=/home/yifan/structral-semantic-features

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_experiment() {
    local exp="$1"
    log "Waiting for $exp workers to finish..."
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

# ── Wait for A1_mid to finish (continuation pipeline handles A1_early→A1_mid) ──
wait_for_experiment "A1_mid"
log "A1_mid complete. Launching B5 on all 8 GPUs..."

# ── Launch B5: IT model, feature_clamp, method12_top100, 4 layer ranges × 10 γ ─
# Workers 6 and 7 were pre-launched on free GPUs — only launch 0-5 here.
mkdir -p logs
for i in 0 1 2 3 4 5; do
    nohup env PYTHONPATH=$PYTHONPATH \
        uv run python src/poc/exp6/run.py \
        --experiment B5 --variant it \
        --device "cuda:$i" \
        --governance-features-path results/exp6/governance_feature_sets.json \
        --mean-feature-acts-path results/exp6/precompute/mean_feature_acts_it \
        --worker-index "$i" --n-workers 8 \
        --run-name "B5_it" \
        >> "logs/exp6_B5_w${i}.log" 2>&1 &
    log "B5 worker $i on cuda:$i (PID $!)"
done

wait_for_experiment "B5"
merge_and_score B5 it 8

log "Generating updated plots..."
uv run python scripts/plot_exp6_B.py 2>&1 | tail -10 || true

log "=== B5 COMPLETE ==="
