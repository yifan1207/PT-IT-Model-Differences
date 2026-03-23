#!/bin/bash
# Launch single_layer_phase experiment across 7 GPUs (one condition per GPU).
# 7 conditions: content/format/corrective × mean/skip, plus corrective_single_neg.
# Baseline is reused from existing phase_it run.
set -e
VARIANT=${1:-it}
N_WORKERS=7
RUN_NAME="single_layer_phase_${VARIANT}"
mkdir -p logs

for i in $(seq 0 $((N_WORKERS - 1))); do
    PYTHONPATH=. nohup uv run python src/poc/exp5/run.py \
        --experiment single_layer_phase \
        --variant "$VARIANT" \
        --device "cuda:${i}" \
        --n-eval-examples 500 \
        --mean-acts-path results/exp5/precompute_it/precompute/mean_mlp_outputs.npz \
        --corrective-direction-path results/exp5/precompute_it/precompute/corrective_directions.npz \
        --run-name "$RUN_NAME" \
        --worker-index "$i" \
        --n-workers "$N_WORKERS" \
        > "logs/exp5_single_layer_w${i}.log" 2>&1 &
    echo "Worker ${i}: PID $!"
done

echo "All $N_WORKERS workers launched. Logs: logs/exp5_single_layer_w*.log"
echo "Monitor: for i in \$(seq 0 6); do echo -n \"w\$i: \"; tail -1 logs/exp5_single_layer_w\${i}.log; done"
