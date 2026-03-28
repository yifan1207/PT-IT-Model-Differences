#!/usr/bin/env bash
# Exp7 0G: Tuned-Lens Replication of Commitment Delay
# Trains per-layer affine probes T_ℓ for both PT and IT models, then recomputes
# the commitment delay using tuned-lens predictions (more calibrated than raw logit-lens).
#
# Steps:
#   1. Train probes for PT (cuda:0) and IT (cuda:1) in parallel (~1 hr each)
#   2. Eval: compare raw vs tuned-lens commitment delays
#
# Expected: IT still commits later than PT under tuned-lens, within ~2 layers of raw.
#
# Usage:
#   bash scripts/run_exp7_0G.sh
#   bash scripts/run_exp7_0G.sh --n-train 20   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

TRAIN_PROMPTS="results/precompute_v2_work/gen_merged.jsonl"
if [[ ! -f "$TRAIN_PROMPTS" ]]; then
    echo "[0G] ERROR: gen_merged.jsonl not found at $TRAIN_PROMPTS"
    exit 1
fi

mkdir -p logs/exp7 results/exp7/0G/probes/pt results/exp7/0G/probes/it

echo "=== Exp7 0G: Tuned-lens probe training (PT on cuda:0, IT on cuda:1) ==="

# Train PT probes on cuda:0
uv run python -m src.poc.exp7.tuned_lens_probes \
    --variant pt \
    --device cuda:0 \
    --train-prompts "$TRAIN_PROMPTS" \
    --output-dir results/exp7/0G/probes/pt/ \
    "${EXTRA_ARGS[@]}" \
    > logs/exp7/0G_pt.log 2>&1 &
PT_PID=$!

# Train IT probes on cuda:1
uv run python -m src.poc.exp7.tuned_lens_probes \
    --variant it \
    --device cuda:1 \
    --train-prompts "$TRAIN_PROMPTS" \
    --output-dir results/exp7/0G/probes/it/ \
    "${EXTRA_ARGS[@]}" \
    > logs/exp7/0G_it.log 2>&1 &
IT_PID=$!

echo "[0G] Training in parallel. Waiting for PT (pid=$PT_PID) and IT (pid=$IT_PID)..."
failed=0
wait "$PT_PID" || { echo "[0G] ERROR: PT probe training failed"; failed=1; }
wait "$IT_PID" || { echo "[0G] ERROR: IT probe training failed"; failed=1; }

if [[ "$failed" -ne 0 ]]; then
    echo "[0G] Check logs/exp7/0G_pt.log and logs/exp7/0G_it.log"
    exit 1
fi

echo "=== [0G] Probes trained. Running evaluation... ==="
uv run python -m src.poc.exp7.tuned_lens_probes \
    --eval-only \
    --probe-dir results/exp7/0G/probes/ \
    --train-prompts "$TRAIN_PROMPTS" \
    --output-dir results/exp7/0G/ \
    2>&1 | tee logs/exp7/0G_eval.log

echo "=== [0G] Done. Results in results/exp7/0G/ ==="
echo "Check: results/exp7/0G/tuned_lens_commitment.json"
echo "  IT mean commitment should exceed PT under tuned-lens (within ~2 layers of raw)"
