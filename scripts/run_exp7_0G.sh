#!/usr/bin/env bash
# Exp7 0G: Tuned-Lens Replication of Commitment Delay
#
# Trains per-layer affine probes T_ℓ for both PT and IT models following
# Belrose et al. (2023) methodology:
#   - Hooks on residual stream (layer output), not MLP output
#   - 2000 prompts × 80 tokens ≈ 160k training activations
#   - 80/20 train/val split with validation loss checkpointing
#   - Cosine annealing LR with warmup
#   - Identity initialisation (warm start)
#
# Evaluation uses KL-to-final commitment metric (KL < 0.1 nats):
#   1. IT probes on IT (matched)
#   2. PT probes on PT (matched)
#   3. PT probes on IT (cross-model transfer — tests universal geometry)
#   4. Raw logit-lens baseline (no probes)
#
# Expected: IT commits later than PT under tuned-lens, within ~2 layers of raw.
# Expected: PT probes work reasonably on IT (cross-model transfer gap < 2 layers).
#
# Usage:
#   bash scripts/run_exp7_0G.sh
#   bash scripts/run_exp7_0G.sh --n-train 40   # quick test

set -euo pipefail

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    EXTRA_ARGS+=("$1")
    shift
done

TRAIN_DATA="results/precompute_v2_work/gen_merged.jsonl"
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "[0G] ERROR: gen_merged.jsonl not found at $TRAIN_DATA"
    exit 1
fi

mkdir -p logs/exp7 results/exp7/0G/probes/pt results/exp7/0G/probes/it

echo "=== Exp7 0G: Tuned-lens probe training ==="
echo "  PT probes on cuda:0, IT probes on cuda:1"
echo "  Default: 2000 prompts, 5000 steps/layer, cosine LR w/ warmup"

# Train PT probes on cuda:0
uv run python -m src.poc.exp7.tuned_lens_probes \
    --variant pt \
    --device cuda:0 \
    --output-dir results/exp7/0G/probes/pt/ \
    "${EXTRA_ARGS[@]}" \
    > logs/exp7/0G_pt.log 2>&1 &
PT_PID=$!

# Train IT probes on cuda:1
uv run python -m src.poc.exp7.tuned_lens_probes \
    --variant it \
    --device cuda:1 \
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

echo ""
echo "=== [0G] Probes trained. Running evaluation (4 conditions)... ==="
echo "  1. IT probes on IT (matched)"
echo "  2. PT probes on PT (matched)"
echo "  3. PT probes on IT (cross-model transfer)"
echo "  4. Raw logit-lens (no probes)"

uv run python -m src.poc.exp7.tuned_lens_probes \
    --eval-only \
    --probe-dir results/exp7/0G/probes/ \
    --output-dir results/exp7/0G/ \
    --n-eval 400 \
    2>&1 | tee logs/exp7/0G_eval.log

echo ""
echo "=== [0G] Done. Results in results/exp7/0G/ ==="
echo "Check: results/exp7/0G/tuned_lens_commitment.json"
echo "  - IT KL commitment should exceed PT under tuned-lens"
echo "  - Cross-model transfer gap (PT probes on IT vs IT probes on IT) should be < 2 layers"
