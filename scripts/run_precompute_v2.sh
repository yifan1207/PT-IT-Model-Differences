#!/usr/bin/env bash
# Precompute corrective direction vectors v2 — unified single-inference pipeline.
#
# Key design: inference runs ONCE.  All layer groups (1-11, 12-19, 20-33) are
# extracted from the SAME 600 high-contrast records and the SAME token pairs
# (a single forward pass hooks layers 1-33 simultaneously).
#
# Pipeline:
#   Phase 1 (gen)   — 8 GPU workers in parallel, one forward pass per record
#   Phase 2 (score) — CPU only; G1 LLM judge + contrast scoring, top-600 selection
#   Phase 3 (acts)  — 8 GPU workers in parallel, all layers 1-33 hooked at once
#   Phase 4 (merge) — CPU only; one npz with layers 1-33 (all groups share same data)
#   Then: launch run_exp6_A_v4.sh with the single direction file
#
# Usage:
#   nohup bash scripts/run_precompute_v2.sh > logs/precompute_v2.log 2>&1 &

set -euo pipefail
LOG_DIR="logs/precompute_v2"
mkdir -p "$LOG_DIR" results/precompute_v2_work

PY="uv run python scripts/precompute_directions_v2.py"

run_worker() {
    local phase=$1 wi=$2 nw=$3 gpu=$4
    local log="$LOG_DIR/${phase}_w${wi}.log"
    echo "[$(date +%T)] START $phase w${wi}/${nw} gpu${gpu}"
    $PY --phase "$phase" \
        --worker-index "$wi" --n-workers "$nw" \
        --device "cuda:${gpu}" > "$log" 2>&1
    echo "[$(date +%T)] DONE  $phase w${wi}"
}

# ═══════════════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] ═══ Phase 1: gen (8 workers, one forward pass per record) ═══"

run_worker gen 0 8 0 &
run_worker gen 1 8 1 &
run_worker gen 2 8 2 &
run_worker gen 3 8 3 &
run_worker gen 4 8 4 &
run_worker gen 5 8 5 &
run_worker gen 6 8 6 &
run_worker gen 7 8 7 &

wait
echo "[$(date +%T)] ═══ Phase 1 complete ═══"

# ═══════════════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] ═══ Phase 2: score (LLM judge + contrast, single selection) ═══"

$PY --phase score --judge-workers 16 > "$LOG_DIR/score.log" 2>&1

echo "[$(date +%T)] ═══ Phase 2 complete ═══"

# ═══════════════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] ═══ Phase 3: acts (8 workers, ALL layers 1-33 in one pass) ═══"

run_worker acts 0 8 0 &
run_worker acts 1 8 1 &
run_worker acts 2 8 2 &
run_worker acts 3 8 3 &
run_worker acts 4 8 4 &
run_worker acts 5 8 5 &
run_worker acts 6 8 6 &
run_worker acts 7 8 7 &

wait
echo "[$(date +%T)] ═══ Phase 3 complete ═══"

# ═══════════════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] ═══ Phase 4: merge (single npz, layers 1-33) ═══"

$PY --phase merge > "$LOG_DIR/merge.log" 2>&1

echo "[$(date +%T)] ═══ Phase 4 complete ═══"

echo ""
echo "Direction vectors ready (single file, all layer groups share same data):"
echo "  results/exp5/precompute_v2/precompute/corrective_directions.npz"
echo "  (contains layers 1-33: early=1-11, mid=12-19, corrective=20-33)"

# ═══════════════════════════════════════════════════════════════════════════════
echo "[$(date +%T)] ═══ Launching A-experiments v4 ═══"
bash scripts/run_exp6_A_v4.sh
