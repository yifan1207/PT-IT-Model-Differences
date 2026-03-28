#!/usr/bin/env bash
# Exp7 Unified Runner — Optimized for minimal GPU passes
#
# This script orchestrates all Tier 0 (0A–0J) experiments with a combined
# data collection strategy to minimize redundant model inference.
#
# ── Key insight ─────────────────────────────────────────────────────────────
# 0A (600 selected) and 0H (random-600 + bottom-600) both need per-record MLP
# activations from all 1400 records with both IT and PT models.
# Instead of two separate collection passes (~60 min each), one unified pass
# over all 1400 records covers BOTH — saving ~1 hour of GPU time.
#
# ── Execution order ──────────────────────────────────────────────────────────
#
# Phase 1 — Single unified data collection (~30 min, all 8 GPUs):
#   UNIFIED:  collect IT+PT MLP acts for all 1400 records
#             → feeds: 0A (selected-600 slice), 0H (random-600/bottom-600 slice)
#   PARALLEL: 0G probe training (2 GPUs) — can use any 2 GPUs
#   PARALLEL: 0J onset analysis (CPU) — no GPU needed
#
# Phase 2 — Direction computation and bootstrap (~10 min, CPU):
#   0A:  bootstrap stability from unified acts (selected-600 slice)
#   0H:  compute random + bottom directions from unified acts
#
# Phase 3 — Force-decode (depends on unified acts), ~30 min:
#   0B:  PT force-decode on IT token sequences
#
# Phase 4 — All A1 experiment variants (~3.5 hrs total, sequential on 8 GPUs):
#   0C:  A1_rand_matched (1 run × 15 conditions)
#   0F:  4 × A1 layer-range variants
#   0H-eval: A1 on held-out 800 (2 runs: random + bottom direction)
#   0I:  A1_formula (4 methods × 5 alpha)
#   0J-gpu: A1 with Gemma 0.5σ and 2σ onset ranges (2 runs)
#
# Phase 5 — CPU post-processing:
#   0D:  Bootstrap CIs on main A1 results
#   0E:  Token classifier robustness
#
# Phase 6 — Plots:
#   All Tier 0 figures
#
# Usage:
#   bash scripts/run_exp7_unified.sh              # full pipeline
#   bash scripts/run_exp7_unified.sh --phase 1    # only Phase 1
#   bash scripts/run_exp7_unified.sh --phase 4    # only Phase 4 (assumes earlier phases done)
#   bash scripts/run_exp7_unified.sh --quick      # 20 records, smoke test

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
PHASE=0   # 0 = all phases
QUICK=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)   shift; PHASE="$1"; shift ;;
        --quick)   QUICK=1; shift ;;
        *)         echo "Unknown arg: $1"; exit 1 ;;
    esac
done

N_RECORDS_ARG=""
N_EVAL_ARG=""
if [[ "$QUICK" -eq 1 ]]; then
    N_RECORDS_ARG="--n-records 20"
    N_EVAL_ARG="--n-eval-examples 40"
    echo "[unified] QUICK MODE: 20 records per worker, 40 eval examples"
fi

CORR_DIR="results/exp5/precompute_v2/precompute/corrective_directions.npz"
NW=8

mkdir -p logs/exp7

run_phase() { [[ "$PHASE" -eq 0 || "$PHASE" -eq "$1" ]]; }

# ── Phase 1: Unified data collection ─────────────────────────────────────────
if run_phase 1; then
    echo ""
    echo "████ Phase 1: Unified MLP activation collection (all 1400 records, 8 GPUs) ████"

    # Check if already done
    if [[ -f "results/exp7/unified_acts/merged.npz" ]]; then
        echo "[Phase 1] merged.npz already exists, skipping collection."
    else
        echo "[Phase 1a] Launching 8 workers for unified activation collection..."
        pids=()
        for i in $(seq 0 $((NW-1))); do
            uv run python -m src.poc.exp7.collect_unified_acts \
                --worker-index "$i" --n-workers "$NW" \
                --device "cuda:${i}" \
                $N_RECORDS_ARG \
                > logs/exp7/unified_w${i}.log 2>&1 &
            pids+=($!)
        done

        failed=0
        for pid in "${pids[@]}"; do
            if ! wait "$pid"; then echo "[Phase 1] Worker $pid failed"; failed=1; fi
        done
        [[ "$failed" -ne 0 ]] && { echo "Check logs/exp7/unified_w*.log"; exit 1; }

        echo "[Phase 1b] Merging workers..."
        uv run python -m src.poc.exp7.collect_unified_acts --merge-only --n-workers "$NW"
    fi

    # Verify unified acts match canonical precompute_v2 direction
    echo "[Phase 1b+] Verifying unified acts against canonical direction..."
    uv run python -m src.poc.exp7.collect_unified_acts --verify

    # 0G probe training can run in parallel on two GPUs while collection happens
    # (If GPUs are shared, run after collection completes)
    echo "[Phase 1c] Starting 0G tuned-lens probe training (cuda:0 and cuda:1)..."
    if [[ -f "results/precompute_v2_work/gen_merged.jsonl" ]]; then
        bash scripts/run_exp7_0G.sh $N_RECORDS_ARG &
        G_PID=$!
        echo "[Phase 1c] 0G running in background (pid=$G_PID)"
    else
        echo "[Phase 1c] WARNING: gen_merged.jsonl not found, skipping 0G"
        G_PID=0
    fi

    # 0J onset analysis runs on CPU immediately (no GPU needed)
    echo "[Phase 1d] Running 0J onset threshold analysis (CPU)..."
    bash scripts/run_exp7_0J.sh --skip-a1-reruns

    echo "[Phase 1] Waiting for 0G to complete..."
    [[ "$G_PID" -ne 0 ]] && wait "$G_PID" && echo "[Phase 1] 0G complete."
fi

# ── Phase 2: Direction computation from unified acts ─────────────────────────
if run_phase 2; then
    echo ""
    echo "████ Phase 2: Direction computation and bootstrap analysis (CPU) ████"

    MERGED="results/exp7/unified_acts/merged.npz"
    if [[ ! -f "$MERGED" ]]; then
        echo "[Phase 2] ERROR: merged.npz not found. Run Phase 1 first."
        exit 1
    fi

    echo "[Phase 2a] Running 0A bootstrap on selected-600 slice..."
    # bootstrap_directions.py reads from acts dir, but we need to create
    # a 0A-compatible acts dir from the unified merged file.
    uv run python -c "
import json, numpy as np
from pathlib import Path
from src.poc.exp7.collect_unified_acts import extract_subset_acts, OUTPUT_DIR

# Load selected-600 IDs
selected_path = Path('results/precompute_v2_work/selected.json')
if selected_path.exists():
    selected_ids = json.loads(selected_path.read_text())
else:
    print('[Phase 2a] selected.json not found — using all records')
    import sys; sys.exit(0)

acts = extract_subset_acts(selected_ids)

# Save in format expected by bootstrap_directions.py (acts/merged.npz)
out_dir = Path('results/exp7/0A/acts')
out_dir.mkdir(parents=True, exist_ok=True)
merged_out = out_dir / 'merged.npz'
np.savez_compressed(str(merged_out), **acts)
print(f'[Phase 2a] Selected-600 acts → {merged_out} ({len(selected_ids)} records)')
" 2>&1 | tee logs/exp7/0A_slice.log

    uv run python -m src.poc.exp7.bootstrap_directions \
        --acts-dir results/exp7/0A/acts/ \
        --canonical-npz "$CORR_DIR" \
        --n-bootstrap 50 --seed 42 \
        --output-dir results/exp7/0A/ \
        2>&1 | tee logs/exp7/0A_bootstrap.log

    echo "[Phase 2b] Computing 0H random and bottom directions..."
    uv run python -c "
import json, numpy as np
from pathlib import Path
from src.poc.exp7.collect_unified_acts import extract_subset_acts, OUTPUT_DIR, ALL_LAYERS, D_MODEL
from src.poc.exp7.precompute_random_split import compute_directions

# Extract random-600 and bottom-600 acts
h_dir = Path('results/exp7/0H')
random_ids = json.loads((h_dir / 'random_600_ids.json').read_text())
bottom_ids = json.loads((h_dir / 'bottom_600_ids.json').read_text())

for split_name, ids in [('random', random_ids), ('bottom', bottom_ids)]:
    acts = extract_subset_acts(ids)
    split_dir = h_dir / f'{split_name}_acts'
    split_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(split_dir / 'merged.npz'), **acts)
    print(f'[Phase 2b] {split_name}-{len(ids)} acts → {split_dir}/merged.npz')

# Compute directions
from src.poc.exp7.precompute_random_split import OUTPUT_DIR as SPLIT_DIR, compute_directions
compute_directions(h_dir, h_dir / 'random_acts')
" 2>&1 | tee logs/exp7/0H_directions.log
fi

# ── Phase 3: Force-decode for 0B ─────────────────────────────────────────────
if run_phase 3; then
    echo ""
    echo "████ Phase 3: PT force-decode on IT tokens (0B) ████"
    bash scripts/run_exp7_0B.sh $N_RECORDS_ARG
fi

# ── Phase 4: All A1 experiment variants (sequential GPU runs) ─────────────────
if run_phase 4; then
    echo ""
    echo "████ Phase 4: A1 experiment variants (sequential, ~3.5 hrs) ████"

    echo "[Phase 4a] 0C: Magnitude-matched random direction control..."
    bash scripts/run_exp7_0C.sh $N_EVAL_ARG

    echo "[Phase 4b] 0F: Layer-range sensitivity (4 A1 runs)..."
    bash scripts/run_exp7_0F.sh $N_EVAL_ARG

    echo "[Phase 4c] 0H: A1 on held-out 800 (random + bottom directions)..."
    RAND_NPZ="results/exp7/0H/random_directions.npz"
    BOTTOM_NPZ="results/exp7/0H/bottom_directions.npz"
    HELD_OUT_IDS="results/exp7/0H/held_out_800_ids.json"

    # Run with random direction (restricted to held-out 800 records)
    if [[ -f "$RAND_NPZ" ]]; then
        pids=()
        for i in $(seq 0 $((NW-1))); do
            uv run python -m src.poc.exp6.run \
                --experiment A1 --variant it \
                --worker-index "$i" --n-workers "$NW" --device "cuda:${i}" \
                --run-name "A1_it_random_dir_w${i}" --output-base "results/exp7/0H" \
                --corrective-direction-path "$RAND_NPZ" \
                --eval-record-ids "$HELD_OUT_IDS" \
                $N_EVAL_ARG > logs/exp7/0H_rand_w${i}.log 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
        src_dirs=(); for i in $(seq 0 $((NW-1))); do src_dirs+=("results/exp7/0H/A1_it_random_dir_w${i}"); done
        uv run python scripts/merge_exp6_workers.py \
            --experiment A1 --variant it --n-workers "$NW" \
            --merged-name "A1_it_random_dir" --output-base "results/exp7/0H" \
            --source-dirs "${src_dirs[@]}"
    fi

    # Run with bottom direction (restricted to held-out 800 records)
    if [[ -f "$BOTTOM_NPZ" ]]; then
        pids=()
        for i in $(seq 0 $((NW-1))); do
            uv run python -m src.poc.exp6.run \
                --experiment A1 --variant it \
                --worker-index "$i" --n-workers "$NW" --device "cuda:${i}" \
                --run-name "A1_it_bottom_dir_w${i}" --output-base "results/exp7/0H" \
                --corrective-direction-path "$BOTTOM_NPZ" \
                --eval-record-ids "$HELD_OUT_IDS" \
                $N_EVAL_ARG > logs/exp7/0H_bottom_w${i}.log 2>&1 &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
        src_dirs=(); for i in $(seq 0 $((NW-1))); do src_dirs+=("results/exp7/0H/A1_it_bottom_dir_w${i}"); done
        uv run python scripts/merge_exp6_workers.py \
            --experiment A1 --variant it --n-workers "$NW" \
            --merged-name "A1_it_bottom_dir" --output-base "results/exp7/0H" \
            --source-dirs "${src_dirs[@]}"
    fi

    echo "[Phase 4d] 0I: Intervention formula sensitivity (24 conditions)..."
    bash scripts/run_exp7_0I.sh $N_EVAL_ARG

    echo "[Phase 4e] 0J: Gemma A1 with alternative onset layer ranges..."
    # Only run if alt ranges were computed
    if [[ -f "results/exp7/0J/gemma_alt_ranges.json" ]]; then
        bash scripts/run_exp7_0J.sh $N_EVAL_ARG
    else
        echo "[Phase 4e] gemma_alt_ranges.json not found, running 0J analysis first..."
        bash scripts/run_exp7_0J.sh $N_EVAL_ARG
    fi
fi

# ── Phase 5: CPU post-processing ──────────────────────────────────────────────
if run_phase 5; then
    echo ""
    echo "████ Phase 5: CPU post-processing (0D, 0E) ████"
    bash scripts/run_exp7_0D.sh
    bash scripts/run_exp7_0E.sh
fi

# ── Phase 6: All Tier 0 figures ───────────────────────────────────────────────
if run_phase 6; then
    echo ""
    echo "████ Phase 6: Generating all Tier 0 figures ████"
    uv run python scripts/plot_exp7_tier0.py
fi

echo ""
echo "████ Exp7 Unified pipeline complete ████"
echo "Results in: results/exp7/{0A,0B,0C,0D,0E,0F,0G,0H,0I,0J}/"
echo "Plots in:   results/exp7/plots/"
