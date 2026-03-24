#!/usr/bin/env bash
# Waits for all running workers to finish, then runs merge → LLM judge → plots
# for single_layer_phase and progressive experiments.
#
# Run once and leave — everything will be done when you come back.
#
# Usage:
#   nohup bash scripts/completion_pipeline.sh > logs/completion_pipeline.log 2>&1 &

set -euo pipefail
PYTHONPATH=/home/yifan/structral-semantic-features
export PYTHONPATH
cd /home/yifan/structral-semantic-features

# Load API key from .env
if [ -f "$(dirname "$0")/../.env" ]; then
    set -a; source "$(dirname "$0")/../.env"; set +a
fi
OPENROUTER_KEY="${OPENROUTER_API_KEY:?'.env missing OPENROUTER_API_KEY'}"
LOG() { echo "[pipeline $(date '+%H:%M:%S')] $*" | tee -a logs/completion_pipeline.log; }

# ─── Helper: wait for a list of PIDs ─────────────────────────────────────────
wait_pids() {
    local label="$1"; shift
    local pids=("$@")
    LOG "Waiting for $label (PIDs: ${pids[*]}) ..."
    while true; do
        local alive=()
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        if [ ${#alive[@]} -eq 0 ]; then
            LOG "$label — all done."
            break
        fi
        LOG "$label — ${#alive[@]} still running: ${alive[*]}"
        sleep 120
    done
}

# ─── 1. Wait for all single_layer_phase workers ──────────────────────────────
# Original 7 workers (GPUs 0-6) + neg parent (GPU 7, sequential)
SLP_PIDS=(3992868 3992875 3992879 3992943 3993010 3993078 3993080 332728)
wait_pids "single_layer_phase workers" "${SLP_PIDS[@]}"

# ─── 2. Merge single_layer_phase results ──────────────────────────────────────
LOG "Merging single_layer_phase results ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp5_workers.py \
    --experiment single_layer_phase \
    --variant it \
    --n-workers 9 \
    --source-dirs \
        results/exp5/single_layer_phase_it_w0 \
        results/exp5/single_layer_phase_it_w1 \
        results/exp5/single_layer_phase_it_w2 \
        results/exp5/single_layer_phase_it_w3 \
        results/exp5/single_layer_phase_it_w4 \
        results/exp5/single_layer_phase_it_w5 \
        results/exp5/single_layer_phase_it_w6 \
        results/exp5/single_layer_phase_neg_w2 \
        results/exp5/single_layer_phase_neg_w5 \
    --baseline-scores results/exp5/phase_it_none_t200/scores.jsonl \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "single_layer_phase merge done."

# ─── 3. Concatenate all SLP sample_outputs for LLM judge ─────────────────────
SLP_MERGED_SAMPLES="results/exp5/merged_single_layer_phase_it/all_sample_outputs.jsonl"
LOG "Concatenating SLP sample outputs → $SLP_MERGED_SAMPLES ..."
for dir in \
    results/exp5/single_layer_phase_it_w0 \
    results/exp5/single_layer_phase_it_w1 \
    results/exp5/single_layer_phase_it_w2 \
    results/exp5/single_layer_phase_it_w3 \
    results/exp5/single_layer_phase_it_w4 \
    results/exp5/single_layer_phase_it_w5 \
    results/exp5/single_layer_phase_it_w6 \
    results/exp5/single_layer_phase_neg_w2 \
    results/exp5/single_layer_phase_neg_w5; do
    f="$dir/sample_outputs.jsonl"
    [ -f "$f" ] && cat "$f" >> "$SLP_MERGED_SAMPLES"
done
LOG "SLP sample outputs concatenated ($(wc -l < "$SLP_MERGED_SAMPLES") rows)."

# ─── 4. LLM judge on single_layer_phase ──────────────────────────────────────
SLP_JUDGE_OUT="results/exp5/merged_single_layer_phase_it/llm_judge_results.jsonl"
LOG "Running LLM judge on single_layer_phase ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/llm_judge_eval.py \
    --samples "$SLP_MERGED_SAMPLES" \
    --output  "$SLP_JUDGE_OUT" \
    --api-key "$OPENROUTER_KEY" \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "SLP LLM judge done."

# ─── 5. Plot LLM judge results for single_layer_phase ────────────────────────
LOG "Plotting SLP LLM judge results ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/plot_llm_judge.py \
    --judge-results "$SLP_JUDGE_OUT" \
    --scores        "results/exp5/merged_single_layer_phase_it/scores.jsonl" \
    --samples       "$SLP_MERGED_SAMPLES" \
    --output-dir    "results/exp5/merged_single_layer_phase_it/plots" \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "SLP LLM judge plots done."

# ─── 6. GCP upload for single_layer_phase dirs ───────────────────────────────
LOG "Uploading single_layer_phase raw files to GCS ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/gcp_upload_and_cleanup.py \
    --bucket gs://pt-vs-it-results/exp5 \
    --run-dirs \
        results/exp5/single_layer_phase_it_w0 \
        results/exp5/single_layer_phase_it_w1 \
        results/exp5/single_layer_phase_it_w2 \
        results/exp5/single_layer_phase_it_w3 \
        results/exp5/single_layer_phase_it_w4 \
        results/exp5/single_layer_phase_it_w5 \
        results/exp5/single_layer_phase_it_w6 \
        results/exp5/single_layer_phase_neg_w2 \
        results/exp5/single_layer_phase_neg_w5 \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "SLP GCS upload done."

# ─── 7. Wait for progressive workers ─────────────────────────────────────────
# w7 (skip_20_33) was killed — restart it on GPU 7 (now free after SLP neg workers done)
LOG "Restarting progressive worker 7 (skip_20_33) on GPU 7 ..."
PYTHONPATH=$PYTHONPATH uv run python src/poc/exp5/run.py \
    --experiment progressive \
    --variant it \
    --device cuda:7 \
    --n-eval-examples 500 \
    --mean-acts-path results/exp5/precompute_it/precompute/mean_mlp_outputs.npz \
    --corrective-direction-path results/exp5/precompute_it/precompute/corrective_directions.npz \
    --run-name progressive_it \
    --worker-index 7 \
    --n-workers 8 \
    >> logs/exp5_progressive_it_w7_restart.log 2>&1 &
PROG_W7_PID=$!
LOG "progressive w7 restarted (PID $PROG_W7_PID)"

PROG_PIDS=(3203907 3203908 3203909 3203973 3204037 3204101 3204146 $PROG_W7_PID)
wait_pids "progressive workers" "${PROG_PIDS[@]}"

# ─── 8. Merge progressive results ────────────────────────────────────────────
LOG "Merging progressive results ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/merge_exp5_workers.py \
    --experiment progressive \
    --variant it \
    --n-workers 8 \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "Progressive merge done."

# ─── 9. LLM judge on progressive ─────────────────────────────────────────────
PROG_MERGED_SAMPLES="results/exp5/merged_progressive_it/all_sample_outputs.jsonl"
LOG "Concatenating progressive sample outputs ..."
for w in 0 1 2 3 4 5 6 7; do
    f="results/exp5/progressive_it_w${w}/sample_outputs.jsonl"
    [ -f "$f" ] && cat "$f" >> "$PROG_MERGED_SAMPLES"
done
LOG "Progressive samples concatenated ($(wc -l < "$PROG_MERGED_SAMPLES") rows)."

PROG_JUDGE_OUT="results/exp5/merged_progressive_it/llm_judge_results.jsonl"
LOG "Running LLM judge on progressive ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/llm_judge_eval.py \
    --samples "$PROG_MERGED_SAMPLES" \
    --output  "$PROG_JUDGE_OUT" \
    --api-key "$OPENROUTER_KEY" \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "Progressive LLM judge done."

# ─── 10. Plot LLM judge results for progressive ───────────────────────────────
LOG "Plotting progressive LLM judge results ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/plot_llm_judge.py \
    --judge-results "$PROG_JUDGE_OUT" \
    --scores        "results/exp5/merged_progressive_it/scores.jsonl" \
    --samples       "$PROG_MERGED_SAMPLES" \
    --output-dir    "results/exp5/merged_progressive_it/plots" \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "Progressive LLM judge plots done."

# ─── 11. GCP upload for progressive dirs ─────────────────────────────────────
LOG "Uploading progressive raw files to GCS ..."
PYTHONPATH=$PYTHONPATH uv run python scripts/gcp_upload_and_cleanup.py \
    --bucket gs://pt-vs-it-results/exp5 \
    --run-dirs results/exp5/progressive_it_w* \
    2>&1 | tee -a logs/completion_pipeline.log
LOG "Progressive GCS upload done."

LOG "=== PIPELINE COMPLETE ==="
LOG "Results:"
LOG "  SLP merged:         results/exp5/merged_single_layer_phase_it/"
LOG "  SLP LLM judge:      $SLP_JUDGE_OUT"
LOG "  SLP plots:          results/exp5/merged_single_layer_phase_it/plots/"
LOG "  Progressive merged: results/exp5/merged_progressive_it/"
LOG "  Progressive plots:  results/exp5/merged_progressive_it/plots/"
