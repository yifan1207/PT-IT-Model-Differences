#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

RUN_ROOT="${RUN_ROOT:-$ROOT/results/exp15_symmetric_behavioral_causality/data}"
RUN_PREFIX="${RUN_PREFIX:-exp15_eval_core_600_t512_dense5}"
WORKERS="${WORKERS:-16}"
PROVIDER="${PROVIDER:-openrouter}"
BULK_MODEL="${BULK_MODEL:-google/gemini-2.5-flash}"
SECOND_MODEL="${SECOND_MODEL:-openai/gpt-4o-mini}"
ESCALATION_MODEL="${ESCALATION_MODEL:-openai/gpt-4o}"

MODELS=(
  gemma3_4b
  qwen3_4b
  llama31_8b
  mistral_7b
  olmo2_7b
)

for model in "${MODELS[@]}"; do
  run_dir="$RUN_ROOT/${RUN_PREFIX}_${model}/${RUN_PREFIX}_${model}"
  echo "[exp15 judge] running $model"
  uv run python "$ROOT/scripts/eval/judge_exp15.py" \
    --run-dir "$run_dir" \
    --bulk-model "$BULK_MODEL" \
    --second-model "$SECOND_MODEL" \
    --escalation-model "$ESCALATION_MODEL" \
    --provider "$PROVIDER" \
    --workers "$WORKERS" \
    "$@"
done
