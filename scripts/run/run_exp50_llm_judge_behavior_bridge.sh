#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PY_RUNNER="${PY_RUNNER:-uv run python}"
MODE="${MODE:-build}"  # build|smoke-api|score-sync|batch-prepare|batch-submit|batch-retrieve|analyze|full-sync
RUN_NAME="${RUN_NAME:-exp50_openai_judge_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp50_llm_judge_behavior_bridge/${RUN_NAME}}"
EXP45_ROOT="${EXP45_ROOT:-results/exp45_behavioral_bridge/exp45_full_a100x8_20260504_0652}"
EXP47_ROOT="${EXP47_ROOT:-}"
EXP47_GCS_URI="${EXP47_GCS_URI:-gs://pt-vs-it-results/results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
PULL_EXP47_GCS="${PULL_EXP47_GCS:-0}"
MODELS="${MODELS:-}"
MAX_DISPLAY_TOKENS="${MAX_DISPLAY_TOKENS:-64}"
MAX_EVENTS_PER_MODEL_CATEGORY="${MAX_EVENTS_PER_MODEL_CATEGORY:-0}"
CONTROL_FRACTION="${CONTROL_FRACTION:-0.10}"
OPENAI_JUDGE_MODEL="${OPENAI_JUDGE_MODEL:-gpt-5.2}"
OPENAI_SMOKE_MODEL="${OPENAI_SMOKE_MODEL:-gpt-5-mini}"
OPENAI_PARALLELISM="${OPENAI_PARALLELISM:-8}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_LIMIT="${SMOKE_LIMIT:-20}"
BATCH_ID="${BATCH_ID:-}"

mkdir -p "$RUN_ROOT"

pull_exp47_if_requested() {
  if [[ "$PULL_EXP47_GCS" != "1" ]]; then
    return
  fi
  EXP47_ROOT="${EXP47_ROOT:-${RUN_ROOT}/inputs/exp47_same_base_recipe_specificity}"
  if command -v gsutil >/dev/null 2>&1; then
    mkdir -p "${EXP47_ROOT}/behavior_bridge"
    gsutil -m rsync -r "${EXP47_GCS_URI%/}/behavior_bridge" "${EXP47_ROOT}/behavior_bridge"
  else
    $PY_RUNNER scripts/infra/download_gcs_prefix.py \
      --uri "${EXP47_GCS_URI%/}/behavior_bridge" \
      --dest "${EXP47_ROOT}/behavior_bridge" \
      --include-regex 'raw/.*/rollout_records\.jsonl\.gz$' \
      --max-workers 8
  fi
}

if [[ -n "$MODELS" ]]; then
  read -r -a model_arr <<< "$MODELS"
  model_args=(--models "${model_arr[@]}")
else
  model_args=()
fi

build_requests() {
  pull_exp47_if_requested
  rollout_args=(--rollout-root "$EXP45_ROOT")
  if [[ -n "$EXP47_ROOT" ]]; then
    rollout_args+=(--rollout-root "$EXP47_ROOT")
  fi
  $PY_RUNNER scripts/scoring/build_exp50_openai_judge_requests.py \
    "${rollout_args[@]}" \
    ${model_args[@]+"${model_args[@]}"} \
    --out-dir "$RUN_ROOT" \
    --max-display-tokens "$MAX_DISPLAY_TOKENS" \
    --max-events-per-model-category "$MAX_EVENTS_PER_MODEL_CATEGORY" \
    --control-fraction "$CONTROL_FRACTION"
}

analyze_outputs() {
  local responses="${RUN_ROOT}/judge_responses.jsonl.gz"
  if [[ ! -f "$responses" ]]; then
    responses="${RUN_ROOT}/judge_responses_smoke.jsonl.gz"
  fi
  $PY_RUNNER scripts/analysis/analyze_exp50_llm_judge_behavior_bridge.py \
    --responses "$responses" \
    --out-dir "$RUN_ROOT/analysis" \
    ${MODELS:+--models $MODELS} \
    --n-boot "$N_BOOT"
}

case "$MODE" in
  build)
    build_requests
    ;;
  smoke-api)
    build_requests
    $PY_RUNNER scripts/scoring/score_exp50_openai_judge.py \
      --requests "$RUN_ROOT/judge_requests.jsonl" \
      --out "$RUN_ROOT/judge_responses_smoke.jsonl.gz" \
      --model "$OPENAI_SMOKE_MODEL" \
      --parallelism 2 \
      --limit "$SMOKE_LIMIT"
    $PY_RUNNER scripts/analysis/analyze_exp50_llm_judge_behavior_bridge.py \
      --responses "$RUN_ROOT/judge_responses_smoke.jsonl.gz" \
      --out-dir "$RUN_ROOT/analysis_smoke" \
      ${MODELS:+--models $MODELS} \
      --n-boot 100
    ;;
  score-sync)
    $PY_RUNNER scripts/scoring/score_exp50_openai_judge.py \
      --requests "$RUN_ROOT/judge_requests.jsonl" \
      --out "$RUN_ROOT/judge_responses.jsonl.gz" \
      --model "$OPENAI_JUDGE_MODEL" \
      --parallelism "$OPENAI_PARALLELISM"
    ;;
  batch-prepare)
    $PY_RUNNER scripts/scoring/score_exp50_openai_judge.py \
      --mode batch-prepare \
      --requests "$RUN_ROOT/judge_requests.jsonl" \
      --out "$RUN_ROOT/judge_responses.jsonl.gz" \
      --batch-input "$RUN_ROOT/openai_batch_input.jsonl" \
      --model "$OPENAI_JUDGE_MODEL"
    ;;
  batch-submit)
    $PY_RUNNER scripts/scoring/score_exp50_openai_judge.py \
      --mode batch-submit \
      --requests "$RUN_ROOT/judge_requests.jsonl" \
      --out "$RUN_ROOT/judge_responses.jsonl.gz" \
      --batch-input "$RUN_ROOT/openai_batch_input.jsonl" \
      --batch-metadata "$RUN_ROOT/openai_batch_job.json"
    ;;
  batch-retrieve)
    $PY_RUNNER scripts/scoring/score_exp50_openai_judge.py \
      --mode batch-retrieve \
      --requests "$RUN_ROOT/judge_requests.jsonl" \
      --out "$RUN_ROOT/judge_responses.jsonl.gz" \
      --batch-id "$BATCH_ID" \
      --batch-metadata "$RUN_ROOT/openai_batch_status.json"
    ;;
  analyze)
    analyze_outputs
    ;;
  full-sync)
    build_requests
    $PY_RUNNER scripts/scoring/score_exp50_openai_judge.py \
      --requests "$RUN_ROOT/judge_requests.jsonl" \
      --out "$RUN_ROOT/judge_responses.jsonl.gz" \
      --model "$OPENAI_JUDGE_MODEL" \
      --parallelism "$OPENAI_PARALLELISM"
    analyze_outputs
    ;;
  *)
    echo "[exp50] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp50] complete ${RUN_ROOT}"
