#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export GCS_PROJECT="${GCS_PROJECT:-studious-hydra-450206-a5}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
MODE="${MODE:-smoke}"
RUN_NAME="${RUN_NAME:-exp45_behavioral_bridge_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp45_behavioral_bridge/${RUN_NAME}}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
MODELS="${MODELS:-}"
PRIMARY_MODELS="${PRIMARY_MODELS:-}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
PART="${PART:-both}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
GCS_DOWNLOAD_WORKERS="${GCS_DOWNLOAD_WORKERS:-12}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp45_behavioral_bridge}"
N_BOOT="${N_BOOT:-1000}"
TOP_K="${TOP_K:-20}"
JUDGE="${JUDGE:-0}"
JUDGE_PARALLELISM="${JUDGE_PARALLELISM:-16}"
JUDGE_MODEL="${JUDGE_MODEL:-google/gemini-2.5-flash}"

case "$MODE" in
  smoke)
    MODELS="${MODELS:-qwen3_4b}"
    PRIMARY_MODELS="${PRIMARY_MODELS:-$MODELS}"
    N_PROMPTS="${N_PROMPTS:-3}"
    MAX_EVENTS="${MAX_EVENTS:-3}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-3}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    N_BOOT="${N_BOOT:-100}"
    ;;
  pilot)
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b}"
    PRIMARY_MODELS="${PRIMARY_MODELS:-$MODELS}"
    N_PROMPTS="${N_PROMPTS:-150}"
    MAX_EVENTS="${MAX_EVENTS:-150}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-150}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    ;;
  full)
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b olmo2_7b}"
    PRIMARY_MODELS="${PRIMARY_MODELS:-$MODELS}"
    N_PROMPTS="${N_PROMPTS:-600}"
    MAX_EVENTS="${MAX_EVENTS:-0}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-600}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    ;;
  one-step-core5)
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b olmo2_7b qwen25_32b}"
    PRIMARY_MODELS="${PRIMARY_MODELS:-$MODELS}"
    PART="${PART:-one_step}"
    N_PROMPTS="${N_PROMPTS:-600}"
    MAX_EVENTS="${MAX_EVENTS:-0}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-0}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    ;;
  analyze-only|merge-only|pull-only)
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b olmo2_7b}"
    PRIMARY_MODELS="${PRIMARY_MODELS:-$MODELS}"
    N_PROMPTS="${N_PROMPTS:-600}"
    MAX_EVENTS="${MAX_EVENTS:-0}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-600}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
    ;;
  *)
    echo "[exp45-runpod] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp45-runpod] mode=${MODE}"
echo "[exp45-runpod] run_root=${RUN_ROOT}"
echo "[exp45-runpod] models=${MODELS}"
echo "[exp45-runpod] primary_models=${PRIMARY_MODELS}"
echo "[exp45-runpod] gpu_list=${GPU_LIST}"
echo "[exp45-runpod] workers_per_model=${WORKERS_PER_MODEL} n_prompts=${N_PROMPTS} max_events=${MAX_EVENTS}"
echo "[exp45-runpod] part=${PART} rollout_events=${ROLLOUT_EVENTS} max_new_tokens=${MAX_NEW_TOKENS}"

download_gcs_prefix() {
  local uri="$1"
  local dest="$2"
  local include_regex="${3:-}"
  if ! $PY_RUNNER - <<'PY'
try:
    import google.cloud.storage  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    uv pip install google-cloud-storage
  fi
  local -a include_args=()
  if [[ -n "$include_regex" ]]; then
    include_args=(--include-regex "$include_regex")
  fi
  $PY_RUNNER scripts/infra/download_gcs_prefix.py \
    --uri "$uri" \
    --dest "$dest" \
    --max-workers "$GCS_DOWNLOAD_WORKERS" \
    "${include_args[@]}"
}

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUNNER scripts/infra/gcs_sync_adc.py upload \
      "$RUN_ROOT" \
      "${GCS_SYNC_DEST%/}/${RUN_NAME}" \
      --workers 24
  else
    echo "[exp45-runpod] no GCS sync helper; skipping ${GCS_SYNC_DEST}" >&2
  fi
}

pull_inputs() {
  local exp20_uri="gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
  local exp20_root="results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
  read -r -a model_arr <<< "$MODELS"
  for model in "${model_arr[@]}"; do
    echo "[exp45-runpod] pull Exp20 manifest ${model}"
    download_gcs_prefix "${exp20_uri}/${PROMPT_MODE}/${model}" "${exp20_root}/${PROMPT_MODE}/${model}" '^exp20_validation_records\.jsonl$'
  done
}

run_collect() {
  mkdir -p "${RUN_ROOT}/logs"
  read -r -a model_arr <<< "$MODELS"
  read -r -a gpu_arr <<< "$GPU_LIST"
  local gpu_count="${#gpu_arr[@]}"
  local job_idx=0
  local pids=()
  for model in "${model_arr[@]}"; do
    for ((worker=0; worker<WORKERS_PER_MODEL; worker++)); do
      local gpu="${gpu_arr[$((job_idx % gpu_count))]}"
      local log_path="${RUN_ROOT}/logs/${model}_w${worker}of${WORKERS_PER_MODEL}.log"
      echo "[exp45-runpod] launch model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu} log=${log_path}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        $PY_RUNNER -m src.poc.exp45_behavioral_bridge collect \
          --model "$model" \
          --out-dir "${RUN_ROOT}/raw/${model}" \
          --prompt-mode "$PROMPT_MODE" \
          --n-prompts "$N_PROMPTS" \
          --max-events "$MAX_EVENTS" \
          --rollout-events "$ROLLOUT_EVENTS" \
          --max-new-tokens "$MAX_NEW_TOKENS" \
          --top-k "$TOP_K" \
          --event-kinds ${EVENT_KINDS} \
          --part "$PART" \
          --worker-index "$worker" \
          --n-workers "$WORKERS_PER_MODEL" \
          --device "cuda:0"
      ) >"$log_path" 2>&1 &
      pids+=("$!")
      job_idx=$((job_idx + 1))
    done
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[exp45-runpod] at least one collect worker failed" >&2
    sync_outputs || true
    return "$status"
  fi
}

merge_outputs() {
  read -r -a model_arr <<< "$MODELS"
  for model in "${model_arr[@]}"; do
    echo "[exp45-runpod] merge ${model}"
    $PY_RUNNER -m src.poc.exp45_behavioral_bridge collect \
      --model "$model" \
      --out-dir "${RUN_ROOT}/raw/${model}" \
      --n-workers "$WORKERS_PER_MODEL" \
      --merge-only
  done
}

analyze_outputs() {
  echo "[exp45-runpod] analyze"
  $PY_RUNNER -m src.poc.exp45_behavioral_bridge analyze \
    --run-root "$RUN_ROOT" \
    --models ${MODELS} \
    --primary-models ${PRIMARY_MODELS} \
    --n-boot "$N_BOOT"
}

score_judge_if_requested() {
  if [[ "$JUDGE" != "1" ]]; then
    return
  fi
  echo "[exp45-runpod] judge"
  $PY_RUNNER scripts/scoring/score_exp45_llm_judge.py \
    --requests "${RUN_ROOT}/analysis/llm_judge_requests.jsonl" \
    --out "${RUN_ROOT}/analysis/llm_judge_results.jsonl" \
    --judge-model "$JUDGE_MODEL" \
    --parallelism "$JUDGE_PARALLELISM"
}

if [[ "$MODE" != "analyze-only" && "$MODE" != "merge-only" ]]; then
  pull_inputs
fi
if [[ "$MODE" == "pull-only" ]]; then
  exit 0
fi
if [[ "$MODE" != "analyze-only" && "$MODE" != "merge-only" ]]; then
  run_collect
fi
if [[ "$MODE" != "analyze-only" ]]; then
  merge_outputs
fi
analyze_outputs
score_judge_if_requested || true
sync_outputs || true
echo "[exp45-runpod] complete ${RUN_ROOT}"

