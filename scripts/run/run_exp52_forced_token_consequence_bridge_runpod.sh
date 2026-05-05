#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${MODE:-smoke}"  # smoke|full|collect-only|analyze-only|sync
RUN_NAME="${RUN_NAME:-exp52_forced_token_consequence_bridge_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp52_forced_token_consequence_bridge/${RUN_NAME}}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP47_ROOT="${EXP47_ROOT:-results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
EXP47_GCS="${EXP47_GCS:-gs://pt-vs-it-results/results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
EXP20_ROOT="${EXP20_ROOT:-${EXP47_ROOT}/exp20}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-${EXP20_ROOT}}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp52_forced_token_consequence_bridge}"
MODELS="${MODELS:-llama31_meta_instruct llama31_tulu3_sft llama31_tulu3_dpo llama31_tulu3_final llama31_openmath2}"
CATEGORY_FILTER="${CATEGORY_FILTER:-GOV-FORMAT SAFETY CONTENT-REASON GOV-CONV}"
GPU_LIST="${GPU_LIST:-}"
EVENT_KIND="${EVENT_KIND:-first_diff}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RESUME="${RESUME:-1}"
N_BOOT="${N_BOOT:-1000}"
MAX_NEW_TOKENS_OVERRIDE="${MAX_NEW_TOKENS_OVERRIDE:-}"

case "$MODE" in
  smoke)
    MODELS="${SMOKE_MODELS:-llama31_tulu3_final}"
    N_PROMPTS="${N_PROMPTS:-80}"
    MAX_EVENTS="${MAX_EVENTS:-16}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
    MAX_NEW_TOKENS_OVERRIDE="${SMOKE_MAX_NEW_TOKENS_OVERRIDE:-24}"
    N_BOOT="${SMOKE_N_BOOT:-100}"
    ;;
  full|collect-only|analyze-only|sync)
    N_PROMPTS="${N_PROMPTS:-1400}"
    MAX_EVENTS="${MAX_EVENTS:-0}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
    ;;
  *)
    echo "[exp52] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

if [[ -n "${PY_RUNNER:-}" ]]; then
  read -r -a PYTHON <<< "$PY_RUNNER"
elif [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  PYTHON=(uv run --no-sync python)
else
  PYTHON=(uv run python)
fi

if [[ -z "$GPU_LIST" ]]; then
  GPU_LIST="$("${PYTHON[@]}" - <<'PY'
try:
    import torch
    print(" ".join(str(i) for i in range(torch.cuda.device_count())))
except Exception:
    print("")
PY
)"
fi
read -r -a GPUS <<< "${GPU_LIST:-}"
if [[ "$MODE" != "analyze-only" && "$MODE" != "sync" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp52] GPU phase requires GPU_LIST or visible CUDA GPUs" >&2
  exit 2
fi

RECORDS_DIR="${RUN_ROOT}/records"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "[exp52] host $(hostname)"
echo "[exp52] mode ${MODE}"
echo "[exp52] run_name ${RUN_NAME}"
echo "[exp52] run_root ${RUN_ROOT}"
echo "[exp52] models ${MODELS}"
echo "[exp52] category_filter ${CATEGORY_FILTER}"
echo "[exp52] gpus ${GPU_LIST:-<none>}"
echo "[exp52] workers_per_model ${WORKERS_PER_MODEL} batch_size ${BATCH_SIZE}"
echo "[exp52] n_prompts ${N_PROMPTS} max_events ${MAX_EVENTS}"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  else
    echo "[exp52] no GCS sync helper; skipping upload" >&2
  fi
}

download_inputs_if_needed() {
  if [[ -f "${EXP47_ROOT}/analysis/effects.csv" ]] && find "${EXP20_ROOT}/raw_shared" -name 'exp20_validation_records.jsonl' -print -quit >/dev/null 2>&1; then
    echo "[exp52] Exp47/Exp20 inputs already present"
    return
  fi
  echo "[exp52] fetching Exp47 analysis and Exp20 manifests from ${EXP47_GCS}"
  mkdir -p "$EXP47_ROOT"
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${EXP47_GCS%/}/analysis" "${EXP47_ROOT}/analysis"
    gsutil -m rsync -r "${EXP47_GCS%/}/exp20/raw_shared" "${EXP20_ROOT}/raw_shared"
  else
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/analysis" "${EXP47_ROOT}/analysis" --workers 24
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/exp20/raw_shared" "${EXP20_ROOT}/raw_shared" --workers 24
  fi
}

wait_batch() {
  local phase="$1"
  shift
  local status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp52] phase ${phase} failed; syncing partial outputs" >&2
    sync_outputs || true
    exit 1
  fi
}

run_collect() {
  echo "[exp52] phase collect"
  local jobs=()
  for model in $MODELS; do
    for ((w=0; w<WORKERS_PER_MODEL; w++)); do
      jobs+=("${model}_w${w}|${model}|${w}")
    done
  done
  local idx=0
  local pids=()
  for job in "${jobs[@]}"; do
    IFS='|' read -r log_stem model worker <<< "$job"
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    local extra=()
    if [[ "$RESUME" == "1" ]]; then extra+=(--resume); fi
    if [[ -n "$MAX_NEW_TOKENS_OVERRIDE" ]]; then extra+=(--max-new-tokens-override "$MAX_NEW_TOKENS_OVERRIDE"); fi
    echo "[exp52] launch model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON[@]}" -m src.poc.exp52_forced_token_consequence_bridge collect \
      --model "$model" \
      --dataset "$DATASET" \
      --exp20-root "$EXP20_ROOT" \
      --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
      --exp47-root "$EXP47_ROOT" \
      --out-dir "$RECORDS_DIR" \
      --device cuda:0 \
      --worker-index "$worker" \
      --n-workers "$WORKERS_PER_MODEL" \
      --n-prompts "$N_PROMPTS" \
      --max-events "$MAX_EVENTS" \
      --batch-size "$BATCH_SIZE" \
      --event-kind "$EVENT_KIND" \
      --prompt-mode "$PROMPT_MODE" \
      --category-filter $CATEGORY_FILTER \
      "${extra[@]}" \
      >"${LOG_DIR}/collect_${log_stem}.log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
    if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
      wait_batch collect "${pids[@]}"
      pids=()
      sync_outputs || true
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_batch collect "${pids[@]}"
  fi
  for model in $MODELS; do
    "${PYTHON[@]}" -m src.poc.exp52_forced_token_consequence_bridge collect \
      --model "$model" \
      --out-dir "$RECORDS_DIR" \
      --n-workers "$WORKERS_PER_MODEL" \
      --merge-only \
      >"${LOG_DIR}/merge_${model}.log" 2>&1
  done
  sync_outputs || true
}

run_analysis() {
  echo "[exp52] phase analyze"
  "${PYTHON[@]}" scripts/analysis/analyze_exp52_forced_token_consequence_bridge.py \
    --records-dir "$RECORDS_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    --dataset "$DATASET" \
    --models "$MODELS" \
    --n-boot "$N_BOOT" \
    >"${LOG_DIR}/analysis.log" 2>&1
  sync_outputs || true
}

download_inputs_if_needed

case "$MODE" in
  smoke|full)
    run_collect
    run_analysis
    ;;
  collect-only)
    run_collect
    ;;
  analyze-only)
    run_analysis
    ;;
  sync)
    sync_outputs
    ;;
esac

echo "[exp52] done ${RUN_ROOT}"

