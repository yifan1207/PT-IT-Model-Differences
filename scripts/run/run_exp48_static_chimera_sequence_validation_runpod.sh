#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PY_RUNNER="${PY_RUNNER:-uv run python}"

MODE="${MODE:-smoke}"  # smoke|preflight|sequence-full|score-auto|rescue-fit|rescue-score|analyze-only|sync|full
RUN_NAME="${RUN_NAME:-exp48_static_chimera_sequence_validation_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp48_static_chimera_sequence_validation/${RUN_NAME}}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp48_static_chimera_sequence_validation}"
EXP47_GCS="${EXP47_GCS:-gs://pt-vs-it-results/results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
EXP47_ROOT="${EXP47_ROOT:-results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
EXP20_ROOT="${EXP20_ROOT:-${EXP47_ROOT}/exp20}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-${EXP47_ROOT}/exp20}"
EXP47_ANALYSIS_DIR="${EXP47_ANALYSIS_DIR:-${EXP47_ROOT}/analysis}"

MODELS="${MODELS:-llama31_meta_instruct llama31_tulu3_sft llama31_tulu3_dpo llama31_tulu3_final llama31_openmath2}"
BOUNDARIES="${BOUNDARIES:-16 19 24 29 31}"
COMPONENT_BOUNDARIES="${COMPONENT_BOUNDARIES:-19 29 31}"
CONTROL_BOUNDARIES="${CONTROL_BOUNDARIES:-19 29 31}"
INTERPOLATION_ALPHAS="${INTERPOLATION_ALPHAS:-0 0.25 0.5 0.75 1}"
RESCUE_KS="${RESCUE_KS:-1 4 16 64 256 full}"
RESCUE_ALPHAS="${RESCUE_ALPHAS:-0 0.25 0.5 1 1.5}"

GPU_LIST="${GPU_LIST:-}"
JOB_SHARD_INDEX="${JOB_SHARD_INDEX:-0}"
JOB_SHARD_COUNT="${JOB_SHARD_COUNT:-1}"
N_SEQUENCE_PROMPTS="${N_SEQUENCE_PROMPTS:-1400}"
N_RESCUE_EXAMPLES="${N_RESCUE_EXAMPLES:-1400}"
SEQUENCE_BATCH_SIZE="${SEQUENCE_BATCH_SIZE:-4}"
RESCUE_BATCH_SIZE="${RESCUE_BATCH_SIZE:-24}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-768}"
HEALTH_EVERY="${HEALTH_EVERY:-8}"
N_BOOT="${N_BOOT:-1000}"

if [[ "$MODE" == "smoke" ]]; then
  MODELS="${SMOKE_MODELS:-llama31_meta_instruct}"
  BOUNDARIES="${SMOKE_BOUNDARIES:-19}"
  COMPONENT_BOUNDARIES="${SMOKE_COMPONENT_BOUNDARIES:-19}"
  CONTROL_BOUNDARIES="${SMOKE_CONTROL_BOUNDARIES:-19}"
  N_SEQUENCE_PROMPTS="${SMOKE_N_SEQUENCE_PROMPTS:-24}"
  N_RESCUE_EXAMPLES="${SMOKE_N_RESCUE_EXAMPLES:-24}"
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-16}"
  N_BOOT="${SMOKE_N_BOOT:-100}"
fi

PREFLIGHT_DIR="${RUN_ROOT}/preflight"
SEQUENCE_ROOT="${RUN_ROOT}/sequence"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

gpu_count="$($PY_RUNNER - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if [[ -z "$GPU_LIST" && "$gpu_count" -gt 0 ]]; then
  GPU_LIST="$(seq -s ' ' 0 $((gpu_count - 1)))"
fi
read -r -a GPUS <<< "${GPU_LIST:-}"

echo "[exp48] host $(hostname)"
echo "[exp48] mode ${MODE}"
echo "[exp48] run_name ${RUN_NAME}"
echo "[exp48] run_root ${RUN_ROOT}"
echo "[exp48] gpus ${GPU_LIST:-<none>} detected=${gpu_count}"
echo "[exp48] shard ${JOB_SHARD_INDEX}/${JOB_SHARD_COUNT}"
echo "[exp48] models ${MODELS}"
echo "[exp48] boundaries ${BOUNDARIES}"
echo "[exp48] gcs ${GCS_SYNC_DEST%/}/${RUN_NAME}"

if [[ "$MODE" != "analyze-only" && "$MODE" != "sync" && "$MODE" != "preflight" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp48] GPU phase requires visible GPUs or GPU_LIST" >&2
  exit 2
fi

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUNNER scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  else
    echo "[exp48] no GCS sync helper; skipping upload" >&2
  fi
}

pull_exp47_support() {
  if [[ -d "$EXP20_ROOT/raw_shared" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    mkdir -p "$EXP47_ROOT"
    echo "[exp48] pulling Exp47 exp20 support from ${EXP47_GCS}/exp20"
    gsutil -m rsync -r "${EXP47_GCS%/}/exp20" "$EXP20_ROOT"
    if [[ ! -d "$EXP47_ANALYSIS_DIR" ]]; then
      gsutil -m rsync -r "${EXP47_GCS%/}/analysis" "$EXP47_ANALYSIS_DIR" || true
    fi
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    mkdir -p "$EXP47_ROOT"
    echo "[exp48] pulling Exp47 exp20 support with ADC helper from ${EXP47_GCS}/exp20"
    $PY_RUNNER scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/exp20" "$EXP20_ROOT" --workers 24
    if [[ ! -d "$EXP47_ANALYSIS_DIR" ]]; then
      $PY_RUNNER scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/analysis" "$EXP47_ANALYSIS_DIR" --workers 12 || true
    fi
  else
    echo "[exp48] Exp47 support not present and gsutil unavailable" >&2
  fi
}

pull_run_from_gcs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  mkdir -p "$RUN_ROOT"
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${GCS_SYNC_DEST%/}/${RUN_NAME}" "$RUN_ROOT" || true
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUNNER scripts/infra/gcs_sync_adc.py download "${GCS_SYNC_DEST%/}/${RUN_NAME}" "$RUN_ROOT" --workers 24 || true
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
    echo "[exp48] phase ${phase} failed; see ${LOG_DIR}" >&2
    sync_outputs || true
    exit "$status"
  fi
}

launch_scheduled() {
  local phase="$1"
  local -n jobs_ref="$2"
  local idx=0
  local launched=0
  local status=0
  local free_gpus=("${GPUS[@]}")
  declare -A pid_gpu=()

  wait_one_dynamic() {
    local done_pid=""
    local rc=0
    if wait -n -p done_pid; then
      rc=0
    else
      rc=$?
    fi
    if [[ -n "${done_pid}" && -n "${pid_gpu[$done_pid]:-}" ]]; then
      free_gpus+=("${pid_gpu[$done_pid]}")
      unset "pid_gpu[$done_pid]"
    fi
    if [[ "$rc" -ne 0 ]]; then
      status=1
    fi
  }

  for job in "${jobs_ref[@]}"; do
    if (( (idx % JOB_SHARD_COUNT) != JOB_SHARD_INDEX )); then
      idx=$((idx + 1))
      continue
    fi
    while [[ "${#free_gpus[@]}" -lt 1 ]]; do
      wait_one_dynamic
      if [[ "$status" -ne 0 ]]; then
        break
      fi
    done
    if [[ "$status" -ne 0 ]]; then
      break
    fi
    local gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    IFS='|' read -r log_name cmd <<< "$job"
    echo "[exp48] launch ${phase} gpu=${gpu} log=${LOG_DIR}/${log_name}.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      eval "$cmd"
    ) >"${LOG_DIR}/${log_name}.log" 2>&1 &
    pid_gpu[$!]="$gpu"
    launched=$((launched + 1))
    idx=$((idx + 1))
  done
  while [[ "${#pid_gpu[@]}" -gt 0 ]]; do
    wait_one_dynamic
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp48] phase ${phase} failed; see ${LOG_DIR}" >&2
    sync_outputs || true
    exit "$status"
  fi
  echo "[exp48] ${phase} launched_on_this_shard=${launched}"
}

run_preflight() {
  echo "[exp48] phase preflight"
  $PY_RUNNER -m src.poc.exp48_static_chimera_sequence_validation adapter-probe \
    --models ${MODELS} \
    --boundaries ${BOUNDARIES} \
    --out-dir "$PREFLIGHT_DIR" \
    >"${LOG_DIR}/preflight.log" 2>&1
  pull_exp47_support || true
  sync_outputs || true
}

sequence_jobs() {
  local -n out_ref="$1"
  out_ref=()
  for model in ${MODELS}; do
    for boundary in ${BOUNDARIES}; do
      local stem="sequence_suite__${model}__b${boundary}"
      out_ref+=("${stem}|$PY_RUNNER -m src.poc.exp48_static_chimera_sequence_validation sequence-suite --model ${model} --dataset ${DATASET} --out-root ${SEQUENCE_ROOT} --device cuda:0 --n-prompts ${N_SEQUENCE_PROMPTS} --prompt-split heldout --boundaries ${boundary} --component-boundaries ${COMPONENT_BOUNDARIES} --control-boundaries ${CONTROL_BOUNDARIES} --interpolation-alphas ${INTERPOLATION_ALPHAS} --batch-size ${SEQUENCE_BATCH_SIZE} --max-new-tokens ${MAX_NEW_TOKENS} --max-prompt-tokens ${MAX_PROMPT_TOKENS} --health-every ${HEALTH_EVERY}")
    done
  done
}

run_sequence_full() {
  echo "[exp48] phase sequence-full"
  local jobs=()
  sequence_jobs jobs
  echo "[exp48] sequence job count total=${#jobs[@]}"
  launch_scheduled "sequence-full" jobs
  sync_outputs || true
}

run_score_auto() {
  echo "[exp48] phase score-auto"
  local count=0
  while IFS= read -r out_dir; do
    local inputs=()
    if [[ -f "${out_dir}/sequence_records.jsonl.gz" ]]; then
      inputs=("${out_dir}/sequence_records.jsonl.gz")
    else
      mapfile -t inputs < <(find "$out_dir" -maxdepth 1 -name 'sequence_records_w*.jsonl.gz' | sort)
    fi
    if [[ "${#inputs[@]}" -eq 0 ]]; then
      continue
    fi
    local log_name="score_$(basename "$out_dir")"
    echo "[exp48] score ${out_dir} inputs=${#inputs[@]}"
    $PY_RUNNER -m src.poc.exp48_static_chimera_sequence_validation score-sequence \
      --dataset "$DATASET" \
      --out-dir "$out_dir" \
      "${inputs[@]}" \
      >"${LOG_DIR}/${log_name}.log" 2>&1
    count=$((count + 1))
  done < <(find "$SEQUENCE_ROOT" -type f \( -name sequence_records.jsonl.gz -o -name 'sequence_records_w*.jsonl.gz' \) -printf '%h\n' | sort -u)
  echo "[exp48] scored files=${count}"
  sync_outputs || true
}

rescue_fit_jobs() {
  local -n out_ref="$1"
  out_ref=()
  for model in ${MODELS}; do
    for boundary in ${BOUNDARIES}; do
      local stem="rescue_fit__${model}__b${boundary}"
      out_ref+=("${stem}|$PY_RUNNER -m src.poc.exp48_static_chimera_sequence_validation rescue --mode fit --model ${model} --model-set ${MODELS} --dataset ${DATASET} --exp20-root ${EXP20_ROOT} --exp20-fallback-root ${EXP20_FALLBACK_ROOT} --out-dir ${RUN_ROOT} --device cuda:0 --boundary ${boundary} --n-examples ${N_RESCUE_EXAMPLES} --event-kinds first_diff --max-k 256")
    done
  done
}

rescue_score_jobs() {
  local -n out_ref="$1"
  out_ref=()
  for model in ${MODELS}; do
    for boundary in ${BOUNDARIES}; do
      local stem="rescue_score__${model}__b${boundary}"
      out_ref+=("${stem}|$PY_RUNNER -m src.poc.exp48_static_chimera_sequence_validation rescue --mode score --model ${model} --model-set ${MODELS} --dataset ${DATASET} --exp20-root ${EXP20_ROOT} --exp20-fallback-root ${EXP20_FALLBACK_ROOT} --out-dir ${RUN_ROOT} --fit-root ${RUN_ROOT} --device cuda:0 --boundary ${boundary} --n-examples ${N_RESCUE_EXAMPLES} --event-kinds first_diff --ks ${RESCUE_KS} --alphas ${RESCUE_ALPHAS} --rescue-batch-size ${RESCUE_BATCH_SIZE}")
    done
  done
}

run_rescue_fit() {
  echo "[exp48] phase rescue-fit"
  pull_exp47_support
  local jobs=()
  rescue_fit_jobs jobs
  echo "[exp48] rescue-fit job count total=${#jobs[@]}"
  launch_scheduled "rescue-fit" jobs
  sync_outputs || true
}

run_rescue_score() {
  echo "[exp48] phase rescue-score"
  pull_exp47_support
  pull_run_from_gcs || true
  local jobs=()
  rescue_score_jobs jobs
  echo "[exp48] rescue-score job count total=${#jobs[@]}"
  launch_scheduled "rescue-score" jobs
  sync_outputs || true
}

run_analyze() {
  echo "[exp48] phase analyze"
  if [[ "${PULL_BEFORE_ANALYZE:-0}" == "1" ]]; then
    pull_run_from_gcs || true
  fi
  $PY_RUNNER scripts/analysis/analyze_exp48_static_chimera_sequence_validation.py \
    --run-root "$RUN_ROOT" \
    --exp47-analysis-dir "$EXP47_ANALYSIS_DIR" \
    --n-boot "$N_BOOT" \
    >"${LOG_DIR}/analyze.log" 2>&1
  sync_outputs || true
}

case "$MODE" in
  preflight)
    run_preflight
    ;;
  sequence-full)
    run_sequence_full
    ;;
  score-auto)
    run_score_auto
    ;;
  rescue-fit)
    run_rescue_fit
    ;;
  rescue-score)
    run_rescue_score
    ;;
  analyze-only)
    run_analyze
    ;;
  sync)
    sync_outputs
    ;;
  smoke)
    run_preflight
    run_sequence_full
    run_score_auto
    run_rescue_fit
    run_rescue_score
    run_analyze
    ;;
  full)
    run_preflight
    run_sequence_full
    run_score_auto
    run_rescue_fit
    run_rescue_score
    run_analyze
    ;;
  *)
    echo "[exp48] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp48] complete ${RUN_ROOT}"
