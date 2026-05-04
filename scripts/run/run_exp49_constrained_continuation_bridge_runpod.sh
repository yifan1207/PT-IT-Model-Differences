#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${MODE:-smoke}"  # smoke|full|analyze-only|score-only|candidates-only
RUN_NAME="${RUN_NAME:-exp49_constrained_continuation_bridge_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp49_constrained_continuation_bridge/${RUN_NAME}}"
EXP47_ROOT="${EXP47_ROOT:-results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
EXP47_GCS="${EXP47_GCS:-gs://pt-vs-it-results/results/exp47_same_base_recipe_specificity/exp47_same_base_recipe_specificity_20260504_0959_a100x24}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp49_constrained_continuation_bridge}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
MODELS="${MODELS:-llama31_meta_instruct llama31_tulu3_sft llama31_tulu3_dpo llama31_tulu3_final llama31_openmath2}"
GPU_LIST="${GPU_LIST:-}"
MAX_TAIL="${MAX_TAIL:-8}"
HORIZONS="${HORIZONS:-0 1 2 4 8}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-8}"
FULL_EXAMPLES="${FULL_EXAMPLES:-}"
EVENT_KIND="${EVENT_KIND:-first_diff}"
RESUME="${RESUME:-1}"
SCORE_WORKERS_PER_MODEL="${SCORE_WORKERS_PER_MODEL:-2}"
CANDIDATE_WORKERS_PER_MODEL="${CANDIDATE_WORKERS_PER_MODEL:-1}"

if [[ "$MODE" == "smoke" ]]; then
  MODELS="${SMOKE_MODELS:-llama31_tulu3_final}"
  N_EXAMPLES="${N_EXAMPLES:-$SMOKE_EXAMPLES}"
  MAX_TAIL="${SMOKE_MAX_TAIL:-2}"
  HORIZONS="${SMOKE_HORIZONS:-0 1 2}"
  SCORE_WORKERS_PER_MODEL="${SMOKE_SCORE_WORKERS_PER_MODEL:-1}"
  CANDIDATE_WORKERS_PER_MODEL="${SMOKE_CANDIDATE_WORKERS_PER_MODEL:-1}"
else
  N_EXAMPLES="${N_EXAMPLES:-$FULL_EXAMPLES}"
fi

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
    n = torch.cuda.device_count()
    print(" ".join(str(i) for i in range(n)))
except Exception:
    print("")
PY
)"
fi
read -r -a GPUS <<< "${GPU_LIST:-}"
if [[ "$MODE" != "analyze-only" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp49] GPU mode requires GPU_LIST or visible CUDA GPUs" >&2
  exit 2
fi

CANDIDATES_DIR="${RUN_ROOT}/candidates"
SCORES_DIR="${RUN_ROOT}/scores"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

echo "[exp49] host $(hostname)"
echo "[exp49] mode ${MODE}"
echo "[exp49] run_name ${RUN_NAME}"
echo "[exp49] run_root ${RUN_ROOT}"
echo "[exp49] exp47_root ${EXP47_ROOT}"
echo "[exp49] models ${MODELS}"
echo "[exp49] gpus ${GPU_LIST:-<none>}"
echo "[exp49] max_tail ${MAX_TAIL} horizons ${HORIZONS}"
echo "[exp49] n_examples ${N_EXAMPLES:-all}"
echo "[exp49] candidate_workers_per_model ${CANDIDATE_WORKERS_PER_MODEL} score_workers_per_model ${SCORE_WORKERS_PER_MODEL}"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  else
    echo "[exp49] no gsutil or ADC sync helper found; skipping GCS sync" >&2
  fi
}

download_exp47_if_needed() {
  if [[ -f "${EXP47_ROOT}/analysis/effects.csv" ]] && find "${EXP47_ROOT}/exp20/raw_shared" -name 'exp20_validation_records.jsonl' -print -quit >/dev/null 2>&1; then
    echo "[exp49] Exp47 artifacts already present"
    return
  fi
  echo "[exp49] fetching required Exp47 artifacts from ${EXP47_GCS}"
  mkdir -p "$EXP47_ROOT"
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${EXP47_GCS%/}/analysis" "${EXP47_ROOT}/analysis"
    gsutil -m rsync -r "${EXP47_GCS%/}/preflight" "${EXP47_ROOT}/preflight" || true
    gsutil -m rsync -r "${EXP47_GCS%/}/exp20/raw_shared" "${EXP47_ROOT}/exp20/raw_shared"
  else
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/analysis" "${EXP47_ROOT}/analysis" --workers 24
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/preflight" "${EXP47_ROOT}/preflight" --workers 24 || true
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py download "${EXP47_GCS%/}/exp20/raw_shared" "${EXP47_ROOT}/exp20/raw_shared" --workers 24
  fi
}

wait_all() {
  local phase="$1"
  shift
  local status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp49] phase ${phase} failed; syncing partial outputs" >&2
    sync_outputs || true
    exit 1
  fi
}

flush_jobs() {
  local phase="$1"
  shift
  if [[ "$#" -gt 0 ]]; then
    wait_all "$phase" "$@"
  fi
}

run_candidate_phase() {
  echo "[exp49] phase candidates"
  local pids=()
  local idx=0
  for model in $MODELS; do
    for ((w=0; w<CANDIDATE_WORKERS_PER_MODEL; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      local extra=()
      if [[ -n "${N_EXAMPLES}" ]]; then extra+=(--n-examples "$N_EXAMPLES"); fi
      if [[ "$RESUME" == "1" ]]; then extra+=(--resume); fi
      CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON[@]}" -m src.poc.exp49_constrained_continuation_bridge collect-candidates \
        --model "$model" \
        --device cuda:0 \
        --exp47-root "$EXP47_ROOT" \
        --dataset "$DATASET" \
        --output-dir "$CANDIDATES_DIR" \
        --event-kind "$EVENT_KIND" \
        --max-tail "$MAX_TAIL" \
        --worker-index "$w" \
        --n-workers "$CANDIDATE_WORKERS_PER_MODEL" \
        "${extra[@]}" \
        >"${LOG_DIR}/candidates_${model}_w${w}.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        flush_jobs candidates "${pids[@]}"
        pids=()
        sync_outputs || true
      fi
    done
  done
  flush_jobs candidates "${pids[@]}"
  sync_outputs || true
}

run_score_phase() {
  echo "[exp49] phase score"
  local pids=()
  local idx=0
  for model in $MODELS; do
    for ((w=0; w<SCORE_WORKERS_PER_MODEL; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      local extra=()
      if [[ -n "${N_EXAMPLES}" ]]; then extra+=(--n-examples "$N_EXAMPLES"); fi
      if [[ "$RESUME" == "1" ]]; then extra+=(--resume); fi
      CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON[@]}" -m src.poc.exp49_constrained_continuation_bridge score-sequences \
        --model "$model" \
        --device cuda:0 \
        --candidates-dir "$CANDIDATES_DIR" \
        --output-dir "$SCORES_DIR" \
        --event-kind "$EVENT_KIND" \
        --worker-index "$w" \
        --n-workers "$SCORE_WORKERS_PER_MODEL" \
        "${extra[@]}" \
        >"${LOG_DIR}/score_${model}_w${w}.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        flush_jobs score "${pids[@]}"
        pids=()
        sync_outputs || true
      fi
    done
  done
  flush_jobs score "${pids[@]}"
  sync_outputs || true
}

run_analysis_phase() {
  echo "[exp49] phase analyze"
  "${PYTHON[@]}" scripts/analysis/analyze_exp49_constrained_continuation_bridge.py \
    --scores-dir "$SCORES_DIR" \
    --output-dir "$ANALYSIS_DIR" \
    --exp47-root "$EXP47_ROOT" \
    --models "$MODELS" \
    --horizons "$HORIZONS" \
    --max-tail "$MAX_TAIL" \
    --n-boot "$N_BOOT" \
    >"${LOG_DIR}/analysis.log" 2>&1
  sync_outputs || true
}

download_exp47_if_needed

case "$MODE" in
  smoke|full)
    run_candidate_phase
    run_score_phase
    run_analysis_phase
    ;;
  candidates-only)
    run_candidate_phase
    ;;
  score-only)
    run_score_phase
    run_analysis_phase
    ;;
  analyze-only)
    run_analysis_phase
    ;;
  *)
    echo "[exp49] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp49] done ${RUN_ROOT}"
