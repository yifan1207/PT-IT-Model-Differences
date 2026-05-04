#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"  # smoke|full|analyze-only
RUN_NAME="${RUN_NAME:-exp46_tulu_fixed_support_stage_sweep_$(date -u +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
N_EXAMPLES="${N_EXAMPLES:-600}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"  # raw_shared|tulu_shared_template
SUPPORT_TARGET_STAGE="${SUPPORT_TARGET_STAGE:-R}"  # R primary; S/D for optional base-anchored appendix checks
GPU_LIST="${GPU_LIST:-}"
BOUNDARY_LAYER="${BOUNDARY_LAYER:-19}"
N_BOOT="${N_BOOT:-2000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-20000}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-64}"
N_TOKENIZER_CHECK_EXAMPLES="${N_TOKENIZER_CHECK_EXAMPLES:-32}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

RUN_ROOT="${RUN_ROOT:-results/exp46_tulu_fixed_support_stage_sweep/${RUN_NAME}}"
PREFLIGHT_DIR="${RUN_ROOT}/preflight"
SUPPORT_DIR="${RUN_ROOT}/support"
STAGE_CACHE_DIR="${RUN_ROOT}/stage_cache"
FIXED_DIR="${RUN_ROOT}/fixed_factorial"
IDENTITY_DIR="${RUN_ROOT}/identity_margin"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"
if [[ "$SUPPORT_TARGET_STAGE" == "R" ]]; then
  SUPPORT_FILE="${SUPPORT_DIR}/base_final_first_diff.jsonl.gz"
elif [[ "$SUPPORT_TARGET_STAGE" == "S" || "$SUPPORT_TARGET_STAGE" == "D" ]]; then
  SUPPORT_TARGET_STAGE_LOWER="$(printf '%s' "$SUPPORT_TARGET_STAGE" | tr '[:upper:]' '[:lower:]')"
  SUPPORT_FILE="${SUPPORT_DIR}/base_${SUPPORT_TARGET_STAGE_LOWER}_first_diff.jsonl.gz"
else
  echo "[exp46] invalid SUPPORT_TARGET_STAGE=${SUPPORT_TARGET_STAGE}; expected S, D, or R" >&2
  exit 2
fi

if [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  PYTHON_CMD=(uv run --no-sync python)
else
  PYTHON_CMD=("${PYTHON_CMD[@]}")
fi

if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp46] invalid MODE=${MODE}" >&2
  exit 2
fi

gpu_count="$(python - <<'PY'
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
if [[ "$MODE" != "analyze-only" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp46] GPU run requires GPU_LIST or visible CUDA GPUs" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  N_TOTAL="$SMOKE_EXAMPLES"
  SUPPORT_WORKERS="${SUPPORT_WORKERS:-1}"
  CACHE_WORKERS_PER_STAGE="${CACHE_WORKERS_PER_STAGE:-1}"
  FIXED_WORKERS_PER_HOST="${FIXED_WORKERS_PER_HOST:-1}"
  IDENTITY_WORKERS_PER_STAGE="${IDENTITY_WORKERS_PER_STAGE:-1}"
else
  N_TOTAL="$N_EXAMPLES"
  SUPPORT_WORKERS="${SUPPORT_WORKERS:-${#GPUS[@]}}"
  CACHE_WORKERS_PER_STAGE="${CACHE_WORKERS_PER_STAGE:-2}"
  FIXED_WORKERS_PER_HOST="${FIXED_WORKERS_PER_HOST:-2}"
  IDENTITY_WORKERS_PER_STAGE="${IDENTITY_WORKERS_PER_STAGE:-2}"
fi

mkdir -p "$LOG_DIR"

echo "[exp46] host $(hostname)"
echo "[exp46] mode ${MODE}"
echo "[exp46] run_name ${RUN_NAME}"
echo "[exp46] run_root ${RUN_ROOT}"
echo "[exp46] dataset ${DATASET}"
echo "[exp46] prompt_mode ${PROMPT_MODE}"
echo "[exp46] support_target_stage ${SUPPORT_TARGET_STAGE}"
echo "[exp46] n_total ${N_TOTAL} max_new_tokens ${MAX_NEW_TOKENS}"
echo "[exp46] gpus ${GPU_LIST:-<none>}"
echo "[exp46] workers support=${SUPPORT_WORKERS} cache_per_stage=${CACHE_WORKERS_PER_STAGE} fixed_per_host=${FIXED_WORKERS_PER_HOST} identity_per_stage=${IDENTITY_WORKERS_PER_STAGE}"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    "${PYTHON_CMD[@]}" scripts/infra/gcs_sync_adc.py upload \
      "$RUN_ROOT" \
      "${GCS_SYNC_DEST%/}/${RUN_NAME}" \
      --workers 24
  else
    echo "[exp46] no gsutil or ADC sync helper; skipping sync to ${GCS_SYNC_DEST}" >&2
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
    echo "[exp46] phase ${phase} failed; see ${LOG_DIR}" >&2
    sync_outputs || true
    exit 1
  fi
}

flush_phase_jobs() {
  local phase="$1"
  shift
  if [[ "$#" -eq 0 ]]; then
    return
  fi
  wait_all "$phase" "$@"
}

run_preflight() {
  echo "[exp46] phase preflight"
  "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.preflight \
    --out-dir "$PREFLIGHT_DIR" \
    --dataset "$DATASET" \
    --n-tokenizer-check-examples "$N_TOKENIZER_CHECK_EXAMPLES" \
    --prompt-mode "$PROMPT_MODE" \
    >"${LOG_DIR}/preflight.log" 2>&1
  sync_outputs || true
}

run_support() {
  echo "[exp46] phase support"
  local pids=()
  for ((w=0; w< SUPPORT_WORKERS; w++)); do
    local gpu="${GPUS[$((w % ${#GPUS[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.support \
      --dataset "$DATASET" \
      --out-dir "$SUPPORT_DIR" \
      --device cuda:0 \
      --n-examples "$N_TOTAL" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --prompt-mode "$PROMPT_MODE" \
      --target-stage "$SUPPORT_TARGET_STAGE" \
      --worker-index "$w" \
      --n-workers "$SUPPORT_WORKERS" \
      >"${LOG_DIR}/support_w${w}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all support "${pids[@]}"
  "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.support \
    --out-dir "$SUPPORT_DIR" \
    --merge-only \
    --target-stage "$SUPPORT_TARGET_STAGE" \
    --n-workers "$SUPPORT_WORKERS"
  sync_outputs || true
}

run_stage_cache() {
  echo "[exp46] phase stage_cache"
  local pids=()
  local idx=0
  for stage in B S D R; do
    for ((w=0; w< CACHE_WORKERS_PER_STAGE; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.cache_stage_states \
        --stage "$stage" \
        --support "$SUPPORT_FILE" \
        --out-dir "$STAGE_CACHE_DIR" \
        --device cuda:0 \
        --n-examples "$N_TOTAL" \
        --worker-index "$w" \
        --n-workers "$CACHE_WORKERS_PER_STAGE" \
        --boundary-layer "$BOUNDARY_LAYER" \
      >"${LOG_DIR}/cache_${stage}_w${w}.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        flush_phase_jobs stage_cache "${pids[@]}"
        pids=()
      fi
    done
  done
  flush_phase_jobs stage_cache "${pids[@]}"
  for stage in B S D R; do
    "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.cache_stage_states \
      --stage "$stage" \
      --support "$SUPPORT_FILE" \
      --out-dir "$STAGE_CACHE_DIR" \
      --merge-only \
      --n-workers "$CACHE_WORKERS_PER_STAGE"
  done
  sync_outputs || true
}

run_fixed_factorial() {
  echo "[exp46] phase fixed_factorial"
  local pids=()
  local idx=0
  for host in B S D R; do
    for ((w=0; w< FIXED_WORKERS_PER_HOST; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.fixed_stage_factorial \
        --host-stage "$host" \
        --support "$SUPPORT_FILE" \
        --stage-cache "$STAGE_CACHE_DIR" \
        --out-dir "$FIXED_DIR" \
        --device cuda:0 \
        --n-examples "$N_TOTAL" \
        --worker-index "$w" \
        --n-workers "$FIXED_WORKERS_PER_HOST" \
        --boundary-layer "$BOUNDARY_LAYER" \
      >"${LOG_DIR}/fixed_${host}_w${w}.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        flush_phase_jobs fixed_factorial "${pids[@]}"
        pids=()
      fi
    done
  done
  flush_phase_jobs fixed_factorial "${pids[@]}"
  for host in B S D R; do
    "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.fixed_stage_factorial \
      --host-stage "$host" \
      --support "$SUPPORT_FILE" \
      --stage-cache "$STAGE_CACHE_DIR" \
      --out-dir "$FIXED_DIR" \
      --merge-only \
      --n-workers "$FIXED_WORKERS_PER_HOST"
  done
  sync_outputs || true
}

run_identity_margin() {
  echo "[exp46] phase identity_margin"
  local pids=()
  local idx=0
  for target in S D R; do
    for ((w=0; w< IDENTITY_WORKERS_PER_STAGE; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.identity_margin \
        --target-stage "$target" \
        --support "$SUPPORT_FILE" \
        --out-dir "$IDENTITY_DIR" \
        --device cuda:0 \
        --n-examples "$N_TOTAL" \
        --worker-index "$w" \
        --n-workers "$IDENTITY_WORKERS_PER_STAGE" \
      >"${LOG_DIR}/identity_${target}_w${w}.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        flush_phase_jobs identity_margin "${pids[@]}"
        pids=()
      fi
    done
  done
  flush_phase_jobs identity_margin "${pids[@]}"
  for target in S D R; do
    "${PYTHON_CMD[@]}" -m src.poc.exp46_tulu_fixed_support_stage_sweep.identity_margin \
      --target-stage "$target" \
      --support "$SUPPORT_FILE" \
      --out-dir "$IDENTITY_DIR" \
      --merge-only \
      --n-workers "$IDENTITY_WORKERS_PER_STAGE"
  done
  sync_outputs || true
}

run_analysis() {
  echo "[exp46] phase analysis"
  local gpu="${GPUS[0]:-0}"
  CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON_CMD[@]}" scripts/analysis/analyze_exp46_tulu_fixed_support_stage_sweep.py \
    --fixed-root "$FIXED_DIR" \
    --stage-cache-root "$STAGE_CACHE_DIR" \
    --identity-root "$IDENTITY_DIR" \
    --preflight-dir "$PREFLIGHT_DIR" \
    --out-dir "$ANALYSIS_DIR" \
    --device cuda:0 \
    --batch-size "$ANALYSIS_BATCH_SIZE" \
    --n-boot "$N_BOOT" \
    --n-permutations "$N_PERMUTATIONS" \
    >"${LOG_DIR}/analysis.log" 2>&1
  sync_outputs || true
}

if [[ "$MODE" != "analyze-only" ]]; then
  run_preflight
  run_support
  run_stage_cache
  run_fixed_factorial
  run_identity_margin
fi
run_analysis

echo "[exp46] complete ${RUN_NAME}"
