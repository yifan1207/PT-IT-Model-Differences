#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"  # smoke|full|analyze-only
RUN_NAME="${RUN_NAME:-exp35_olmo_base_anchored_stage_decomposition_$(date -u +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
N_EXAMPLES="${N_EXAMPLES:-600}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-25}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GPU_LIST="${GPU_LIST:-}"
BOUNDARY_LAYER="${BOUNDARY_LAYER:-19}"
N_BOOT="${N_BOOT:-2000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-20000}"
ANALYSIS_BATCH_SIZE="${ANALYSIS_BATCH_SIZE:-64}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

RUN_ROOT="${RUN_ROOT:-results/exp35_olmo_base_anchored_stage_decomposition/${RUN_NAME}}"
SUPPORT_DIR="${RUN_ROOT}/support"
STAGE_CACHE_DIR="${RUN_ROOT}/stage_cache"
FIXED_DIR="${RUN_ROOT}/fixed_factorial"
IDENTITY_DIR="${RUN_ROOT}/identity_margin"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"

if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp35] invalid MODE=${MODE}" >&2
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
  echo "[exp35] GPU run requires GPU_LIST or visible CUDA GPUs" >&2
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

echo "[exp35] host $(hostname)"
echo "[exp35] mode ${MODE}"
echo "[exp35] run_name ${RUN_NAME}"
echo "[exp35] run_root ${RUN_ROOT}"
echo "[exp35] dataset ${DATASET}"
echo "[exp35] n_total ${N_TOTAL} max_new_tokens ${MAX_NEW_TOKENS}"
echo "[exp35] gpus ${GPU_LIST:-<none>}"
echo "[exp35] workers support=${SUPPORT_WORKERS} cache_per_stage=${CACHE_WORKERS_PER_STAGE} fixed_per_host=${FIXED_WORKERS_PER_HOST} identity_per_stage=${IDENTITY_WORKERS_PER_STAGE}"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    uv run python scripts/infra/gcs_sync_adc.py upload \
      "$RUN_ROOT" \
      "${GCS_SYNC_DEST%/}/${RUN_NAME}" \
      --workers 24
  else
    echo "[exp35] no gsutil or ADC sync helper; skipping sync to ${GCS_SYNC_DEST}" >&2
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
    echo "[exp35] phase ${phase} failed; see ${LOG_DIR}" >&2
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

run_support() {
  echo "[exp35] phase support"
  local pids=()
  for ((w=0; w< SUPPORT_WORKERS; w++)); do
    local gpu="${GPUS[$((w % ${#GPUS[@]}))]}"
    CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.support \
      --dataset "$DATASET" \
      --out-dir "$SUPPORT_DIR" \
      --device cuda:0 \
      --n-examples "$N_TOTAL" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --worker-index "$w" \
      --n-workers "$SUPPORT_WORKERS" \
      >"${LOG_DIR}/support_w${w}.log" 2>&1 &
    pids+=("$!")
  done
  wait_all support "${pids[@]}"
  uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.support \
    --out-dir "$SUPPORT_DIR" \
    --merge-only \
    --n-workers "$SUPPORT_WORKERS"
  sync_outputs || true
}

run_stage_cache() {
  echo "[exp35] phase stage_cache"
  local pids=()
  local idx=0
  for stage in B S D R; do
    for ((w=0; w< CACHE_WORKERS_PER_STAGE; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.cache_stage_states \
        --stage "$stage" \
        --support "${SUPPORT_DIR}/base_rlvr_first_diff.jsonl.gz" \
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
    uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.cache_stage_states \
      --stage "$stage" \
      --support "${SUPPORT_DIR}/base_rlvr_first_diff.jsonl.gz" \
      --out-dir "$STAGE_CACHE_DIR" \
      --merge-only \
      --n-workers "$CACHE_WORKERS_PER_STAGE"
  done
  sync_outputs || true
}

run_fixed_factorial() {
  echo "[exp35] phase fixed_factorial"
  local pids=()
  local idx=0
  for host in B S D R; do
    for ((w=0; w< FIXED_WORKERS_PER_HOST; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.fixed_stage_factorial \
        --host-stage "$host" \
        --support "${SUPPORT_DIR}/base_rlvr_first_diff.jsonl.gz" \
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
    uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.fixed_stage_factorial \
      --host-stage "$host" \
      --support "${SUPPORT_DIR}/base_rlvr_first_diff.jsonl.gz" \
      --stage-cache "$STAGE_CACHE_DIR" \
      --out-dir "$FIXED_DIR" \
      --merge-only \
      --n-workers "$FIXED_WORKERS_PER_HOST"
  done
  sync_outputs || true
}

run_identity_margin() {
  echo "[exp35] phase identity_margin"
  local pids=()
  local idx=0
  for target in S D R; do
    for ((w=0; w< IDENTITY_WORKERS_PER_STAGE; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.identity_margin \
        --target-stage "$target" \
        --support "${SUPPORT_DIR}/base_rlvr_first_diff.jsonl.gz" \
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
    uv run python -m src.poc.exp35_olmo_base_anchored_stage_decomposition.identity_margin \
      --target-stage "$target" \
      --support "${SUPPORT_DIR}/base_rlvr_first_diff.jsonl.gz" \
      --out-dir "$IDENTITY_DIR" \
      --merge-only \
      --n-workers "$IDENTITY_WORKERS_PER_STAGE"
  done
  sync_outputs || true
}

run_analysis() {
  echo "[exp35] phase analysis"
  local gpu="${GPUS[0]:-0}"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python scripts/analysis/analyze_exp35_olmo_base_anchored_stage_decomposition.py \
    --fixed-root "$FIXED_DIR" \
    --stage-cache-root "$STAGE_CACHE_DIR" \
    --identity-root "$IDENTITY_DIR" \
    --out-dir "$ANALYSIS_DIR" \
    --device cuda:0 \
    --batch-size "$ANALYSIS_BATCH_SIZE" \
    --n-boot "$N_BOOT" \
    --n-permutations "$N_PERMUTATIONS" \
    >"${LOG_DIR}/analysis.log" 2>&1
  sync_outputs || true
}

if [[ "$MODE" != "analyze-only" ]]; then
  run_support
  run_stage_cache
  run_fixed_factorial
  run_identity_margin
fi
run_analysis

echo "[exp35] complete ${RUN_NAME}"
