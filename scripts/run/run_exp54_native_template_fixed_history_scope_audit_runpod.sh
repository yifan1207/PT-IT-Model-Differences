#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${MODE:-smoke}"  # smoke|full|analyze-only|sync
RUN_NAME_BASE="${RUN_NAME_BASE:-exp54_native_template_fixed_history_scope_audit_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT_BASE="${RUN_ROOT_BASE:-results/exp22_fixed_history_template_audit/${RUN_NAME_BASE}}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
MODELS="${MODELS:-llama31_8b qwen3_4b mistral_7b olmo2_7b}"
TEACHER_SOURCES="${TEACHER_SOURCES:-pt_raw it_native}"
GPU_LIST="${GPU_LIST:-}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
PROCS_PER_GPU="${PROCS_PER_GPU:-1}"
N_EXAMPLES="${N_EXAMPLES:-600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROBE_FAMILIES="${PROBE_FAMILIES:-raw tuned}"
TUNED_LENS_DIR="${TUNED_LENS_DIR:-/workspace/tuned_lens_probes}"
N_BOOT="${N_BOOT:-2000}"
N_BINS="${N_BINS:-10}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp22_fixed_history_template_audit}"
FAIL_ON_QUALITY="${FAIL_ON_QUALITY:-0}"
WRITE_PAPER_SYNTHESIS="${WRITE_PAPER_SYNTHESIS:-0}"

if [[ -n "${PY_RUNNER:-}" ]]; then
  read -r -a PYTHON <<< "$PY_RUNNER"
elif [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  PYTHON=(uv run --no-sync python)
else
  PYTHON=(uv run python)
fi

if [[ "$MODE" == "smoke" ]]; then
  MODELS="${SMOKE_MODELS:-qwen3_4b}"
  TEACHER_SOURCES="${SMOKE_TEACHER_SOURCES:-pt_raw it_native}"
  N_EXAMPLES="${SMOKE_EXAMPLES:-8}"
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-16}"
  PROBE_FAMILIES="${SMOKE_PROBE_FAMILIES:-raw}"
  WORKERS_PER_MODEL="${SMOKE_WORKERS_PER_MODEL:-1}"
  N_BOOT="${SMOKE_N_BOOT:-50}"
  GCS_SYNC_DEST="${SMOKE_GCS_SYNC_DEST:-none}"
fi

if [[ "$MODE" == "sync" ]]; then
  if [[ -z "$GCS_SYNC_DEST" || "$GCS_SYNC_DEST" == "none" || "$GCS_SYNC_DEST" == "NONE" ]]; then
    echo "[exp54] GCS sync disabled"
    exit 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT_BASE" "${GCS_SYNC_DEST%/}/${RUN_NAME_BASE}"
  else
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT_BASE" "${GCS_SYNC_DEST%/}/${RUN_NAME_BASE}" --workers 24
  fi
  exit 0
fi

echo "[exp54] host $(hostname)"
echo "[exp54] mode ${MODE}"
echo "[exp54] run_name_base ${RUN_NAME_BASE}"
echo "[exp54] run_root_base ${RUN_ROOT_BASE}"
echo "[exp54] dataset ${DATASET}"
echo "[exp54] models ${MODELS}"
echo "[exp54] teacher_sources ${TEACHER_SOURCES}"
echo "[exp54] probe_families ${PROBE_FAMILIES}"
echo "[exp54] workers_per_model ${WORKERS_PER_MODEL} procs_per_gpu ${PROCS_PER_GPU}"

if [[ "$MODE" != "analyze-only" ]]; then
  for teacher_source in $TEACHER_SOURCES; do
    echo "[exp54] running teacher_source=${teacher_source}"
    MODE=full \
    RUN_NAME="${RUN_NAME_BASE}_${teacher_source}" \
    RUN_ROOT="${RUN_ROOT_BASE}/${teacher_source}" \
    DATASET="$DATASET" \
    MODELS="$MODELS" \
    GPU_LIST="$GPU_LIST" \
    WORKERS_PER_MODEL="$WORKERS_PER_MODEL" \
    PROCS_PER_GPU="$PROCS_PER_GPU" \
    N_EXAMPLES="$N_EXAMPLES" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    PROBE_FAMILIES="$PROBE_FAMILIES" \
    TUNED_LENS_DIR="$TUNED_LENS_DIR" \
    N_BOOT="$N_BOOT" \
    N_BINS="$N_BINS" \
    TEACHER_SOURCE="$teacher_source" \
    FAIL_ON_QUALITY="$FAIL_ON_QUALITY" \
    WRITE_PAPER_SYNTHESIS=0 \
    GCS_SYNC_DEST=none \
      bash scripts/run/run_exp22_fixed_history_template_audit_runpod.sh
  done
fi

echo "[exp54] combined analysis"
mkdir -p "${RUN_ROOT_BASE}/analysis"
"${PYTHON[@]}" scripts/analysis/analyze_exp22_fixed_history_template_audit.py \
  --root "$RUN_ROOT_BASE" \
  --out-dir "${RUN_ROOT_BASE}/analysis" \
  --models $MODELS \
  --n-boot "$N_BOOT" \
  --n-bins "$N_BINS"

if [[ "$WRITE_PAPER_SYNTHESIS" == "1" ]]; then
  "${PYTHON[@]}" scripts/analysis/build_exp22_fixed_history_template_audit.py \
    --run-root "$RUN_ROOT_BASE" \
    --out-dir results/paper_synthesis
fi

if [[ -n "$GCS_SYNC_DEST" && "$GCS_SYNC_DEST" != "none" && "$GCS_SYNC_DEST" != "NONE" ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT_BASE" "${GCS_SYNC_DEST%/}/${RUN_NAME_BASE}"
  else
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT_BASE" "${GCS_SYNC_DEST%/}/${RUN_NAME_BASE}" --workers 24 || true
  fi
fi

echo "[exp54] complete ${RUN_ROOT_BASE}"
