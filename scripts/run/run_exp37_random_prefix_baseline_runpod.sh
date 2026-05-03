#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"  # smoke|pilot|full|analyze-only
RUN_NAME="${RUN_NAME:-exp37_random_prefix_baseline_$(date -u +%Y%m%d_%H%M%S)}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
DEFAULT_MODELS="gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b"
SMOKE_MODELS="${SMOKE_MODELS:-llama31_8b}"
if [[ -z "${MODELS+x}" ]]; then
  if [[ "$MODE" == "smoke" ]]; then
    MODELS="$SMOKE_MODELS"
  else
    MODELS="$DEFAULT_MODELS"
  fi
fi
ARMS="${ARMS:-first_diff_reference random_local_disagreement prediv_future_pair}"
PREFIX_SOURCES="${PREFIX_SOURCES:-pt_rollout it_rollout shared_prediv}"
GPU_LIST="${GPU_LIST:-}"
N_EXAMPLES="${N_EXAMPLES:-600}"
PILOT_EXAMPLES="${PILOT_EXAMPLES:-150}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
K_CANDIDATE_PREFIXES="${K_CANDIDATE_PREFIXES:-8}"
COLLECT_WORKERS_PER_MODEL="${COLLECT_WORKERS_PER_MODEL:-1}"
FACTORIAL_WORKERS_PER_MODEL="${FACTORIAL_WORKERS_PER_MODEL:-1}"
N_BOOT="${N_BOOT:-2000}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

RUN_ROOT="${RUN_ROOT:-results/exp37_random_prefix_baseline/${RUN_NAME}}"
MANIFEST_ROOT="${RUN_ROOT}/manifests"
FACTORIAL_ROOT="${RUN_ROOT}/residual_factorial"
ANALYSIS_DIR="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"

if [[ "$MODE" != "smoke" && "$MODE" != "pilot" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp37] invalid MODE=${MODE}" >&2
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
read -r -a MODEL_ARRAY <<< "$MODELS"

if [[ "$MODE" != "analyze-only" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp37] GPU run requires GPU_LIST or visible CUDA GPUs" >&2
  exit 2
fi

case "$MODE" in
  smoke) N_TOTAL="$SMOKE_EXAMPLES" ;;
  pilot) N_TOTAL="$PILOT_EXAMPLES" ;;
  full) N_TOTAL="$N_EXAMPLES" ;;
  analyze-only) N_TOTAL="$N_EXAMPLES" ;;
esac

mkdir -p "$LOG_DIR"

echo "[exp37] host $(hostname)"
echo "[exp37] mode=${MODE} run_name=${RUN_NAME}"
echo "[exp37] run_root=${RUN_ROOT}"
echo "[exp37] dataset=${DATASET} n_total=${N_TOTAL} max_new_tokens=${MAX_NEW_TOKENS}"
echo "[exp37] models=${MODELS}"
echo "[exp37] arms=${ARMS}"
echo "[exp37] prefix_sources=${PREFIX_SOURCES}"
echo "[exp37] gpus=${GPU_LIST:-<none>}"
echo "[exp37] workers collect=${COLLECT_WORKERS_PER_MODEL} factorial=${FACTORIAL_WORKERS_PER_MODEL}"

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
    echo "[exp37] no gsutil or ADC sync helper; skipping sync to ${GCS_SYNC_DEST}" >&2
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
    echo "[exp37] phase ${phase} failed; see ${LOG_DIR}" >&2
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

arm_source_keys() {
  python - "$ARMS" "$PREFIX_SOURCES" <<'PY'
import sys
arms = set(sys.argv[1].split())
sources = set(sys.argv[2].split())
keys = []
if "first_diff_reference" in arms:
    keys.append("first_diff__reference")
if "random_local_disagreement" in arms:
    if "pt_rollout" in sources:
        keys.append("random_local_disagreement__pt_rollout")
    if "it_rollout" in sources:
        keys.append("random_local_disagreement__it_rollout")
if "prediv_future_pair" in arms and "shared_prediv" in sources:
    keys.append("prediv_future_pair__shared_prediv")
print(" ".join(keys))
PY
}

run_collect() {
  echo "[exp37] phase collect manifests"
  local pids=()
  local idx=0
  for ((w=0; w<COLLECT_WORKERS_PER_MODEL; w++)); do
    for model in "${MODEL_ARRAY[@]}"; do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp37_random_prefix_baseline.collect_prefix_manifests \
        --model "$model" \
        --dataset "$DATASET" \
        --out-root "$MANIFEST_ROOT" \
        --device cuda:0 \
        --worker-index "$w" \
        --n-workers "$COLLECT_WORKERS_PER_MODEL" \
        --n-eval-examples "$N_TOTAL" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --k-candidate-prefixes "$K_CANDIDATE_PREFIXES" \
        --arms $ARMS \
        --prefix-sources $PREFIX_SOURCES \
        >"${LOG_DIR}/collect_${model}_w${w}.log" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        flush_jobs collect "${pids[@]}"
        pids=()
      fi
  done
  done
  flush_jobs collect "${pids[@]}"
  for model in "${MODEL_ARRAY[@]}"; do
    uv run python -m src.poc.exp37_random_prefix_baseline.collect_prefix_manifests \
      --model "$model" \
      --out-root "$MANIFEST_ROOT" \
      --n-workers "$COLLECT_WORKERS_PER_MODEL" \
      --arms $ARMS \
      --prefix-sources $PREFIX_SOURCES \
      --merge-only
  done
  sync_outputs || true
}

run_factorial() {
  echo "[exp37] phase residual factorial"
  local keys
  read -r -a keys <<< "$(arm_source_keys)"
  local pids=()
  local idx=0
  for key in "${keys[@]}"; do
    for ((w=0; w<FACTORIAL_WORKERS_PER_MODEL; w++)); do
      for model in "${MODEL_ARRAY[@]}"; do
        local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
        local out_dir="${FACTORIAL_ROOT}/${key}/raw_shared/${model}"
        CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
          --model "$model" \
          --dataset "$DATASET" \
          --prompt-mode raw_shared \
          --event-kinds first_diff \
          --exp20-root "${MANIFEST_ROOT}/${key}" \
          --out-dir "$out_dir" \
          --device cuda:0 \
          --worker-index "$w" \
          --n-workers "$FACTORIAL_WORKERS_PER_MODEL" \
          --n-eval-examples "$N_TOTAL" \
          --experiment-name exp37_random_prefix_baseline \
          --boundary-mode full_late \
          >"${LOG_DIR}/factorial_${key}_${model}_w${w}.log" 2>&1 &
        pids+=("$!")
        idx=$((idx + 1))
        if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
          flush_jobs factorial "${pids[@]}"
          pids=()
        fi
    done
    done
  done
  flush_jobs factorial "${pids[@]}"
  for key in "${keys[@]}"; do
    for model in "${MODEL_ARRAY[@]}"; do
      uv run python -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
        --model "$model" \
        --out-dir "${FACTORIAL_ROOT}/${key}/raw_shared/${model}" \
        --n-workers "$FACTORIAL_WORKERS_PER_MODEL" \
        --merge-only
    done
  done
  sync_outputs || true
}

run_analysis() {
  echo "[exp37] phase analysis"
  uv run python scripts/analysis/analyze_exp37_random_prefix_baseline.py \
    --run-root "$RUN_ROOT" \
    --out-dir "$ANALYSIS_DIR" \
    --models "${MODEL_ARRAY[@]}" \
    --n-bootstrap "$N_BOOT"
  sync_outputs || true
}

if [[ "$MODE" != "analyze-only" ]]; then
  run_collect
  run_factorial
fi
run_analysis

echo "[exp37] complete run_root=${RUN_ROOT}"
