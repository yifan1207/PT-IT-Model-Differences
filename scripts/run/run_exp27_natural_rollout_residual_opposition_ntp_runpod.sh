#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-pilot}"  # smoke|pilot|full|analyze-only
RUN_NAME="${RUN_NAME:-exp27_natural_rollout_residopp_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp27_natural_rollout_residual_opposition_ntp/${RUN_NAME}}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
GPU_LIST="${GPU_LIST:-}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
N_BOOT="${N_BOOT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-1024}"
INCLUDE_BOUNDARY_SOURCE="${INCLUDE_BOUNDARY_SOURCE:-0}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"
PY_RUNNER="${PY_RUNNER:-uv run python}"

SMOKE_MODEL="${SMOKE_MODEL:-qwen3_4b}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-5}"
SMOKE_VARIANTS="${SMOKE_VARIANTS:-full noopp normpres_noopp flipopp randorth randremove randremove_resnorm}"
SMOKE_RAND_SEEDS="${SMOKE_RAND_SEEDS:-0}"

PILOT_MODELS="${PILOT_MODELS:-gemma3_4b llama31_8b olmo2_7b}"
PILOT_EXAMPLES="${PILOT_EXAMPLES:-200}"
PILOT_VARIANTS="${PILOT_VARIANTS:-full noopp normpres_noopp flipopp randorth randremove randremove_resnorm}"
PILOT_RAND_SEEDS="${PILOT_RAND_SEEDS:-0}"

FULL_MODELS="${FULL_MODELS:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
FULL_EXAMPLES="${FULL_EXAMPLES:-600}"
FULL_VARIANTS="${FULL_VARIANTS:-full noopp normpres_noopp flipopp randorth randremove randremove_resnorm}"
FULL_RAND_SEEDS="${FULL_RAND_SEEDS:-0 1 2}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|pilot|full|analyze-only bash scripts/run/run_exp27_natural_rollout_residual_opposition_ntp_runpod.sh

Important env vars:
  RUN_NAME RUN_ROOT DATASET GPU_LIST="0 1 ..." WORKERS_PER_MODEL=1 BATCH_SIZE=2
  INCLUDE_BOUNDARY_SOURCE=0|1 GCS_SYNC_DEST=gs://...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "smoke" && "$MODE" != "pilot" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp27] invalid MODE=${MODE}" >&2
  usage
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
if [[ "$MODE" != "analyze-only" && "$gpu_count" -lt 1 ]]; then
  echo "[exp27] GPU mode requires at least one CUDA device" >&2
  exit 2
fi

case "$MODE" in
  smoke)
    MODELS="$SMOKE_MODEL"
    N_EXAMPLES="$SMOKE_EXAMPLES"
    VARIANTS="$SMOKE_VARIANTS"
    RAND_SEEDS="$SMOKE_RAND_SEEDS"
    ;;
  pilot)
    MODELS="$PILOT_MODELS"
    N_EXAMPLES="$PILOT_EXAMPLES"
    VARIANTS="$PILOT_VARIANTS"
    RAND_SEEDS="$PILOT_RAND_SEEDS"
    ;;
  full|analyze-only)
    MODELS="$FULL_MODELS"
    N_EXAMPLES="$FULL_EXAMPLES"
    VARIANTS="$FULL_VARIANTS"
    RAND_SEEDS="$FULL_RAND_SEEDS"
    ;;
esac

mkdir -p "$RUN_ROOT/logs"

echo "[exp27] host $(hostname)"
echo "[exp27] mode ${MODE}"
echo "[exp27] run_name ${RUN_NAME}"
echo "[exp27] run_root ${RUN_ROOT}"
echo "[exp27] models ${MODELS}"
echo "[exp27] variants ${VARIANTS}"
echo "[exp27] rand_seeds ${RAND_SEEDS}"
echo "[exp27] gpu_count ${gpu_count} gpu_list ${GPU_LIST:-<none>} slots_per_gpu ${SLOTS_PER_GPU}"
echo "[exp27] batch_size ${BATCH_SIZE} max_new_tokens ${MAX_NEW_TOKENS} include_boundary_source ${INCLUDE_BOUNDARY_SOURCE}"
echo "[exp27] py_runner ${PY_RUNNER}"

declare -a gpu_pool=()
if [[ -n "$GPU_LIST" ]]; then
  read -r -a requested_gpus <<< "$GPU_LIST"
else
  requested_gpus=()
fi
for gpu in "${requested_gpus[@]}"; do
  for _slot in $(seq 1 "$SLOTS_PER_GPU"); do
    gpu_pool+=("$gpu")
  done
done
if [[ "${#gpu_pool[@]}" -lt 1 && "$MODE" != "analyze-only" ]]; then
  echo "[exp27] no GPUs in pool" >&2
  exit 2
fi

boundary_args=()
if [[ "$INCLUDE_BOUNDARY_SOURCE" == "1" ]]; then
  boundary_args+=(--include-boundary-source)
fi

run_collect_one() {
  local gpu="$1"
  local model="$2"
  local worker="$3"
  local out_dir="${RUN_ROOT}/records/${model}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp27_natural_rollout_residual_opposition_ntp \
    --model "$model" \
    --dataset "$DATASET" \
    --out-dir "$out_dir" \
    --device cuda:0 \
    --model-variants pt it \
    --variants ${VARIANTS} \
    --rand-seeds ${RAND_SEEDS} \
    --worker-index "$worker" \
    --n-workers "$WORKERS_PER_MODEL" \
    --n-eval-examples "$N_EXAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    "${boundary_args[@]}" \
    >"${RUN_ROOT}/logs/collect_${model}_w${worker}of${WORKERS_PER_MODEL}.log" 2>&1
}

merge_model() {
  local model="$1"
  local out_dir="${RUN_ROOT}/records/${model}"
  $PY_RUNNER -m src.poc.exp27_natural_rollout_residual_opposition_ntp.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --merge-only \
    --model-variants pt it \
    --n-workers "$WORKERS_PER_MODEL" \
    >"${RUN_ROOT}/logs/merge_${model}.log" 2>&1
}

run_jobs() {
  local -a jobs=("$@")
  local -a free=("${gpu_pool[@]}")
  local -a active_pids=()
  declare -A pid_to_gpu=()
  local job_index=0
  local status=0
  while [[ "$job_index" -lt "${#jobs[@]}" || "${#active_pids[@]}" -gt 0 ]]; do
    while [[ "${#free[@]}" -gt 0 && "$job_index" -lt "${#jobs[@]}" ]]; do
      local gpu="${free[0]}"
      free=("${free[@]:1}")
      IFS='|' read -r model worker <<< "${jobs[$job_index]}"
      echo "[exp27] launch collect model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
      run_collect_one "$gpu" "$model" "$worker" &
      local pid="$!"
      active_pids+=("$pid")
      pid_to_gpu["$pid"]="$gpu"
      job_index=$((job_index + 1))
    done
    if [[ "${#active_pids[@]}" -gt 0 ]]; then
      local done_pid=""
      if ! wait -n -p done_pid "${active_pids[@]}"; then
        status=1
        echo "[exp27] collect worker pid=${done_pid} failed" >&2
      fi
      if [[ -n "$done_pid" ]]; then
        free+=("${pid_to_gpu[$done_pid]}")
        unset "pid_to_gpu[$done_pid]"
        local -a next=()
        for pid in "${active_pids[@]}"; do
          [[ "$pid" != "$done_pid" ]] && next+=("$pid")
        done
        active_pids=("${next[@]}")
      fi
    fi
  done
  return "$status"
}

run_collect_all() {
  local -a jobs=()
  for model in ${MODELS}; do
    for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
      jobs+=("${model}|${worker}")
    done
  done
  run_jobs "${jobs[@]}"
  for model in ${MODELS}; do
    merge_model "$model"
  done
}

run_analyze() {
  $PY_RUNNER scripts/analysis/analyze_exp27_natural_rollout_residual_opposition_ntp.py \
    --exp27-root "$RUN_ROOT" \
    --out-dir "${RUN_ROOT}/analysis" \
    --models ${MODELS} \
    --n-boot "$N_BOOT"
}

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp27] GCS_SYNC_DEST set but gsutil not found; skipping sync" >&2
  fi
}

case "$MODE" in
  smoke|pilot|full)
    run_collect_all
    run_analyze
    sync_outputs
    ;;
  analyze-only)
    run_analyze
    sync_outputs
    ;;
esac

echo "[exp27] complete run_name=${RUN_NAME}"
