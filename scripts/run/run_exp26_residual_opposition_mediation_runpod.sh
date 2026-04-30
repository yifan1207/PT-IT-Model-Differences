#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-pilot}"  # smoke|pilot|full|analyze-only|calibrate-only
RUN_NAME="${RUN_NAME:-exp26_residopp_mediation_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp26_residual_opposition_mediation/${RUN_NAME}}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
EXP23_ROOT="${EXP23_ROOT:-results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
LATE_TARGET="${LATE_TARGET:-it}"  # it|pt
GPU_LIST="${GPU_LIST:-}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-1}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
N_BOOT="${N_BOOT:-2000}"
N_CALIBRATION_EXAMPLES="${N_CALIBRATION_EXAMPLES:-200}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"
PY_RUNNER="${PY_RUNNER:-uv run python}"

SMOKE_MODEL="${SMOKE_MODEL:-qwen3_4b}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-8}"
PILOT_MODELS="${PILOT_MODELS:-gemma3_4b llama31_8b olmo2_7b}"
PILOT_EXAMPLES="${PILOT_EXAMPLES:-200}"
FULL_MODELS="${FULL_MODELS:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
FULL_EXAMPLES="${FULL_EXAMPLES:-600}"

if [[ "$LATE_TARGET" == "pt" ]]; then
  SMOKE_VARIANTS="${SMOKE_VARIANTS:-opp_scale_0p5 noopp flipopp normpres_noopp randorth}"
  PILOT_VARIANTS="${PILOT_VARIANTS:-opp_scale_0p5 noopp flipopp normpres_noopp randorth}"
  FULL_VARIANTS="${FULL_VARIANTS:-opp_scale_0p5 noopp flipopp normpres_noopp randorth}"
else
  SMOKE_VARIANTS="${SMOKE_VARIANTS:-opp_scale_0p5 noopp flipopp normpres_noopp randorth}"
  PILOT_VARIANTS="${PILOT_VARIANTS:-opp_scale_0p5 noopp flipopp normpres_noopp randorth}"
  FULL_VARIANTS="${FULL_VARIANTS:-opp_scale_0p5 noopp flipopp normpres_noopp ptlevel_opp randorth}"
fi
SMOKE_RAND_SEEDS="${SMOKE_RAND_SEEDS:-0}"
PILOT_RAND_SEEDS="${PILOT_RAND_SEEDS:-0}"
FULL_RAND_SEEDS="${FULL_RAND_SEEDS:-0 1 2}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|pilot|full|analyze-only|calibrate-only bash scripts/run/run_exp26_residual_opposition_mediation_runpod.sh

Important env vars:
  RUN_NAME RUN_ROOT EXP20_ROOT EXP23_ROOT DATASET GPU_LIST="0 1 ..."
  WORKERS_PER_MODEL=1 SLOTS_PER_GPU=1 N_BOOT=2000 GCS_SYNC_DEST=gs://...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "smoke" && "$MODE" != "pilot" && "$MODE" != "full" && "$MODE" != "analyze-only" && "$MODE" != "calibrate-only" ]]; then
  echo "[exp26] invalid MODE=${MODE}" >&2
  usage
  exit 2
fi
if [[ "$LATE_TARGET" != "it" && "$LATE_TARGET" != "pt" ]]; then
  echo "[exp26] invalid LATE_TARGET=${LATE_TARGET}; expected it or pt" >&2
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
  echo "[exp26] GPU mode requires at least one CUDA device" >&2
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
  full|analyze-only|calibrate-only)
    MODELS="$FULL_MODELS"
    N_EXAMPLES="$FULL_EXAMPLES"
    VARIANTS="$FULL_VARIANTS"
    RAND_SEEDS="$FULL_RAND_SEEDS"
    ;;
esac

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/ptlevel_alpha"

echo "[exp26] host $(hostname)"
echo "[exp26] mode ${MODE}"
echo "[exp26] run_name ${RUN_NAME}"
echo "[exp26] run_root ${RUN_ROOT}"
echo "[exp26] models ${MODELS}"
echo "[exp26] variants ${VARIANTS}"
echo "[exp26] late_target ${LATE_TARGET}"
echo "[exp26] rand_seeds ${RAND_SEEDS}"
echo "[exp26] gpu_count ${gpu_count} gpu_list ${GPU_LIST:-<none>} slots_per_gpu ${SLOTS_PER_GPU}"
echo "[exp26] py_runner ${PY_RUNNER}"

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
  echo "[exp26] no GPUs in pool" >&2
  exit 2
fi

run_calibration_one() {
  local gpu="$1"
  local model="$2"
  local out_path="${RUN_ROOT}/ptlevel_alpha/${model}.json"
  if [[ -s "$out_path" ]]; then
    echo "[exp26] skip calibrate model=${model} existing=${out_path}"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp26_residual_opposition_mediation.calibrate \
    --model "$model" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-path "$out_path" \
    --device cuda:0 \
    --prompt-mode "$PROMPT_MODE" \
    --event-kind first_diff \
    --n-eval-examples "$N_CALIBRATION_EXAMPLES" \
    >"${RUN_ROOT}/logs/calibrate_${model}.log" 2>&1
}

run_collect_one() {
  local gpu="$1"
  local model="$2"
  local worker="$3"
  local out_dir="${RUN_ROOT}/records/${PROMPT_MODE}/${model}"
  local alpha_path="${RUN_ROOT}/ptlevel_alpha/${model}.json"
  local alpha_args=()
  if [[ -f "$alpha_path" ]]; then
    alpha_args+=(--ptlevel-alpha-path "$alpha_path")
  fi
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp26_residual_opposition_mediation \
    --model "$model" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-dir "$out_dir" \
    --device cuda:0 \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    --late-target "$LATE_TARGET" \
    --variants ${VARIANTS} \
    --rand-seeds ${RAND_SEEDS} \
    --worker-index "$worker" \
    --n-workers "$WORKERS_PER_MODEL" \
    --n-eval-examples "$N_EXAMPLES" \
    "${alpha_args[@]}" \
    >"${RUN_ROOT}/logs/collect_${PROMPT_MODE}_${model}_w${worker}of${WORKERS_PER_MODEL}.log" 2>&1
}

merge_model() {
  local model="$1"
  local out_dir="${RUN_ROOT}/records/${PROMPT_MODE}/${model}"
  $PY_RUNNER -m src.poc.exp26_residual_opposition_mediation.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --merge-only \
    --n-workers "$WORKERS_PER_MODEL" \
    >"${RUN_ROOT}/logs/merge_${PROMPT_MODE}_${model}.log" 2>&1
}

run_jobs() {
  local kind="$1"
  shift
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
      IFS='|' read -r a b <<< "${jobs[$job_index]}"
      if [[ "$kind" == "calibrate" ]]; then
        echo "[exp26] launch calibrate model=${a} gpu=${gpu}"
        run_calibration_one "$gpu" "$a" &
      else
        echo "[exp26] launch collect model=${a} worker=${b}/${WORKERS_PER_MODEL} gpu=${gpu}"
        run_collect_one "$gpu" "$a" "$b" &
      fi
      local pid="$!"
      active_pids+=("$pid")
      pid_to_gpu["$pid"]="$gpu"
      job_index=$((job_index + 1))
    done
    if [[ "${#active_pids[@]}" -gt 0 ]]; then
      local done_pid=""
      if ! wait -n -p done_pid "${active_pids[@]}"; then
        status=1
        echo "[exp26] ${kind} worker pid=${done_pid} failed" >&2
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

run_calibration_all() {
  if [[ " ${VARIANTS} " != *" ptlevel_opp "* ]]; then
    return
  fi
  local -a jobs=()
  for model in ${MODELS}; do
    jobs+=("${model}|0")
  done
  run_jobs calibrate "${jobs[@]}"
}

run_collect_all() {
  local -a jobs=()
  for model in ${MODELS}; do
    for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
      jobs+=("${model}|${worker}")
    done
  done
  run_jobs collect "${jobs[@]}"
  for model in ${MODELS}; do
    merge_model "$model"
  done
}

run_analyze() {
  $PY_RUNNER scripts/analysis/analyze_exp26_residual_opposition_mediation.py \
    --exp26-root "$RUN_ROOT" \
    --exp23-root "$EXP23_ROOT" \
    --out-dir "${RUN_ROOT}/analysis" \
    --models ${MODELS} \
    --prompt-mode "$PROMPT_MODE" \
    --event-kind first_diff \
    --late-target "$LATE_TARGET" \
    --n-boot "$N_BOOT"
}

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp26] GCS_SYNC_DEST set but gsutil not found; skipping sync" >&2
  fi
}

case "$MODE" in
  smoke|pilot|full)
    run_calibration_all
    run_collect_all
    run_analyze
    sync_outputs
    ;;
  calibrate-only)
    run_calibration_all
    sync_outputs
    ;;
  analyze-only)
    run_analyze
    sync_outputs
    ;;
esac

echo "[exp26] complete run_name=${RUN_NAME}"
