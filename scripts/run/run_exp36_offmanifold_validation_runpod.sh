#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"  # smoke|pilot|full|analyze-only
MODEL_LIST="${MODEL_LIST:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
N_PROMPTS="${N_PROMPTS:-600}"
PILOT_PROMPTS="${PILOT_PROMPTS:-150}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-20}"
N_WORKERS_PER_MODEL="${N_WORKERS_PER_MODEL:-2}"
SMOKE_WORKERS_PER_MODEL="${SMOKE_WORKERS_PER_MODEL:-1}"
PILOT_WORKERS_PER_MODEL="${PILOT_WORKERS_PER_MODEL:-1}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
ALPHAS="${ALPHAS:-0.00 0.25 0.50 0.75 1.00}"
N_RANDOM="${N_RANDOM:-3}"
RANDOM_CONTROL="${RANDOM_CONTROL:-signed_permutation}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
POSITION_ROBUSTNESS_MIN_POS="${POSITION_ROBUSTNESS_MIN_POS:-3}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-200}"
PILOT_N_BOOT="${PILOT_N_BOOT:-500}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
EXP23_SUMMARY="${EXP23_SUMMARY:-results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4/analysis/exp23_summary.json}"
RUN_NAME="${RUN_NAME:-exp36_offmanifold_dense5_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp36_offmanifold_validation/${RUN_NAME}}"
PY_RUN="${PY_RUN:-uv run python}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|pilot|full|analyze-only bash scripts/run/run_exp36_offmanifold_validation_runpod.sh

Environment overrides:
  MODEL_LIST, N_PROMPTS, PILOT_PROMPTS, SMOKE_PROMPTS, N_WORKERS_PER_MODEL,
  GPU_LIST, ALPHAS, N_RANDOM, DATASET, EXP20_ROOT, EXP20_FALLBACK_ROOT,
  EXP23_SUMMARY, RUN_NAME, RUN_ROOT, GCS_SYNC_DEST.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "smoke" && "$MODE" != "pilot" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp36] invalid MODE=${MODE}" >&2
  usage
  exit 2
fi
if [[ "$PROMPT_MODE" != "raw_shared" ]]; then
  echo "[exp36] Exp36 is raw_shared-only; got PROMPT_MODE=${PROMPT_MODE}" >&2
  exit 2
fi
if [[ "$EVENT_KINDS" != "first_diff" ]]; then
  echo "[exp36] Exp36 v1 is first_diff-only; got EVENT_KINDS=${EVENT_KINDS}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mkdir -p "${RUN_ROOT}/logs"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py && -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
    $PY_RUN scripts/infra/gcs_sync_adc.py upload \
      "$RUN_ROOT" \
      "${GCS_SYNC_DEST%/}/${RUN_NAME}" \
      --credentials-file "$GOOGLE_APPLICATION_CREDENTIALS" \
      --workers 24
  else
    echo "[exp36] no GCS sync tool configured; skipping ${GCS_SYNC_DEST}" >&2
  fi
}

preflight() {
  test -f "$DATASET"
  for model in $MODEL_LIST; do
    test -f "${EXP20_ROOT}/${PROMPT_MODE}/${model}/exp20_validation_records.jsonl" || \
      test -f "${EXP20_ROOT}/${PROMPT_MODE}/${model}/exp20_records.jsonl" || \
      test -f "${EXP20_FALLBACK_ROOT}/${PROMPT_MODE}/${model}/exp20_validation_records.jsonl" || \
      test -f "${EXP20_FALLBACK_ROOT}/${PROMPT_MODE}/${model}/exp20_records.jsonl"
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
}

collect_worker() {
  local gpu="$1"
  local model="$2"
  local n_examples="$3"
  local worker_index="$4"
  local n_workers="$5"
  local out_dir="${RUN_ROOT}/${PROMPT_MODE}/${model}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp36_offmanifold_validation.collect \
    --model "$model" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-dir "$out_dir" \
    --device cuda:0 \
    --prompt-mode "$PROMPT_MODE" \
    --n-eval-examples "$n_examples" \
    --worker-index "$worker_index" \
    --n-workers "$n_workers" \
    --event-kinds ${EVENT_KINDS} \
    --alphas ${ALPHAS} \
    --n-random "$N_RANDOM" \
    --random-control "$RANDOM_CONTROL" \
    >"${RUN_ROOT}/logs/collect_${model}_w${worker_index}of${n_workers}.log" 2>&1
}

merge_model() {
  local model="$1"
  local n_workers="$2"
  local out_dir="${RUN_ROOT}/${PROMPT_MODE}/${model}"
  $PY_RUN -m src.poc.exp36_offmanifold_validation.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/merge_${model}.log" 2>&1
}

analyze() {
  local n_boot="$1"
  $PY_RUN scripts/analysis/analyze_exp36_offmanifold_validation.py \
    --run-root "$RUN_ROOT" \
    --models ${MODEL_LIST} \
    --prompt-mode "$PROMPT_MODE" \
    --n-bootstrap "$n_boot" \
    --position-min "$POSITION_ROBUSTNESS_MIN_POS" \
    --exp23-summary "$EXP23_SUMMARY" \
    >"${RUN_ROOT}/logs/analyze_exp36.log" 2>&1
  $PY_RUN - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "analysis" / "summary.json").read_text())
for readout in ("common_it", "common_pt"):
    payload = summary["readouts"][readout]
    endpoint = payload["endpoint_interaction"]
    slope = payload["slope"]
    print(
        f"[exp36] {readout} endpoint={endpoint['estimate']:.6f} "
        f"ci=[{endpoint['ci95_low']:.6f},{endpoint['ci95_high']:.6f}] "
        f"slope={slope['estimate']:.6f} ci=[{slope['ci95_low']:.6f},{slope['ci95_high']:.6f}]"
    )
PY
}

preflight

if [[ "$MODE" == "analyze-only" ]]; then
  analyze "$N_BOOT"
  sync_outputs || true
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  N_EXAMPLES="$SMOKE_PROMPTS"
  WORKERS_PER_MODEL="$SMOKE_WORKERS_PER_MODEL"
  BOOT="$SMOKE_N_BOOT"
elif [[ "$MODE" == "pilot" ]]; then
  N_EXAMPLES="$PILOT_PROMPTS"
  WORKERS_PER_MODEL="$PILOT_WORKERS_PER_MODEL"
  BOOT="$PILOT_N_BOOT"
else
  N_EXAMPLES="$N_PROMPTS"
  WORKERS_PER_MODEL="$N_WORKERS_PER_MODEL"
  BOOT="$N_BOOT"
fi

read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp36] GPU_LIST must not be empty" >&2
  exit 2
fi

echo "[exp36] mode=${MODE} run_root=${RUN_ROOT}"
echo "[exp36] models=${MODEL_LIST}"
echo "[exp36] n_examples=${N_EXAMPLES} workers_per_model=${WORKERS_PER_MODEL} gpus=${GPU_LIST}"
echo "[exp36] alphas=${ALPHAS} n_random=${N_RANDOM} random_control=${RANDOM_CONTROL}"

declare -a job_models=()
declare -a job_worker_indices=()
for ((worker=0; worker<WORKERS_PER_MODEL; worker++)); do
  for model in $MODEL_LIST; do
    job_models+=("$model")
    job_worker_indices+=("$worker")
  done
done

status=0
next_idx=0
total_jobs="${#job_models[@]}"
batch_size="${#GPUS[@]}"
declare -a active_pids=()
declare -a active_gpus=()

gpu_is_active() {
  local candidate="$1"
  local active_gpu
  for active_gpu in "${active_gpus[@]:-}"; do
    if [[ "$active_gpu" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

first_free_gpu() {
  local candidate
  for candidate in "${GPUS[@]}"; do
    if ! gpu_is_active "$candidate"; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

drop_finished_pid() {
  local finished_pid="$1"
  local -a kept_pids=()
  local -a kept_gpus=()
  local i
  for i in "${!active_pids[@]}"; do
    if [[ "${active_pids[$i]}" != "$finished_pid" ]]; then
      kept_pids+=("${active_pids[$i]}")
      kept_gpus+=("${active_gpus[$i]}")
    fi
  done
  active_pids=("${kept_pids[@]}")
  active_gpus=("${kept_gpus[@]}")
}

while [[ "$next_idx" -lt "$total_jobs" || "${#active_pids[@]}" -gt 0 ]]; do
  while [[ "${#active_pids[@]}" -lt "$batch_size" && "$next_idx" -lt "$total_jobs" ]]; do
    gpu="$(first_free_gpu)"
    model="${job_models[$next_idx]}"
    worker_index="${job_worker_indices[$next_idx]}"
    echo "[exp36] launch model=${model} worker=${worker_index}/${WORKERS_PER_MODEL} gpu=${gpu}"
    collect_worker "$gpu" "$model" "$N_EXAMPLES" "$worker_index" "$WORKERS_PER_MODEL" &
    active_pids+=("$!")
    active_gpus+=("$gpu")
    next_idx=$((next_idx + 1))
  done
  if [[ "${#active_pids[@]}" -gt 0 ]]; then
    finished_pid=""
    if wait -n -p finished_pid; then
      echo "[exp36] finished worker"
    else
      echo "[exp36] failed worker" >&2
      status=1
    fi
    if [[ -n "$finished_pid" ]]; then
      drop_finished_pid "$finished_pid"
    else
      active_pids=()
      active_gpus=()
    fi
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[exp36] at least one worker failed; inspect ${RUN_ROOT}/logs" >&2
  sync_outputs || true
  exit "$status"
fi

for model in $MODEL_LIST; do
  merge_model "$model" "$WORKERS_PER_MODEL"
done

analyze "$BOOT"
sync_outputs || true
echo "[exp36] ${MODE} complete -> ${RUN_ROOT}"
