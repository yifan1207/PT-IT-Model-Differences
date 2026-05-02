#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
MODEL_LIST="${MODEL_LIST:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
BOUNDARY_MODES="${BOUNDARY_MODES:-last3 last1}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
N_PROMPTS="${N_PROMPTS:-600}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-20}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SMOKE_GPU_LIST="${SMOKE_GPU_LIST:-${GPU_LIST}}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-200}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
FULL_LATE_ROOT="${FULL_LATE_ROOT:-results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4}"
RUN_NAME="${RUN_NAME:-exp31_terminal_depth_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp31_terminal_depth_factorial/${RUN_NAME}}"
PY_RUN="${PY_RUN:-uv run python}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|analyze-only bash scripts/run/run_exp31_terminal_depth_factorial_runpod.sh

Environment overrides:
  MODEL_LIST, BOUNDARY_MODES, GPU_LIST, SMOKE_GPU_LIST, N_PROMPTS,
  SMOKE_PROMPTS, N_BOOT, SMOKE_N_BOOT, DATASET, EXP20_ROOT,
  EXP20_FALLBACK_ROOT, FULL_LATE_ROOT, RUN_NAME, RUN_ROOT, PY_RUN.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "Invalid MODE=${MODE}" >&2
  usage
  exit 2
fi
if [[ "$PROMPT_MODE" != "raw_shared" ]]; then
  echo "Exp31 is intentionally raw_shared-only; got PROMPT_MODE=${PROMPT_MODE}" >&2
  exit 2
fi
if [[ "$EVENT_KINDS" != "first_diff" ]]; then
  echo "Exp31 is intentionally first_diff-only; got EVENT_KINDS=${EVENT_KINDS}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "${RUN_ROOT}/logs"

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

resolve_boundary_layer() {
  local model="$1"
  local boundary_mode="$2"
  $PY_RUN - "$model" "$boundary_mode" <<'PY'
import sys
from src.poc.cross_model.config import get_spec
from src.poc.exp20_divergence_token_counterfactual.collect import DEPTH_ABLATION_WINDOWS

model = sys.argv[1]
mode = sys.argv[2]
spec = get_spec(model)
if mode == "full_late":
    print(int(DEPTH_ABLATION_WINDOWS[model]["late"][0]))
elif mode == "last3":
    print(int(spec.n_layers - 3))
elif mode == "last1":
    print(int(spec.n_layers - 1))
else:
    raise SystemExit(f"unknown boundary mode: {mode}")
PY
}

collect_worker() {
  local gpu="$1"
  local model="$2"
  local boundary_mode="$3"
  local n_examples="$4"
  local worker_index="$5"
  local n_workers="$6"
  local include_noop="$7"
  local boundary_layer
  boundary_layer="$(resolve_boundary_layer "$model" "$boundary_mode")"
  local out_dir="${RUN_ROOT}/residual_factorial/${boundary_mode}/${PROMPT_MODE}/${model}"
  mkdir -p "$out_dir"
  local extra=()
  if [[ "$include_noop" -eq 0 ]]; then
    extra+=(--no-noop-patch)
  fi
  if [[ "$boundary_mode" != "full_late" ]]; then
    extra+=(--boundary-layer-override "$boundary_layer")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
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
    --experiment-name exp31_terminal_depth_factorial \
    --boundary-mode "$boundary_mode" \
    "${extra[@]}" \
    >"${RUN_ROOT}/logs/residual_${boundary_mode}_${model}_w${worker_index}of${n_workers}.log" 2>&1
}

merge_boundary_model() {
  local model="$1"
  local boundary_mode="$2"
  local n_workers="$3"
  local out_dir="${RUN_ROOT}/residual_factorial/${boundary_mode}/${PROMPT_MODE}/${model}"
  $PY_RUN -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
    --model "$model" \
    --out-dir "$out_dir" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/merge_${boundary_mode}_${model}.log" 2>&1
}

analyze() {
  local n_boot="$1"
  $PY_RUN scripts/analysis/analyze_exp31_terminal_depth_factorial.py \
    --run-root "$RUN_ROOT" \
    --full-late-root "$FULL_LATE_ROOT" \
    --models ${MODEL_LIST} \
    --boundary-modes ${BOUNDARY_MODES} \
    --prompt-mode "$PROMPT_MODE" \
    --readouts common_it common_pt \
    --n-bootstrap "$n_boot" \
    >"${RUN_ROOT}/logs/analyze_exp31.log" 2>&1
  $PY_RUN - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "analysis" / "terminal_depth_summary.json").read_text())
for boundary in summary["boundary_modes"]:
    for readout in summary["readouts"]:
        row = summary["summary_rows"].get("Dense-5 family mean", {}).get(boundary, {}).get(readout, {}).get("all_positions")
        if row:
            print(
                f"[exp31] {boundary} {readout} interaction={row['interaction']:.6f} "
                f"ci=[{row['interaction_ci_low']:.6f},{row['interaction_ci_high']:.6f}] "
                f"retention={row['retention_fraction']:.4f}"
            )
PY
}

preflight

if [[ "$MODE" == "analyze-only" ]]; then
  analyze "$N_BOOT"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  N_EXAMPLES="$SMOKE_PROMPTS"
  declare -a gpus=($SMOKE_GPU_LIST)
  include_noop=1
  n_boot="$SMOKE_N_BOOT"
  echo "[exp31] smoke root=${RUN_ROOT} models=${MODEL_LIST} boundaries=${BOUNDARY_MODES} gpus=${SMOKE_GPU_LIST}"
else
  N_EXAMPLES="$N_PROMPTS"
  declare -a gpus=($GPU_LIST)
  include_noop=0
  n_boot="$N_BOOT"
  echo "[exp31] full root=${RUN_ROOT} models=${MODEL_LIST} boundaries=${BOUNDARY_MODES} gpus=${GPU_LIST}"
fi

if [[ "${#gpus[@]}" -lt 1 ]]; then
  echo "GPU list must not be empty" >&2
  exit 2
fi

declare -a job_models=()
declare -a job_modes=()
for boundary_mode in $BOUNDARY_MODES; do
  for model in $MODEL_LIST; do
    job_models+=("$model")
    job_modes+=("$boundary_mode")
  done
done

status=0
next_idx=0
total_jobs="${#job_models[@]}"
batch_size="${#gpus[@]}"
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
  for candidate in "${gpus[@]}"; do
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
    boundary_mode="${job_modes[$next_idx]}"
    echo "[exp31] launch boundary=${boundary_mode} model=${model} gpu=${gpu}"
    collect_worker "$gpu" "$model" "$boundary_mode" "$N_EXAMPLES" 0 1 "$include_noop" &
    active_pids+=("$!")
    active_gpus+=("$gpu")
    next_idx=$((next_idx + 1))
  done
  if [[ "${#active_pids[@]}" -gt 0 ]]; then
    finished_pid=""
    if wait -n -p finished_pid; then
      echo "[exp31] finished worker"
    else
      echo "[exp31] failed worker" >&2
      status=1
    fi
    if [[ -n "$finished_pid" ]]; then
      drop_finished_pid "$finished_pid"
    else
      # Defensive fallback for shells where wait -p cannot identify the child.
      active_pids=()
      active_gpus=()
    fi
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[exp31] at least one worker failed; inspect ${RUN_ROOT}/logs" >&2
  exit "$status"
fi

for boundary_mode in $BOUNDARY_MODES; do
  for model in $MODEL_LIST; do
    merge_boundary_model "$model" "$boundary_mode" 1
  done
done

analyze "$n_boot"
echo "[exp31] ${MODE} complete -> ${RUN_ROOT}"
