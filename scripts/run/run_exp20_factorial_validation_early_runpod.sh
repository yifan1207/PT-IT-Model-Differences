#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-factorial_validation_early_$(date -u +%Y%m%d_%H%M%S)}"
BASELINE_ROOT="${BASELINE_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009}"
N_EXAMPLES="${N_EXAMPLES:-600}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
PROMPT_MODES="${PROMPT_MODES:-raw_shared native}"
MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"

ROOT="results/exp20_divergence_token_counterfactual/${RUN_NAME}"
mkdir -p "${ROOT}" logs

echo "[exp20-validation-early-runpod] host $(hostname)"
echo "[exp20-validation-early-runpod] run_name ${RUN_NAME}"
echo "[exp20-validation-early-runpod] baseline ${BASELINE_ROOT}"
echo "[exp20-validation-early-runpod] modes ${PROMPT_MODES}"
echo "[exp20-validation-early-runpod] models ${MODELS}"
echo "[exp20-validation-early-runpod] workers_per_model ${WORKERS_PER_MODEL}"

run_model_worker() {
  local mode="$1"
  local model="$2"
  local worker="$3"
  local gpu="$4"
  local out_dir="${ROOT}/${mode}/${model}"
  local baseline="${BASELINE_ROOT}/${mode}/${model}/exp20_validation_records.jsonl"
  local log_path="logs/${RUN_NAME}.${mode}.${model}.w${worker}.log"
  mkdir -p "${out_dir}"
  echo "[exp20-validation-early-runpod] launch mode=${mode} model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.poc.exp20_divergence_token_counterfactual.augment_factorial_validation_early \
    --model "${model}" \
    --baseline-jsonl "${baseline}" \
    --dataset "${DATASET}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE_PREFIX}:0" \
    --worker-index "${worker}" \
    --n-workers "${WORKERS_PER_MODEL}" \
    --n-eval-examples "${N_EXAMPLES}" \
    --prompt-mode "${mode}" \
    > "${log_path}" 2>&1 &
}

merge_mode() {
  local mode="$1"
  for model in ${MODELS}; do
    local out_dir="${ROOT}/${mode}/${model}"
    uv run python -m src.poc.exp20_divergence_token_counterfactual.augment_factorial_validation_early \
      --model "${model}" \
      --baseline-jsonl "${BASELINE_ROOT}/${mode}/${model}/exp20_validation_records.jsonl" \
      --out-dir "${out_dir}" \
      --n-workers "${WORKERS_PER_MODEL}" \
      --merge-only \
      > "logs/${RUN_NAME}.${mode}.${model}.merge.log" 2>&1
  done
}

gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [ "${gpu_count}" -lt 1 ]; then
  echo "[exp20-validation-early-runpod] expected at least 1 GPU, found ${gpu_count}" >&2
  exit 2
fi
echo "[exp20-validation-early-runpod] detected ${gpu_count} GPUs"

declare -a jobs=()
for mode in ${PROMPT_MODES}; do
  for model in ${MODELS}; do
    for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
      jobs+=("${mode}|${model}|${worker}")
    done
  done
done

declare -a free_gpus=()
for gpu in $(seq 0 $((gpu_count - 1))); do
  free_gpus+=("${gpu}")
done

declare -a active_pids=()
declare -A pid_to_gpu=()
job_index=0
status=0

remove_active_pid() {
  local done_pid="$1"
  local next=()
  for pid in "${active_pids[@]}"; do
    if [ "${pid}" != "${done_pid}" ]; then
      next+=("${pid}")
    fi
  done
  active_pids=("${next[@]}")
}

echo "[exp20-validation-early-runpod] queued ${#jobs[@]} worker jobs"
while [ "${job_index}" -lt "${#jobs[@]}" ] || [ "${#active_pids[@]}" -gt 0 ]; do
  while [ "${#free_gpus[@]}" -gt 0 ] && [ "${job_index}" -lt "${#jobs[@]}" ]; do
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    IFS='|' read -r mode model worker <<< "${jobs[$job_index]}"
    run_model_worker "${mode}" "${model}" "${worker}" "${gpu}"
    pid="$!"
    active_pids+=("${pid}")
    pid_to_gpu["${pid}"]="${gpu}"
    job_index=$((job_index + 1))
  done

  if [ "${#active_pids[@]}" -gt 0 ]; then
    done_pid=""
    if ! wait -n -p done_pid "${active_pids[@]}"; then
      status=1
      echo "[exp20-validation-early-runpod] worker pid=${done_pid} failed" >&2
    fi
    if [ -n "${done_pid}" ]; then
      free_gpus+=("${pid_to_gpu[${done_pid}]}")
      unset "pid_to_gpu[${done_pid}]"
      remove_active_pid "${done_pid}"
    fi
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "[exp20-validation-early-runpod] at least one worker failed" >&2
  exit "${status}"
fi

for mode in ${PROMPT_MODES}; do
  merge_mode "${mode}"
  echo "[exp20-validation-early-runpod] complete mode=${mode}"
done

python - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
expected = ${N_EXAMPLES}
summary = {"root": str(root), "counts": {}, "missing": [], "ok": True}
for mode in "${PROMPT_MODES}".split():
    for model in "${MODELS}".split():
        path = root / mode / model / "exp20_validation_early_records.jsonl"
        if not path.exists():
            summary["missing"].append(str(path))
            continue
        count = sum(1 for line in path.open("rb") if line.strip())
        summary["counts"][f"{mode}/{model}"] = count
        if count != expected:
            summary["missing"].append(f"{mode}/{model}: {count}/{expected}")
summary["ok"] = not summary["missing"]
(root / "early_collect_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["ok"]:
    raise SystemExit(4)
PY

echo "[exp20-validation-early-runpod] complete ${ROOT}"
