#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-full_runpod_$(date -u +%Y%m%d_%H%M%S)}"
N_EXAMPLES="${N_EXAMPLES:-600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
PROMPT_MODES="${PROMPT_MODES:-raw_shared native}"
MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b deepseek_v2_lite}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"

ROOT="results/exp20_divergence_token_counterfactual/${RUN_NAME}"
mkdir -p "${ROOT}" logs

echo "[exp20-runpod] host $(hostname)"
echo "[exp20-runpod] run_name ${RUN_NAME}"
echo "[exp20-runpod] modes ${PROMPT_MODES}"
echo "[exp20-runpod] models ${MODELS}"
echo "[exp20-runpod] workers_per_model ${WORKERS_PER_MODEL}"
echo "[exp20-runpod] n_examples ${N_EXAMPLES} max_new_tokens ${MAX_NEW_TOKENS}"
echo "[exp20-runpod] dataset ${DATASET}"

run_model_worker() {
  local mode="$1"
  local model="$2"
  local worker="$3"
  local gpu="$4"
  local out_dir="${ROOT}/${mode}/${model}"
  local log_path="logs/${RUN_NAME}.${mode}.${model}.w${worker}.log"
  mkdir -p "${out_dir}"
  echo "[exp20-runpod] launch mode=${mode} model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.poc.exp20_divergence_token_counterfactual.collect \
    --model "${model}" \
    --dataset "${DATASET}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE_PREFIX}:0" \
    --worker-index "${worker}" \
    --n-workers "${WORKERS_PER_MODEL}" \
    --n-eval-examples "${N_EXAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt-mode "${mode}" \
    > "${log_path}" 2>&1 &
}

merge_mode() {
  local mode="$1"
  for model in ${MODELS}; do
    local out_dir="${ROOT}/${mode}/${model}"
    uv run python -m src.poc.exp20_divergence_token_counterfactual.collect \
      --model "${model}" \
      --out-dir "${out_dir}" \
      --n-workers "${WORKERS_PER_MODEL}" \
      --merge-only \
      > "logs/${RUN_NAME}.${mode}.${model}.merge.log" 2>&1
  done
  uv run python scripts/analysis/analyze_exp20.py \
    --run-dir "${ROOT}/${mode}" \
    --out "${ROOT}/${mode}/summary.json" \
    > "logs/${RUN_NAME}.${mode}.analysis.log" 2>&1
  uv run python scripts/plot/plot_exp20.py \
    --summary "${ROOT}/${mode}/summary.json" \
    --pool all6 \
    --out "${ROOT}/${mode}/exp20_all6_overview.png" \
    > "logs/${RUN_NAME}.${mode}.plot_all6.log" 2>&1
  uv run python scripts/plot/plot_exp20.py \
    --summary "${ROOT}/${mode}/summary.json" \
    --pool dense5 \
    --out "${ROOT}/${mode}/exp20_dense5_overview.png" \
    > "logs/${RUN_NAME}.${mode}.plot_dense5.log" 2>&1
}

check_mode_quality() {
  local mode="$1"
  python - <<PY
import json
from pathlib import Path
root = Path("${ROOT}") / "${mode}"
summary = json.loads((root / "summary.json").read_text())
print(json.dumps({
    "mode": "${mode}",
    "quality": summary["quality"],
    "records_all6": summary["pooled"]["all6"]["n_records"],
    "records_dense5": summary["pooled"]["dense5"]["n_records"],
    "first_diff": summary["pooled"]["all6"].get("divergence", {}).get("first_diff", {}),
}, indent=2, sort_keys=True))
if not summary["quality"].get("ok"):
    raise SystemExit(3)
PY
}

gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [ "${gpu_count}" -lt 1 ]; then
  echo "[exp20-runpod] expected at least 1 GPU, found ${gpu_count}" >&2
  exit 2
fi
echo "[exp20-runpod] detected ${gpu_count} GPUs"

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

echo "[exp20-runpod] queued ${#jobs[@]} worker jobs"
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
      echo "[exp20-runpod] worker pid=${done_pid} failed" >&2
    fi
    if [ -n "${done_pid}" ]; then
      free_gpus+=("${pid_to_gpu[${done_pid}]}")
      unset "pid_to_gpu[${done_pid}]"
      remove_active_pid "${done_pid}"
    fi
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "[exp20-runpod] at least one worker failed" >&2
  exit "${status}"
fi

for mode in ${PROMPT_MODES}; do
  merge_mode "${mode}"
  check_mode_quality "${mode}"
  echo "[exp20-runpod] complete mode=${mode}"
done

echo "[exp20-runpod] complete ${ROOT}"
