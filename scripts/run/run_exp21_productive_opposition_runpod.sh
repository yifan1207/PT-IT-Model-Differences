#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-exp21_productive_opposition_$(date -u +%Y%m%d_%H%M%S)}"
N_EXAMPLES="${N_EXAMPLES:-600}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
PROMPT_MODES="${PROMPT_MODES:-native raw_shared}"
MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b deepseek_v2_lite}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
GPU_LIST="${GPU_LIST:-}"
TOP_K="${TOP_K:-10}"
N_BOOT="${N_BOOT:-2000}"
CONDITIONS="${CONDITIONS:-}"
EVENT_KINDS="${EVENT_KINDS:-}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE

ROOT="results/exp21_productive_opposition/${RUN_NAME}"
mkdir -p "${ROOT}" logs "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

echo "[exp21-runpod] host $(hostname)"
echo "[exp21-runpod] run_name ${RUN_NAME}"
echo "[exp21-runpod] root ${ROOT}"
echo "[exp21-runpod] modes ${PROMPT_MODES}"
echo "[exp21-runpod] models ${MODELS}"
echo "[exp21-runpod] workers_per_model ${WORKERS_PER_MODEL}"
echo "[exp21-runpod] gpu_list ${GPU_LIST:-<all detected>}"
echo "[exp21-runpod] dataset ${DATASET}"
echo "[exp21-runpod] exp20_root ${EXP20_ROOT}"
echo "[exp21-runpod] exp20_fallback_root ${EXP20_FALLBACK_ROOT}"
echo "[exp21-runpod] conditions ${CONDITIONS:-<default all>}"
echo "[exp21-runpod] event_kinds ${EVENT_KINDS:-<default all>}"
echo "[exp21-runpod] hf_home ${HF_HOME}"

run_model_worker() {
  local mode="$1"
  local model="$2"
  local worker="$3"
  local gpu="$4"
  local out_dir="${ROOT}/${mode}/${model}"
  local log_path="logs/${RUN_NAME}.${mode}.${model}.w${worker}.log"
  mkdir -p "${out_dir}"
  echo "[exp21-runpod] launch mode=${mode} model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
  local condition_args=()
  if [ -n "${CONDITIONS}" ]; then
    condition_args+=(--conditions)
    # shellcheck disable=SC2206
    condition_args+=(${CONDITIONS})
  fi
  local event_args=()
  if [ -n "${EVENT_KINDS}" ]; then
    event_args+=(--event-kinds)
    # shellcheck disable=SC2206
    event_args+=(${EVENT_KINDS})
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.poc.exp21_productive_opposition.collect \
    --model "${model}" \
    --dataset "${DATASET}" \
    --exp20-root "${EXP20_ROOT}" \
    --exp20-fallback-root "${EXP20_FALLBACK_ROOT}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE_PREFIX}:0" \
    --worker-index "${worker}" \
    --n-workers "${WORKERS_PER_MODEL}" \
    --n-eval-examples "${N_EXAMPLES}" \
    --prompt-mode "${mode}" \
    --top-k "${TOP_K}" \
    "${condition_args[@]}" \
    "${event_args[@]}" \
    > "${log_path}" 2>&1 &
}

merge_mode() {
  local mode="$1"
  for model in ${MODELS}; do
    local out_dir="${ROOT}/${mode}/${model}"
    uv run python -m src.poc.exp21_productive_opposition.collect \
      --model "${model}" \
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
  echo "[exp21-runpod] expected at least 1 GPU, found ${gpu_count}" >&2
  exit 2
fi
echo "[exp21-runpod] detected ${gpu_count} GPUs"

declare -a jobs=()
for mode in ${PROMPT_MODES}; do
  for model in ${MODELS}; do
    for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
      jobs+=("${mode}|${model}|${worker}")
    done
  done
done

declare -a free_gpus=()
if [ -n "${GPU_LIST}" ]; then
  # shellcheck disable=SC2206
  free_gpus=(${GPU_LIST})
  for gpu in "${free_gpus[@]}"; do
    if [ "${gpu}" -lt 0 ] || [ "${gpu}" -ge "${gpu_count}" ]; then
      echo "[exp21-runpod] requested GPU ${gpu}, but detected GPUs are 0..$((gpu_count - 1))" >&2
      exit 2
    fi
  done
else
  for gpu in $(seq 0 $((gpu_count - 1))); do
    free_gpus+=("${gpu}")
  done
fi

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

echo "[exp21-runpod] queued ${#jobs[@]} worker jobs"
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
      echo "[exp21-runpod] worker pid=${done_pid} failed" >&2
    fi
    if [ -n "${done_pid}" ]; then
      free_gpus+=("${pid_to_gpu[${done_pid}]}")
      unset "pid_to_gpu[${done_pid}]"
      remove_active_pid "${done_pid}"
    fi
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "[exp21-runpod] at least one worker failed" >&2
  exit "${status}"
fi

for mode in ${PROMPT_MODES}; do
  merge_mode "${mode}"
  echo "[exp21-runpod] complete mode=${mode}"
done

python - <<PY
import gzip, json
from pathlib import Path
root = Path("${ROOT}")
expected = ${N_EXAMPLES}
summary = {"root": str(root), "counts": {}, "missing": [], "ok": True}
for mode in "${PROMPT_MODES}".split():
    for model in "${MODELS}".split():
        path = root / mode / model / "records.jsonl.gz"
        if not path.exists():
            summary["missing"].append(str(path))
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        summary["counts"][f"{mode}/{model}"] = count
        if count != expected:
            summary["missing"].append(f"{mode}/{model}: {count}/{expected}")
summary["ok"] = not summary["missing"]
(root / "collect_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["ok"]:
    raise SystemExit(4)
PY

uv run python scripts/analysis/analyze_exp21_productive_opposition.py \
  --root "${ROOT}" \
  --out-dir "${ROOT}/analysis" \
  --models ${MODELS} \
  --n-boot "${N_BOOT}"

echo "[exp21-runpod] complete ${ROOT}"
