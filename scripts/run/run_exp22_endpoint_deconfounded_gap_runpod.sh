#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-exp22_endpoint_deconfounded_gap_$(date -u +%Y%m%d_%H%M%S)}"
N_EXAMPLES="${N_EXAMPLES:-600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b}"
VARIANTS="${VARIANTS:-pt it}"
WORKERS_PER_BRANCH="${WORKERS_PER_BRANCH:-1}"
PROBE_FAMILIES="${PROBE_FAMILIES:-raw tuned}"
TUNED_LENS_DIR="${TUNED_LENS_DIR:-/workspace/tuned_lens_probes}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
TOP_K="${TOP_K:-5}"
N_BOOT="${N_BOOT:-2000}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"
export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE

ROOT="results/exp22_endpoint_deconfounded_gap/${RUN_NAME}"
mkdir -p "${ROOT}" logs "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_HUB_CACHE}"

echo "[exp22-runpod] host $(hostname)"
echo "[exp22-runpod] run_name ${RUN_NAME}"
echo "[exp22-runpod] root ${ROOT}"
echo "[exp22-runpod] models ${MODELS}"
echo "[exp22-runpod] variants ${VARIANTS}"
echo "[exp22-runpod] workers_per_branch ${WORKERS_PER_BRANCH}"
echo "[exp22-runpod] n_examples ${N_EXAMPLES}"
echo "[exp22-runpod] max_new_tokens ${MAX_NEW_TOKENS}"
echo "[exp22-runpod] probe_families ${PROBE_FAMILIES}"
echo "[exp22-runpod] tuned_lens_dir ${TUNED_LENS_DIR}"
echo "[exp22-runpod] hf_home ${HF_HOME}"

gpu_count="$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [ "${gpu_count}" -lt 1 ]; then
  echo "[exp22-runpod] expected at least 1 GPU, found ${gpu_count}" >&2
  exit 2
fi
echo "[exp22-runpod] detected ${gpu_count} GPUs"

if [[ " ${PROBE_FAMILIES} " == *" tuned "* ]]; then
  if [ ! -d "${TUNED_LENS_DIR}" ]; then
    echo "[exp22-runpod] tuned probes requested but TUNED_LENS_DIR missing: ${TUNED_LENS_DIR}" >&2
    exit 3
  fi
fi

run_branch_worker() {
  local model="$1"
  local variant="$2"
  local worker="$3"
  local gpu="$4"
  local out_dir="${ROOT}/${model}/${variant}"
  local log_path="logs/${RUN_NAME}.${model}.${variant}.w${worker}.log"
  mkdir -p "${out_dir}"
  echo "[exp22-runpod] launch model=${model} variant=${variant} worker=${worker}/${WORKERS_PER_BRANCH} gpu=${gpu}"
  # shellcheck disable=SC2206
  local probe_args=(${PROBE_FAMILIES})
  CUDA_VISIBLE_DEVICES="${gpu}" uv run python -m src.poc.exp22_endpoint_deconfounded_gap.collect \
    --model "${model}" \
    --variant "${variant}" \
    --dataset "${DATASET}" \
    --out-dir "${out_dir}" \
    --device "${DEVICE_PREFIX}:0" \
    --worker-index "${worker}" \
    --n-workers "${WORKERS_PER_BRANCH}" \
    --n-eval-examples "${N_EXAMPLES}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --probe-families "${probe_args[@]}" \
    --tuned-lens-dir "${TUNED_LENS_DIR}" \
    --top-k "${TOP_K}" \
    > "${log_path}" 2>&1 &
}

merge_branch() {
  local model="$1"
  local variant="$2"
  local out_dir="${ROOT}/${model}/${variant}"
  uv run python -m src.poc.exp22_endpoint_deconfounded_gap.collect \
    --model "${model}" \
    --variant "${variant}" \
    --out-dir "${out_dir}" \
    --n-workers "${WORKERS_PER_BRANCH}" \
    --merge-only \
    > "logs/${RUN_NAME}.${model}.${variant}.merge.log" 2>&1
}

declare -a jobs=()
for model in ${MODELS}; do
  for variant in ${VARIANTS}; do
    for worker in $(seq 0 $((WORKERS_PER_BRANCH - 1))); do
      jobs+=("${model}|${variant}|${worker}")
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

echo "[exp22-runpod] queued ${#jobs[@]} worker jobs"
while [ "${job_index}" -lt "${#jobs[@]}" ] || [ "${#active_pids[@]}" -gt 0 ]; do
  while [ "${#free_gpus[@]}" -gt 0 ] && [ "${job_index}" -lt "${#jobs[@]}" ]; do
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    IFS='|' read -r model variant worker <<< "${jobs[$job_index]}"
    run_branch_worker "${model}" "${variant}" "${worker}" "${gpu}"
    pid="$!"
    active_pids+=("${pid}")
    pid_to_gpu["${pid}"]="${gpu}"
    job_index=$((job_index + 1))
  done

  if [ "${#active_pids[@]}" -gt 0 ]; then
    done_pid=""
    if ! wait -n -p done_pid "${active_pids[@]}"; then
      status=1
      echo "[exp22-runpod] worker pid=${done_pid} failed" >&2
    fi
    if [ -n "${done_pid}" ]; then
      free_gpus+=("${pid_to_gpu[${done_pid}]}")
      unset "pid_to_gpu[${done_pid}]"
      remove_active_pid "${done_pid}"
    fi
  fi
done

if [ "${status}" -ne 0 ]; then
  echo "[exp22-runpod] at least one worker failed" >&2
  exit "${status}"
fi

for model in ${MODELS}; do
  for variant in ${VARIANTS}; do
    merge_branch "${model}" "${variant}"
  done
done

python - <<PY
import gzip, json
from pathlib import Path
root = Path("${ROOT}")
expected = ${N_EXAMPLES}
summary = {"root": str(root), "counts": {}, "malformed": {}, "missing": [], "ok": True}
for model in "${MODELS}".split():
    for variant in "${VARIANTS}".split():
        path = root / model / variant / "records.jsonl.gz"
        key = f"{model}/{variant}"
        if not path.exists():
            summary["missing"].append(str(path))
            continue
        count = 0
        malformed = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    rec = json.loads(line)
                    malformed += int(bool(rec.get("malformed")))
                except Exception:
                    malformed += 1
        summary["counts"][key] = count
        summary["malformed"][key] = malformed
        if count != expected:
            summary["missing"].append(f"{key}: {count}/{expected}")
        if malformed / max(count, 1) > 0.01:
            summary["missing"].append(f"{key}: malformed {malformed}/{count}")
summary["ok"] = not summary["missing"]
(root / "collect_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["ok"]:
    raise SystemExit(4)
PY

uv run python scripts/analysis/analyze_exp22_endpoint_deconfounded_gap.py \
  --root "${ROOT}" \
  --out-dir "${ROOT}/analysis" \
  --n-boot "${N_BOOT}" \
  --fail-on-quality

if [ -n "${GCS_SYNC_DEST}" ]; then
  if command -v gsutil >/dev/null 2>&1; then
    echo "[exp22-runpod] syncing ${ROOT} to ${GCS_SYNC_DEST%/}/${RUN_NAME}"
    gsutil -m rsync -r "${ROOT}" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp22-runpod] GCS_SYNC_DEST set but gsutil not found; skipping sync" >&2
  fi
fi

echo "[exp22-runpod] complete ${ROOT}"

