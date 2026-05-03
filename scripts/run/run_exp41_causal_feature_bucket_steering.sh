#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-logit-smoke}"
RUN_NAME="${RUN_NAME:-exp41_terminal_bucket_steering_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-results/exp41_causal_feature_bucket_steering/${RUN_NAME}}"
MANIFEST_DIR="${MANIFEST_DIR:-${RUN_DIR}/bucket_manifest}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
MANIFEST_MODE="${MANIFEST_MODE:-strict_primary}"
ACTIVATION_MODE="${ACTIVATION_MODE:-mediation_topk}"
BATCH_SIZE="${BATCH_SIZE:-4}"
DTYPE="${DTYPE:-bfloat16}"
CROSSCODER_DTYPE="${CROSSCODER_DTYPE:-bfloat16}"
EVENT_KIND="${EVENT_KIND:-first_diff}"
N_EVENTS="${N_EVENTS:-}"
MODELS_STR="${MODELS:-gemma3_4b llama31_8b mistral_7b qwen3_4b}"

if [[ -d /workspace/hf_cache ]]; then
  export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
fi
if [[ -z "${HF_TOKEN:-}" && -f "${HF_HOME:-$HOME/.cache/huggingface}/token" ]]; then
  export HF_TOKEN="$(cat "${HF_HOME:-$HOME/.cache/huggingface}/token")"
fi
if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

mkdir -p "${RUN_DIR}" logs/exp41

run_manifest() {
  uv run python -m src.poc.exp41_causal_feature_bucket_steering manifest \
    --out-dir "${MANIFEST_DIR}"
}

run_analyze() {
  uv run python -m src.poc.exp41_causal_feature_bucket_steering analyze \
    --run-dir "${RUN_DIR}"
}

cuda_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' '
  else
    echo 0
  fi
}

run_logit() {
  local mode="$1"
  local buckets
  local alphas
  if [[ "${mode}" == "logit-full" ]]; then
    buckets=(structure_readout format_control mcq_scaffold surface_punctuation safety_advice_boundary artifact_repetition)
    alphas=(0 0.5 1 1.5 2)
  else
    buckets=(structure_readout format_control surface_punctuation artifact_repetition)
    alphas=(0 1 2)
  fi

  if [[ ! -f "${MANIFEST_DIR}/${MANIFEST_MODE}/bucket_features.csv" ]]; then
    run_manifest
  fi

  local n_gpus
  n_gpus="$(cuda_count)"
  if [[ "${n_gpus}" -le 0 ]]; then
    echo "[exp41] No CUDA GPUs visible; set MODE=manifest or run on a GPU node." >&2
    exit 1
  fi

  local workers_per_model
  workers_per_model="${WORKERS_PER_MODEL:-1}"
  if [[ -z "${WORKERS_PER_MODEL:-}" && "${n_gpus}" -ge 8 ]]; then
    workers_per_model=2
  fi

  read -r -a models <<< "${MODELS_STR}"
  local pids=()
  local gpu="${GPU_OFFSET:-0}"
  for model in "${models[@]}"; do
    for ((worker=0; worker<workers_per_model; worker++)); do
      local device="cuda:$((gpu % n_gpus))"
      local log_path="logs/exp41/${RUN_NAME}_${model}_w${worker}.log"
      echo "[exp41] launch ${model} worker ${worker}/${workers_per_model} on ${device} -> ${log_path}"
      cmd=(
        uv run python -m src.poc.exp41_causal_feature_bucket_steering logit-replay
        --model "${model}"
        --out-dir "${RUN_DIR}"
        --manifest-dir "${MANIFEST_DIR}"
        --manifest-mode "${MANIFEST_MODE}"
        --prompt-mode "${PROMPT_MODE}"
        --event-kind "${EVENT_KIND}"
        --device "${device}"
        --dtype "${DTYPE}"
        --crosscoder-dtype "${CROSSCODER_DTYPE}"
        --activation-mode "${ACTIVATION_MODE}"
        --batch-size "${BATCH_SIZE}"
        --worker-index "${worker}"
        --n-workers "${workers_per_model}"
        --buckets "${buckets[@]}"
        --alphas "${alphas[@]}"
      )
      if [[ -n "${N_EVENTS}" ]]; then
        cmd+=(--n-events "${N_EVENTS}")
      fi
      "${cmd[@]}" > "${log_path}" 2>&1 &
      pids+=("$!")
      gpu=$((gpu + 1))
    done
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "[exp41] At least one logit replay worker failed. Check logs/exp41/${RUN_NAME}_*.log" >&2
    exit 1
  fi
  if [[ "${ANALYZE_AFTER_LOGIT:-1}" != "0" ]]; then
    run_analyze
  fi
}

case "${MODE}" in
  manifest)
    run_manifest
    ;;
  logit-smoke|logit-full)
    run_logit "${MODE}"
    ;;
  analyze-only)
    run_analyze
    ;;
  *)
    echo "Unknown MODE=${MODE}. Use manifest, logit-smoke, logit-full, or analyze-only." >&2
    exit 2
    ;;
esac

if [[ -n "${GCS_DEST:-}" ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${RUN_DIR}" "${GCS_DEST%/}/${RUN_NAME}"
  else
    echo "[exp41] GCS_DEST set but gsutil not found; skipping upload." >&2
  fi
fi

echo "[exp41] done: ${RUN_DIR}"
