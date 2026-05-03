#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export GCS_PROJECT="${GCS_PROJECT:-studious-hydra-450206-a5}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
MODE="${MODE:-smoke}"
RUN_NAME="${RUN_NAME:-exp44_middle_terminal_feature_handoff_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp44_middle_terminal_feature_handoff/${RUN_NAME}}"
MODELS="${MODELS:-}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-}"
GCS_DOWNLOAD_WORKERS="${GCS_DOWNLOAD_WORKERS:-12}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp44_middle_terminal_feature_handoff}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
K_LIST="${K_LIST:-20 50 100 200}"
WINDOWS="${WINDOWS:-early mid late_preterminal midlate_preterminal terminal_entry}"
DIRECTIONS="${DIRECTIONS:-rescue degrade}"
CROSSCODER_DTYPE="${CROSSCODER_DTYPE:-bfloat16}"
N_BOOT="${N_BOOT:-2000}"
N_PERM="${N_PERM:-2000}"
PRIMARY_MODELS="${PRIMARY_MODELS:-llama31_8b mistral_7b qwen3_4b}"
PRIMARY_K="${PRIMARY_K:-200}"

case "$MODE" in
  smoke)
    N_PROMPTS="${N_PROMPTS:-2}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    MODELS="${MODELS:-qwen3_4b}"
    K_LIST="${K_LIST:-20}"
    RANDOM_SEEDS="${RANDOM_SEEDS:-0}"
    WINDOWS="${WINDOWS:-mid terminal_entry}"
    DIRECTIONS="${DIRECTIONS:-rescue}"
    N_BOOT="${N_BOOT:-100}"
    N_PERM="${N_PERM:-100}"
    ;;
  pilot)
    N_PROMPTS="${N_PROMPTS:-50}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b}"
    ;;
  full)
    N_PROMPTS="${N_PROMPTS:-600}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b}"
    ;;
  diagnostic-gemma)
    N_PROMPTS="${N_PROMPTS:-600}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
    MODELS="${MODELS:-gemma3_4b}"
    PRIMARY_MODELS="${PRIMARY_MODELS_OVERRIDE:-gemma3_4b}"
    ;;
  pull-only|analyze-only|merge-only)
    N_PROMPTS="${N_PROMPTS:-600}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
    MODELS="${MODELS:-llama31_8b mistral_7b qwen3_4b}"
    ;;
  *)
    echo "[exp44-runpod] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp44-runpod] mode=${MODE}"
echo "[exp44-runpod] run_root=${RUN_ROOT}"
echo "[exp44-runpod] models=${MODELS}"
echo "[exp44-runpod] primary_models=${PRIMARY_MODELS}"
echo "[exp44-runpod] gpu_list=${GPU_LIST}"
echo "[exp44-runpod] workers_per_model=${WORKERS_PER_MODEL} n_prompts=${N_PROMPTS}"
echo "[exp44-runpod] k_list=${K_LIST} seeds=${RANDOM_SEEDS}"
echo "[exp44-runpod] windows=${WINDOWS} directions=${DIRECTIONS}"

download_gcs_prefix() {
  local uri="$1"
  local dest="$2"
  local include_regex="${3:-}"
  if ! $PY_RUNNER - <<'PY'
try:
    import google.cloud.storage  # noqa: F401
except Exception:
    raise SystemExit(1)
PY
  then
    uv pip install google-cloud-storage
  fi
  local -a include_args=()
  if [[ -n "$include_regex" ]]; then
    include_args=(--include-regex "$include_regex")
  fi
  $PY_RUNNER scripts/infra/download_gcs_prefix.py \
    --uri "$uri" \
    --dest "$dest" \
    --max-workers "$GCS_DOWNLOAD_WORKERS" \
    "${include_args[@]}"
}

family_root() {
  case "$1" in
    gemma3_4b)
      printf '%s' "results/exp34_dense5_final_readout_crosscoder/exp34_gemma3_4b_full_20260502_2110_a100x8_bs16/gemma3_4b/selected_d81920_k64"
      ;;
    llama31_8b)
      printf '%s' "results/exp30_final_readout_crosscoder_mediation/exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/selected_d131072_k64"
      ;;
    mistral_7b)
      printf '%s' "results/exp34_dense5_final_readout_crosscoder/exp34_mistral_7b_full_20260502_1124/mistral_7b/selected_d131072_k64"
      ;;
    qwen3_4b)
      printf '%s' "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/exp38_qwen3_4b_final2_d81920_k64_20260503_0451_a100x2/selected_d81920_k64"
      ;;
    *)
      echo "[exp44-runpod] unknown family $1" >&2
      exit 2
      ;;
  esac
}

family_gcs() {
  case "$1" in
    gemma3_4b)
      printf '%s' "gs://pt-vs-it-results/results/exp34_dense5_final_readout_crosscoder/exp34_gemma3_4b_full_20260502_2110_a100x8_bs16/gemma3_4b/selected_d81920_k64"
      ;;
    llama31_8b)
      printf '%s' "gs://pt-vs-it-results/results/exp30_final_readout_crosscoder_mediation/exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/selected_d131072_k64"
      ;;
    mistral_7b)
      printf '%s' "gs://pt-vs-it-results/results/exp34_dense5_final_readout_crosscoder/exp34_mistral_7b_full_20260502_1124/mistral_7b/selected_d131072_k64"
      ;;
    qwen3_4b)
      printf '%s' "gs://pt-vs-it-results/results/exp38_qwen_olmo_final_layer_crosscoder_hardening/exp38_qwen3_4b_final2_d81920_k64_20260503_0451_a100x2/selected_d81920_k64"
      ;;
    *)
      echo "[exp44-runpod] unknown family $1" >&2
      exit 2
      ;;
  esac
}

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUNNER scripts/infra/gcs_sync_adc.py upload \
      "$RUN_ROOT" \
      "${GCS_SYNC_DEST%/}/${RUN_NAME}" \
      --workers 24
  else
    echo "[exp44-runpod] no gsutil or ADC sync helper; skipping sync to ${GCS_SYNC_DEST}" >&2
  fi
}

pull_inputs() {
  local exp20_uri="gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
  local exp20_root="results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
  read -r -a model_arr <<< "$MODELS"
  for model in "${model_arr[@]}"; do
    echo "[exp44-runpod] pull Exp20 manifest ${model}"
    download_gcs_prefix "${exp20_uri}/${PROMPT_MODE}/${model}" "${exp20_root}/${PROMPT_MODE}/${model}" '^exp20_validation_records\.jsonl$'
  done
  for model in "${model_arr[@]}"; do
    local root
    local uri
    root="$(family_root "$model")"
    uri="$(family_gcs "$model")"
    echo "[exp44-runpod] pull crosscoder inputs ${model}"
    mkdir -p "$root"
    download_gcs_prefix "${uri}/dictionaries" "${root}/dictionaries" '(^|/)(crosscoder\.pt|config\.json|crosscoder_sliced_causal\.pt)$'
    download_gcs_prefix "${uri}/feature_stats" "${root}/feature_stats" '^causal_feature_scores\.csv$'
  done
}

run_collect() {
  mkdir -p "${RUN_ROOT}/logs"
  read -r -a model_arr <<< "$MODELS"
  read -r -a gpu_arr <<< "$GPU_LIST"
  local gpu_count="${#gpu_arr[@]}"
  local job_idx=0
  local pids=()
  for model in "${model_arr[@]}"; do
    for ((worker=0; worker<WORKERS_PER_MODEL; worker++)); do
      local gpu="${gpu_arr[$((job_idx % gpu_count))]}"
      local root
      root="$(family_root "$model")"
      local log_path="${RUN_ROOT}/logs/${model}_w${worker}of${WORKERS_PER_MODEL}.log"
      echo "[exp44-runpod] launch model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu} log=${log_path}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        $PY_RUNNER -m src.poc.exp44_middle_terminal_feature_handoff collect \
          --model "$model" \
          --crosscoder-root "$root" \
          --out-dir "${RUN_ROOT}/raw/${model}" \
          --prompt-mode "$PROMPT_MODE" \
          --n-prompts "$N_PROMPTS" \
          --worker-index "$worker" \
          --n-workers "$WORKERS_PER_MODEL" \
          --k-list ${K_LIST} \
          --random-seeds ${RANDOM_SEEDS} \
          --windows ${WINDOWS} \
          --directions ${DIRECTIONS} \
          --crosscoder-dtype "$CROSSCODER_DTYPE" \
          --device "cuda:0"
      ) >"$log_path" 2>&1 &
      pids+=("$!")
      job_idx=$((job_idx + 1))
    done
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" != "0" ]]; then
    echo "[exp44-runpod] at least one collect worker failed" >&2
    sync_outputs || true
    return "$status"
  fi
}

merge_outputs() {
  read -r -a model_arr <<< "$MODELS"
  for model in "${model_arr[@]}"; do
    echo "[exp44-runpod] merge ${model}"
    $PY_RUNNER -m src.poc.exp44_middle_terminal_feature_handoff collect \
      --model "$model" \
      --crosscoder-root "$(family_root "$model")" \
      --out-dir "${RUN_ROOT}/raw/${model}" \
      --n-workers "$WORKERS_PER_MODEL" \
      --merge-only
  done
}

analyze_outputs() {
  echo "[exp44-runpod] analyze"
  $PY_RUNNER -m src.poc.exp44_middle_terminal_feature_handoff analyze \
    --run-root "$RUN_ROOT" \
    --models ${MODELS} \
    --primary-models ${PRIMARY_MODELS} \
    --primary-k "$PRIMARY_K" \
    --n-boot "$N_BOOT" \
    --n-perm "$N_PERM"
}

if [[ "$MODE" != "analyze-only" && "$MODE" != "merge-only" ]]; then
  pull_inputs
fi
if [[ "$MODE" == "pull-only" ]]; then
  exit 0
fi
if [[ "$MODE" != "analyze-only" && "$MODE" != "merge-only" ]]; then
  run_collect
fi
if [[ "$MODE" != "analyze-only" ]]; then
  merge_outputs
fi
analyze_outputs
sync_outputs || true
echo "[exp44-runpod] complete ${RUN_ROOT}"
