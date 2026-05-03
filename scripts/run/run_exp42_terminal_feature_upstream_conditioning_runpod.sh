#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export GCS_PROJECT="${GCS_PROJECT:-studious-hydra-450206-a5}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
MODE="${MODE:-full}"
RUN_NAME="${RUN_NAME:-exp42_terminal_feature_upstream_conditioning_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp42_terminal_feature_upstream_conditioning/${RUN_NAME}}"
MODELS="${MODELS:-}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-}"
N_BOOT="${N_BOOT:-2000}"
GCS_DOWNLOAD_WORKERS="${GCS_DOWNLOAD_WORKERS:-12}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
K_LIST="${K_LIST:-1 2 5 10 20 50 100 200}"
CROSSCODER_DTYPE="${CROSSCODER_DTYPE:-bfloat16}"

case "$MODE" in
  smoke)
    N_PROMPTS="${N_PROMPTS:-4}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    MODELS="${MODELS:-qwen3_4b}"
    K_LIST="${K_LIST:-1 5 20}"
    N_BOOT="${N_BOOT:-200}"
    ;;
  full)
    N_PROMPTS="${N_PROMPTS:-600}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
    MODELS="${MODELS:-gemma3_4b llama31_8b mistral_7b qwen3_4b}"
    ;;
  pull-only|analyze-only|merge-only)
    N_PROMPTS="${N_PROMPTS:-600}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
    MODELS="${MODELS:-gemma3_4b llama31_8b mistral_7b qwen3_4b}"
    ;;
  *)
    echo "[exp42-runpod] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp42-runpod] mode=${MODE}"
echo "[exp42-runpod] run_root=${RUN_ROOT}"
echo "[exp42-runpod] models=${MODELS}"
echo "[exp42-runpod] gpu_list=${GPU_LIST}"
echo "[exp42-runpod] workers_per_model=${WORKERS_PER_MODEL} n_prompts=${N_PROMPTS}"
echo "[exp42-runpod] k_list=${K_LIST} seeds=${RANDOM_SEEDS}"

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
      echo "[exp42-runpod] unknown family $1" >&2
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
      echo "[exp42-runpod] unknown family $1" >&2
      exit 2
      ;;
  esac
}

pull_inputs() {
  local exp20_uri="gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
  local exp20_root="results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
  read -r -a model_arr <<< "$MODELS"
  for model in "${model_arr[@]}"; do
    echo "[exp42-runpod] pull Exp20 manifest ${model}"
    download_gcs_prefix "${exp20_uri}/${PROMPT_MODE}/${model}" "${exp20_root}/${PROMPT_MODE}/${model}" '^exp20_validation_records\.jsonl$'
  done
  for model in "${model_arr[@]}"; do
    local root
    local uri
    root="$(family_root "$model")"
    uri="$(family_gcs "$model")"
    echo "[exp42-runpod] pull crosscoder inputs ${model}"
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
      echo "[exp42-runpod] launch model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu} log=${log_path}"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        $PY_RUNNER -m src.poc.exp42_terminal_feature_upstream_conditioning collect \
          --model "$model" \
          --crosscoder-root "$root" \
          --out-dir "${RUN_ROOT}/raw/${model}" \
          --prompt-mode "$PROMPT_MODE" \
          --n-prompts "$N_PROMPTS" \
          --worker-index "$worker" \
          --n-workers "$WORKERS_PER_MODEL" \
          --k-list ${K_LIST} \
          --random-seeds ${RANDOM_SEEDS} \
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
    echo "[exp42-runpod] at least one collect worker failed" >&2
    return "$status"
  fi
}

merge_outputs() {
  read -r -a model_arr <<< "$MODELS"
  for model in "${model_arr[@]}"; do
    echo "[exp42-runpod] merge ${model}"
    $PY_RUNNER -m src.poc.exp42_terminal_feature_upstream_conditioning collect \
      --model "$model" \
      --crosscoder-root "$(family_root "$model")" \
      --out-dir "${RUN_ROOT}/raw/${model}" \
      --n-workers "$WORKERS_PER_MODEL" \
      --merge-only
  done
}

analyze_outputs() {
  echo "[exp42-runpod] analyze"
  $PY_RUNNER -m src.poc.exp42_terminal_feature_upstream_conditioning analyze \
    --run-root "$RUN_ROOT" \
    --models ${MODELS} \
    --k-list ${K_LIST} \
    --n-boot "$N_BOOT"
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
echo "[exp42-runpod] complete ${RUN_ROOT}"
