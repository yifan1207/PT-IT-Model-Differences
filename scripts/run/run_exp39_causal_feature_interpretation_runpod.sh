#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

PY_RUNNER="${PY_RUNNER:-uv run python}"
RUN_NAME="${RUN_NAME:-exp39_causal_feature_interpretation_$(date -u +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-results/exp39_causal_feature_interpretation}"
FAMILIES="${FAMILIES:-gemma3_4b llama31_8b mistral_7b qwen3_4b}"
TOP_N="${TOP_N:-25}"
CONTROL_PER_FEATURE="${CONTROL_PER_FEATURE:-1}"
N_PROMPTS="${N_PROMPTS:-3000}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
APPEND_PT_GREEDY_TOKENS="${APPEND_PT_GREEDY_TOKENS:-384}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EXAMPLES_PER_KIND="${EXAMPLES_PER_KIND:-20}"
VALIDATION_EXAMPLES_PER_KIND="${VALIDATION_EXAMPLES_PER_KIND:-12}"
CONTEXT_BEFORE="${CONTEXT_BEFORE:-64}"
CONTEXT_AFTER="${CONTEXT_AFTER:-16}"
CROSSCODER_DTYPE="${CROSSCODER_DTYPE:-bfloat16}"
PROJECTION_TOP_K="${PROJECTION_TOP_K:-30}"
PULL_GCS="${PULL_GCS:-1}"
PULL_CACHE="${PULL_CACHE:-0}"
RUN_AUTOINTERP="${RUN_AUTOINTERP:-1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.5}"
OPENAI_PARALLELISM="${OPENAI_PARALLELISM:-8}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"
GPU_LIST="${GPU_LIST:-}"
SHARDS_PER_FAMILY="${SHARDS_PER_FAMILY:-auto}"
GCS_PROJECT="${GCS_PROJECT:-studious-hydra-450206-a5}"
GCS_DOWNLOAD_WORKERS="${GCS_DOWNLOAD_WORKERS:-4}"
export PATH="${HOME}/.local/bin:${PATH}"
export GCS_PROJECT

if [[ -z "$GPU_LIST" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
  else
    GPU_LIST="0"
  fi
fi

read -r -a FAMILY_ARR <<< "$FAMILIES"
read -r -a GPU_ARR <<< "$GPU_LIST"
if [[ "${#GPU_ARR[@]}" -lt 1 ]]; then
  echo "[exp39] no GPUs found in GPU_LIST=${GPU_LIST}" >&2
  exit 2
fi
if [[ "$SHARDS_PER_FAMILY" == "auto" ]]; then
  SHARDS_PER_FAMILY=$(( ${#GPU_ARR[@]} / ${#FAMILY_ARR[@]} ))
  if [[ "$SHARDS_PER_FAMILY" -lt 1 ]]; then
    SHARDS_PER_FAMILY=1
  fi
fi

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
    olmo2_7b)
      printf '%s' "results/exp38_qwen_olmo_final_layer_crosscoder_hardening/exp38_olmo2_7b_final2_d131072_k64_20260503_0441_a100x2/selected_d131072_k64"
      ;;
    *)
      echo "[exp39] unknown family $1" >&2
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
    olmo2_7b)
      printf '%s' "gs://pt-vs-it-results/results/exp38_qwen_olmo_final_layer_crosscoder_hardening/exp38_olmo2_7b_final2_d131072_k64_20260503_0441_a100x2/selected_d131072_k64"
      ;;
    *)
      echo "[exp39] unknown family $1" >&2
      exit 2
      ;;
  esac
}

pull_family_artifacts() {
  local family="$1"
  local root
  local uri
  root="$(family_root "$family")"
  uri="$(family_gcs "$family")"
  mkdir -p "$root"
  echo "[exp39] pulling ${family} artifacts from ${uri}"
  download_gcs_prefix "${uri%/}/dictionaries" "${root}/dictionaries" '(^|/)(crosscoder\.pt|config\.json)$'
  download_gcs_prefix "${uri%/}/feature_stats" "${root}/feature_stats" '^(causal_feature_scores\.csv|causal_top_features\.csv|causal_feature_scores_summary\.json)$'
  download_gcs_prefix "${uri%/}/analysis" "${root}/analysis" '^(summary\.json|effects\.csv)$'
  if [[ "$PULL_CACHE" == "1" ]]; then
    download_gcs_prefix "${uri%/}/cache" "${root}/cache" 'layer_.*\.pt$'
  fi
}

download_gcs_prefix() {
  local uri="$1"
  local dest="$2"
  local include_regex="${3:-}"
  if [[ -z "$include_regex" ]] && command -v gsutil >/dev/null 2>&1 && gsutil -q ls "${uri%/}/**" >/dev/null 2>&1; then
    gsutil -m rsync -r "$uri" "$dest"
    return
  fi
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

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "${OUT_ROOT}/${RUN_NAME}" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp39] GCS_SYNC_DEST set but gsutil missing; skipping sync" >&2
  fi
}

echo "[exp39] host $(hostname)"
echo "[exp39] run_name ${RUN_NAME}"
echo "[exp39] families ${FAMILIES}"
echo "[exp39] gpu_list ${GPU_LIST}"
echo "[exp39] shards_per_family ${SHARDS_PER_FAMILY}"
echo "[exp39] top_n ${TOP_N} controls_per_feature ${CONTROL_PER_FEATURE} n_prompts ${N_PROMPTS}"
echo "[exp39] append_pt_greedy_tokens ${APPEND_PT_GREEDY_TOKENS} max_seq_len ${MAX_SEQ_LEN}"

$PY_RUNNER -m src.poc.exp39_causal_feature_interpretation preflight \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --families "${FAMILY_ARR[@]}"

if [[ "$PULL_GCS" == "1" ]]; then
  pull_pids=()
  for family in "${FAMILY_ARR[@]}"; do
    pull_family_artifacts "$family" &
    pull_pids+=("$!")
  done
  pull_status=0
  for pid in "${pull_pids[@]}"; do
    if ! wait "$pid"; then
      pull_status=1
    fi
  done
  if [[ "$pull_status" -ne 0 ]]; then
    echo "[exp39] one or more artifact pulls failed" >&2
    exit "$pull_status"
  fi
fi

$PY_RUNNER -m src.poc.exp39_causal_feature_interpretation preflight \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --families "${FAMILY_ARR[@]}" \
  --strict

$PY_RUNNER -m src.poc.exp39_causal_feature_interpretation select \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --families "${FAMILY_ARR[@]}" \
  --top-n "$TOP_N" \
  --control-per-feature "$CONTROL_PER_FEATURE"

mkdir -p "${OUT_ROOT}/${RUN_NAME}/logs"
pids=()
suffixes=()
idx=0
for family in "${FAMILY_ARR[@]}"; do
  for shard in $(seq 0 $((SHARDS_PER_FAMILY - 1))); do
    gpu="${GPU_ARR[$((idx % ${#GPU_ARR[@]}))]}"
    suffix="${family}_s${shard}of${SHARDS_PER_FAMILY}"
    suffixes+=("$suffix")
    idx=$((idx + 1))
    echo "[exp39] dashboard family=${family} shard=${shard}/${SHARDS_PER_FAMILY} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp39_causal_feature_interpretation dashboard \
      --out-root "$OUT_ROOT" \
      --run-name "$RUN_NAME" \
      --families "$family" \
      --n-prompts "$N_PROMPTS" \
      --max-seq-len "$MAX_SEQ_LEN" \
      --append-pt-greedy-tokens "$APPEND_PT_GREEDY_TOKENS" \
      --batch-size "$BATCH_SIZE" \
      --examples-per-kind "$EXAMPLES_PER_KIND" \
      --validation-examples-per-kind "$VALIDATION_EXAMPLES_PER_KIND" \
      --context-before "$CONTEXT_BEFORE" \
      --context-after "$CONTEXT_AFTER" \
      --crosscoder-dtype "$CROSSCODER_DTYPE" \
      --projection-top-k "$PROJECTION_TOP_K" \
      --prompt-shard-index "$shard" \
      --prompt-shard-count "$SHARDS_PER_FAMILY" \
      --device cuda:0 \
      --output-suffix "$suffix" \
      >"${OUT_ROOT}/${RUN_NAME}/logs/dashboard_${suffix}.log" 2>&1 &
    pids+=("$!")
  done
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[exp39] one or more dashboard workers failed" >&2
  exit "$status"
fi

$PY_RUNNER -m src.poc.exp39_causal_feature_interpretation merge-dashboards \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME" \
  --suffixes "${suffixes[@]}"

if [[ "$RUN_AUTOINTERP" == "1" ]]; then
  $PY_RUNNER -m src.poc.exp39_causal_feature_interpretation autointerp \
    --out-root "$OUT_ROOT" \
    --run-name "$RUN_NAME" \
    --model "$OPENAI_MODEL" \
    --parallelism "$OPENAI_PARALLELISM" \
    --include-controls
  $PY_RUNNER -m src.poc.exp39_causal_feature_interpretation validate \
    --out-root "$OUT_ROOT" \
    --run-name "$RUN_NAME" \
    --model "$OPENAI_MODEL" \
    --parallelism "$OPENAI_PARALLELISM"
fi

$PY_RUNNER -m src.poc.exp39_causal_feature_interpretation analyze \
  --out-root "$OUT_ROOT" \
  --run-name "$RUN_NAME"

sync_outputs
echo "[exp39] complete run_name=${RUN_NAME}"
