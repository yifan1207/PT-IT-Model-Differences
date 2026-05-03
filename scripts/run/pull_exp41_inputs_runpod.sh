#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PATH="${HOME}/.local/bin:${PATH}"
export GCS_PROJECT="${GCS_PROJECT:-studious-hydra-450206-a5}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
GCS_DOWNLOAD_WORKERS="${GCS_DOWNLOAD_WORKERS:-8}"
MODELS="${MODELS:-gemma3_4b llama31_8b mistral_7b qwen3_4b}"

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
      echo "[exp41-pull] unknown family $1" >&2
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
      echo "[exp41-pull] unknown family $1" >&2
      exit 2
      ;;
  esac
}

EXP39_URI="gs://pt-vs-it-results/results/exp39_causal_feature_interpretation/exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345"
EXP39_ROOT="results/exp39_causal_feature_interpretation/exp39_reinterp_specific_labels_ctrl_h100x8_20260503_110345"
EXP20_URI="gs://pt-vs-it-results/results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
EXP20_ROOT="results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"

echo "[exp41-pull] pulling Exp39 manifest inputs"
download_gcs_prefix "${EXP39_URI}/autointerp" "${EXP39_ROOT}/autointerp" '^(causal_paper_taxonomy_v3\.jsonl|all_paper_taxonomy_v4\.jsonl)$'
download_gcs_prefix "${EXP39_URI}/feature_selection" "${EXP39_ROOT}/feature_selection" '^(selected_features\.csv|control_features\.csv)$'

read -r -a MODEL_ARR <<< "$MODELS"
for model in "${MODEL_ARR[@]}"; do
  echo "[exp41-pull] pulling Exp20 raw_shared records for ${model}"
  download_gcs_prefix "${EXP20_URI}/raw_shared/${model}" "${EXP20_ROOT}/raw_shared/${model}" '^exp20_validation_records\.jsonl$'
done

for family in "${MODEL_ARR[@]}"; do
  root="$(family_root "$family")"
  uri="$(family_gcs "$family")"
  echo "[exp41-pull] pulling crosscoder dictionaries for ${family}"
  mkdir -p "$root"
  download_gcs_prefix "${uri}/dictionaries" "${root}/dictionaries" '(^|/)(crosscoder\.pt|config\.json)$'
done

echo "[exp41-pull] complete"

