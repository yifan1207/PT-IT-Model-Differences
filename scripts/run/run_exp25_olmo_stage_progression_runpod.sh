#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"  # preflight|smoke|full|analyze-only
RUN_NAME="${RUN_NAME:-exp25_olmo_stage_progression_$(date -u +%Y%m%d_%H%M%S)}"
STAGE_MODELS="${STAGE_MODELS:-olmo2_7b_pt_sft olmo2_7b_sft_dpo olmo2_7b_dpo_rlvr olmo2_7b}"
SMOKE_MODEL="${SMOKE_MODEL:-olmo2_7b_dpo_rlvr}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-50}"
N_EXAMPLES="${N_EXAMPLES:-600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
EXP20_WORKERS_PER_MODEL="${EXP20_WORKERS_PER_MODEL:-2}"
EXP23_RESIDUAL_WORKERS="${EXP23_RESIDUAL_WORKERS:-2}"
GPU_LIST="${GPU_LIST:-}"
N_BOOT="${N_BOOT:-2000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-20000}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

EXP20_RUN_NAME="${EXP20_RUN_NAME:-${RUN_NAME}_exp20}"
EXP23_RUN_NAME="${EXP23_RUN_NAME:-${RUN_NAME}_exp23}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/${EXP20_RUN_NAME}}"
EXP23_ROOT="${EXP23_ROOT:-results/exp23_midlate_interaction_suite/${EXP23_RUN_NAME}}"
SYNTHESIS_DIR="${SYNTHESIS_DIR:-results/paper_synthesis/${RUN_NAME}}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|analyze-only|preflight bash scripts/run/run_exp25_olmo_stage_progression_runpod.sh

Environment overrides:
  RUN_NAME, STAGE_MODELS, SMOKE_MODEL, DATASET, N_EXAMPLES, SMOKE_EXAMPLES
  GPU_LIST="0 1 ..." EXP20_WORKERS_PER_MODEL=2 EXP23_RESIDUAL_WORKERS=2
  N_BOOT=2000 N_PERMUTATIONS=20000 GCS_SYNC_DEST=gs://...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "preflight" && "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp25] invalid MODE=${MODE}" >&2
  usage
  exit 2
fi

gpu_count="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if [[ -z "$GPU_LIST" && "$gpu_count" -gt 0 ]]; then
  GPU_LIST="$(seq -s ' ' 0 $((gpu_count - 1)))"
fi

echo "[exp25] host $(hostname)"
echo "[exp25] mode ${MODE}"
echo "[exp25] run_name ${RUN_NAME}"
echo "[exp25] stage_models ${STAGE_MODELS}"
echo "[exp25] dataset ${DATASET}"
echo "[exp25] gpu_count ${gpu_count} gpu_list ${GPU_LIST:-<none>}"
echo "[exp25] exp20_root ${EXP20_ROOT}"
echo "[exp25] exp23_root ${EXP23_ROOT}"

run_preflight() {
  uv run python -m src.poc.exp25_olmo_stage_progression.preflight \
    --models ${STAGE_MODELS} \
    --dataset "$DATASET" \
    --n-examples 50 \
    --out "${SYNTHESIS_DIR}/olmo_stage_preflight.json"
}

run_exp20() {
  local models="$1"
  local n_examples="$2"
  local workers="$3"
  RUN_NAME="$EXP20_RUN_NAME" \
  MODELS="$models" \
  PROMPT_MODES="$PROMPT_MODE" \
  DATASET="$DATASET" \
  N_EXAMPLES="$n_examples" \
  MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  WORKERS_PER_MODEL="$workers" \
  GPU_LIST="$GPU_LIST" \
  GCS_SYNC_DEST="" \
  bash scripts/run/run_exp20_factorial_validation_runpod.sh

  uv run python scripts/analysis/analyze_exp20_factorial_validation.py \
    --root "$EXP20_ROOT" \
    --models ${models} \
    --out-dir "${EXP20_ROOT}/validation_analysis" \
    --n-boot "$N_BOOT"
}

run_exp23() {
  local models="$1"
  local n_examples="$2"
  local workers="$3"
  local mode="$4"
  local mode_args=(--mode "$mode")
  if [[ "$mode" == "smoke" ]]; then
    mode_args+=(--model "$models" --smoke-prompts "$n_examples")
  else
    mode_args+=(--models "$models" --n-prompts "$n_examples")
  fi
  bash scripts/run/run_exp23_midlate_interaction_suite.sh \
    "${mode_args[@]}" \
    --parts residual \
    --run-name "$EXP23_RUN_NAME" \
    --run-root "$EXP23_ROOT" \
    --prompt-mode "$PROMPT_MODE" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_ROOT" \
    --event-kinds first_diff \
    --residual-workers "$workers" \
    --gpu-list "$GPU_LIST" \
    --smoke-gpu "${GPU_LIST%% *}" \
    --n-boot "$N_BOOT" \
    --no-trajectories
}

run_label_swap_per_model() {
  local models="$1"
  local compat_root="${EXP23_ROOT}/analysis/compatibility_permutation"
  for model in ${models}; do
    uv run python scripts/analysis/analyze_exp23_compatibility_permutation.py \
      --run-root "$EXP23_ROOT" \
      --out-dir "${compat_root}/${model}" \
      --models "$model" \
      --prompt-mode "$PROMPT_MODE" \
      --readout common_it \
      --n-permutations "$N_PERMUTATIONS"
  done
}

run_synthesis() {
  uv run python scripts/analysis/analyze_olmo_stage_progression.py \
    --exp20-root "$EXP20_ROOT" \
    --exp23-summary "${EXP23_ROOT}/analysis/exp23_summary.json" \
    --compatibility-root "${EXP23_ROOT}/analysis/compatibility_permutation" \
    --out-dir "$SYNTHESIS_DIR" \
    --models ${STAGE_MODELS} \
    --prompt-mode "$PROMPT_MODE"
}

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if ! command -v gsutil >/dev/null 2>&1; then
    echo "[exp25] GCS_SYNC_DEST set but gsutil not found; skipping sync" >&2
    return
  fi
  gsutil -m rsync -r "$EXP20_ROOT" "${GCS_SYNC_DEST%/}/exp20/${EXP20_RUN_NAME}"
  gsutil -m rsync -r "$EXP23_ROOT" "${GCS_SYNC_DEST%/}/exp23/${EXP23_RUN_NAME}"
  gsutil -m rsync -r "$SYNTHESIS_DIR" "${GCS_SYNC_DEST%/}/paper_synthesis/${RUN_NAME}"
}

case "$MODE" in
  preflight)
    run_preflight
    ;;
  smoke)
    if [[ "$gpu_count" -lt 1 ]]; then
      echo "[exp25] smoke requires a GPU with enough memory for two OLMo 7B checkpoints" >&2
      exit 2
    fi
    STAGE_MODELS="$SMOKE_MODEL"
    run_preflight
    run_exp20 "$SMOKE_MODEL" "$SMOKE_EXAMPLES" 1
    run_exp23 "$SMOKE_MODEL" "$SMOKE_EXAMPLES" 1 smoke
    run_label_swap_per_model "$SMOKE_MODEL"
    run_synthesis
    sync_outputs
    ;;
  full)
    if [[ "$gpu_count" -lt 1 ]]; then
      echo "[exp25] full run requires GPUs" >&2
      exit 2
    fi
    run_preflight
    run_exp20 "$STAGE_MODELS" "$N_EXAMPLES" "$EXP20_WORKERS_PER_MODEL"
    run_exp23 "$STAGE_MODELS" "$N_EXAMPLES" "$EXP23_RESIDUAL_WORKERS" full
    run_label_swap_per_model "$STAGE_MODELS"
    run_synthesis
    sync_outputs
    ;;
  analyze-only)
    run_label_swap_per_model "$STAGE_MODELS"
    run_synthesis
    sync_outputs
    ;;
esac

echo "[exp25] complete run_name=${RUN_NAME}"
