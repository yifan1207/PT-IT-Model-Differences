#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
PY_RUNNER="${PY_RUNNER:-uv run python}"

MODE="${MODE:-smoke}"  # smoke|full|preflight|support|native-audit|token|bridge-one-step|bridge-rollout|analyze-only|sync
RUN_NAME="${RUN_NAME:-exp47_same_base_recipe_specificity_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp47_same_base_recipe_specificity/${RUN_NAME}}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp47_same_base_recipe_specificity}"
GPU_LIST="${GPU_LIST:-}"
MODELS="${MODELS:-llama31_meta_instruct llama31_tulu3_sft llama31_tulu3_dpo llama31_tulu3_final llama31_openmath2}"
NATIVE_AUDIT_MODELS="${NATIVE_AUDIT_MODELS:-llama31_meta_instruct llama31_tulu3_sft llama31_tulu3_dpo llama31_tulu3_final}"
ROLLOUT_MODELS="${ROLLOUT_MODELS:-llama31_meta_instruct llama31_tulu3_final llama31_openmath2}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BOUNDARY_LAYER="${BOUNDARY_LAYER:-19}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
N_BOOT="${N_BOOT:-1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-20000}"
N_TOKENIZER_CHECK_EXAMPLES="${N_TOKENIZER_CHECK_EXAMPLES:-64}"
RUN_NATIVE_AUDIT="${RUN_NATIVE_AUDIT:-1}"
ALLOW_MAIN_DRIFT="${ALLOW_MAIN_DRIFT:-1}"

case "$MODE" in
  smoke)
    N_EXAMPLES="${N_EXAMPLES:-12}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    ROLLOUT_WORKERS_PER_MODEL="${ROLLOUT_WORKERS_PER_MODEL:-1}"
    NATIVE_AUDIT_EXAMPLES="${NATIVE_AUDIT_EXAMPLES:-12}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-8}"
    ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-16}"
    N_BOOT="${N_BOOT:-100}"
    N_PERMUTATIONS="${N_PERMUTATIONS:-1000}"
    ;;
  full|support|token|bridge-one-step|bridge-rollout|analyze-only|sync|native-audit|preflight)
    N_EXAMPLES="${N_EXAMPLES:-1400}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-3}"
    ROLLOUT_WORKERS_PER_MODEL="${ROLLOUT_WORKERS_PER_MODEL:-2}"
    NATIVE_AUDIT_EXAMPLES="${NATIVE_AUDIT_EXAMPLES:-200}"
    ROLLOUT_EVENTS="${ROLLOUT_EVENTS:-450}"
    ROLLOUT_MAX_NEW_TOKENS="${ROLLOUT_MAX_NEW_TOKENS:-256}"
    ;;
  *)
    echo "[exp47] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

PREFLIGHT_DIR="${RUN_ROOT}/preflight"
EXP20_ROOT="${RUN_ROOT}/exp20"
EXP23_ROOT="${RUN_ROOT}/residual_factorial"
BRIDGE_ROOT="${RUN_ROOT}/behavior_bridge"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR"

gpu_count="$($PY_RUNNER - <<'PY'
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
read -r -a GPUS <<< "${GPU_LIST:-}"

echo "[exp47] host $(hostname)"
echo "[exp47] mode ${MODE}"
echo "[exp47] run_name ${RUN_NAME}"
echo "[exp47] run_root ${RUN_ROOT}"
echo "[exp47] dataset ${DATASET}"
echo "[exp47] models ${MODELS}"
echo "[exp47] rollout_models ${ROLLOUT_MODELS}"
echo "[exp47] gpus ${GPU_LIST:-<none>} detected=${gpu_count}"
echo "[exp47] workers_per_model ${WORKERS_PER_MODEL} rollout_workers_per_model ${ROLLOUT_WORKERS_PER_MODEL}"
echo "[exp47] n_examples ${N_EXAMPLES} native_audit_examples ${NATIVE_AUDIT_EXAMPLES} rollout_events ${ROLLOUT_EVENTS}"

if [[ "$MODE" != "analyze-only" && "$MODE" != "sync" && "$MODE" != "preflight" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp47] GPU phase requires visible GPUs or GPU_LIST" >&2
  exit 2
fi

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUNNER scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  else
    echo "[exp47] no GCS sync helper; skipping ${GCS_SYNC_DEST}" >&2
  fi
}

wait_batch() {
  local phase="$1"
  shift
  local status=0
  for pid in "$@"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp47] phase ${phase} failed; see ${LOG_DIR}" >&2
    sync_outputs || true
    exit "$status"
  fi
}

launch_scheduled() {
  local phase="$1"
  local -n jobs_ref="$2"
  local idx=0
  local pids=()
  for job in "${jobs_ref[@]}"; do
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    IFS='|' read -r log_name cmd <<< "$job"
    echo "[exp47] launch ${phase} gpu=${gpu} log=${LOG_DIR}/${log_name}.log"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      eval "$cmd"
    ) >"${LOG_DIR}/${log_name}.log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
    if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
      wait_batch "$phase" "${pids[@]}"
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait_batch "$phase" "${pids[@]}"
  fi
}

run_preflight() {
  echo "[exp47] phase preflight"
  local drift_args=()
  if [[ "$ALLOW_MAIN_DRIFT" == "1" ]]; then
    drift_args=(--allow-main-drift)
  fi
  $PY_RUNNER scripts/analysis/preflight_exp47_same_base_recipe_specificity.py \
    --out-dir "$PREFLIGHT_DIR" \
    --dataset "$DATASET" \
    --models ${MODELS} \
    --n-tokenizer-check-examples "$N_TOKENIZER_CHECK_EXAMPLES" \
    "${drift_args[@]}" \
    >"${LOG_DIR}/preflight.log" 2>&1
  sync_outputs || true
}

run_support_mode() {
  local prompt_mode="$1"
  local models="$2"
  local n_examples="$3"
  local workers="$4"
  echo "[exp47] phase exp20-support mode=${prompt_mode} models=${models}"
  local jobs=()
  for model in ${models}; do
    for ((w=0; w<workers; w++)); do
      local out_dir="${EXP20_ROOT}/${prompt_mode}/${model}"
      mkdir -p "$out_dir"
      jobs+=("exp20_${prompt_mode}_${model}_w${w}|$PY_RUNNER -m src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation --model ${model} --dataset ${DATASET} --out-dir ${out_dir} --device cuda:0 --worker-index ${w} --n-workers ${workers} --n-eval-examples ${n_examples} --max-new-tokens ${MAX_NEW_TOKENS} --prompt-mode ${prompt_mode}")
    done
  done
  launch_scheduled "exp20-${prompt_mode}" jobs
  for model in ${models}; do
    $PY_RUNNER -m src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation \
      --model "$model" \
      --dataset "$DATASET" \
      --out-dir "${EXP20_ROOT}/${prompt_mode}/${model}" \
      --n-workers "$workers" \
      --merge-only \
      >"${LOG_DIR}/exp20_${prompt_mode}_${model}_merge.log" 2>&1
  done
  sync_outputs || true
}

run_token_factorial() {
  echo "[exp47] phase residual-factorial"
  local jobs=()
  for model in ${MODELS}; do
    for ((w=0; w<WORKERS_PER_MODEL; w++)); do
      local out_dir="${EXP23_ROOT}/raw_shared/${model}"
      mkdir -p "$out_dir"
      jobs+=("exp23_${model}_w${w}|$PY_RUNNER -m src.poc.exp23_midlate_interaction_suite.residual_factorial --model ${model} --dataset ${DATASET} --exp20-root ${EXP20_ROOT} --exp20-fallback-root ${EXP20_ROOT} --out-dir ${out_dir} --device cuda:0 --worker-index ${w} --n-workers ${WORKERS_PER_MODEL} --n-eval-examples ${N_EXAMPLES} --prompt-mode raw_shared --event-kinds ${EVENT_KINDS} --boundary-layer-override ${BOUNDARY_LAYER} --experiment-name exp47_same_base_recipe_specificity --boundary-mode full_late")
    done
  done
  launch_scheduled "residual-factorial" jobs
  for model in ${MODELS}; do
    $PY_RUNNER -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
      --model "$model" \
      --out-dir "${EXP23_ROOT}/raw_shared/${model}" \
      --n-workers "$WORKERS_PER_MODEL" \
      --merge-only \
      >"${LOG_DIR}/exp23_${model}_merge.log" 2>&1
  done
  sync_outputs || true
}

run_bridge_one_step() {
  echo "[exp47] phase bridge-one-step"
  local jobs=()
  for model in ${MODELS}; do
    for ((w=0; w<WORKERS_PER_MODEL; w++)); do
      local out_dir="${BRIDGE_ROOT}/raw/${model}"
      mkdir -p "$out_dir"
      jobs+=("bridge_one_${model}_w${w}|$PY_RUNNER -m src.poc.exp45_behavioral_bridge collect --model ${model} --dataset ${DATASET} --exp20-root ${EXP20_ROOT} --exp20-fallback-root ${EXP20_ROOT} --out-dir ${out_dir} --prompt-mode raw_shared --n-prompts ${N_EXAMPLES} --max-events 0 --rollout-events 0 --max-new-tokens 1 --top-k 20 --event-kinds ${EVENT_KINDS} --part one_step --boundary-layer-override ${BOUNDARY_LAYER} --worker-index ${w} --n-workers ${WORKERS_PER_MODEL} --device cuda:0")
    done
  done
  launch_scheduled "bridge-one-step" jobs
  for model in ${MODELS}; do
    $PY_RUNNER -m src.poc.exp45_behavioral_bridge collect \
      --model "$model" \
      --out-dir "${BRIDGE_ROOT}/raw/${model}" \
      --n-workers "$WORKERS_PER_MODEL" \
      --merge-part one_step \
      --merge-only \
      >"${LOG_DIR}/bridge_one_${model}_merge.log" 2>&1
  done
  sync_outputs || true
}

run_bridge_rollout() {
  echo "[exp47] phase bridge-rollout"
  local jobs=()
  for model in ${ROLLOUT_MODELS}; do
    for ((w=0; w<ROLLOUT_WORKERS_PER_MODEL; w++)); do
      local out_dir="${BRIDGE_ROOT}/raw/${model}"
      mkdir -p "$out_dir"
      jobs+=("bridge_rollout_${model}_w${w}|$PY_RUNNER -m src.poc.exp45_behavioral_bridge collect --model ${model} --dataset ${DATASET} --exp20-root ${EXP20_ROOT} --exp20-fallback-root ${EXP20_ROOT} --out-dir ${out_dir} --prompt-mode raw_shared --n-prompts ${N_EXAMPLES} --max-events 0 --rollout-events ${ROLLOUT_EVENTS} --max-new-tokens ${ROLLOUT_MAX_NEW_TOKENS} --top-k 20 --event-kinds ${EVENT_KINDS} --part rollout --category-filter GOV-FORMAT CONTENT-REASON --boundary-layer-override ${BOUNDARY_LAYER} --worker-index ${w} --n-workers ${ROLLOUT_WORKERS_PER_MODEL} --device cuda:0")
    done
  done
  launch_scheduled "bridge-rollout" jobs
  for model in ${ROLLOUT_MODELS}; do
    $PY_RUNNER -m src.poc.exp45_behavioral_bridge collect \
      --model "$model" \
      --out-dir "${BRIDGE_ROOT}/raw/${model}" \
      --n-workers "$ROLLOUT_WORKERS_PER_MODEL" \
      --merge-part rollout \
      --merge-only \
      >"${LOG_DIR}/bridge_rollout_${model}_merge.log" 2>&1
  done
  sync_outputs || true
}

run_analyze() {
  echo "[exp47] phase analyze"
  $PY_RUNNER scripts/analysis/analyze_exp47_same_base_recipe_specificity.py \
    --run-root "$RUN_ROOT" \
    --dataset "$DATASET" \
    --models ${MODELS} \
    --n-boot "$N_BOOT" \
    --n-permutations "$N_PERMUTATIONS" \
    >"${LOG_DIR}/analyze.log" 2>&1
  sync_outputs || true
}

case "$MODE" in
  preflight)
    run_preflight
    ;;
  support)
    run_support_mode raw_shared "$MODELS" "$N_EXAMPLES" "$WORKERS_PER_MODEL"
    ;;
  native-audit)
    run_support_mode native "$NATIVE_AUDIT_MODELS" "$NATIVE_AUDIT_EXAMPLES" 1
    ;;
  token)
    run_token_factorial
    ;;
  bridge-one-step)
    run_bridge_one_step
    ;;
  bridge-rollout)
    run_bridge_rollout
    ;;
  analyze-only)
    run_analyze
    ;;
  sync)
    sync_outputs
    ;;
  smoke|full)
    run_preflight
    run_support_mode raw_shared "$MODELS" "$N_EXAMPLES" "$WORKERS_PER_MODEL"
    if [[ "$RUN_NATIVE_AUDIT" == "1" ]]; then
      run_support_mode native "$NATIVE_AUDIT_MODELS" "$NATIVE_AUDIT_EXAMPLES" 1
    fi
    run_token_factorial
    run_bridge_one_step
    run_bridge_rollout
    run_analyze
    ;;
esac

echo "[exp47] complete ${RUN_ROOT}"
