#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
TORCHRUN="${TORCHRUN:-uv run torchrun}"

MODE="${MODE:-smoke}"  # smoke|full|prepare|train|merge|sanity|factorial|analyze-only|sync
RUN_NAME="${RUN_NAME:-exp53_controlled_domain_finetunes_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp53_controlled_domain_finetunes/${RUN_NAME}}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp53_controlled_domain_finetunes}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B}"
BASE_REVISION="${BASE_REVISION:-d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}"
DOMAINS="${DOMAINS:-code biomed}"
MODELS="${MODELS:-llama31_code_cpt_lora llama31_biomed_cpt_lora}"
DATASET_MAIN="${DATASET_MAIN:-data/eval_dataset_v2.jsonl}"
GPU_LIST="${GPU_LIST:-}"
BOUNDARY_LAYER="${BOUNDARY_LAYER:-19}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
N_BOOT="${N_BOOT:-1000}"
N_PERMUTATIONS="${N_PERMUTATIONS:-20000}"

case "$MODE" in
  smoke)
    TRAIN_TOKENS="${TRAIN_TOKENS:-262144}"
    PREP_TRAIN_TOKENS="${PREP_TRAIN_TOKENS:-524288}"
    EVAL_RECORDS="${EVAL_RECORDS:-16}"
    SUPPORT_RECORDS="${SUPPORT_RECORDS:-24}"
    N_EXAMPLES_MAIN="${N_EXAMPLES_MAIN:-24}"
    N_EXAMPLES_DOMAIN="${N_EXAMPLES_DOMAIN:-24}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
    SEQ_LEN="${SEQ_LEN:-1024}"
    EVAL_TOKENS="${EVAL_TOKENS:-65536}"
    N_BOOT="${N_BOOT:-100}"
    N_PERMUTATIONS="${N_PERMUTATIONS:-1000}"
    ;;
  full|prepare|train|merge|sanity|factorial|analyze-only|sync)
    TRAIN_TOKENS="${TRAIN_TOKENS:-20000000}"
    PREP_TRAIN_TOKENS="${PREP_TRAIN_TOKENS:-24000000}"
    EVAL_RECORDS="${EVAL_RECORDS:-512}"
    SUPPORT_RECORDS="${SUPPORT_RECORDS:-600}"
    N_EXAMPLES_MAIN="${N_EXAMPLES_MAIN:-1400}"
    N_EXAMPLES_DOMAIN="${N_EXAMPLES_DOMAIN:-600}"
    WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-4}"
    SEQ_LEN="${SEQ_LEN:-4096}"
    EVAL_TOKENS="${EVAL_TOKENS:-1000000}"
    ;;
  *)
    echo "[exp53] unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

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

echo "[exp53] host $(hostname)"
echo "[exp53] mode ${MODE}"
echo "[exp53] run_name ${RUN_NAME}"
echo "[exp53] run_root ${RUN_ROOT}"
echo "[exp53] domains ${DOMAINS}"
echo "[exp53] models ${MODELS}"
echo "[exp53] gpus ${GPU_LIST:-<none>} detected=${gpu_count}"
echo "[exp53] train_tokens ${TRAIN_TOKENS} seq_len ${SEQ_LEN}"

if [[ "$MODE" != "analyze-only" && "$MODE" != "sync" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp53] GPU phase requires GPUs" >&2
  exit 2
fi

ensure_deps() {
  if [[ "${EXP53_SKIP_DEP_INSTALL:-0}" == "1" ]]; then
    return
  fi
  echo "[exp53] ensuring training/data dependencies"
  uv pip install "datasets>=3.0.0" "peft>=0.14.0" "accelerate>=1.0.0" >/dev/null
}

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" || "$GCS_SYNC_DEST" == "none" || "$GCS_SYNC_DEST" == "NONE" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUNNER scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  else
    echo "[exp53] no GCS sync helper; skipping ${GCS_SYNC_DEST}" >&2
  fi
}

publish_latest_symlink() {
  mkdir -p results/exp53_controlled_domain_finetunes
  ln -sfn "$(basename "$RUN_ROOT")" results/exp53_controlled_domain_finetunes/latest
}

run_prepare() {
  ensure_deps
  echo "[exp53] phase prepare"
  $PY_RUNNER -m src.poc.exp53_controlled_domain_finetunes.prepare_data \
    --run-root "$RUN_ROOT" \
    --domains ${DOMAINS} \
    --base-model "$BASE_MODEL" \
    --base-revision "$BASE_REVISION" \
    --hf-token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
    --train-tokens "$PREP_TRAIN_TOKENS" \
    --eval-records "$EVAL_RECORDS" \
    --support-records "$SUPPORT_RECORDS" \
    --force \
    >"${LOG_DIR}/prepare.log" 2>&1
  sync_outputs || true
}

gpu_csv() {
  local start="$1"
  local count="$2"
  local vals=()
  for ((i=0; i<count; i++)); do
    vals+=("${GPUS[$((start + i))]}")
  done
  local IFS=,
  echo "${vals[*]}"
}

run_train_domain() {
  local domain="$1"
  local gpu_start="$2"
  local gpu_count_local="$3"
  local port="$4"
  local gpus
  gpus="$(gpu_csv "$gpu_start" "$gpu_count_local")"
  echo "[exp53] train ${domain} on visible GPUs ${gpus}"
  (
    export CUDA_VISIBLE_DEVICES="$gpus"
    $TORCHRUN --standalone --nnodes=1 --nproc_per_node="$gpu_count_local" --master_port="$port" \
      -m src.poc.exp53_controlled_domain_finetunes.train_lora \
      --run-root "$RUN_ROOT" \
      --domain "$domain" \
      --base-model "$BASE_MODEL" \
      --base-revision "$BASE_REVISION" \
      --hf-token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
      --train-tokens "$TRAIN_TOKENS" \
      --seq-len "$SEQ_LEN" \
      --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
      --grad-accum "${GRAD_ACCUM:-1}" \
      --rank "${LORA_RANK:-32}" \
      --alpha "${LORA_ALPHA:-64}" \
      --learning-rate "${LEARNING_RATE:-1e-4}" \
      --dataloader-workers "${DATALOADER_WORKERS:-2}"
  ) >"${LOG_DIR}/train_${domain}.log" 2>&1
}

run_train() {
  ensure_deps
  echo "[exp53] phase train"
  local n="${#GPUS[@]}"
  if [[ "$n" -ge 16 ]]; then
    run_train_domain code 0 8 29531 &
    pid_a=$!
    run_train_domain biomed 8 8 29532 &
    pid_b=$!
    wait "$pid_a"
    wait "$pid_b"
  elif [[ "$n" -ge 8 ]]; then
    run_train_domain code 0 8 29531
    sync_outputs || true
    run_train_domain biomed 0 8 29532
  else
    local per="$n"
    run_train_domain code 0 "$per" 29531
    sync_outputs || true
    run_train_domain biomed 0 "$per" 29532
  fi
  sync_outputs || true
}

run_merge() {
  ensure_deps
  echo "[exp53] phase merge"
  local idx=0
  local pids=()
  for domain in ${DOMAINS}; do
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      $PY_RUNNER -m src.poc.exp53_controlled_domain_finetunes.merge_lora \
        --run-root "$RUN_ROOT" \
        --domain "$domain" \
        --base-model "$BASE_MODEL" \
        --base-revision "$BASE_REVISION" \
        --hf-token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
        --device cuda:0
    ) >"${LOG_DIR}/merge_${domain}.log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
    if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
      wait "${pids[@]}"
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait "${pids[@]}"
  fi
  publish_latest_symlink
  sync_outputs || true
}

run_sanity() {
  ensure_deps
  echo "[exp53] phase sanity"
  local idx=0
  local pids=()
  for domain in ${DOMAINS}; do
    local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      $PY_RUNNER -m src.poc.exp53_controlled_domain_finetunes.eval_domain_nll \
        --run-root "$RUN_ROOT" \
        --domain "$domain" \
        --base-model "$BASE_MODEL" \
        --base-revision "$BASE_REVISION" \
        --hf-token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
        --device cuda:0 \
        --max-tokens "$EVAL_TOKENS" \
        --seq-len "$SEQ_LEN"
      $PY_RUNNER -m src.poc.exp53_controlled_domain_finetunes.check_merge_equivalence \
        --run-root "$RUN_ROOT" \
        --domain "$domain" \
        --base-model "$BASE_MODEL" \
        --base-revision "$BASE_REVISION" \
        --hf-token "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}" \
        --device cuda:0
      $PY_RUNNER -m src.poc.exp53_controlled_domain_finetunes.generation_health \
        --run-root "$RUN_ROOT" \
        --domain "$domain" \
        --device cuda:0
    ) >"${LOG_DIR}/sanity_${domain}.log" 2>&1 &
    pids+=("$!")
    idx=$((idx + 1))
    if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
      wait "${pids[@]}"
      pids=()
    fi
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    wait "${pids[@]}"
  fi
  sync_outputs || true
}

run_exp47_slice() {
  local slice="$1"
  local dataset="$2"
  local n_examples="$3"
  local models="$4"
  local out_root="${RUN_ROOT}/factorial/${slice}"
  echo "[exp53] exp47 slice=${slice} dataset=${dataset} models=${models}"
  MODE=preflight RUN_ROOT="$out_root" DATASET="$dataset" MODELS="$models" GPU_LIST="$GPU_LIST" \
    WORKERS_PER_MODEL="$WORKERS_PER_MODEL" N_EXAMPLES="$n_examples" RUN_NATIVE_AUDIT=0 \
    N_BOOT="$N_BOOT" N_PERMUTATIONS="$N_PERMUTATIONS" \
    bash scripts/run/run_exp47_same_base_recipe_specificity_runpod.sh
  MODE=support RUN_ROOT="$out_root" DATASET="$dataset" MODELS="$models" GPU_LIST="$GPU_LIST" \
    WORKERS_PER_MODEL="$WORKERS_PER_MODEL" N_EXAMPLES="$n_examples" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" RUN_NATIVE_AUDIT=0 \
    N_BOOT="$N_BOOT" N_PERMUTATIONS="$N_PERMUTATIONS" \
    bash scripts/run/run_exp47_same_base_recipe_specificity_runpod.sh
  MODE=token RUN_ROOT="$out_root" DATASET="$dataset" MODELS="$models" GPU_LIST="$GPU_LIST" \
    WORKERS_PER_MODEL="$WORKERS_PER_MODEL" N_EXAMPLES="$n_examples" BOUNDARY_LAYER="$BOUNDARY_LAYER" RUN_NATIVE_AUDIT=0 \
    N_BOOT="$N_BOOT" N_PERMUTATIONS="$N_PERMUTATIONS" \
    bash scripts/run/run_exp47_same_base_recipe_specificity_runpod.sh
  MODE=analyze-only RUN_ROOT="$out_root" DATASET="$dataset" MODELS="$models" GPU_LIST="$GPU_LIST" \
    WORKERS_PER_MODEL="$WORKERS_PER_MODEL" N_EXAMPLES="$n_examples" RUN_NATIVE_AUDIT=0 \
    N_BOOT="$N_BOOT" N_PERMUTATIONS="$N_PERMUTATIONS" \
    bash scripts/run/run_exp47_same_base_recipe_specificity_runpod.sh
}

run_factorial() {
  publish_latest_symlink
  echo "[exp53] phase factorial"
  run_exp47_slice main_eval "$DATASET_MAIN" "$N_EXAMPLES_MAIN" "$MODELS"
  run_exp47_slice code_support "${RUN_ROOT}/data/code/support_dataset.jsonl" "$N_EXAMPLES_DOMAIN" "llama31_code_cpt_lora"
  run_exp47_slice biomed_support "${RUN_ROOT}/data/biomed/support_dataset.jsonl" "$N_EXAMPLES_DOMAIN" "llama31_biomed_cpt_lora"
  sync_outputs || true
}

run_analyze() {
  echo "[exp53] phase analyze"
  $PY_RUNNER scripts/analysis/analyze_exp53_controlled_domain_foils.py \
    --run-root "$RUN_ROOT" \
    >"${LOG_DIR}/analyze_exp53.log" 2>&1
  sync_outputs || true
}

case "$MODE" in
  prepare)
    run_prepare
    ;;
  train)
    run_train
    ;;
  merge)
    run_merge
    ;;
  sanity)
    run_sanity
    ;;
  factorial)
    run_factorial
    ;;
  analyze-only)
    run_analyze
    ;;
  sync)
    sync_outputs
    ;;
  smoke|full)
    run_prepare
    run_train
    run_merge
    run_sanity
    run_factorial
    run_analyze
    ;;
esac

echo "[exp53] complete ${RUN_ROOT}"
