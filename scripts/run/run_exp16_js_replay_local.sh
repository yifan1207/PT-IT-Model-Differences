#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="full"
RUN_NAME="exp16_js_replay_$(date +%Y%m%d_%H%M%S)"
SOURCE_RUN_ROOT="results/exp14_symmetric_matched_prefix_causality/exp13exp14_full_20260416"
DATASET="data/exp3_dataset.jsonl"
RUN_ROOT=""
LOG_DIR=""
MAX_NEW_TOKENS=512
SMOKE_MODEL="gemma3_4b"
SMOKE_PROMPTS=8
SMOKE_GPU=0
INCLUDE_DEEPSEEK=0

declare -A SOURCE_NUM_SHARDS=(
  [gemma3_4b]=1
  [qwen3_4b]=1
  [llama31_8b]=2
  [mistral_7b]=2
  [olmo2_7b]=2
  [deepseek_v2_lite]=7
)

usage() {
  cat <<EOF
Usage:
  bash scripts/run/run_exp16_js_replay_local.sh [options]

Options:
  --mode smoke|full
  --run-name NAME
  --source-run-root PATH         (default: canonical exp14 full run)
  --dataset PATH                 (default: data/exp3_dataset.jsonl)
  --run-root PATH                (default: results/exp16_matched_prefix_js_gap/<run-name>)
  --log-dir PATH                 (default: <run-root>/logs)
  --max-new-tokens N             (default: 512; kept as a consistency assertion)
  --smoke-model MODEL            (default: gemma3_4b)
  --smoke-prompts N              (default: 8)
  --smoke-gpu IDX                (default: 0)
  --include-deepseek             (appendix run after dense-5)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --source-run-root) SOURCE_RUN_ROOT="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --smoke-model) SMOKE_MODEL="$2"; shift 2 ;;
    --smoke-prompts) SMOKE_PROMPTS="$2"; shift 2 ;;
    --smoke-gpu) SMOKE_GPU="$2"; shift 2 ;;
    --include-deepseek) INCLUDE_DEEPSEEK=1; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="results/exp16_matched_prefix_js_gap/${RUN_NAME}"
fi
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${RUN_ROOT}/logs"
fi
mkdir -p "$RUN_ROOT" "$LOG_DIR" "${RUN_ROOT}/shards" "${RUN_ROOT}/merged"

batch_hint() {
  case "$1" in
    gemma3_4b|qwen3_4b) echo 64 ;;
    *) echo 48 ;;
  esac
}

ensure_source_merged() {
  local model="$1"
  local source_merged_dir="${SOURCE_RUN_ROOT}/merged/${model}"
  if [[ -f "${source_merged_dir}/prompts.jsonl" && -f "${source_merged_dir}/generated_texts.jsonl" ]]; then
    return 0
  fi
  local num_shards="${SOURCE_NUM_SHARDS[$model]:-}"
  if [[ -z "$num_shards" ]]; then
    echo "[exp16] unknown source shard count for ${model}" >&2
    exit 1
  fi
  echo "[exp16] merging source exp14 shards for ${model}" >&2
  uv run python scripts/merge/merge_exp11_shards_local.py \
    --run-root "$SOURCE_RUN_ROOT" \
    --model "$model" \
    --num-shards "$num_shards"
}

source_model_dir() {
  local model="$1"
  ensure_source_merged "$model"
  echo "${SOURCE_RUN_ROOT}/merged/${model}"
}

run_single() {
  local gpu="$1"
  local model="$2"
  local limit_prompts="$3"
  local shard_index="$4"
  local num_shards="$5"
  local out_dir="$6"
  local log_path="$7"
  local source_dir
  local batch_size
  source_dir="$(source_model_dir "$model")"
  batch_size="$(batch_hint "$model")"
  local -a cmd=(
    uv run python -m src.poc.exp16_matched_prefix_js_gap
    --model "$model"
    --dataset "$DATASET"
    --n-prompts 600
    --prompt-manifest "${source_dir}/prompts.jsonl"
    --teacher-token-manifest "${source_dir}/generated_texts.jsonl"
    --teacher-forced
    --causal-combined
    --js-only
    --readout-mode raw
    --max-new-tokens "$MAX_NEW_TOKENS"
    --chunk-size 32
    --batch-size "$batch_size"
    --num-shards "$num_shards"
    --shard-index "$shard_index"
    --out-dir "$out_dir"
  )
  if [[ -n "$limit_prompts" ]]; then
    cmd+=(--limit-prompts "$limit_prompts")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_path" 2>&1
}

if [[ "$MODE" == "smoke" ]]; then
  echo "[exp16] starting smoke on ${SMOKE_MODEL} gpu=${SMOKE_GPU}"
  run_single \
    "$SMOKE_GPU" \
    "$SMOKE_MODEL" \
    "$SMOKE_PROMPTS" \
    0 \
    1 \
    "${RUN_ROOT}/${SMOKE_MODEL}" \
    "${LOG_DIR}/${SMOKE_MODEL}_smoke.log"
  uv run python scripts/analysis/analyze_exp16.py --run-root "$RUN_ROOT" --out "${RUN_ROOT}/js_summary.json"
  uv run python scripts/plot/plot_exp16.py --summary "${RUN_ROOT}/js_summary.json" --out-dir "$RUN_ROOT"
  echo "[exp16] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

launch_bg() {
  local gpu="$1"
  local model="$2"
  local shard_index="$3"
  local num_shards="$4"
  local out_dir="${RUN_ROOT}/shards/${model}__shard${shard_index}of${num_shards}"
  local log_path="${LOG_DIR}/${model}__shard${shard_index}of${num_shards}.log"
  echo "[exp16] launch model=${model} shard=${shard_index}/${num_shards} gpu=${gpu}" >&2
  run_single "$gpu" "$model" "" "$shard_index" "$num_shards" "$out_dir" "$log_path" &
  LAUNCHED_PID=$!
}

launch_bg 0 llama31_8b 0 2; pid_llama0="$LAUNCHED_PID"
launch_bg 1 llama31_8b 1 2; pid_llama1="$LAUNCHED_PID"
launch_bg 2 mistral_7b 0 2; pid_mistral0="$LAUNCHED_PID"
launch_bg 3 mistral_7b 1 2; pid_mistral1="$LAUNCHED_PID"
launch_bg 4 olmo2_7b 0 2; pid_olmo0="$LAUNCHED_PID"
launch_bg 5 olmo2_7b 1 2; pid_olmo1="$LAUNCHED_PID"
launch_bg 6 gemma3_4b 0 1; pid_gemma="$LAUNCHED_PID"
launch_bg 7 qwen3_4b 0 1; pid_qwen="$LAUNCHED_PID"

wait "$pid_llama0"; echo "[exp16] finished llama31_8b shard 0/2"
wait "$pid_llama1"; echo "[exp16] finished llama31_8b shard 1/2"
wait "$pid_mistral0"; echo "[exp16] finished mistral_7b shard 0/2"
wait "$pid_mistral1"; echo "[exp16] finished mistral_7b shard 1/2"
wait "$pid_olmo0"; echo "[exp16] finished olmo2_7b shard 0/2"
wait "$pid_olmo1"; echo "[exp16] finished olmo2_7b shard 1/2"
wait "$pid_gemma"; echo "[exp16] finished gemma3_4b shard 0/1"
wait "$pid_qwen"; echo "[exp16] finished qwen3_4b shard 0/1"

uv run python scripts/merge/merge_exp16_shards.py --run-root "$RUN_ROOT" --model gemma3_4b --num-shards 1
uv run python scripts/merge/merge_exp16_shards.py --run-root "$RUN_ROOT" --model qwen3_4b --num-shards 1
uv run python scripts/merge/merge_exp16_shards.py --run-root "$RUN_ROOT" --model llama31_8b --num-shards 2
uv run python scripts/merge/merge_exp16_shards.py --run-root "$RUN_ROOT" --model mistral_7b --num-shards 2
uv run python scripts/merge/merge_exp16_shards.py --run-root "$RUN_ROOT" --model olmo2_7b --num-shards 2

if [[ "$INCLUDE_DEEPSEEK" == "1" ]]; then
  launch_bg 6 deepseek_v2_lite 0 1; pid_deepseek="$LAUNCHED_PID"
  wait "$pid_deepseek"; echo "[exp16] finished deepseek_v2_lite shard 0/1"
  uv run python scripts/merge/merge_exp16_shards.py --run-root "$RUN_ROOT" --model deepseek_v2_lite --num-shards 1
fi

uv run python scripts/analysis/analyze_exp16.py --run-root "$RUN_ROOT" --out "${RUN_ROOT}/js_summary.json"
uv run python scripts/plot/plot_exp16.py --summary "${RUN_ROOT}/js_summary.json" --out-dir "$RUN_ROOT"

echo "[exp16] full run complete -> ${RUN_ROOT}"
