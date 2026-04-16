#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="full"
RUN_NAME="exp13_exp14_$(date +%Y%m%d_%H%M%S)"
MODEL="gemma3_4b"
N_PROMPTS=600
PROMPT_SEED=0
MAX_NEW_TOKENS=512
READOUT_MODE="both"
TUNED_LENS_DIR="${HOME}/.cache/exp11_tuned_lens_probes_v3"
DATASET="data/exp3_dataset.jsonl"
LOG_DIR=""
RUN_ROOT=""
SMOKE_PROMPTS=8
SMOKE_GPU=0

usage() {
  cat <<EOF
Usage:
  bash scripts/run_exp13_exp14_local.sh [options]

Options:
  --mode smoke|full
  --run-name NAME
  --model MODEL                  (smoke mode only, default: gemma3_4b)
  --n-prompts N                  (full mode default: 600)
  --smoke-prompts N              (smoke mode default: 8)
  --prompt-seed N                (default: 0)
  --max-new-tokens N             (default: 512)
  --tuned-lens-dir PATH
  --dataset PATH                 (default: data/exp3_dataset.jsonl)
  --run-root PATH                (default: results/exp13/<run-name>)
  --log-dir PATH                 (default: <run-root>/logs)
  --smoke-gpu IDX                (smoke mode default: 0)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --n-prompts) N_PROMPTS="$2"; shift 2 ;;
    --smoke-prompts) SMOKE_PROMPTS="$2"; shift 2 ;;
    --prompt-seed) PROMPT_SEED="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --tuned-lens-dir) TUNED_LENS_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-dir) LOG_DIR="$2"; shift 2 ;;
    --smoke-gpu) SMOKE_GPU="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="results/exp13/${RUN_NAME}"
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

run_single() {
  local gpu="$1"
  local model="$2"
  local n_prompts="$3"
  local shard_index="$4"
  local num_shards="$5"
  local out_dir="$6"
  local log_path="$7"
  local batch_size
  batch_size="$(batch_hint "$model")"
  CUDA_VISIBLE_DEVICES="$gpu" \
  uv run python -m src.poc.exp11.run \
    --model "$model" \
    --dataset "$DATASET" \
    --n-prompts "$n_prompts" \
    --prompt-seed "$PROMPT_SEED" \
    --teacher-forced \
    --causal-combined \
    --readout-mode "$READOUT_MODE" \
    --tuned-lens-dir "$TUNED_LENS_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --chunk-size 64 \
    --batch-size "$batch_size" \
    --num-shards "$num_shards" \
    --shard-index "$shard_index" \
    --out-dir "$out_dir" \
    >"$log_path" 2>&1
}

if [[ "$MODE" == "smoke" ]]; then
  echo "[exp13/14] starting smoke on ${MODEL} gpu=${SMOKE_GPU}"
  run_single \
    "$SMOKE_GPU" \
    "$MODEL" \
    "$SMOKE_PROMPTS" \
    0 \
    1 \
    "${RUN_ROOT}/${MODEL}" \
    "${LOG_DIR}/${MODEL}_smoke.log"
  uv run python scripts/analyze_exp13_full.py --run-root "$RUN_ROOT" --out "${RUN_ROOT}/exp13_full_summary.json"
  uv run python scripts/plot_exp13_full.py --summary "${RUN_ROOT}/exp13_full_summary.json" --out-dir "$RUN_ROOT"
  echo "[exp13/14] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

launch_bg() {
  local gpu="$1"
  local model="$2"
  local shard_index="$3"
  local num_shards="$4"
  local out_dir="${RUN_ROOT}/shards/${model}__shard${shard_index}of${num_shards}"
  local log_path="${LOG_DIR}/${model}__shard${shard_index}of${num_shards}.log"
  echo "[exp13/14] launch model=${model} shard=${shard_index}/${num_shards} gpu=${gpu}"
  run_single "$gpu" "$model" "$N_PROMPTS" "$shard_index" "$num_shards" "$out_dir" "$log_path" &
  echo $!
}

pid_llama0="$(launch_bg 0 llama31_8b 0 2)"
pid_llama1="$(launch_bg 1 llama31_8b 1 2)"
pid_mistral0="$(launch_bg 2 mistral_7b 0 2)"
pid_mistral1="$(launch_bg 3 mistral_7b 1 2)"
pid_olmo0="$(launch_bg 4 olmo2_7b 0 2)"
pid_olmo1="$(launch_bg 5 olmo2_7b 1 2)"
pid_gemma="$(launch_bg 6 gemma3_4b 0 1)"
pid_qwen="$(launch_bg 7 qwen3_4b 0 1)"

wait "$pid_gemma"
echo "[exp13/14] finished gemma3_4b shard 0/1"
pid_deepseek="$(launch_bg 6 deepseek_v2_lite 0 1)"
echo "[exp13/14] queued deepseek on gpu 6"

wait "$pid_llama0"; echo "[exp13/14] finished llama31_8b shard 0/2"
wait "$pid_llama1"; echo "[exp13/14] finished llama31_8b shard 1/2"
wait "$pid_mistral0"; echo "[exp13/14] finished mistral_7b shard 0/2"
wait "$pid_mistral1"; echo "[exp13/14] finished mistral_7b shard 1/2"
wait "$pid_olmo0"; echo "[exp13/14] finished olmo2_7b shard 0/2"
wait "$pid_olmo1"; echo "[exp13/14] finished olmo2_7b shard 1/2"
wait "$pid_qwen"; echo "[exp13/14] finished qwen3_4b shard 0/1"
wait "$pid_deepseek"; echo "[exp13/14] finished deepseek_v2_lite shard 0/1"

uv run python scripts/merge_exp11_shards_local.py --run-root "$RUN_ROOT" --model gemma3_4b --num-shards 1
uv run python scripts/merge_exp11_shards_local.py --run-root "$RUN_ROOT" --model qwen3_4b --num-shards 1
uv run python scripts/merge_exp11_shards_local.py --run-root "$RUN_ROOT" --model deepseek_v2_lite --num-shards 1
uv run python scripts/merge_exp11_shards_local.py --run-root "$RUN_ROOT" --model llama31_8b --num-shards 2
uv run python scripts/merge_exp11_shards_local.py --run-root "$RUN_ROOT" --model mistral_7b --num-shards 2
uv run python scripts/merge_exp11_shards_local.py --run-root "$RUN_ROOT" --model olmo2_7b --num-shards 2

uv run python scripts/analyze_exp13_full.py --run-root "$RUN_ROOT" --out "${RUN_ROOT}/exp13_full_summary.json"
uv run python scripts/plot_exp13_full.py --summary "${RUN_ROOT}/exp13_full_summary.json" --out-dir "$RUN_ROOT"

echo "[exp13/14] full run complete -> ${RUN_ROOT}"
