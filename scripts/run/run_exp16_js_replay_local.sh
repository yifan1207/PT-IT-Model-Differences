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
TEACHER_SOURCE="it"
TEACHER_MANIFEST_ROOT=""

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
  --teacher-source it|pt         (default: it)
  --teacher-manifest-root PATH   (default: <run-root>/teacher_manifests for pt source)
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
    --teacher-source) TEACHER_SOURCE="$2"; shift 2 ;;
    --teacher-manifest-root) TEACHER_MANIFEST_ROOT="$2"; shift 2 ;;
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
if [[ -z "$TEACHER_MANIFEST_ROOT" ]]; then
  TEACHER_MANIFEST_ROOT="${RUN_ROOT}/teacher_manifests"
fi
mkdir -p "$RUN_ROOT" "$LOG_DIR" "${RUN_ROOT}/shards" "${RUN_ROOT}/merged"

if [[ "$TEACHER_SOURCE" != "it" && "$TEACHER_SOURCE" != "pt" ]]; then
  echo "[exp16] --teacher-source must be one of: it, pt" >&2
  exit 1
fi

batch_hint() {
  case "$1" in
    gemma3_4b|qwen3_4b) echo 64 ;;
    *) echo 48 ;;
  esac
}

teacher_batch_hint() {
  case "$1" in
    gemma3_4b|qwen3_4b) echo 96 ;;
    *) echo 64 ;;
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

teacher_manifest_path_for_model() {
  local model="$1"
  local source_dir
  source_dir="$(source_model_dir "$model")"
  if [[ "$TEACHER_SOURCE" == "it" ]]; then
    echo "${source_dir}/generated_texts.jsonl"
  else
    mkdir -p "$TEACHER_MANIFEST_ROOT"
    echo "${TEACHER_MANIFEST_ROOT}/${model}_pt_teacher_manifest.jsonl"
  fi
}

teacher_pipeline_for_source() {
  if [[ "$TEACHER_SOURCE" == "it" ]]; then
    echo "C_it_chat"
  else
    echo "A_pt_raw"
  fi
}

assert_teacher_manifest_ready() {
  local model="$1"
  if [[ "$TEACHER_SOURCE" == "it" ]]; then
    local source_dir
    source_dir="$(source_model_dir "$model")"
    if [[ ! -s "${source_dir}/generated_texts.jsonl" ]]; then
      echo "[exp16] missing IT teacher manifest rows for ${model} at ${source_dir}/generated_texts.jsonl" >&2
      exit 1
    fi
    return 0
  fi

  local teacher_manifest_path
  teacher_manifest_path="$(teacher_manifest_path_for_model "$model")"
  if [[ ! -s "$teacher_manifest_path" ]]; then
    echo "[exp16] missing PT teacher manifest for ${model} at ${teacher_manifest_path}" >&2
    exit 1
  fi
}

build_teacher_manifest() {
  local gpu="$1"
  local model="$2"
  local limit_prompts="${3:-}"
  local source_dir teacher_manifest_path batch_size
  source_dir="$(source_model_dir "$model")"
  teacher_manifest_path="$(teacher_manifest_path_for_model "$model")"
  if [[ -s "$teacher_manifest_path" && -z "$limit_prompts" ]]; then
    echo "[exp16] PT teacher manifest already present for ${model}" >&2
    return 0
  fi
  if [[ -n "$limit_prompts" ]]; then
    teacher_manifest_path="${TEACHER_MANIFEST_ROOT}/${model}_pt_teacher_manifest_smoke_${limit_prompts}.jsonl"
  fi
  batch_size="$(teacher_batch_hint "$model")"
  echo "[exp16] building PT teacher manifest for ${model} on gpu=${gpu}" >&2
  local -a cmd=(
    uv run python scripts/precompute/build_exp16_teacher_manifest.py
    --model "$model"
    --dataset "$DATASET"
    --prompt-manifest "${source_dir}/prompts.jsonl"
    --variant pt
    --prompt-mode raw_format_b
    --pipeline-name A_pt_raw
    --max-new-tokens "$MAX_NEW_TOKENS"
    --batch-size "$batch_size"
    --device cuda:0
    --out "$teacher_manifest_path"
    --resume
  )
  if [[ -n "$limit_prompts" ]]; then
    cmd+=(--limit-prompts "$limit_prompts")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" \
  "${cmd[@]}"
}

build_teacher_manifest_path() {
  local model="$1"
  local limit_prompts="${2:-}"
  if [[ -n "$limit_prompts" ]]; then
    echo "${TEACHER_MANIFEST_ROOT}/${model}_pt_teacher_manifest_smoke_${limit_prompts}.jsonl"
  else
    echo "${TEACHER_MANIFEST_ROOT}/${model}_pt_teacher_manifest.jsonl"
  fi
}

run_single() {
  local gpu="$1"
  local model="$2"
  local limit_prompts="$3"
  local shard_index="$4"
  local num_shards="$5"
  local out_dir="$6"
  local log_path="$7"
  local teacher_manifest_override="${8:-}"
  local source_dir
  local teacher_manifest_path
  local teacher_pipeline
  local batch_size
  source_dir="$(source_model_dir "$model")"
  if [[ -n "$teacher_manifest_override" ]]; then
    teacher_manifest_path="$teacher_manifest_override"
  else
    assert_teacher_manifest_ready "$model"
    teacher_manifest_path="$(teacher_manifest_path_for_model "$model")"
  fi
  teacher_pipeline="$(teacher_pipeline_for_source)"
  batch_size="$(batch_hint "$model")"
  local -a cmd=(
    uv run python -m src.poc.exp16_matched_prefix_js_gap
    --model "$model"
    --dataset "$DATASET"
    --n-prompts 600
    --prompt-manifest "${source_dir}/prompts.jsonl"
    --teacher-token-manifest "$teacher_manifest_path"
    --teacher-pipeline "$teacher_pipeline"
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
    --resume
  )
  if [[ -n "$limit_prompts" ]]; then
    cmd+=(--limit-prompts "$limit_prompts")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" >"$log_path" 2>&1
}

if [[ "$MODE" == "smoke" ]]; then
  echo "[exp16] starting smoke on ${SMOKE_MODEL} gpu=${SMOKE_GPU}"
  local_teacher_manifest_path=""
  if [[ "$TEACHER_SOURCE" == "pt" ]]; then
    build_teacher_manifest "$SMOKE_GPU" "$SMOKE_MODEL" "$SMOKE_PROMPTS"
    local_teacher_manifest_path="$(build_teacher_manifest_path "$SMOKE_MODEL" "$SMOKE_PROMPTS")"
  fi
  run_single \
    "$SMOKE_GPU" \
    "$SMOKE_MODEL" \
    "$SMOKE_PROMPTS" \
    0 \
    1 \
    "${RUN_ROOT}/${SMOKE_MODEL}" \
    "${LOG_DIR}/${SMOKE_MODEL}_smoke.log" \
    "$local_teacher_manifest_path"
  uv run python scripts/analysis/analyze_exp16.py --run-root "$RUN_ROOT" --out "${RUN_ROOT}/js_summary.json"
  uv run python scripts/plot/plot_exp16.py --summary "${RUN_ROOT}/js_summary.json" --out-dir "$RUN_ROOT"
  echo "[exp16] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

if [[ "$TEACHER_SOURCE" == "pt" ]]; then
  build_teacher_manifest 0 llama31_8b &
  pid_teacher_llama=$!
  build_teacher_manifest 1 mistral_7b &
  pid_teacher_mistral=$!
  build_teacher_manifest 2 olmo2_7b &
  pid_teacher_olmo=$!
  build_teacher_manifest 3 gemma3_4b &
  pid_teacher_gemma=$!
  build_teacher_manifest 4 qwen3_4b &
  pid_teacher_qwen=$!

  wait "$pid_teacher_llama"; echo "[exp16] PT teacher ready for llama31_8b"
  wait "$pid_teacher_mistral"; echo "[exp16] PT teacher ready for mistral_7b"
  wait "$pid_teacher_olmo"; echo "[exp16] PT teacher ready for olmo2_7b"
  wait "$pid_teacher_gemma"; echo "[exp16] PT teacher ready for gemma3_4b"
  wait "$pid_teacher_qwen"; echo "[exp16] PT teacher ready for qwen3_4b"

  if [[ "$INCLUDE_DEEPSEEK" == "1" ]]; then
    build_teacher_manifest 6 deepseek_v2_lite
    echo "[exp16] PT teacher ready for deepseek_v2_lite"
  fi
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
