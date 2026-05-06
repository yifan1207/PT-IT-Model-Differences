#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${MODE:-smoke}"  # smoke|full|analyze-only|sync
RUN_NAME="${RUN_NAME:-exp55_late_window_robustness_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp55_late_window_robustness/${RUN_NAME}}"
DATASET="${DATASET:-data/exp3_dataset.jsonl}"
MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SHARDS_BY_MODEL="${SHARDS_BY_MODEL:-}"
SHARD_INDICES_BY_MODEL="${SHARD_INDICES_BY_MODEL:-}"
N_PROMPTS="${N_PROMPTS:-600}"
PROMPT_SEED="${PROMPT_SEED:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
TUNED_LENS_DIR="${TUNED_LENS_DIR:-/workspace/tuned_lens_probes}"
WINDOW_SPECS="${WINDOW_SPECS:-prelate_half late_full late_front_half late_center_half late_terminal_half terminal_quarter}"
N_BOOT="${N_BOOT:-2000}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp55_late_window_robustness}"
WRITE_PAPER_SYNTHESIS="${WRITE_PAPER_SYNTHESIS:-0}"
MERGE_AFTER="${MERGE_AFTER:-1}"
ANALYZE_AFTER="${ANALYZE_AFTER:-1}"

if [[ -n "${PY_RUNNER:-}" ]]; then
  read -r -a PYTHON <<< "$PY_RUNNER"
elif [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  PYTHON=(uv run --no-sync python)
else
  PYTHON=(uv run python)
fi

read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp55] GPU_LIST must contain at least one GPU id" >&2
  exit 1
fi

if [[ "$MODE" == "smoke" ]]; then
  MODELS="${SMOKE_MODELS:-gemma3_4b}"
  N_PROMPTS="${SMOKE_PROMPTS:-8}"
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-16}"
  WINDOW_SPECS="${SMOKE_WINDOW_SPECS:-late_full late_terminal_half}"
  N_BOOT="${SMOKE_N_BOOT:-50}"
  GCS_SYNC_DEST="${SMOKE_GCS_SYNC_DEST:-none}"
fi

if [[ "$MODE" == "sync" ]]; then
  if [[ -z "$GCS_SYNC_DEST" || "$GCS_SYNC_DEST" == "none" || "$GCS_SYNC_DEST" == "NONE" ]]; then
    echo "[exp55] GCS sync disabled"
    exit 0
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  fi
  exit 0
fi

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/shards" "$RUN_ROOT/merged"

echo "[exp55] host $(hostname)"
echo "[exp55] mode ${MODE}"
echo "[exp55] run_root ${RUN_ROOT}"
echo "[exp55] dataset ${DATASET}"
echo "[exp55] models ${MODELS}"
echo "[exp55] window_specs ${WINDOW_SPECS}"
echo "[exp55] max_new_tokens ${MAX_NEW_TOKENS}"
echo "[exp55] chunk_size ${CHUNK_SIZE}"
echo "[exp55] gpu_list ${GPU_LIST}"
echo "[exp55] shards_by_model ${SHARDS_BY_MODEL:-default}"
echo "[exp55] shard_indices_by_model ${SHARD_INDICES_BY_MODEL:-all}"
echo "[exp55] merge_after ${MERGE_AFTER}"
echo "[exp55] analyze_after ${ANALYZE_AFTER}"

batch_hint() {
  case "$1" in
    gemma3_4b|qwen3_4b) echo 48 ;;
    *) echo 32 ;;
  esac
}

num_shards_for_model() {
  if [[ -n "$SHARDS_BY_MODEL" ]]; then
    local item key value
    for item in $SHARDS_BY_MODEL; do
      key="${item%%:*}"
      value="${item#*:}"
      if [[ "$key" == "$1" && "$value" =~ ^[0-9]+$ && "$value" -gt 0 ]]; then
        echo "$value"
        return
      fi
    done
  fi
  case "$1" in
    llama31_8b|mistral_7b|olmo2_7b) echo 2 ;;
    *) echo 1 ;;
  esac
}

shard_indices_for_model() {
  local model="$1"
  local num_shards="$2"
  if [[ -n "$SHARD_INDICES_BY_MODEL" ]]; then
    local item key value
    for item in $SHARD_INDICES_BY_MODEL; do
      key="${item%%:*}"
      value="${item#*:}"
      if [[ "$key" == "$model" ]]; then
        echo "${value//,/ }"
        return
      fi
    done
  fi
  local indices=()
  local shard
  for ((shard=0; shard<num_shards; shard++)); do
    indices+=("$shard")
  done
  echo "${indices[*]}"
}

run_single() {
  local gpu="$1"
  local model="$2"
  local shard_index="$3"
  local num_shards="$4"
  local out_dir="${RUN_ROOT}/shards/${model}__shard${shard_index}of${num_shards}"
  local log_path="${RUN_ROOT}/logs/${model}__shard${shard_index}of${num_shards}.log"
  local batch_size
  batch_size="$(batch_hint "$model")"
  echo "[exp55] launch model=${model} shard=${shard_index}/${num_shards} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" \
  "${PYTHON[@]}" -m src.poc.exp14_symmetric_matched_prefix_causality \
    --model "$model" \
    --dataset "$DATASET" \
    --n-prompts "$N_PROMPTS" \
    --prompt-seed "$PROMPT_SEED" \
    --teacher-forced \
    --late-window-sweep \
    --late-window-specs $WINDOW_SPECS \
    --readout-mode raw \
    --tuned-lens-dir "$TUNED_LENS_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --chunk-size "$CHUNK_SIZE" \
    --batch-size "$batch_size" \
    --num-shards "$num_shards" \
    --shard-index "$shard_index" \
    --out-dir "$out_dir" \
    >"$log_path" 2>&1
}

if [[ "$MODE" != "analyze-only" ]]; then
  pids=()
  gpu_cursor=0
  for model in $MODELS; do
    shards="$(num_shards_for_model "$model")"
    read -r -a shard_indices <<< "$(shard_indices_for_model "$model" "$shards")"
    for shard in "${shard_indices[@]}"; do
      if [[ ! "$shard" =~ ^[0-9]+$ || "$shard" -lt 0 || "$shard" -ge "$shards" ]]; then
        echo "[exp55] invalid shard index ${shard} for model=${model} num_shards=${shards}" >&2
        exit 1
      fi
      gpu="${GPUS[$((gpu_cursor % ${#GPUS[@]}))]}"
      run_single "$gpu" "$model" "$shard" "$shards" &
      pids+=("$!")
      gpu_cursor=$((gpu_cursor + 1))
    done
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  if [[ "$MERGE_AFTER" == "1" ]]; then
    for model in $MODELS; do
      shards="$(num_shards_for_model "$model")"
      "${PYTHON[@]}" scripts/merge_exp11_shards_local.py \
        --run-root "$RUN_ROOT" \
        --model "$model" \
        --num-shards "$shards"
    done
  fi
fi

if [[ "$ANALYZE_AFTER" == "1" ]]; then
  echo "[exp55] analyzing"
  "${PYTHON[@]}" scripts/analysis/analyze_exp55_late_window_robustness.py \
    --run-root "$RUN_ROOT" \
    --out-dir "$RUN_ROOT/analysis" \
    --models $MODELS \
    --n-boot "$N_BOOT"
fi

if [[ "$WRITE_PAPER_SYNTHESIS" == "1" && "$ANALYZE_AFTER" == "1" ]]; then
  cp "$RUN_ROOT/analysis/exp55_late_window_robustness.json" results/paper_synthesis/
  cp "$RUN_ROOT/analysis/exp55_late_window_robustness_effects.csv" results/paper_synthesis/
  cp "$RUN_ROOT/analysis/exp55_late_window_robustness_note.md" results/paper_synthesis/
  if [[ -f "$RUN_ROOT/analysis/exp55_late_window_robustness.png" ]]; then
    cp "$RUN_ROOT/analysis/exp55_late_window_robustness.png" results/paper_synthesis/
  fi
fi

if [[ -n "$GCS_SYNC_DEST" && "$GCS_SYNC_DEST" != "none" && "$GCS_SYNC_DEST" != "NONE" ]]; then
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24 || true
  fi
fi

echo "[exp55] complete ${RUN_ROOT}"
