#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODE="${MODE:-preflight}"  # preflight|smoke|run-one|full|analyze-only
RUN_NAME="${RUN_NAME:-exp34_dense5_final_readout_crosscoder_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp34_dense5_final_readout_crosscoder/${RUN_NAME}}"
OUT_DIR="${OUT_DIR:-results/paper_synthesis/exp34_dense5_final_readout_crosscoder/${RUN_NAME}}"
MODEL_LIST="${MODEL_LIST:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
MODEL_NAME="${MODEL_NAME:-}"
LLAMA_MODE="${LLAMA_MODE:-reuse}"  # reuse|rerun
LLAMA_ROOT="${LLAMA_ROOT:-results/exp30_final_readout_crosscoder_mediation/exp30_l31_paperfaithful_runpod_20260502_012105_a100x8/selected_d131072_k64}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SMOKE_GPU_LIST="${SMOKE_GPU_LIST:-$GPU_LIST}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
EXP30_RUNNER="${EXP30_RUNNER:-scripts/run/run_exp30_final_readout_crosscoder_mediation.sh}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-gs://pt-vs-it-results/results/exp34_dense5_final_readout_crosscoder}"

DATASETS_TRAIN="${DATASETS_TRAIN:-data/eval_dataset_v2.jsonl data/exp3_dataset.jsonl data/exp6_dataset.jsonl}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
DATASET_EVAL="${DATASET_EVAL:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"

# Full selected-run defaults, inherited from Exp30.
N_TOKENS="${N_TOKENS:-2000000}"
SELECTED_STEPS="${SELECTED_STEPS:-24000}"
SELECTED_BATCH_TOKENS="${SELECTED_BATCH_TOKENS:-512}"
SELECTED_AUXK="${SELECTED_AUXK:-1024}"
CAUSAL_RANK_PROMPTS="${CAUSAL_RANK_PROMPTS:-160}"
CAUSAL_MEDIATE_SKIP_PROMPTS="${CAUSAL_MEDIATE_SKIP_PROMPTS:-160}"
FULL_PROMPTS="${FULL_PROMPTS:-440}"
CAUSAL_K_LIST="${CAUSAL_K_LIST:-25 50 100 200 500}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
N_BOOT="${N_BOOT:-1000}"

# Fast all-model GPU smoke. This still exercises final-three cache/train/rank/mediate.
SMOKE_N_TOKENS="${SMOKE_N_TOKENS:-8000}"
SMOKE_N_TRAIN_PROMPTS="${SMOKE_N_TRAIN_PROMPTS:-30}"
SMOKE_DICT_SIZE="${SMOKE_DICT_SIZE:-1024}"
SMOKE_K="${SMOKE_K:-8}"
SMOKE_STEPS="${SMOKE_STEPS:-40}"
SMOKE_BATCH_TOKENS="${SMOKE_BATCH_TOKENS:-256}"
SMOKE_AUXK="${SMOKE_AUXK:-32}"
SMOKE_RANK_PROMPTS="${SMOKE_RANK_PROMPTS:-8}"
SMOKE_MEDIATE_SKIP_PROMPTS="${SMOKE_MEDIATE_SKIP_PROMPTS:-8}"
SMOKE_FULL_PROMPTS="${SMOKE_FULL_PROMPTS:-8}"
SMOKE_CAUSAL_K_LIST="${SMOKE_CAUSAL_K_LIST:-2 4 8}"
SMOKE_RANDOM_SEEDS="${SMOKE_RANDOM_SEEDS:-0}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-50}"

usage() {
  cat <<EOF
Usage:
  MODE=preflight|smoke|run-one|full|analyze-only bash scripts/run/run_exp34_dense5_final_readout_crosscoder.sh

Key environment:
  RUN_ROOT, OUT_DIR, MODEL_LIST, MODEL_NAME, LLAMA_MODE=reuse|rerun,
  GPU_LIST, SMOKE_GPU_LIST, GCS_SYNC_DEST, PY_RUNNER.

Full mode runs models sequentially on one 8-GPU node. For multiple RunPod
nodes, launch this script once per pod with MODEL_LIST or MODEL_NAME set to
one model, then run analyze-only on the shared/pulled result root.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  preflight|smoke|run-one|full|analyze-only) ;;
  *)
    echo "[exp34] invalid MODE=${MODE}" >&2
    usage
    exit 2
    ;;
esac

if [[ "$PROMPT_MODE" != "raw_shared" ]]; then
  echo "[exp34] raw_shared only; got PROMPT_MODE=${PROMPT_MODE}" >&2
  exit 2
fi
if [[ "$EVENT_KINDS" != "first_diff" ]]; then
  echo "[exp34] first_diff only; got EVENT_KINDS=${EVENT_KINDS}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$RUN_ROOT/logs" "$OUT_DIR"

model_config_env() {
  local model="$1"
  $PY_RUNNER - "$model" <<'PY'
import sys
from src.poc.cross_model.config import get_spec

model = sys.argv[1]
spec = get_spec(model)
layers = " ".join(str(x) for x in range(spec.n_layers - 3, spec.n_layers))
if spec.d_model == 4096:
    name = "d131072_k64"
    dict_size = 131072
elif spec.d_model == 2560:
    name = "d81920_k64"
    dict_size = 81920
else:
    dict_size = 32 * spec.d_model
    name = f"d{dict_size}_k64"
print(f"EXP34_LAYERS='{layers}'")
print(f"EXP34_SELECTED_NAME='{name}'")
print(f"EXP34_DICT_SIZE='{dict_size}'")
print("EXP34_K='64'")
print(f"EXP34_D_MODEL='{spec.d_model}'")
PY
}

preflight() {
  echo "[exp34] preflight root=${RUN_ROOT}"
  $PY_RUNNER scripts/analysis/analyze_exp34_dense5_final_readout_crosscoder.py \
    --preflight-only \
    --run-root "$RUN_ROOT" \
    --out-dir "$OUT_DIR" \
    --models ${MODEL_LIST} \
    --dataset "$DATASET_EVAL" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    --rank-prompts "$CAUSAL_RANK_PROMPTS" \
    --mediate-prompts "$FULL_PROMPTS" \
    >"${RUN_ROOT}/logs/preflight.log" 2>&1
  cat "${RUN_ROOT}/logs/preflight.log"
}

run_model() {
  local model="$1"
  local gpu_list="$2"
  local model_mode="$3"
  eval "$(model_config_env "$model")"
  local model_root="${RUN_ROOT}/${model}"
  local selected_name="$EXP34_SELECTED_NAME"
  local dict_size="$EXP34_DICT_SIZE"
  local kk="$EXP34_K"
  local steps="$SELECTED_STEPS"
  local batch_tokens="$SELECTED_BATCH_TOKENS"
  local auxk="$SELECTED_AUXK"
  local n_tokens="$N_TOKENS"
  local n_train_prompts=0
  local rank_prompts="$CAUSAL_RANK_PROMPTS"
  local skip_prompts="$CAUSAL_MEDIATE_SKIP_PROMPTS"
  local full_prompts="$FULL_PROMPTS"
  local k_list="$CAUSAL_K_LIST"
  local random_seeds="$RANDOM_SEEDS"
  local n_boot="$N_BOOT"
  local cache_batch_size="${CACHE_BATCH_SIZE:-8}"
  local cache_workers="${CACHE_WORKERS:-0}"
  local sync_dest="$GCS_SYNC_DEST"

  if [[ "$model_mode" == "smoke" ]]; then
    selected_name="smoke_d${SMOKE_DICT_SIZE}_k${SMOKE_K}"
    dict_size="$SMOKE_DICT_SIZE"
    kk="$SMOKE_K"
    steps="$SMOKE_STEPS"
    batch_tokens="$SMOKE_BATCH_TOKENS"
    auxk="$SMOKE_AUXK"
    n_tokens="$SMOKE_N_TOKENS"
    n_train_prompts="$SMOKE_N_TRAIN_PROMPTS"
    rank_prompts="$SMOKE_RANK_PROMPTS"
    skip_prompts="$SMOKE_MEDIATE_SKIP_PROMPTS"
    full_prompts="$SMOKE_FULL_PROMPTS"
    k_list="$SMOKE_CAUSAL_K_LIST"
    random_seeds="$SMOKE_RANDOM_SEEDS"
    n_boot="$SMOKE_N_BOOT"
    cache_batch_size="${SMOKE_CACHE_BATCH_SIZE:-2}"
    cache_workers=1
    sync_dest="${SMOKE_GCS_SYNC_DEST:-}"
  fi

  echo "[exp34] ${model_mode} model=${model} layers=${EXP34_LAYERS} d_model=${EXP34_D_MODEL} config=${selected_name} gpus=${gpu_list}"
  MODEL="$model" \
  RUN_NAME="${RUN_NAME}_${model}" \
  RUN_ROOT="$model_root" \
  MODE=full-selected \
  DATASETS_TRAIN="$DATASETS_TRAIN" \
  EXCLUDE_DATASETS="$EXCLUDE_DATASETS" \
  DATASET_EVAL="$DATASET_EVAL" \
  EXP20_ROOT="$EXP20_ROOT" \
  EXP20_FALLBACK_ROOT="$EXP20_FALLBACK_ROOT" \
  PROMPT_MODE="$PROMPT_MODE" \
  EVENT_KINDS="$EVENT_KINDS" \
  GPU_LIST="$gpu_list" \
  N_TOKENS="$n_tokens" \
  N_TRAIN_PROMPTS="$n_train_prompts" \
  SELECTED_LAYERS="$EXP34_LAYERS" \
  SELECTED_NAME="$selected_name" \
  SELECTED_DICT_SIZE="$dict_size" \
  SELECTED_K="$kk" \
  SELECTED_STEPS="$steps" \
  SELECTED_BATCH_TOKENS="$batch_tokens" \
  SELECTED_AUXK="$auxk" \
  CAUSAL_RANK_PROMPTS="$rank_prompts" \
  CAUSAL_MEDIATE_SKIP_PROMPTS="$skip_prompts" \
  FULL_PROMPTS="$full_prompts" \
  CAUSAL_K_LIST="$k_list" \
  RANDOM_SEEDS="$random_seeds" \
  N_BOOT="$n_boot" \
  CACHE_BATCH_SIZE="$cache_batch_size" \
  CACHE_WORKERS="$cache_workers" \
  GCS_SYNC_DEST="$sync_dest" \
  bash "$EXP30_RUNNER" \
    >"${RUN_ROOT}/logs/${model}_${model_mode}.log" 2>&1
}

analyze() {
  echo "[exp34] synthesize root=${RUN_ROOT} out=${OUT_DIR}"
  $PY_RUNNER scripts/analysis/analyze_exp34_dense5_final_readout_crosscoder.py \
    --run-root "$RUN_ROOT" \
    --out-dir "$OUT_DIR" \
    --models ${MODEL_LIST} \
    --llama-root "$LLAMA_ROOT" \
    --dataset "$DATASET_EVAL" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    >"${RUN_ROOT}/logs/analyze_exp34.log" 2>&1
  cat "${RUN_ROOT}/logs/analyze_exp34.log"
}

if [[ "$MODE" == "preflight" ]]; then
  preflight
  exit 0
fi

if [[ "$MODE" == "analyze-only" ]]; then
  analyze
  exit 0
fi

if [[ "$MODE" == "run-one" ]]; then
  if [[ -z "$MODEL_NAME" ]]; then
    echo "[exp34] MODE=run-one requires MODEL_NAME" >&2
    exit 2
  fi
  preflight
  run_model "$MODEL_NAME" "$GPU_LIST" "full"
  analyze
  echo "[exp34] run-one complete -> ${RUN_ROOT}"
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  preflight
  read -r -a smoke_gpus <<< "$SMOKE_GPU_LIST"
  read -r -a models <<< "$MODEL_LIST"
  if [[ "${#smoke_gpus[@]}" -lt "${#models[@]}" ]]; then
    echo "[exp34] smoke needs at least one GPU per model; gpus=${SMOKE_GPU_LIST} models=${MODEL_LIST}" >&2
    exit 2
  fi
  pids=()
  status=0
  for idx in "${!models[@]}"; do
    model="${models[$idx]}"
    gpu="${smoke_gpus[$idx]}"
    run_model "$model" "$gpu" "smoke" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp34] smoke failed; inspect ${RUN_ROOT}/logs" >&2
    exit "$status"
  fi
  analyze
  echo "[exp34] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

if [[ "$MODE" == "full" ]]; then
  preflight
  for model in $MODEL_LIST; do
    if [[ "$model" == "llama31_8b" && "$LLAMA_MODE" == "reuse" ]]; then
      echo "[exp34] reuse llama31_8b from ${LLAMA_ROOT}"
      continue
    fi
    run_model "$model" "$GPU_LIST" "full"
  done
  analyze
  echo "[exp34] full complete -> ${RUN_ROOT}"
fi
