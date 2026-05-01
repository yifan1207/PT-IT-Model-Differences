#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODE="${MODE:-smoke}"  # smoke|full|cache|train|rank|mediate-dev|mediate-full|analyze-only
RUN_NAME="${RUN_NAME:-exp28_llama31_8b_late_mlp_btopk_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp28_late_mlp_crosscoder_mediation/${RUN_NAME}}"
MODEL="${MODEL:-llama31_8b}"
DATASET_TRAIN="${DATASET_TRAIN:-data/eval_dataset_v2.jsonl}"
DATASET_EVAL="${DATASET_EVAL:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
LAYERS="${LAYERS:-19 20 21 22 23 24 25 26 27 28 29 30 31}"
SMOKE_LAYERS="${SMOKE_LAYERS:-19 25 31}"
GPU_LIST="${GPU_LIST:-}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

SMOKE_TOKENS="${SMOKE_TOKENS:-20000}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-200}"
SMOKE_DICT_SIZE="${SMOKE_DICT_SIZE:-8192}"
SMOKE_K="${SMOKE_K:-32}"
SMOKE_STEPS="${SMOKE_STEPS:-500}"
SMOKE_BATCH_TOKENS="${SMOKE_BATCH_TOKENS:-1024}"
SMOKE_MEDIATE_PROMPTS="${SMOKE_MEDIATE_PROMPTS:-12}"
SMOKE_K_LIST="${SMOKE_K_LIST:-10 50}"
SMOKE_MIN_VE="${SMOKE_MIN_VE:-0.10}"

N_TOKENS="${N_TOKENS:-300000}"
N_TRAIN_PROMPTS="${N_TRAIN_PROMPTS:-600}"
DICT_SIZE="${DICT_SIZE:-32768}"
K="${K:-64}"
STEPS="${STEPS:-4000}"
BATCH_TOKENS="${BATCH_TOKENS:-4096}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
APPEND_PT_GREEDY_TOKENS="${APPEND_PT_GREEDY_TOKENS:-0}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
DEV_PROMPTS="${DEV_PROMPTS:-200}"
FULL_PROMPTS="${FULL_PROMPTS:-600}"
K_LIST="${K_LIST:-50 200 1000}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
N_BOOT="${N_BOOT:-1000}"
MIN_VE="${MIN_VE:-0.80}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|cache|train|rank|mediate-dev|mediate-full|analyze-only bash scripts/run/run_exp28_late_mlp_crosscoder_mediation.sh

Important env vars:
  RUN_NAME RUN_ROOT MODEL LAYERS GPU_LIST DATASET_TRAIN DATASET_EVAL
  N_TOKENS DICT_SIZE K STEPS BATCH_TOKENS K_LIST RANDOM_SEEDS GCS_SYNC_DEST
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "cache" && "$MODE" != "train" && "$MODE" != "rank" && "$MODE" != "mediate-dev" && "$MODE" != "mediate-full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp28] invalid MODE=${MODE}" >&2
  usage
  exit 2
fi

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
if [[ "$MODE" != "analyze-only" && "$MODE" != "rank" && "$gpu_count" -lt 1 ]]; then
  echo "[exp28] GPU mode requires CUDA; detected ${gpu_count}" >&2
  exit 2
fi
read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt 1 && "$MODE" != "analyze-only" ]]; then
  echo "[exp28] no GPUs in GPU_LIST" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"
echo "[exp28] host $(hostname)"
echo "[exp28] mode ${MODE}"
echo "[exp28] run_name ${RUN_NAME}"
echo "[exp28] run_root ${RUN_ROOT}"
echo "[exp28] model ${MODEL}"
echo "[exp28] gpu_count ${gpu_count} gpu_list ${GPU_LIST:-<none>}"
echo "[exp28] py_runner ${PY_RUNNER}"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp28] GCS_SYNC_DEST set but gsutil not found; skipping sync" >&2
  fi
}

run_cache() {
  local layers="$1"
  local tokens="$2"
  local prompts="$3"
  local gpu="${GPUS[0]}"
  echo "[exp28] cache gpu=${gpu} layers=${layers} tokens=${tokens}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation cache \
    --model "$MODEL" \
    --dataset "$DATASET_TRAIN" \
    --out-dir "$RUN_ROOT" \
    --layers ${layers} \
    --n-prompts "$prompts" \
    --n-tokens "$tokens" \
    --batch-size "$CACHE_BATCH_SIZE" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --append-pt-greedy-tokens "$APPEND_PT_GREEDY_TOKENS" \
    --device cuda:0 \
    >"${RUN_ROOT}/logs/cache.log" 2>&1
}

train_one_layer() {
  local gpu="$1"
  local layer="$2"
  local dict_size="$3"
  local kk="$4"
  local steps="$5"
  local batch_tokens="$6"
  echo "[exp28] train layer=${layer} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation train \
    --run-root "$RUN_ROOT" \
    --layers "$layer" \
    --dict-size "$dict_size" \
    --k "$kk" \
    --steps "$steps" \
    --batch-tokens "$batch_tokens" \
    --device cuda:0 \
    >"${RUN_ROOT}/logs/train_layer_${layer}.log" 2>&1
}

run_train() {
  local layers="$1"
  local dict_size="$2"
  local kk="$3"
  local steps="$4"
  local batch_tokens="$5"
  read -r -a layer_arr <<< "$layers"
  local -a free=("${GPUS[@]}")
  local -a active=()
  declare -A pid_gpu=()
  local idx=0
  local status=0
  while [[ "$idx" -lt "${#layer_arr[@]}" || "${#active[@]}" -gt 0 ]]; do
    while [[ "${#free[@]}" -gt 0 && "$idx" -lt "${#layer_arr[@]}" ]]; do
      local gpu="${free[0]}"
      free=("${free[@]:1}")
      train_one_layer "$gpu" "${layer_arr[$idx]}" "$dict_size" "$kk" "$steps" "$batch_tokens" &
      local pid="$!"
      active+=("$pid")
      pid_gpu["$pid"]="$gpu"
      idx=$((idx + 1))
    done
    if [[ "${#active[@]}" -gt 0 ]]; then
      local done_pid=""
      if ! wait -n -p done_pid "${active[@]}"; then
        status=1
        echo "[exp28] train worker pid=${done_pid} failed" >&2
      fi
      if [[ -n "$done_pid" ]]; then
        free+=("${pid_gpu[$done_pid]}")
        unset "pid_gpu[$done_pid]"
        local -a next=()
        for pid in "${active[@]}"; do
          [[ "$pid" != "$done_pid" ]] && next+=("$pid")
        done
        active=("${next[@]}")
      fi
    fi
  done
  return "$status"
}

validate_quality() {
  local min_ve="$1"
  $PY_RUNNER - "$RUN_ROOT" "$min_ve" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_ve = float(sys.argv[2])
summary_paths = sorted((root / "dictionaries").glob("layer_*/config.json"))
bad = []
for path in summary_paths:
    data = json.loads(path.read_text())
    m = data.get("metrics", {})
    ve_pt = float(m.get("heldout_variance_explained_pt", -1))
    ve_it = float(m.get("heldout_variance_explained_it", -1))
    if ve_pt < min_ve or ve_it < min_ve:
        bad.append((str(path), ve_pt, ve_it))
print(f"[exp28] quality gate min_ve={min_ve} checked={len(summary_paths)} bad={len(bad)}")
for item in bad[:20]:
    print(f"[exp28] LOW_VE path={item[0]} ve_pt={item[1]:.3f} ve_it={item[2]:.3f}")
if bad:
    raise SystemExit(3)
PY
}

run_rank() {
  local layers="$1"
  local gpu="${GPUS[0]:-0}"
  echo "[exp28] rank gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation rank \
    --run-root "$RUN_ROOT" \
    --model "$MODEL" \
    --layers ${layers} \
    --prompt-mode "$PROMPT_MODE" \
    --event-kind first_diff \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --device cuda:0 \
    >"${RUN_ROOT}/logs/rank.log" 2>&1
}

mediate_one_worker() {
  local gpu="$1"
  local worker="$2"
  local n_workers="$3"
  local prompts="$4"
  local tag="$5"
  echo "[exp28] mediate ${tag} worker=${worker}/${n_workers} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation mediate \
    --run-root "$RUN_ROOT" \
    --out-dir "${RUN_ROOT}/mediation" \
    --model "$MODEL" \
    --dataset "$DATASET_EVAL" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    --n-prompts "$prompts" \
    --k-list ${K_LIST} \
    --random-seeds ${RANDOM_SEEDS} \
    --worker-index "$worker" \
    --n-workers "$n_workers" \
    --device cuda:0 \
    >"${RUN_ROOT}/logs/mediate_${tag}_w${worker}of${n_workers}.log" 2>&1
}

run_mediate() {
  local prompts="$1"
  local tag="$2"
  local n_workers="${#GPUS[@]}"
  local -a pids=()
  for worker in $(seq 0 $((n_workers - 1))); do
    mediate_one_worker "${GPUS[$worker]}" "$worker" "$n_workers" "$prompts" "$tag" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp28] mediation ${tag} failed" >&2
    exit "$status"
  fi
  $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation mediate \
    --run-root "$RUN_ROOT" \
    --out-dir "${RUN_ROOT}/mediation" \
    --model "$MODEL" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/mediate_${tag}_merge.log" 2>&1
}

run_analyze() {
  $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation analyze \
    --run-root "$RUN_ROOT" \
    --n-boot "$N_BOOT" \
    >"${RUN_ROOT}/logs/analyze.log" 2>&1
}

case "$MODE" in
  smoke)
    LAYERS="$SMOKE_LAYERS"
    K_LIST="$SMOKE_K_LIST"
    RANDOM_SEEDS="0"
    run_cache "$LAYERS" "$SMOKE_TOKENS" "$SMOKE_PROMPTS"
    run_train "$LAYERS" "$SMOKE_DICT_SIZE" "$SMOKE_K" "$SMOKE_STEPS" "$SMOKE_BATCH_TOKENS"
    validate_quality "$SMOKE_MIN_VE"
    run_rank "$LAYERS"
    run_mediate "$SMOKE_MEDIATE_PROMPTS" "smoke"
    run_analyze
    sync_outputs
    ;;
  full)
    run_cache "$LAYERS" "$N_TOKENS" "$N_TRAIN_PROMPTS"
    run_train "$LAYERS" "$DICT_SIZE" "$K" "$STEPS" "$BATCH_TOKENS"
    validate_quality "$MIN_VE"
    run_rank "$LAYERS"
    run_mediate "$DEV_PROMPTS" "dev"
    run_analyze
    run_mediate "$FULL_PROMPTS" "full"
    run_analyze
    sync_outputs
    ;;
  cache)
    run_cache "$LAYERS" "$N_TOKENS" "$N_TRAIN_PROMPTS"
    sync_outputs
    ;;
  train)
    run_train "$LAYERS" "$DICT_SIZE" "$K" "$STEPS" "$BATCH_TOKENS"
    validate_quality "$MIN_VE"
    sync_outputs
    ;;
  rank)
    run_rank "$LAYERS"
    sync_outputs
    ;;
  mediate-dev)
    run_mediate "$DEV_PROMPTS" "dev"
    run_analyze
    sync_outputs
    ;;
  mediate-full)
    run_mediate "$FULL_PROMPTS" "full"
    run_analyze
    sync_outputs
    ;;
  analyze-only)
    run_analyze
    sync_outputs
    ;;
esac

echo "[exp28] complete run_name=${RUN_NAME}"
