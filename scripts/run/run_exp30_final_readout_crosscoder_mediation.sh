#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODE="${MODE:-pilot-grid}"  # smoke|cache|pilot-grid|score-grid|full-selected|causal-rank|causal-mediate-dev|causal-mediate-full|analyze-only
RUN_NAME="${RUN_NAME:-exp30_final_readout_crosscoder_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp30_final_readout_crosscoder_mediation/${RUN_NAME}}"
MODEL="${MODEL:-llama31_8b}"
DATASETS_TRAIN="${DATASETS_TRAIN:-data/eval_dataset_v2.jsonl data/exp3_dataset.jsonl data/exp6_dataset.jsonl}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
DATASET_EVAL="${DATASET_EVAL:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
GPU_LIST="${GPU_LIST:-}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

# Exp30 is the paper-facing late-boundary version of Exp28: a focused
# BatchTopK crosscoder followed by causal feature ranking and held-out mediation.
LAYERS="${LAYERS:-31}"
SELECTED_LAYERS="${SELECTED_LAYERS:-29 30 31}"
N_TOKENS="${N_TOKENS:-2000000}"
N_TRAIN_PROMPTS="${N_TRAIN_PROMPTS:-0}"  # 0 means all deduped non-holdout prompts.
VAL_SPLIT="${VAL_SPLIT:-prompt}"
VAL_FRACTION="${VAL_FRACTION:-0.10}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
APPEND_PT_GREEDY_TOKENS="${APPEND_PT_GREEDY_TOKENS:-384}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CACHE_WORKERS="${CACHE_WORKERS:-0}"  # 0 means min(num GPUs, 8).
FORCE_CACHE="${FORCE_CACHE:-0}"

LR="${LR:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"
K_ANNEAL_START="${K_ANNEAL_START:-256}"
K_ANNEAL_STEPS="${K_ANNEAL_STEPS:-2000}"
THRESHOLD_START_STEP="${THRESHOLD_START_STEP:-6000}"
THRESHOLD_CALIBRATION_BATCHES="${THRESHOLD_CALIBRATION_BATCHES:-64}"
AUXK_ALPHA="${AUXK_ALPHA:-0.03125}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1.0}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
LOG_EVERY="${LOG_EVERY:-100}"
RESUME_TRAIN="${RESUME_TRAIN:-1}"
SAME_INIT_FOR_ALL_BRANCHES="${SAME_INIT_FOR_ALL_BRANCHES:-1}"
SCALE_TOPK_BY_DECODER_NORM="${SCALE_TOPK_BY_DECODER_NORM:-1}"
NORM_INIT_SCALE="${NORM_INIT_SCALE:-0.1}"
DECODER_NORM_SCALE="${DECODER_NORM_SCALE:-1.0}"

PILOT_CONFIGS="${PILOT_CONFIGS:-l31_d65536_k64,31,65536,64,20000,1024,512,0;l31_d65536_k128,31,65536,128,20000,1024,512,1;l31_d131072_k64,31,131072,64,24000,512,1024,2}"
INCLUDE_K32="${INCLUDE_K32:-0}"
SELECTED_NAME="${SELECTED_NAME:-d131072_k64}"
SELECTED_DICT_SIZE="${SELECTED_DICT_SIZE:-131072}"
SELECTED_K="${SELECTED_K:-64}"
SELECTED_STEPS="${SELECTED_STEPS:-24000}"
SELECTED_BATCH_TOKENS="${SELECTED_BATCH_TOKENS:-512}"
SELECTED_AUXK="${SELECTED_AUXK:-1024}"
SELECTED_SEED="${SELECTED_SEED:-0}"

CAUSAL_LAYERS="${CAUSAL_LAYERS:-31}"
CAUSAL_RANK_PROMPTS="${CAUSAL_RANK_PROMPTS:-160}"
CAUSAL_RANK_SKIP_PROMPTS="${CAUSAL_RANK_SKIP_PROMPTS:-0}"
CAUSAL_MEDIATE_SKIP_PROMPTS="${CAUSAL_MEDIATE_SKIP_PROMPTS:-160}"
DEV_PROMPTS="${DEV_PROMPTS:-200}"
FULL_PROMPTS="${FULL_PROMPTS:-440}"
CAUSAL_K_LIST="${CAUSAL_K_LIST:-25 50 100 200 500}"
RANDOM_SEEDS="${RANDOM_SEEDS:-0 1 2}"
N_BOOT="${N_BOOT:-1000}"
BOUNDARY_LAYER_OVERRIDE="${BOUNDARY_LAYER_OVERRIDE:-}"
MATERIALIZE_CAUSAL_SLICES="${MATERIALIZE_CAUSAL_SLICES:-1}"
CROSSCODER_DTYPE="${CROSSCODER_DTYPE:-bfloat16}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|cache|pilot-grid|score-grid|full-selected|causal-rank|causal-mediate-dev|causal-mediate-full|analyze-only bash scripts/run/run_exp30_final_readout_crosscoder_mediation.sh

Pilot config format:
  name,layer,dict_size,k,steps,batch_tokens,auxk,seed;...
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

case "$MODE" in
  smoke|cache|pilot-grid|score-grid|full-selected|causal-rank|causal-mediate-dev|causal-mediate-full|analyze-only) ;;
  *)
    echo "[exp30] invalid MODE=${MODE}" >&2
    usage
    exit 2
    ;;
esac

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
read -r -a GPUS <<< "$GPU_LIST"
if [[ "$MODE" != "analyze-only" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp30] no GPUs in GPU_LIST" >&2
  exit 2
fi

mkdir -p "$RUN_ROOT/logs"
echo "[exp30] host $(hostname)"
echo "[exp30] mode ${MODE}"
echo "[exp30] run_name ${RUN_NAME}"
echo "[exp30] run_root ${RUN_ROOT}"
echo "[exp30] model ${MODEL}"
echo "[exp30] gpu_count ${gpu_count} gpu_list ${GPU_LIST:-<none>}"
echo "[exp30] boundary_layer_override ${BOUNDARY_LAYER_OVERRIDE:-<late-default>}"
echo "[exp30] train_datasets ${DATASETS_TRAIN}"
echo "[exp30] exclude_datasets ${EXCLUDE_DATASETS:-<none>}"

read -r -a TRAIN_DATASET_ARR <<< "$DATASETS_TRAIN"
read -r -a EXCLUDE_DATASET_ARR <<< "$EXCLUDE_DATASETS"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp30] GCS_SYNC_DEST set but gsutil not found; skipping sync" >&2
  fi
}

boundary_args() {
  if [[ -n "$BOUNDARY_LAYER_OVERRIDE" ]]; then
    printf -- '--boundary-layer-override %s' "$BOUNDARY_LAYER_OVERRIDE"
  fi
  return 0
}

cache_complete() {
  local root="$1"
  local layers="$2"
  for layer in $layers; do
    [[ -f "${root}/cache/layer_${layer}.pt" ]] || return 1
  done
  return 0
}

run_cache() {
  local root="$1"
  local layers="$2"
  local tokens="$3"
  local prompts="$4"
  if [[ "$FORCE_CACHE" != "1" ]] && cache_complete "$root" "$layers"; then
    echo "[exp30] cache already present root=${root} layers=${layers}"
    return
  fi
  local -a exclude_args=()
  if [[ "${#EXCLUDE_DATASET_ARR[@]}" -gt 0 && -n "${EXCLUDE_DATASET_ARR[0]:-}" ]]; then
    exclude_args=(--exclude-dataset "${EXCLUDE_DATASET_ARR[@]}")
  fi
  mkdir -p "${root}/logs"
  local workers="$CACHE_WORKERS"
  if [[ "$workers" -le 0 ]]; then
    workers="${#GPUS[@]}"
    [[ "$workers" -gt 8 ]] && workers=8
  fi
  [[ "$workers" -lt 1 ]] && workers=1
  if [[ "$workers" -eq 1 ]]; then
    local gpu="${GPUS[0]}"
    echo "[exp30] cache gpu=${gpu} root=${root} layers=${layers} tokens=${tokens}"
    CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation cache \
      --model "$MODEL" \
      --dataset "${TRAIN_DATASET_ARR[@]}" \
      "${exclude_args[@]}" \
      --out-dir "$root" \
      --layers ${layers} \
      --n-prompts "$prompts" \
      --n-tokens "$tokens" \
      --batch-size "$CACHE_BATCH_SIZE" \
      --max-seq-len "$MAX_SEQ_LEN" \
      --append-pt-greedy-tokens "$APPEND_PT_GREEDY_TOKENS" \
      --val-fraction "$VAL_FRACTION" \
      --val-split "$VAL_SPLIT" \
      --device cuda:0 \
      >"${root}/logs/cache.log" 2>&1
    return
  fi

  local per_worker_tokens=$(( (tokens + workers - 1) / workers ))
  echo "[exp30] cache workers=${workers} root=${root} layers=${layers} total_tokens=${tokens} per_worker_tokens=${per_worker_tokens}"
  FREE_GPUS=("${GPUS[@]}")
  ACTIVE_PIDS=()
  ACTIVE_GPUS=()
  WAIT_STATUS=0
  for worker in $(seq 0 $((workers - 1))); do
    while [[ "${#FREE_GPUS[@]}" -lt 1 ]]; do
      wait_for_any_active_train
    done
    local gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    echo "[exp30] cache worker=${worker}/${workers} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation cache \
      --model "$MODEL" \
      --dataset "${TRAIN_DATASET_ARR[@]}" \
      "${exclude_args[@]}" \
      --out-dir "$root" \
      --layers ${layers} \
      --n-prompts "$prompts" \
      --n-tokens "$per_worker_tokens" \
      --batch-size "$CACHE_BATCH_SIZE" \
      --max-seq-len "$MAX_SEQ_LEN" \
      --append-pt-greedy-tokens "$APPEND_PT_GREEDY_TOKENS" \
      --val-fraction "$VAL_FRACTION" \
      --val-split "$VAL_SPLIT" \
      --worker-index "$worker" \
      --n-workers "$workers" \
      --device cuda:0 \
      >"${root}/logs/cache_w${worker}of${workers}.log" 2>&1 &
    local pid="$!"
    ACTIVE_PIDS+=("$pid")
    ACTIVE_GPUS+=("$gpu")
  done
  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    wait_for_any_active_train
  done
  if [[ "$WAIT_STATUS" -ne 0 ]]; then
    return "$WAIT_STATUS"
  fi
  $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation cache \
    --model "$MODEL" \
    --out-dir "$root" \
    --layers ${layers} \
    --val-fraction "$VAL_FRACTION" \
    --val-split "$VAL_SPLIT" \
    --n-workers "$workers" \
    --merge-workers \
    >"${root}/logs/cache_merge.log" 2>&1
  find "${root}/cache" -name 'layer_*_w*.pt' -delete
}

train_one() {
  local root="$1"
  local gpu="$2"
  local layer="$3"
  local dict_size="$4"
  local kk="$5"
  local steps="$6"
  local batch_tokens="$7"
  local auxk="$8"
  local seed="$9"
  local tag="${10}"
  local -a resume_args=()
  local -a init_args=()
  [[ "$RESUME_TRAIN" == "1" ]] && resume_args+=(--resume)
  [[ "$SAME_INIT_FOR_ALL_BRANCHES" == "1" ]] && init_args+=(--same-init-for-all-branches)
  [[ "$SCALE_TOPK_BY_DECODER_NORM" == "1" ]] && init_args+=(--scale-topk-by-decoder-norm)
  mkdir -p "${root}/logs"
  echo "[exp30] train tag=${tag} layer=${layer} gpu=${gpu} dict=${dict_size} k=${kk} steps=${steps}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation train \
    --run-root "$root" \
    --layers "$layer" \
    --dict-size "$dict_size" \
    --k "$kk" \
    --steps "$steps" \
    --batch-tokens "$batch_tokens" \
    --lr "$LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --threshold-start-step "$THRESHOLD_START_STEP" \
    --threshold-calibration-batches "$THRESHOLD_CALIBRATION_BATCHES" \
    --auxk "$auxk" \
    --auxk-alpha "$AUXK_ALPHA" \
    --grad-clip-norm "$GRAD_CLIP_NORM" \
    --norm-init-scale "$NORM_INIT_SCALE" \
    --decoder-norm-scale "$DECODER_NORM_SCALE" \
    --k-anneal-start "$K_ANNEAL_START" \
    --k-anneal-steps "$K_ANNEAL_STEPS" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --log-every "$LOG_EVERY" \
    --seed "$seed" \
    --device cuda:0 \
    "${init_args[@]}" \
    "${resume_args[@]}" \
    >"${root}/logs/train_${tag}_layer_${layer}.log" 2>&1
}

wait_for_any_active_train() {
  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    local running
    running="$(jobs -pr)"
    local i
    for i in "${!ACTIVE_PIDS[@]}"; do
      local pid="${ACTIVE_PIDS[$i]}"
      if ! printf '%s\n' "$running" | grep -qx "$pid"; then
        if ! wait "$pid"; then
          WAIT_STATUS=1
          echo "[exp30] train worker pid=${pid} failed" >&2
        fi
        FREE_GPUS+=("${ACTIVE_GPUS[$i]}")
        unset "ACTIVE_PIDS[$i]"
        unset "ACTIVE_GPUS[$i]"
        ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
        ACTIVE_GPUS=("${ACTIVE_GPUS[@]}")
        return
      fi
    done
    sleep 5
  done
}

run_pilot_grid() {
  run_cache "$RUN_ROOT" "$LAYERS" "$N_TOKENS" "$N_TRAIN_PROMPTS"
  local configs="$PILOT_CONFIGS"
  if [[ "$INCLUDE_K32" == "1" ]]; then
    configs="${configs};l31_d131072_k32,31,131072,32,24000,4096,1024,3"
  fi
  IFS=';' read -r -a cfgs <<< "$configs"
  FREE_GPUS=("${GPUS[@]}")
  ACTIVE_PIDS=()
  ACTIVE_GPUS=()
  WAIT_STATUS=0
  for cfg in "${cfgs[@]}"; do
    [[ -z "$cfg" ]] && continue
    IFS=',' read -r name layer dict_size kk steps batch_tokens auxk seed <<< "$cfg"
    local cfg_root="${RUN_ROOT}/grid/${name}"
    mkdir -p "$cfg_root"
    ln -sfn "${RUN_ROOT}/cache" "${cfg_root}/cache"
    while [[ "${#FREE_GPUS[@]}" -lt 1 ]]; do
      wait_for_any_active_train
    done
    local gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    train_one "$cfg_root" "$gpu" "$layer" "$dict_size" "$kk" "$steps" "$batch_tokens" "$auxk" "$seed" "$name" &
    local pid="$!"
    ACTIVE_PIDS+=("$pid")
    ACTIVE_GPUS+=("$gpu")
  done
  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    wait_for_any_active_train
  done
  summarize_grid
  return "$WAIT_STATUS"
}

run_selected_full() {
  local selected_root="${SELECTED_ROOT:-${RUN_ROOT}/selected_${SELECTED_NAME}}"
  run_cache "$selected_root" "$SELECTED_LAYERS" "$N_TOKENS" "$N_TRAIN_PROMPTS"
  FREE_GPUS=("${GPUS[@]}")
  ACTIVE_PIDS=()
  ACTIVE_GPUS=()
  WAIT_STATUS=0
  for layer in $SELECTED_LAYERS; do
    while [[ "${#FREE_GPUS[@]}" -lt 1 ]]; do
      wait_for_any_active_train
    done
    local gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    train_one "$selected_root" "$gpu" "$layer" "$SELECTED_DICT_SIZE" "$SELECTED_K" "$SELECTED_STEPS" "$SELECTED_BATCH_TOKENS" "$SELECTED_AUXK" "$SELECTED_SEED" "$SELECTED_NAME" &
    local pid="$!"
    ACTIVE_PIDS+=("$pid")
    ACTIVE_GPUS+=("$gpu")
  done
  while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    wait_for_any_active_train
  done
  if [[ "$WAIT_STATUS" -ne 0 ]]; then
    return "$WAIT_STATUS"
  fi
  RUN_ROOT="$selected_root" CAUSAL_LAYERS="$SELECTED_LAYERS" run_causal_rank
  RUN_ROOT="$selected_root" CAUSAL_LAYERS="$SELECTED_LAYERS" run_causal_mediate "$FULL_PROMPTS" "full"
  RUN_ROOT="$selected_root" run_analyze
}

summarize_grid() {
  $PY_RUNNER - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for cfg in sorted((root / "grid").glob("*")):
    for path in sorted((cfg / "dictionaries").glob("layer_*/config.json")):
        data = json.loads(path.read_text())
        metrics = data.get("metrics", {})
        rows.append({
            "config": cfg.name,
            "layer": data.get("layer"),
            "dict_size": data.get("crosscoder", {}).get("dict_size"),
            "k": data.get("crosscoder", {}).get("k"),
            "steps": data.get("steps"),
            "ve_pt": metrics.get("heldout_variance_explained_pt"),
            "ve_it": metrics.get("heldout_variance_explained_it"),
            "effective_l0": metrics.get("effective_l0"),
            "alive_fraction": metrics.get("alive_fraction"),
            "inference_threshold": metrics.get("inference_threshold"),
        })
out = root / "grid_summary.json"
out.write_text(json.dumps(rows, indent=2) + "\n")
print(f"[exp30] wrote {out} rows={len(rows)}")
for row in rows:
    print(
        "[exp30] grid {config} layer={layer} k={k} ve_pt={ve_pt:.3f} "
        "ve_it={ve_it:.3f} l0={effective_l0:.1f} alive={alive_fraction:.3f}".format(**row)
    )
PY
}

causal_rank_worker() {
  local root="$1"
  local gpu="$2"
  local worker="$3"
  local n_workers="$4"
  mkdir -p "${root}/logs"
  echo "[exp30] causal-rank root=${root} worker=${worker}/${n_workers} gpu=${gpu} layers=${CAUSAL_LAYERS}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation causal-rank \
    --run-root "$root" \
    --out-dir "${root}/feature_stats" \
    --model "$MODEL" \
    --dataset "$DATASET_EVAL" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    --n-prompts "$CAUSAL_RANK_PROMPTS" \
    --skip-prompts "$CAUSAL_RANK_SKIP_PROMPTS" \
    --layers ${CAUSAL_LAYERS} \
    $(boundary_args) \
    --crosscoder-dtype "$CROSSCODER_DTYPE" \
    --worker-index "$worker" \
    --n-workers "$n_workers" \
    --device cuda:0 \
    >"${root}/logs/causal_rank_w${worker}of${n_workers}.log" 2>&1
}

run_causal_rank() {
  local root="$RUN_ROOT"
  local n_workers="${#GPUS[@]}"
  local -a pids=()
  for worker in $(seq 0 $((n_workers - 1))); do
    causal_rank_worker "$root" "${GPUS[$worker]}" "$worker" "$n_workers" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp30] causal-rank failed root=${root}" >&2
    return "$status"
  fi
  $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation causal-rank \
    --run-root "$root" \
    --out-dir "${root}/feature_stats" \
    --model "$MODEL" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${root}/logs/causal_rank_merge.log" 2>&1
}

mediate_worker() {
  local root="$1"
  local gpu="$2"
  local worker="$3"
  local n_workers="$4"
  local prompts="$5"
  local tag="$6"
  mkdir -p "${root}/logs"
  echo "[exp30] mediate root=${root} tag=${tag} worker=${worker}/${n_workers} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation mediate \
    --run-root "$root" \
    --out-dir "${root}/mediation" \
    --model "$MODEL" \
    --dataset "$DATASET_EVAL" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    --n-prompts "$prompts" \
    --skip-prompts "$CAUSAL_MEDIATE_SKIP_PROMPTS" \
    --k-list ${CAUSAL_K_LIST} \
    --random-seeds ${RANDOM_SEEDS} \
    $(boundary_args) \
    --crosscoder-dtype "$CROSSCODER_DTYPE" \
    --selection-suite causal \
    --causal-feature-csv "${root}/feature_stats/causal_feature_scores.csv" \
    --worker-index "$worker" \
    --n-workers "$n_workers" \
    --device cuda:0 \
    >"${root}/logs/mediate_${tag}_w${worker}of${n_workers}.log" 2>&1
}

run_causal_mediate() {
  local prompts="$1"
  local tag="$2"
  local root="$RUN_ROOT"
  if [[ "$MATERIALIZE_CAUSAL_SLICES" == "1" && -f "${root}/feature_stats/causal_feature_scores.csv" ]]; then
    echo "[exp30] materialize causal slices root=${root} layers=${CAUSAL_LAYERS}"
    $PY_RUNNER scripts/infra/materialize_exp30_causal_slices.py \
      --run-root "$root" \
      --layers ${CAUSAL_LAYERS} \
      --k-list ${CAUSAL_K_LIST} \
      --random-seeds ${RANDOM_SEEDS} \
      --dtype "$CROSSCODER_DTYPE" \
      >"${root}/logs/materialize_causal_slices_${tag}.log" 2>&1
  fi
  local n_workers="${#GPUS[@]}"
  local -a pids=()
  for worker in $(seq 0 $((n_workers - 1))); do
    mediate_worker "$root" "${GPUS[$worker]}" "$worker" "$n_workers" "$prompts" "$tag" &
    pids+=("$!")
  done
  local status=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp30] mediation failed root=${root} tag=${tag}" >&2
    return "$status"
  fi
  $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation mediate \
    --run-root "$root" \
    --out-dir "${root}/mediation" \
    --model "$MODEL" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${root}/logs/mediate_${tag}_merge.log" 2>&1
}

run_analyze() {
  local root="$RUN_ROOT"
  mkdir -p "${root}/logs"
  $PY_RUNNER -m src.poc.exp28_late_mlp_crosscoder_mediation analyze \
    --run-root "$root" \
    --n-boot "$N_BOOT" \
    >"${root}/logs/analyze.log" 2>&1
}

run_score_grid() {
  local status=0
  local roots=("${RUN_ROOT}"/grid/*)
  for cfg_root in "${roots[@]}"; do
    [[ -d "$cfg_root" ]] || continue
    echo "[exp30] scoring grid config ${cfg_root}"
    RUN_ROOT="$cfg_root" run_causal_rank || status=1
    RUN_ROOT="$cfg_root" run_causal_mediate "$DEV_PROMPTS" "dev" || status=1
    RUN_ROOT="$cfg_root" run_analyze || status=1
  done
  return "$status"
}

case "$MODE" in
  smoke)
    LAYERS="31"
    N_TOKENS="8000"
    N_TRAIN_PROMPTS="30"
    PILOT_CONFIGS="smoke_l31_d1024_k8,31,1024,8,40,256,32,0"
    run_pilot_grid
    ;;
  cache)
    run_cache "$RUN_ROOT" "$LAYERS" "$N_TOKENS" "$N_TRAIN_PROMPTS"
    sync_outputs
    ;;
  pilot-grid)
    run_pilot_grid
    sync_outputs
    ;;
  score-grid)
    run_score_grid
    sync_outputs
    ;;
  full-selected)
    run_selected_full
    sync_outputs
    ;;
  causal-rank)
    run_causal_rank
    sync_outputs
    ;;
  causal-mediate-dev)
    run_causal_mediate "$DEV_PROMPTS" "dev"
    run_analyze
    sync_outputs
    ;;
  causal-mediate-full)
    run_causal_mediate "$FULL_PROMPTS" "full"
    run_analyze
    sync_outputs
    ;;
  analyze-only)
    run_analyze
    sync_outputs
    ;;
esac

echo "[exp30] complete run_name=${RUN_NAME}"
