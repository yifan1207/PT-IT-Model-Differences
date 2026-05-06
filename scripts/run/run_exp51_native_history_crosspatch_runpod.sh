#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
RUN_NAME="${RUN_NAME:-exp51_native_history_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp51_native_history_crosspatch/${RUN_NAME}}"
MODELS="${MODELS:-qwen3_4b llama31_8b mistral_7b olmo2_7b}"
SMOKE_MODEL="${SMOKE_MODEL:-qwen3_4b}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
HORIZONS="${HORIZONS:-0 1 2 4 8 16 32}"
SMOKE_HORIZONS="${SMOKE_HORIZONS:-4 8}"
MAX_HISTORY_TOKENS="${MAX_HISTORY_TOKENS:-64}"
N_PROMPTS="${N_PROMPTS:-600}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-3}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-2}"
TOTAL_WORKERS_PER_MODEL="${TOTAL_WORKERS_PER_MODEL:-${WORKERS_PER_MODEL}}"
WORKER_OFFSET="${WORKER_OFFSET:-0}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SMOKE_GPU="${SMOKE_GPU:-0}"
RUN_PT_MIRROR="${RUN_PT_MIRROR:-1}"
PT_MIRROR_PROMPTS="${PT_MIRROR_PROMPTS:-150}"
N_BOOT="${N_BOOT:-1000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-100}"
NO_NOOP_PATCH="${NO_NOOP_PATCH:-0}"
MERGE_AFTER="${MERGE_AFTER:-1}"
ANALYZE_AFTER="${ANALYZE_AFTER:-1}"
GCS_SYNC_DEST="${GCS_SYNC_DEST-gs://pt-vs-it-results/results/exp51_native_history_crosspatch}"
PY_RUN="${PY_RUN:-uv run python}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|analyze-only bash scripts/run/run_exp51_native_history_crosspatch_runpod.sh

Important env:
  RUN_NAME RUN_ROOT MODELS DATASET GPU_LIST WORKERS_PER_MODEL TOTAL_WORKERS_PER_MODEL
  WORKER_OFFSET RUN_PT_MIRROR PT_MIRROR_PROMPTS N_PROMPTS HORIZONS NO_NOOP_PATCH
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp51] invalid MODE=${MODE}" >&2
  usage
  exit 2
fi
if [[ "$PROMPT_MODE" != "raw_shared" ]]; then
  echo "[exp51] Exp51 v1 is raw_shared-only; got PROMPT_MODE=${PROMPT_MODE}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$RUN_ROOT/logs"

preflight() {
  echo "[exp51] host=$(hostname)"
  echo "[exp51] mode=${MODE} run_root=${RUN_ROOT}"
  echo "[exp51] models=${MODELS}"
  echo "[exp51] dataset=${DATASET}"
  echo "[exp51] gpu_list=${GPU_LIST}"
  echo "[exp51] workers_per_model=${WORKERS_PER_MODEL} total_workers_per_model=${TOTAL_WORKERS_PER_MODEL} offset=${WORKER_OFFSET}"
  test -f "$DATASET"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
}

collect_worker() {
  local gpu="$1"
  local model="$2"
  local history_source="$3"
  local n_examples="$4"
  local worker_index="$5"
  local n_workers="$6"
  local horizons="$7"
  local log_name="$8"
  local out_dir="${RUN_ROOT}/${history_source}/${PROMPT_MODE}/${model}"
  local extra=()
  mkdir -p "$out_dir"
  if [[ "$NO_NOOP_PATCH" -eq 1 ]]; then
    extra+=(--no-noop-patch)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp51_native_history_crosspatch.collect \
    --model "$model" \
    --dataset "$DATASET" \
    --out-dir "$out_dir" \
    --device cuda:0 \
    --prompt-mode "$PROMPT_MODE" \
    --history-source "$history_source" \
    --n-eval-examples "$n_examples" \
    --worker-index "$worker_index" \
    --n-workers "$n_workers" \
    --max-history-tokens "$MAX_HISTORY_TOKENS" \
    --horizons ${horizons} \
    "${extra[@]}" \
    >"${RUN_ROOT}/logs/${log_name}" 2>&1
}

merge_one() {
  local model="$1"
  local history_source="$2"
  local n_workers="$3"
  local out_dir="${RUN_ROOT}/${history_source}/${PROMPT_MODE}/${model}"
  $PY_RUN -m src.poc.exp51_native_history_crosspatch.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/merge_${history_source}_${model}.log" 2>&1
}

analyze() {
  local histories="$1"
  local n_boot="$2"
  $PY_RUN -m src.poc.exp51_native_history_crosspatch.analyze \
    --run-root "$RUN_ROOT" \
    --models ${MODELS} \
    --history-sources ${histories} \
    --prompt-mode "$PROMPT_MODE" \
    --readouts common_it common_pt \
    --primary-horizons 4 8 16 \
    --n-bootstrap "$n_boot" \
    >"${RUN_ROOT}/logs/analyze.log" 2>&1
}

quick_readout() {
  python - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
path = root / "analysis" / "summary.json"
if not path.exists():
    print(f"[exp51] missing summary: {path}")
    raise SystemExit(0)
summary = json.loads(path.read_text())
for hist, by_readout in summary.get("effects", {}).items():
    payload = by_readout.get("common_it", {}).get("primary_horizons", {}).get("interaction", {})
    print(
        f"[exp51] {hist} common_it primary interaction="
        f"{payload.get('estimate')} ci=[{payload.get('ci95_low')},{payload.get('ci95_high')}] "
        f"n={payload.get('n_units')} clusters={payload.get('n_prompt_clusters')}"
    )
print("[exp51] support rows", summary.get("support", {}).get("rows"))
print("[exp51] noop max", summary.get("support", {}).get("noop_margin_abs_delta_max"))
PY
}

sync_gcs() {
  if [[ -z "${GCS_SYNC_DEST:-}" || "${GCS_SYNC_DEST:-}" == "none" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    echo "[exp51] syncing ${RUN_ROOT} to ${GCS_SYNC_DEST%/}/${RUN_NAME}"
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUN scripts/infra/gcs_sync_adc.py "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  else
    echo "[exp51] no gsutil or ADC sync helper; skipping sync" >&2
  fi
}

preflight

if [[ "$MODE" == "analyze-only" ]]; then
  histories="it"
  if [[ "$RUN_PT_MIRROR" -eq 1 ]]; then
    histories="it pt"
  fi
  analyze "$histories" "$N_BOOT"
  quick_readout
  sync_gcs
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  echo "[exp51] smoke model=${SMOKE_MODEL} gpu=${SMOKE_GPU} prompts=${SMOKE_PROMPTS}"
  collect_worker "$SMOKE_GPU" "$SMOKE_MODEL" "it" "$SMOKE_PROMPTS" 0 1 "$SMOKE_HORIZONS" "smoke_it_${SMOKE_MODEL}.log"
  merge_one "$SMOKE_MODEL" "it" 1
  MODELS="$SMOKE_MODEL" analyze "it" "$SMOKE_N_BOOT"
  quick_readout
  sync_gcs
  echo "[exp51] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

declare -a gpus=($GPU_LIST)
if [[ "${#gpus[@]}" -lt 1 ]]; then
  echo "[exp51] GPU_LIST must not be empty" >&2
  exit 2
fi

run_history_batch() {
  local history_source="$1"
  local n_examples="$2"
  local local_workers="$3"
  local total_workers="$4"
  local worker_offset="$5"
  local horizons="$6"
  local status=0
  local job_count=0
  for model in $MODELS; do
    for local_idx in $(seq 0 $((local_workers - 1))); do
      local global_idx=$((worker_offset + local_idx))
      local gpu="${gpus[$((job_count % ${#gpus[@]}))]}"
      echo "[exp51] launch history=${history_source} model=${model} worker=${global_idx}/${total_workers} gpu=${gpu}"
      collect_worker "$gpu" "$model" "$history_source" "$n_examples" "$global_idx" "$total_workers" "$horizons" "collect_${history_source}_${model}_w${global_idx}of${total_workers}.log" &
      job_count=$((job_count + 1))
    done
  done
  for _ in $(seq 1 "$job_count"); do
    if ! wait -n; then
      status=1
    fi
  done
  if [[ "$status" -ne 0 ]]; then
    echo "[exp51] at least one ${history_source} worker failed; inspect ${RUN_ROOT}/logs" >&2
    exit "$status"
  fi
}

echo "[exp51] full IT-history run"
run_history_batch "it" "$N_PROMPTS" "$WORKERS_PER_MODEL" "$TOTAL_WORKERS_PER_MODEL" "$WORKER_OFFSET" "$HORIZONS"

if [[ "$RUN_PT_MIRROR" -eq 1 ]]; then
  echo "[exp51] PT-history mirror run prompts=${PT_MIRROR_PROMPTS}"
  # Mirror is deliberately smaller; keep one local worker per model unless the
  # user explicitly overrides PT_MIRROR_WORKERS_PER_MODEL.
  PT_MIRROR_WORKERS_PER_MODEL="${PT_MIRROR_WORKERS_PER_MODEL:-1}"
  run_history_batch "pt" "$PT_MIRROR_PROMPTS" "$PT_MIRROR_WORKERS_PER_MODEL" "$PT_MIRROR_WORKERS_PER_MODEL" 0 "$HORIZONS"
fi

if [[ "$MERGE_AFTER" -eq 1 ]]; then
  for model in $MODELS; do
    merge_one "$model" "it" "$TOTAL_WORKERS_PER_MODEL"
    if [[ "$RUN_PT_MIRROR" -eq 1 ]]; then
      merge_one "$model" "pt" "${PT_MIRROR_WORKERS_PER_MODEL:-1}"
    fi
  done
fi

if [[ "$ANALYZE_AFTER" -eq 1 ]]; then
  histories="it"
  if [[ "$RUN_PT_MIRROR" -eq 1 ]]; then
    histories="it pt"
  fi
  analyze "$histories" "$N_BOOT"
  quick_readout
fi

sync_gcs
echo "[exp51] full complete -> ${RUN_ROOT}"
