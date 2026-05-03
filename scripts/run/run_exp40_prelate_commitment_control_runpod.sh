#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"  # smoke|pilot|full|analyze-only
RUN_NAME="${RUN_NAME:-exp40_prelate_commitment_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp40_prelate_commitment_control/${RUN_NAME}}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
EXP23_ROOT="${EXP23_ROOT:-results/exp23_midlate_interaction_suite/exp23_dense5_full_h100x8_20260426_sh4_rw4}"
MODEL_LIST="${MODEL_LIST:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
N_PROMPTS="${N_PROMPTS:-600}"
PILOT_PROMPTS="${PILOT_PROMPTS:-150}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-20}"
N_WORKERS_PER_MODEL="${N_WORKERS_PER_MODEL:-1}"
SMOKE_WORKERS_PER_MODEL="${SMOKE_WORKERS_PER_MODEL:-1}"
PILOT_WORKERS_PER_MODEL="${PILOT_WORKERS_PER_MODEL:-1}"
GPU_LIST="${GPU_LIST:-}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-200}"
PILOT_N_BOOT="${PILOT_N_BOOT:-500}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"
PY_RUN="${PY_RUN:-uv run python}"

if [[ "$MODE" != "smoke" && "$MODE" != "pilot" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "[exp40] invalid MODE=${MODE}" >&2
  exit 2
fi
if [[ "$PROMPT_MODE" != "raw_shared" ]]; then
  echo "[exp40] Exp40 is raw_shared-only; got PROMPT_MODE=${PROMPT_MODE}" >&2
  exit 2
fi
if [[ "$EVENT_KINDS" != "first_diff" ]]; then
  echo "[exp40] Exp40 v1 is first_diff-only; got EVENT_KINDS=${EVENT_KINDS}" >&2
  exit 2
fi

gpu_count="$($PY_RUN - <<'PY'
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
if [[ "$MODE" != "analyze-only" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp40] GPU run requires GPU_LIST or visible CUDA GPUs" >&2
  exit 2
fi

if [[ "$MODE" == "smoke" ]]; then
  N_EXAMPLES="$SMOKE_PROMPTS"
  WORKERS_PER_MODEL="$SMOKE_WORKERS_PER_MODEL"
  BOOT="$SMOKE_N_BOOT"
elif [[ "$MODE" == "pilot" ]]; then
  N_EXAMPLES="$PILOT_PROMPTS"
  WORKERS_PER_MODEL="$PILOT_WORKERS_PER_MODEL"
  BOOT="$PILOT_N_BOOT"
else
  N_EXAMPLES="$N_PROMPTS"
  WORKERS_PER_MODEL="$N_WORKERS_PER_MODEL"
  BOOT="$N_BOOT"
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

mkdir -p "${RUN_ROOT}/logs"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    $PY_RUN scripts/infra/gcs_sync_adc.py upload \
      "$RUN_ROOT" \
      "${GCS_SYNC_DEST%/}/${RUN_NAME}" \
      --workers 24
  else
    echo "[exp40] no GCS sync tool configured; skipping ${GCS_SYNC_DEST}" >&2
  fi
}

preflight() {
  test -f "$DATASET"
  test -d "$EXP23_ROOT"
  for model in $MODEL_LIST; do
    test -f "${EXP20_ROOT}/${PROMPT_MODE}/${model}/exp20_validation_records.jsonl" || \
      test -f "${EXP20_ROOT}/${PROMPT_MODE}/${model}/exp20_records.jsonl" || \
      test -f "${EXP20_FALLBACK_ROOT}/${PROMPT_MODE}/${model}/exp20_validation_records.jsonl" || \
      test -f "${EXP20_FALLBACK_ROOT}/${PROMPT_MODE}/${model}/exp20_records.jsonl"
  done
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
}

collect_worker() {
  local gpu="$1"
  local model="$2"
  local worker_index="$3"
  local n_workers="$4"
  local out_dir="${RUN_ROOT}/${PROMPT_MODE}/${model}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp40_prelate_commitment_control.collect \
    --model "$model" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-dir "$out_dir" \
    --device cuda:0 \
    --prompt-mode "$PROMPT_MODE" \
    --event-kinds ${EVENT_KINDS} \
    --n-eval-examples "$N_EXAMPLES" \
    --worker-index "$worker_index" \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/collect_${model}_w${worker_index}of${n_workers}.log" 2>&1
}

merge_model() {
  local model="$1"
  local n_workers="$2"
  local out_dir="${RUN_ROOT}/${PROMPT_MODE}/${model}"
  $PY_RUN -m src.poc.exp40_prelate_commitment_control.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --n-workers "$n_workers" \
    --merge-only \
    >"${RUN_ROOT}/logs/merge_${model}.log" 2>&1
}

analyze() {
  local n_boot="$1"
  $PY_RUN scripts/analysis/analyze_exp40_prelate_commitment_control.py \
    --run-root "$RUN_ROOT" \
    --exp23-root "$EXP23_ROOT" \
    --models ${MODEL_LIST} \
    --prompt-mode "$PROMPT_MODE" \
    --n-bootstrap "$n_boot" \
    >"${RUN_ROOT}/logs/analyze_exp40.log" 2>&1
  $PY_RUN - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "analysis" / "summary.json").read_text())
common = summary["readouts"]["common_it"]
all_interaction = common["scopes"]["all"]["metrics"]["interaction"]
low = common["scopes"]["boundary_it_low_tercile"]["metrics"]["interaction"]
reg = common["regressions"]["state_level_all_support"]["it_upstream_provenance_coef"]
print(
    f"[exp40] common_it all={all_interaction['estimate']:.6f} "
    f"ci=[{all_interaction['ci95_low']:.6f},{all_interaction['ci95_high']:.6f}]"
)
print(
    f"[exp40] common_it low_boundary={low['estimate']:.6f} "
    f"ci=[{low['ci95_low']:.6f},{low['ci95_high']:.6f}]"
)
print(
    f"[exp40] common_it state_reg_it_coef={reg['estimate']:.6f} "
    f"ci=[{reg['ci95_low']:.6f},{reg['ci95_high']:.6f}]"
)
PY
}

preflight

echo "[exp40] mode=${MODE} run_root=${RUN_ROOT}"
echo "[exp40] models=${MODEL_LIST}"
echo "[exp40] n_examples=${N_EXAMPLES} workers_per_model=${WORKERS_PER_MODEL} gpus=${GPU_LIST:-<none>}"

if [[ "$MODE" != "analyze-only" ]]; then
  declare -a pids=()
  idx=0
  for ((worker=0; worker<WORKERS_PER_MODEL; worker++)); do
    for model in $MODEL_LIST; do
      gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      collect_worker "$gpu" "$model" "$worker" "$WORKERS_PER_MODEL" &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        status=0
        for pid in "${pids[@]}"; do
          if ! wait "$pid"; then
            status=1
          fi
        done
        if [[ "$status" -ne 0 ]]; then
          echo "[exp40] collect failed; see ${RUN_ROOT}/logs" >&2
          sync_outputs || true
          exit 1
        fi
        pids=()
      fi
    done
  done
  if [[ "${#pids[@]}" -gt 0 ]]; then
    status=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    if [[ "$status" -ne 0 ]]; then
      echo "[exp40] collect failed; see ${RUN_ROOT}/logs" >&2
      sync_outputs || true
      exit 1
    fi
  fi
  for model in $MODEL_LIST; do
    merge_model "$model" "$WORKERS_PER_MODEL"
  done
  sync_outputs || true
fi

analyze "$BOOT"
sync_outputs || true
echo "[exp40] complete run_root=${RUN_ROOT}"
