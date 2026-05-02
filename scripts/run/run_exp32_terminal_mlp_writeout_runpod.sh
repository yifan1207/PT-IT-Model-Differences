#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-smoke}"
RUN_NAME="${RUN_NAME:-exp32_terminal_mlp_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
ROOT="results/exp32_terminal_mlp_writeout/${RUN_NAME}"
N_EXAMPLES="${N_EXAMPLES:-600}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-50}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
REFERENCE_ROOT="${REFERENCE_ROOT:-results/exp21_productive_opposition/exp21_full_productive_opposition_clean_20260426_053736}"
PROMPT_MODES="${PROMPT_MODES:-raw_shared}"
MODELS="${MODELS:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
SMOKE_MODELS="${SMOKE_MODELS:-llama31_8b}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
GPU_LIST="${GPU_LIST:-}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
TOP_K="${TOP_K:-10}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-200}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
CONDITIONS="${CONDITIONS:-A_pt_raw C_it_chat B_last3_raw B_last1_raw D_last3_ptswap D_last1_ptswap}"
PY_RUN="${PY_RUN:-uv run python}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|analyze-only bash scripts/run/run_exp32_terminal_mlp_writeout_runpod.sh
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "Invalid MODE=${MODE}" >&2
  usage
  exit 2
fi

export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

mkdir -p "$ROOT/logs" logs

run_model_worker() {
  local mode="$1"
  local model="$2"
  local worker="$3"
  local gpu="$4"
  local n_examples="$5"
  local out_dir="${ROOT}/${mode}/${model}"
  mkdir -p "$out_dir"
  local condition_args=(--conditions)
  # shellcheck disable=SC2206
  condition_args+=(${CONDITIONS})
  local event_args=(--event-kinds)
  # shellcheck disable=SC2206
  event_args+=(${EVENT_KINDS})
  echo "[exp32] launch mode=${mode} model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp21_productive_opposition.collect \
    --model "$model" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-dir "$out_dir" \
    --device "${DEVICE_PREFIX}:0" \
    --worker-index "$worker" \
    --n-workers "$WORKERS_PER_MODEL" \
    --n-eval-examples "$n_examples" \
    --prompt-mode "$mode" \
    --top-k "$TOP_K" \
    "${condition_args[@]}" \
    "${event_args[@]}" \
    >"${ROOT}/logs/${mode}.${model}.w${worker}.log" 2>&1 &
}

merge_model() {
  local mode="$1"
  local model="$2"
  local out_dir="${ROOT}/${mode}/${model}"
  $PY_RUN -m src.poc.exp21_productive_opposition.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --n-workers "$WORKERS_PER_MODEL" \
    --merge-only \
    >"${ROOT}/logs/${mode}.${model}.merge.log" 2>&1
}

analyze() {
  local models="$1"
  local n_boot="$2"
  $PY_RUN scripts/analysis/analyze_exp32_terminal_mlp_writeout.py \
    --root "$ROOT" \
    --reference-root "$REFERENCE_ROOT" \
    --out-dir "$ROOT/analysis" \
    --models ${models} \
    --modes ${PROMPT_MODES} \
    --event-kinds ${EVENT_KINDS} \
    --n-bootstrap "$n_boot" \
    >"${ROOT}/logs/analyze_exp32.log" 2>&1
  $PY_RUN - "$ROOT" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root / "analysis" / "exp32_terminal_mlp_effects.csv").open()))
for row in rows:
    if (
        row["model"] in {"Dense-5 family mean", "Gemma-removed Dense-4"}
        and row["prompt_mode"] == "raw_shared"
        and row["event_kind"] == "first_diff"
        and row["effect"] == "terminal_interaction"
        and row["metric"] == "margin_writein_it_vs_pt"
    ):
        print(
            f"[exp32] {row['model']} {row['window']} interaction={float(row['mean']):.6f} "
            f"ci=[{float(row['ci_low']):.6f},{float(row['ci_high']):.6f}] "
            f"ret={float(row['retention_fraction']):.4f}"
        )
PY
}

if [[ "$MODE" == "smoke" ]]; then
  MODELS_TO_RUN="$SMOKE_MODELS"
  EXAMPLES="$SMOKE_EXAMPLES"
  BOOT="$SMOKE_N_BOOT"
else
  MODELS_TO_RUN="$MODELS"
  EXAMPLES="$N_EXAMPLES"
  BOOT="$N_BOOT"
fi

if [[ "$MODE" == "analyze-only" ]]; then
  analyze "$MODELS" "$N_BOOT"
  exit 0
fi

test -f "$DATASET"
for model in $MODELS_TO_RUN; do
  test -f "${EXP20_ROOT}/raw_shared/${model}/exp20_validation_records.jsonl" || \
    test -f "${EXP20_FALLBACK_ROOT}/raw_shared/${model}/exp20_validation_records.jsonl" || \
    test -f "${EXP20_FALLBACK_ROOT}/raw_shared/${model}/exp20_records.jsonl"
done

gpu_count="$($PY_RUN - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$gpu_count" -lt 1 ]]; then
  echo "[exp32] expected GPU, found ${gpu_count}" >&2
  exit 2
fi
if [[ -n "$GPU_LIST" ]]; then
  # shellcheck disable=SC2206
  free_gpus=(${GPU_LIST})
else
  free_gpus=($(seq 0 $((gpu_count - 1))))
fi

declare -a jobs=()
for mode in $PROMPT_MODES; do
  for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
    for model in $MODELS_TO_RUN; do
      jobs+=("${mode}|${model}|${worker}")
    done
  done
done

declare -a active_pids=()
declare -A pid_to_gpu=()
job_index=0
status=0

remove_active_pid() {
  local done_pid="$1"
  local next=()
  for pid in "${active_pids[@]}"; do
    [[ "$pid" != "$done_pid" ]] && next+=("$pid")
  done
  active_pids=("${next[@]}")
}

echo "[exp32] ${MODE} root=${ROOT} models=${MODELS_TO_RUN} examples=${EXAMPLES} gpus=${free_gpus[*]}"
while [[ "$job_index" -lt "${#jobs[@]}" || "${#active_pids[@]}" -gt 0 ]]; do
  while [[ "${#free_gpus[@]}" -gt 0 && "$job_index" -lt "${#jobs[@]}" ]]; do
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    IFS='|' read -r mode model worker <<<"${jobs[$job_index]}"
    run_model_worker "$mode" "$model" "$worker" "$gpu" "$EXAMPLES"
    pid="$!"
    active_pids+=("$pid")
    pid_to_gpu["$pid"]="$gpu"
    job_index=$((job_index + 1))
  done
  if [[ "${#active_pids[@]}" -gt 0 ]]; then
    done_pid=""
    if ! wait -n -p done_pid "${active_pids[@]}"; then
      status=1
      echo "[exp32] worker pid=${done_pid} failed" >&2
    fi
    if [[ -n "$done_pid" ]]; then
      free_gpus+=("${pid_to_gpu[$done_pid]}")
      unset "pid_to_gpu[$done_pid]"
      remove_active_pid "$done_pid"
    fi
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[exp32] at least one worker failed; inspect ${ROOT}/logs" >&2
  exit "$status"
fi

for mode in $PROMPT_MODES; do
  for model in $MODELS_TO_RUN; do
    merge_model "$mode" "$model"
  done
done

$PY_RUN - "$ROOT" "$EXAMPLES" "$PROMPT_MODES" "$MODELS_TO_RUN" <<'PY'
import gzip, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
modes = sys.argv[3].split()
models = sys.argv[4].split()
summary = {"ok": True, "counts": {}, "missing": []}
for mode in modes:
    for model in models:
        path = root / mode / model / "records.jsonl.gz"
        if not path.exists():
            summary["ok"] = False
            summary["missing"].append(str(path))
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            count = sum(1 for line in handle if line.strip())
        summary["counts"][f"{mode}/{model}"] = count
        if count != expected:
            summary["ok"] = False
            summary["missing"].append(f"{mode}/{model}: {count}/{expected}")
(root / "collect_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["ok"]:
    raise SystemExit(4)
PY

analyze "$MODELS_TO_RUN" "$BOOT"
echo "[exp32] ${MODE} complete -> ${ROOT}"
