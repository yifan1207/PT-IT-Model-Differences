#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:-smoke}"
RUN_NAME="${RUN_NAME:-exp33_terminal_identity_margin_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
ROOT="results/exp33_terminal_identity_margin/${RUN_NAME}"
N_EXAMPLES="${N_EXAMPLES:-600}"
SMOKE_EXAMPLES="${SMOKE_EXAMPLES:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-32}"
DATASET="${DATASET:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
PROMPT_MODES="${PROMPT_MODES:-raw_shared}"
MODELS="${MODELS:-gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b}"
SMOKE_MODELS="${SMOKE_MODELS:-llama31_8b}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
GPU_LIST="${GPU_LIST:-}"
DEVICE_PREFIX="${DEVICE_PREFIX:-cuda}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-200}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
PY_RUN="${PY_RUN:-uv run python}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|analyze-only bash scripts/run/run_exp33_terminal_identity_margin_runpod.sh
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
  local max_new_tokens="$6"
  local out_dir="${ROOT}/${mode}/${model}"
  mkdir -p "$out_dir"
  echo "[exp33] launch mode=${mode} model=${model} worker=${worker}/${WORKERS_PER_MODEL} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation \
    --model "$model" \
    --dataset "$DATASET" \
    --out-dir "$out_dir" \
    --device "${DEVICE_PREFIX}:0" \
    --worker-index "$worker" \
    --n-workers "$WORKERS_PER_MODEL" \
    --n-eval-examples "$n_examples" \
    --max-new-tokens "$max_new_tokens" \
    --prompt-mode "$mode" \
    >"${ROOT}/logs/${mode}.${model}.w${worker}.log" 2>&1 &
}

merge_model() {
  local mode="$1"
  local model="$2"
  local out_dir="${ROOT}/${mode}/${model}"
  $PY_RUN -m src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation \
    --model "$model" \
    --dataset "$DATASET" \
    --out-dir "$out_dir" \
    --n-workers "$WORKERS_PER_MODEL" \
    --merge-only \
    >"${ROOT}/logs/${mode}.${model}.merge.log" 2>&1
}

analyze() {
  local models="$1"
  local n_boot="$2"
  $PY_RUN scripts/analysis/analyze_exp33_terminal_identity_margin.py \
    --root "$ROOT" \
    --out-dir "$ROOT/analysis" \
    --models ${models} \
    --modes ${PROMPT_MODES} \
    --event-kinds ${EVENT_KINDS} \
    --n-bootstrap "$n_boot" \
    >"${ROOT}/logs/analyze_exp33.log" 2>&1
  $PY_RUN - "$ROOT" <<'PY'
import csv, pathlib, sys
root = pathlib.Path(sys.argv[1])
rows = list(csv.DictReader((root / "analysis" / "exp33_terminal_identity_margin_effects.csv").open()))
wanted = {
    "B_last3 IT-token transfer",
    "B_last1 IT-token transfer",
    "PT host margin: last3 minus A",
    "PT host margin: last1 minus A",
    "IT host margin: C minus D_last3",
    "IT host margin: C minus D_last1",
    "terminal last3 margin interaction",
    "terminal last1 margin interaction",
}
for row in rows:
    if row["model"] == "dense5" and row["mode"] == "raw_shared" and row["metric"] in wanted:
        mean = "nan" if row["mean"] in {"", "None"} else f"{float(row['mean']):.6f}"
        lo = "nan" if row["ci_low"] in {"", "None"} else f"{float(row['ci_low']):.6f}"
        hi = "nan" if row["ci_high"] in {"", "None"} else f"{float(row['ci_high']):.6f}"
        print(f"[exp33] dense5 {row['metric']} mean={mean} ci=[{lo},{hi}] n={row['n']}")
PY
}

if [[ "$MODE" == "smoke" ]]; then
  MODELS_TO_RUN="$SMOKE_MODELS"
  EXAMPLES="$SMOKE_EXAMPLES"
  TOKENS="$SMOKE_MAX_NEW_TOKENS"
  BOOT="$SMOKE_N_BOOT"
else
  MODELS_TO_RUN="$MODELS"
  EXAMPLES="$N_EXAMPLES"
  TOKENS="$MAX_NEW_TOKENS"
  BOOT="$N_BOOT"
fi

if [[ "$MODE" == "analyze-only" ]]; then
  analyze "$MODELS" "$N_BOOT"
  exit 0
fi

test -f "$DATASET"
gpu_count="$($PY_RUN - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if [[ "$gpu_count" -lt 1 ]]; then
  echo "[exp33] expected GPU, found ${gpu_count}" >&2
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

echo "[exp33] ${MODE} root=${ROOT} models=${MODELS_TO_RUN} examples=${EXAMPLES} tokens=${TOKENS} gpus=${free_gpus[*]}"
while [[ "$job_index" -lt "${#jobs[@]}" || "${#active_pids[@]}" -gt 0 ]]; do
  while [[ "${#free_gpus[@]}" -gt 0 && "$job_index" -lt "${#jobs[@]}" ]]; do
    gpu="${free_gpus[0]}"
    free_gpus=("${free_gpus[@]:1}")
    IFS='|' read -r mode model worker <<<"${jobs[$job_index]}"
    run_model_worker "$mode" "$model" "$worker" "$gpu" "$EXAMPLES" "$TOKENS"
    pid="$!"
    active_pids+=("$pid")
    pid_to_gpu["$pid"]="$gpu"
    job_index=$((job_index + 1))
  done
  if [[ "${#active_pids[@]}" -gt 0 ]]; then
    done_pid=""
    if ! wait -n -p done_pid "${active_pids[@]}"; then
      status=1
      echo "[exp33] worker pid=${done_pid} failed" >&2
    fi
    if [[ -n "$done_pid" ]]; then
      free_gpus+=("${pid_to_gpu[$done_pid]}")
      unset "pid_to_gpu[$done_pid]"
      remove_active_pid "$done_pid"
    fi
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "[exp33] at least one worker failed; inspect ${ROOT}/logs" >&2
  exit "$status"
fi

for mode in $PROMPT_MODES; do
  for model in $MODELS_TO_RUN; do
    merge_model "$mode" "$model"
  done
done

$PY_RUN - "$ROOT" "$EXAMPLES" "$PROMPT_MODES" "$MODELS_TO_RUN" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
expected = int(sys.argv[2])
modes = sys.argv[3].split()
models = sys.argv[4].split()
summary = {"ok": True, "counts": {}, "missing": []}
for mode in modes:
    for model in models:
        path = root / mode / model / "exp20_validation_records.jsonl"
        if not path.exists():
            summary["ok"] = False
            summary["missing"].append(str(path))
            continue
        count = sum(1 for line in path.open("rb") if line.strip())
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
echo "[exp33] ${MODE} complete -> ${ROOT}"
