#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="smoke"
RUN_NAME="exp23_midlate_interaction_suite_$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT=""
MODELS="gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b"
MODEL="qwen3_4b"
PARTS="part_a residual"
PROMPT_MODE="raw_shared"
EVENT_KINDS="first_diff first_nonformat_diff first_assistant_marker_diff"
N_PROMPTS=600
SMOKE_PROMPTS=4
MAX_NEW_TOKENS=32
PART_A_BATCH_SIZE=16
PART_A_CHUNK_SIZE=32
PART_A_SHARDS=1
RESIDUAL_WORKERS=1
DATASET="data/eval_dataset_v2.jsonl"
EXP20_ROOT="results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early"
EXP20_FALLBACK_ROOT="results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final"
GPU_LIST="0"
SMOKE_GPU="0"
N_BOOT=500
COLLECT_TRAJECTORIES=1
INCLUDE_NOOP_PATCH=1
READOUT_MODE="raw"
TUNED_LENS_DIR="${HOME}/.cache/exp11_tuned_lens_probes_v3"

usage() {
  cat <<EOF
Usage:
  bash scripts/run/run_exp23_midlate_interaction_suite.sh [options]

Options:
  --mode smoke|full|analyze-only        (default: smoke)
  --run-name NAME
  --run-root PATH
  --parts "part_a residual"             (default: both)
  --model MODEL                         (smoke default: qwen3_4b)
  --models "M1 M2 ..."                  (full/analyze default: dense five)
  --prompt-mode raw_shared|native       (default: raw_shared; residual primary is raw_shared)
  --event-kinds "first_diff ..."        (default: all Exp20 divergence event kinds)
  --n-prompts N                         (full default: 600)
  --smoke-prompts N                     (default: 4)
  --max-new-tokens N                    (Part A teacher length, default: 32)
  --part-a-batch-size N                 (default: 16)
  --part-a-chunk-size N                 (default: 32)
  --part-a-shards N                     (full mode prompt shards per model, default: 1)
  --residual-workers N                  (full mode workers per model, default: 1)
  --gpu-list "0 1 ..."                  (full default: 0)
  --smoke-gpu IDX                       (default: 0)
  --dataset PATH
  --exp20-root PATH
  --exp20-fallback-root PATH
  --n-boot N                            (analysis bootstrap draws, default: 500)
  --readout-mode raw|tuned_pt_shared|both (Part A default: raw)
  --tuned-lens-dir PATH
  --no-trajectories                     (skip residual layerwise compact trajectories)
  --no-noop-patch                       (skip no-op boundary patch checks)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --parts) PARTS="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --prompt-mode) PROMPT_MODE="$2"; shift 2 ;;
    --event-kinds) EVENT_KINDS="$2"; shift 2 ;;
    --n-prompts) N_PROMPTS="$2"; shift 2 ;;
    --smoke-prompts) SMOKE_PROMPTS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --part-a-batch-size) PART_A_BATCH_SIZE="$2"; shift 2 ;;
    --part-a-chunk-size) PART_A_CHUNK_SIZE="$2"; shift 2 ;;
    --part-a-shards) PART_A_SHARDS="$2"; shift 2 ;;
    --residual-workers) RESIDUAL_WORKERS="$2"; shift 2 ;;
    --gpu-list) GPU_LIST="$2"; shift 2 ;;
    --smoke-gpu) SMOKE_GPU="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --exp20-root) EXP20_ROOT="$2"; shift 2 ;;
    --exp20-fallback-root) EXP20_FALLBACK_ROOT="$2"; shift 2 ;;
    --n-boot) N_BOOT="$2"; shift 2 ;;
    --readout-mode) READOUT_MODE="$2"; shift 2 ;;
    --tuned-lens-dir) TUNED_LENS_DIR="$2"; shift 2 ;;
    --no-trajectories) COLLECT_TRAJECTORIES=0; shift ;;
    --no-noop-patch) INCLUDE_NOOP_PATCH=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "Invalid --mode: ${MODE}" >&2
  usage
  exit 2
fi
if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="results/exp23_midlate_interaction_suite/${RUN_NAME}"
fi
mkdir -p "$RUN_ROOT/logs"
if [[ "$PART_A_SHARDS" -lt 1 ]]; then
  echo "--part-a-shards must be >= 1" >&2
  exit 2
fi
if [[ "$RESIDUAL_WORKERS" -lt 1 ]]; then
  echo "--residual-workers must be >= 1" >&2
  exit 2
fi

has_part() {
  [[ " ${PARTS} " == *" $1 "* ]]
}

residual_args_common() {
  local model="$1"
  local out_dir="$2"
  local n_examples="$3"
  local device="$4"
  local worker_index="$5"
  local n_workers="$6"
  local extra=()
  if [[ "$COLLECT_TRAJECTORIES" -eq 1 ]]; then
    extra+=(--collect-trajectories)
  fi
  if [[ "$INCLUDE_NOOP_PATCH" -eq 0 ]]; then
    extra+=(--no-noop-patch)
  fi
  uv run python -m src.poc.exp23_midlate_interaction_suite \
    --model "$model" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-dir "$out_dir" \
    --device "$device" \
    --prompt-mode "$PROMPT_MODE" \
    --n-eval-examples "$n_examples" \
    --worker-index "$worker_index" \
    --n-workers "$n_workers" \
    --event-kinds ${EVENT_KINDS} \
    "${extra[@]}"
}

run_residual_worker() {
  local gpu="$1"
  local model="$2"
  local n_examples="$3"
  local worker_index="${4:-0}"
  local n_workers="${5:-1}"
  local out_dir="${RUN_ROOT}/residual_factorial/${PROMPT_MODE}/${model}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" residual_args_common "$model" "$out_dir" "$n_examples" "cuda:0" "$worker_index" "$n_workers" \
    >"${RUN_ROOT}/logs/residual_${PROMPT_MODE}_${model}_w${worker_index}of${n_workers}.log" 2>&1
}

merge_residual_model() {
  local model="$1"
  local n_workers="$2"
  local out_dir="${RUN_ROOT}/residual_factorial/${PROMPT_MODE}/${model}"
  uv run python -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
    --model "$model" \
    --out-dir "$out_dir" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/residual_${PROMPT_MODE}_${model}_merge.log" 2>&1
}

run_part_a_one() {
  local gpu="$1"
  local model="$2"
  local n_examples="$3"
  local shard_index="${4:-0}"
  local num_shards="${5:-1}"
  local out_dir="${6:-${RUN_ROOT}/part_a_mlp_kl/${model}}"
  mkdir -p "$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python -m src.poc.exp23_midlate_kl_factorial \
    --model "$model" \
    --dataset "$DATASET" \
    --n-prompts "$n_examples" \
    --seed 0 \
    --teacher-forced \
    --causal-combined \
    --include-midlate-factorial \
    --readout-mode "$READOUT_MODE" \
    --tuned-lens-dir "$TUNED_LENS_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --chunk-size "$PART_A_CHUNK_SIZE" \
    --batch-size "$PART_A_BATCH_SIZE" \
    --shard-index "$shard_index" \
    --num-shards "$num_shards" \
    --resume \
    --out-dir "$out_dir" \
    --device cuda:0 \
    >"${RUN_ROOT}/logs/part_a_${model}_shard${shard_index}of${num_shards}.log" 2>&1
}

merge_part_a_model() {
  local model="$1"
  local num_shards="$2"
  python - "$RUN_ROOT" "$model" "$num_shards" <<'PY'
import json
import shutil
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
model = sys.argv[2]
num_shards = int(sys.argv[3])
merged_dir = run_root / "part_a_mlp_kl" / model
if num_shards == 1:
    merged_dir.mkdir(parents=True, exist_ok=True)
    raise SystemExit(0)

shard_root = run_root / "part_a_mlp_kl" / "_shards" / model
shard_dirs = [shard_root / f"shard{idx}of{num_shards}" for idx in range(num_shards)]
missing = [str(path) for path in shard_dirs if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing Part-A shard dirs for {model}: {missing[:3]}")

if merged_dir.exists():
    for child in merged_dir.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
else:
    merged_dir.mkdir(parents=True, exist_ok=True)

jsonl_names = [
    "prompts.jsonl",
    "teacher_token_manifest.jsonl",
    "prompt_summaries.jsonl",
    "generated_texts.jsonl",
    "step_metrics.jsonl",
    "mechanism_metrics.jsonl",
    "js_prompt_region_metrics.jsonl",
    "js_audit_sample.jsonl",
]
for name in jsonl_names:
    out_path = merged_dir / name
    wrote = False
    with out_path.open("w", encoding="utf-8") as fout:
        for shard_dir in shard_dirs:
            in_path = shard_dir / name
            if not in_path.exists():
                continue
            with in_path.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        wrote = True
    if not wrote:
        out_path.unlink(missing_ok=True)

config_path = shard_dirs[0] / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text())
    config["merged_from_shards"] = [str(path) for path in shard_dirs]
    config["num_shards"] = num_shards
    config["shard_index"] = None
    config["n_prompts_sampled_after_sharding"] = sum(
        1 for shard_dir in shard_dirs for _ in (shard_dir / "prompts.jsonl").open("r", encoding="utf-8")
    )
    (merged_dir / "config.json").write_text(json.dumps(config, indent=2))

for name in [
    "secondary_trajectory_stats.json",
    "teacher_cap_diagnostics.json",
    "js_layer_stats.json",
    "mechanism_gradient_directions.json",
]:
    src = shard_dirs[0] / name
    if src.exists():
        shutil.copy2(src, merged_dir / name)
PY
}

analyze_suite() {
  local part_a_summary=""
  if has_part part_a; then
    uv run python scripts/analysis/analyze_exp23_midlate_kl_factorial.py \
      --run-root "${RUN_ROOT}/part_a_mlp_kl" \
      --out-dir "${RUN_ROOT}/analysis/part_a_mlp_kl" \
      --n-bootstrap "$N_BOOT" \
      --models ${MODELS}
    part_a_summary="${RUN_ROOT}/analysis/part_a_mlp_kl/exp23_midlate_kl_factorial_summary.json"
  fi
  local maybe_part_a=()
  if [[ -n "$part_a_summary" && -f "$part_a_summary" ]]; then
    maybe_part_a=(--part-a-summary "$part_a_summary")
  fi
  uv run python scripts/analysis/analyze_exp23_midlate_interaction_suite.py \
    --run-root "$RUN_ROOT" \
    --out-dir "${RUN_ROOT}/analysis" \
    --models ${MODELS} \
    --prompt-mode "$PROMPT_MODE" \
    --n-bootstrap "$N_BOOT" \
    "${maybe_part_a[@]}"
}

if [[ "$MODE" == "analyze-only" ]]; then
  analyze_suite
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  MODELS="$MODEL"
  echo "[exp23] smoke root=${RUN_ROOT} model=${MODEL} gpu=${SMOKE_GPU} parts=${PARTS}"
  if has_part part_a; then
    run_part_a_one "$SMOKE_GPU" "$MODEL" "$SMOKE_PROMPTS"
  fi
  if has_part residual; then
    run_residual_worker "$SMOKE_GPU" "$MODEL" "$SMOKE_PROMPTS" 0 1
    merge_residual_model "$MODEL" 1
  fi
  analyze_suite
  echo "[exp23] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

echo "[exp23] full root=${RUN_ROOT} models=${MODELS} gpus=${GPU_LIST} parts=${PARTS}"
declare -a gpus=($GPU_LIST)
if [[ "${#gpus[@]}" -lt 1 ]]; then
  echo "--gpu-list must not be empty" >&2
  exit 2
fi

declare -a jobs=($MODELS)
status=0
batch_size="${#gpus[@]}"

if has_part part_a; then
  declare -a part_a_models=()
  declare -a part_a_shards=()
  for shard_index in $(seq 0 $((PART_A_SHARDS - 1))); do
    for model in "${jobs[@]}"; do
      part_a_models+=("$model")
      part_a_shards+=("$shard_index")
    done
  done
  next_idx=0
  active=0
  total_jobs="${#part_a_models[@]}"
  while [[ "$next_idx" -lt "$total_jobs" || "$active" -gt 0 ]]; do
    while [[ "$active" -lt "$batch_size" && "$next_idx" -lt "$total_jobs" ]]; do
      slot=$((next_idx % batch_size))
      model="${part_a_models[$next_idx]}"
      shard_index="${part_a_shards[$next_idx]}"
      gpu="${gpus[$slot]}"
      shard_dir="${RUN_ROOT}/part_a_mlp_kl/_shards/${model}/shard${shard_index}of${PART_A_SHARDS}"
      if [[ "$PART_A_SHARDS" -eq 1 ]]; then
        shard_dir="${RUN_ROOT}/part_a_mlp_kl/${model}"
      fi
      echo "[exp23] launch part_a model=${model} shard=${shard_index}/${PART_A_SHARDS} gpu=${gpu}"
      run_part_a_one "$gpu" "$model" "$N_PROMPTS" "$shard_index" "$PART_A_SHARDS" "$shard_dir" &
      active=$((active + 1))
      next_idx=$((next_idx + 1))
    done
    if [[ "$active" -gt 0 ]]; then
      if wait -n; then
        echo "[exp23] finished part_a worker"
      else
        echo "[exp23] failed part_a worker" >&2
        status=1
      fi
      active=$((active - 1))
    fi
  done
  if [[ "$status" -eq 0 ]]; then
    for model in "${jobs[@]}"; do
      echo "[exp23] merge part_a model=${model} shards=${PART_A_SHARDS}"
      merge_part_a_model "$model" "$PART_A_SHARDS"
    done
  fi
fi

if [[ "$status" -eq 0 ]] && has_part residual; then
  declare -a residual_models=()
  declare -a residual_workers=()
  for worker_index in $(seq 0 $((RESIDUAL_WORKERS - 1))); do
    for model in "${jobs[@]}"; do
      residual_models+=("$model")
      residual_workers+=("$worker_index")
    done
  done
  next_idx=0
  active=0
  total_jobs="${#residual_models[@]}"
  while [[ "$next_idx" -lt "$total_jobs" || "$active" -gt 0 ]]; do
    while [[ "$active" -lt "$batch_size" && "$next_idx" -lt "$total_jobs" ]]; do
      slot=$((next_idx % batch_size))
      model="${residual_models[$next_idx]}"
      worker_index="${residual_workers[$next_idx]}"
      gpu="${gpus[$slot]}"
      echo "[exp23] launch residual model=${model} worker=${worker_index}/${RESIDUAL_WORKERS} gpu=${gpu}"
      run_residual_worker "$gpu" "$model" "$N_PROMPTS" "$worker_index" "$RESIDUAL_WORKERS" &
      active=$((active + 1))
      next_idx=$((next_idx + 1))
    done
    if [[ "$active" -gt 0 ]]; then
      if wait -n; then
        echo "[exp23] finished residual worker"
      else
        echo "[exp23] failed residual worker" >&2
        status=1
      fi
      active=$((active - 1))
    fi
  done
  if [[ "$status" -eq 0 ]]; then
    for model in "${jobs[@]}"; do
      echo "[exp23] merge residual model=${model} workers=${RESIDUAL_WORKERS}"
      merge_residual_model "$model" "$RESIDUAL_WORKERS"
    done
  fi
fi

if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
analyze_suite
echo "[exp23] full complete -> ${RUN_ROOT}"
