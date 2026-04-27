#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

RUN_NAME="exp24_32b_external_validity_$(date -u +%Y%m%d_%H%M%S)"
RUN_ROOT=""
PHASE="smoke"
MODEL="qwen25_32b"
MODELS="qwen25_32b olmo2_32b"
GPU_GROUP="2,3,4,5"
GPU_GROUPS=""
WORKERS_PER_MODEL=1
PART_A_SHARDS=1
DATASET="data/eval_dataset_v2.jsonl"
N_PROMPTS=1400
SMOKE_PROMPTS=8
MAX_NEW_TOKENS=128
PROMPT_MODES="raw_shared native"
EVENT_KINDS="first_diff"
EXP21_CONDITIONS="A_pt_raw B_mid_raw B_late_raw B_midlate_raw C_it_chat D_mid_ptswap D_late_ptswap D_midlate_ptswap"
PART_A_BATCH_SIZE=1
PART_A_CHUNK_SIZE=4
N_BOOT=2000
TUNED_LENS_DIR="${HOME}/.cache/exp11_tuned_lens_probes_v3"

usage() {
  cat <<EOF
Usage:
  bash scripts/run/run_exp24_32b_external_validity.sh [options]

Phases:
  smoke              Exp20 raw/native, Exp23 residual raw, Exp21 raw/native, 8 prompts by default
  exp20-raw          Collect Exp20 factorial validation for raw_shared
  exp20-native       Collect Exp20 factorial validation for native
  exp20-support      Collect Exp20 factorial validation for raw_shared and native
  exp23-primary      Collect Exp23 residual-state x late-stack factorial on raw_shared
  exp21-raw          Collect Exp21 productive-opposition support for raw_shared
  exp21-native       Collect Exp21 productive-opposition support for native
  exp21-support      Collect Exp21 productive-opposition support for raw_shared and native
  bridge-raw-kl      Run raw-only Exp23 Part-A MLP KL factorial bridge
  analyze            Build Exp20/Exp21/Exp23/bridge analyses from existing outputs
  synthesize         Build paper-facing Exp24 synthesis artifacts
  all                exp20-support, exp23-primary, exp21-support, bridge-raw-kl, analyze, synthesize

Options:
  --phase PHASE
  --run-name NAME
  --run-root PATH                 (default: results/exp24_32b_external_validity/<run-name>)
  --model MODEL|all               (default: qwen25_32b; all expands --models)
  --models "M1 M2 ..."            (default: qwen25_32b olmo2_32b)
  --gpu-group "2,3,4,5"           (default: 2,3,4,5; visible as cuda:0.. inside worker)
  --gpu-groups "0,1,2,3 4,5,6,7" (data-parallel worker groups; overrides --gpu-group)
  --workers-per-model N           (default: 1; use <= number of --gpu-groups)
  --part-a-shards N               (default: 1; use <= number of --gpu-groups)
  --dataset PATH                  (default: full eval_dataset_v2.jsonl)
  --n-prompts N                   (default: 1400)
  --smoke-prompts N               (default: 8)
  --max-new-tokens N              (default: 128)
  --prompt-modes "raw_shared native"
  --event-kinds "first_diff ..."
  --exp21-conditions "A_pt_raw ..."
  --part-a-batch-size N           (default: 1)
  --part-a-chunk-size N           (default: 4)
  --n-boot N                      (default: 2000)
  --tuned-lens-dir PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase) PHASE="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --gpu-group) GPU_GROUP="$2"; shift 2 ;;
    --gpu-groups) GPU_GROUPS="$2"; shift 2 ;;
    --workers-per-model) WORKERS_PER_MODEL="$2"; shift 2 ;;
    --part-a-shards) PART_A_SHARDS="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --n-prompts) N_PROMPTS="$2"; shift 2 ;;
    --smoke-prompts) SMOKE_PROMPTS="$2"; shift 2 ;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
    --prompt-modes) PROMPT_MODES="$2"; shift 2 ;;
    --event-kinds) EVENT_KINDS="$2"; shift 2 ;;
    --exp21-conditions) EXP21_CONDITIONS="$2"; shift 2 ;;
    --part-a-batch-size) PART_A_BATCH_SIZE="$2"; shift 2 ;;
    --part-a-chunk-size) PART_A_CHUNK_SIZE="$2"; shift 2 ;;
    --n-boot) N_BOOT="$2"; shift 2 ;;
    --tuned-lens-dir) TUNED_LENS_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$RUN_ROOT" ]]; then
  RUN_ROOT="results/exp24_32b_external_validity/${RUN_NAME}"
fi

EXP20_ROOT="${RUN_ROOT}/exp20_factorial_validation"
EXP23_ROOT="${RUN_ROOT}/exp23_midlate_interaction_suite"
EXP21_ROOT="${RUN_ROOT}/exp21_productive_opposition"
ANALYSIS_ROOT="${RUN_ROOT}/analysis"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$EXP20_ROOT" "$EXP23_ROOT" "$EXP21_ROOT" "$ANALYSIS_ROOT" "$LOG_DIR"

models_to_run() {
  if [[ "$MODEL" == "all" ]]; then
    echo "$MODELS"
  else
    echo "$MODEL"
  fi
}

n_for_phase() {
  if [[ "$PHASE" == "smoke" ]]; then
    echo "$SMOKE_PROMPTS"
  else
    echo "$N_PROMPTS"
  fi
}

cuda_run() {
  local group="$1"
  shift
  CUDA_VISIBLE_DEVICES="$group" "$@"
}

gpu_groups() {
  if [[ -n "$GPU_GROUPS" ]]; then
    echo "$GPU_GROUPS"
  else
    echo "$GPU_GROUP"
  fi
}

group_count() {
  local groups=($(gpu_groups))
  echo "${#groups[@]}"
}

check_parallelism() {
  local groups=($(gpu_groups))
  if [[ "${#groups[@]}" -lt 1 ]]; then
    echo "[exp24] no GPU groups configured" >&2
    exit 2
  fi
  if [[ "$WORKERS_PER_MODEL" -lt 1 ]]; then
    echo "[exp24] --workers-per-model must be >= 1" >&2
    exit 2
  fi
  if [[ "$PART_A_SHARDS" -lt 1 ]]; then
    echo "[exp24] --part-a-shards must be >= 1" >&2
    exit 2
  fi
  if [[ "$WORKERS_PER_MODEL" -gt "${#groups[@]}" ]]; then
    echo "[exp24] --workers-per-model=${WORKERS_PER_MODEL} exceeds GPU groups (${#groups[@]}): ${groups[*]}" >&2
    exit 2
  fi
  if [[ "$PART_A_SHARDS" -gt "${#groups[@]}" ]]; then
    echo "[exp24] --part-a-shards=${PART_A_SHARDS} exceeds GPU groups (${#groups[@]}): ${groups[*]}" >&2
    exit 2
  fi
}

wait_for_jobs() {
  local label="$1"
  local status=0
  while [[ "$#" -gt 1 ]]; do
    shift
    if ! wait "$1"; then
      echo "[exp24] ${label} worker pid=$1 failed" >&2
      status=1
    fi
  done
  return "$status"
}

count_jsonl() {
  local path="$1"
  python - "$path" <<'PY'
import gzip
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print(-1)
    raise SystemExit(0)
opener = gzip.open if path.suffix == ".gz" else open
with opener(path, "rt", encoding="utf-8") as handle:
    print(sum(1 for line in handle if line.strip()))
PY
}

validate_count() {
  local label="$1"
  local path="$2"
  local expected="$3"
  local got
  got="$(count_jsonl "$path")"
  echo "[exp24] count ${label}: ${got}/${expected}"
  if [[ "$got" != "$expected" ]]; then
    echo "[exp24] count check failed for ${label}: ${path}" >&2
    exit 4
  fi
}

run_exp20_mode() {
  local model="$1"
  local mode="$2"
  local n_examples="$3"
  local out_dir="${EXP20_ROOT}/${mode}/${model}"
  local groups=($(gpu_groups))
  local pids=()
  mkdir -p "$out_dir"
  echo "[exp24] Exp20 ${mode}/${model} gpu_groups=$(gpu_groups) workers=${WORKERS_PER_MODEL} n=${n_examples}"
  for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
    local group="${groups[$worker]}"
    cuda_run "$group" uv run python -m src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation \
      --model "$model" \
      --dataset "$DATASET" \
      --out-dir "$out_dir" \
      --device cuda:0 \
      --worker-index "$worker" \
      --n-workers "$WORKERS_PER_MODEL" \
      --n-eval-examples "$n_examples" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --prompt-mode "$mode" \
      >"${LOG_DIR}/exp20_${mode}_${model}_w${worker}of${WORKERS_PER_MODEL}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_jobs "Exp20 ${mode}/${model}" "${pids[@]}"
  uv run python -m src.poc.exp20_divergence_token_counterfactual.collect_factorial_validation \
    --model "$model" \
    --out-dir "$out_dir" \
    --n-workers "$WORKERS_PER_MODEL" \
    --merge-only \
    >"${LOG_DIR}/exp20_${mode}_${model}_merge.log" 2>&1
  validate_count "exp20/${mode}/${model}" "${out_dir}/exp20_validation_records.jsonl" "$n_examples"
}

run_exp23_primary_model() {
  local model="$1"
  local n_examples="$2"
  local out_dir="${EXP23_ROOT}/residual_factorial/raw_shared/${model}"
  local groups=($(gpu_groups))
  local pids=()
  mkdir -p "$out_dir"
  echo "[exp24] Exp23 residual raw_shared/${model} gpu_groups=$(gpu_groups) workers=${WORKERS_PER_MODEL} n=${n_examples}"
  for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
    local group="${groups[$worker]}"
    cuda_run "$group" uv run python -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
      --model "$model" \
      --dataset "$DATASET" \
      --exp20-root "$EXP20_ROOT" \
      --out-dir "$out_dir" \
      --device cuda:0 \
      --prompt-mode raw_shared \
      --n-eval-examples "$n_examples" \
      --worker-index "$worker" \
      --n-workers "$WORKERS_PER_MODEL" \
      --event-kinds ${EVENT_KINDS} \
      >"${LOG_DIR}/exp23_residual_raw_shared_${model}_w${worker}of${WORKERS_PER_MODEL}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_jobs "Exp23 residual/${model}" "${pids[@]}"
  uv run python -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
    --model "$model" \
    --out-dir "$out_dir" \
    --n-workers "$WORKERS_PER_MODEL" \
    --merge-only \
    >"${LOG_DIR}/exp23_residual_raw_shared_${model}_merge.log" 2>&1
  validate_count "exp23/raw_shared/${model}" "${out_dir}/records.jsonl.gz" "$n_examples"
}

run_exp21_mode() {
  local model="$1"
  local mode="$2"
  local n_examples="$3"
  local out_dir="${EXP21_ROOT}/${mode}/${model}"
  local groups=($(gpu_groups))
  local pids=()
  mkdir -p "$out_dir"
  echo "[exp24] Exp21 ${mode}/${model} gpu_groups=$(gpu_groups) workers=${WORKERS_PER_MODEL} n=${n_examples}"
  for worker in $(seq 0 $((WORKERS_PER_MODEL - 1))); do
    local group="${groups[$worker]}"
    cuda_run "$group" uv run python -m src.poc.exp21_productive_opposition.collect \
      --model "$model" \
      --dataset "$DATASET" \
      --exp20-root "$EXP20_ROOT" \
      --out-dir "$out_dir" \
      --device cuda:0 \
      --worker-index "$worker" \
      --n-workers "$WORKERS_PER_MODEL" \
      --n-eval-examples "$n_examples" \
      --prompt-mode "$mode" \
      --event-kinds ${EVENT_KINDS} \
      --conditions ${EXP21_CONDITIONS} \
      >"${LOG_DIR}/exp21_${mode}_${model}_w${worker}of${WORKERS_PER_MODEL}.log" 2>&1 &
    pids+=("$!")
  done
  wait_for_jobs "Exp21 ${mode}/${model}" "${pids[@]}"
  uv run python -m src.poc.exp21_productive_opposition.collect \
    --model "$model" \
    --out-dir "$out_dir" \
    --n-workers "$WORKERS_PER_MODEL" \
    --merge-only \
    >"${LOG_DIR}/exp21_${mode}_${model}_merge.log" 2>&1
  validate_count "exp21/${mode}/${model}" "${out_dir}/records.jsonl.gz" "$n_examples"
}

run_bridge_shard() {
  local group="$1"
  local model="$2"
  local n_examples="$3"
  local shard_index="$4"
  local out_dir="$5"
  mkdir -p "$out_dir"
  cuda_run "$group" uv run python -m src.poc.exp23_midlate_kl_factorial \
    --model "$model" \
    --dataset "$DATASET" \
    --n-prompts "$n_examples" \
    --seed 0 \
    --teacher-forced \
    --causal-combined \
    --include-midlate-factorial \
    --readout-mode raw \
    --tuned-lens-dir "$TUNED_LENS_DIR" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --chunk-size "$PART_A_CHUNK_SIZE" \
    --batch-size "$PART_A_BATCH_SIZE" \
    --num-shards "$PART_A_SHARDS" \
    --shard-index "$shard_index" \
    --device cuda:0 \
    --resume \
    --out-dir "$out_dir" \
    >"${LOG_DIR}/bridge_raw_kl_${model}_shard${shard_index}of${PART_A_SHARDS}.log" 2>&1
}

merge_bridge_model() {
  local model="$1"
  python - "$EXP23_ROOT" "$model" "$PART_A_SHARDS" <<'PY'
import json
import shutil
import sys
from pathlib import Path

exp23_root = Path(sys.argv[1])
model = sys.argv[2]
num_shards = int(sys.argv[3])
merged_dir = exp23_root / "part_a_mlp_kl" / "merged" / model
if num_shards == 1:
    src = exp23_root / "part_a_mlp_kl" / model
    merged_dir.parent.mkdir(parents=True, exist_ok=True)
    if merged_dir.exists() or merged_dir.is_symlink():
        if merged_dir.is_symlink() or merged_dir.is_file():
            merged_dir.unlink()
        else:
            shutil.rmtree(merged_dir)
    shutil.copytree(src, merged_dir)
    raise SystemExit(0)

shard_root = exp23_root / "part_a_mlp_kl" / "_shards" / model
shard_dirs = [shard_root / f"shard{idx}of{num_shards}" for idx in range(num_shards)]
missing = [str(path) for path in shard_dirs if not path.exists()]
if missing:
    raise FileNotFoundError(f"Missing bridge shard dirs for {model}: {missing}")
if merged_dir.exists():
    shutil.rmtree(merged_dir)
merged_dir.mkdir(parents=True)

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
        1
        for shard_dir in shard_dirs
        for _ in (shard_dir / "prompts.jsonl").open("r", encoding="utf-8")
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

run_bridge_model() {
  local model="$1"
  local n_examples="$2"
  local groups=($(gpu_groups))
  local pids=()
  echo "[exp24] raw KL bridge ${model} gpu_groups=$(gpu_groups) shards=${PART_A_SHARDS} n=${n_examples}"
  for shard_index in $(seq 0 $((PART_A_SHARDS - 1))); do
    local group="${groups[$shard_index]}"
    local out_dir="${EXP23_ROOT}/part_a_mlp_kl/${model}"
    if [[ "$PART_A_SHARDS" -gt 1 ]]; then
      out_dir="${EXP23_ROOT}/part_a_mlp_kl/_shards/${model}/shard${shard_index}of${PART_A_SHARDS}"
    fi
    run_bridge_shard "$group" "$model" "$n_examples" "$shard_index" "$out_dir" &
    pids+=("$!")
  done
  wait_for_jobs "bridge/${model}" "${pids[@]}"
  merge_bridge_model "$model"
  validate_count "bridge/${model}/prompts" "${EXP23_ROOT}/part_a_mlp_kl/merged/${model}/prompts.jsonl" "$n_examples"
}

analyze_all() {
  local models="$1"
  mkdir -p "$ANALYSIS_ROOT"
  uv run python scripts/analysis/analyze_exp20_factorial_validation.py \
    --root "$EXP20_ROOT" \
    --out-dir "${ANALYSIS_ROOT}/exp20_factorial_validation" \
    --models ${models} \
    --n-boot "$N_BOOT"
  uv run python scripts/analysis/analyze_exp21_productive_opposition.py \
    --root "$EXP21_ROOT" \
    --out-dir "${ANALYSIS_ROOT}/exp21_productive_opposition" \
    --models ${models} \
    --n-boot "$N_BOOT"
  local part_a_summary="${ANALYSIS_ROOT}/part_a_mlp_kl/exp23_midlate_kl_factorial_summary.json"
  if [[ -d "${EXP23_ROOT}/part_a_mlp_kl" ]]; then
    uv run python scripts/analysis/analyze_exp23_midlate_kl_factorial.py \
      --run-root "${EXP23_ROOT}/part_a_mlp_kl" \
      --out-dir "${ANALYSIS_ROOT}/part_a_mlp_kl" \
      --models ${models} \
      --n-bootstrap "$N_BOOT"
  fi
  local part_a_args=()
  if [[ -f "$part_a_summary" ]]; then
    part_a_args=(--part-a-summary "$part_a_summary")
  fi
  uv run python scripts/analysis/analyze_exp23_midlate_interaction_suite.py \
    --run-root "$EXP23_ROOT" \
    --out-dir "${ANALYSIS_ROOT}/exp23_midlate_interaction_suite" \
    --models ${models} \
    --prompt-mode raw_shared \
    --n-bootstrap "$N_BOOT" \
    "${part_a_args[@]}"
}

synthesize_all() {
  uv run python scripts/analysis/build_exp24_32b_external_validity_synthesis.py \
    --run-root "$RUN_ROOT" \
    --out-dir "results/paper_synthesis/exp24_32b_external_validity" \
    --models $(models_to_run)
}

run_for_modes() {
  local modes="$1"
  local n_examples="$2"
  local collector="$3"
  for model in $(models_to_run); do
    for mode in ${modes}; do
      "$collector" "$model" "$mode" "$n_examples"
    done
  done
}

run_for_models() {
  local n_examples="$1"
  local collector="$2"
  for model in $(models_to_run); do
    "$collector" "$model" "$n_examples"
  done
}

check_parallelism
echo "[exp24] run_root=${RUN_ROOT}"
echo "[exp24] phase=${PHASE} models=$(models_to_run)"
echo "[exp24] dataset=${DATASET} n_prompts=${N_PROMPTS} smoke_prompts=${SMOKE_PROMPTS}"
echo "[exp24] gpu_groups=$(gpu_groups) workers_per_model=${WORKERS_PER_MODEL} part_a_shards=${PART_A_SHARDS}"
echo "[exp24] prompt_modes=${PROMPT_MODES} event_kinds=${EVENT_KINDS}"

case "$PHASE" in
  smoke)
    n_examples="$(n_for_phase)"
    run_for_modes "$PROMPT_MODES" "$n_examples" run_exp20_mode
    run_for_models "$n_examples" run_exp23_primary_model
    run_for_modes "$PROMPT_MODES" "$n_examples" run_exp21_mode
    analyze_all "$(models_to_run)"
    synthesize_all
    ;;
  exp20-raw)
    run_for_modes "raw_shared" "$(n_for_phase)" run_exp20_mode
    ;;
  exp20-native)
    run_for_modes "native" "$(n_for_phase)" run_exp20_mode
    ;;
  exp20-support)
    run_for_modes "$PROMPT_MODES" "$(n_for_phase)" run_exp20_mode
    ;;
  exp23-primary)
    run_for_models "$(n_for_phase)" run_exp23_primary_model
    ;;
  exp21-raw)
    run_for_modes "raw_shared" "$(n_for_phase)" run_exp21_mode
    ;;
  exp21-native)
    run_for_modes "native" "$(n_for_phase)" run_exp21_mode
    ;;
  exp21-support)
    run_for_modes "$PROMPT_MODES" "$(n_for_phase)" run_exp21_mode
    ;;
  bridge-raw-kl)
    run_for_models "$(n_for_phase)" run_bridge_model
    uv run python scripts/analysis/analyze_exp23_midlate_kl_factorial.py \
      --run-root "${EXP23_ROOT}/part_a_mlp_kl" \
      --out-dir "${ANALYSIS_ROOT}/part_a_mlp_kl" \
      --models $(models_to_run) \
      --n-bootstrap "$N_BOOT"
    ;;
  analyze)
    analyze_all "$(models_to_run)"
    ;;
  synthesize)
    synthesize_all
    ;;
  all)
    run_for_modes "$PROMPT_MODES" "$N_PROMPTS" run_exp20_mode
    run_for_models "$N_PROMPTS" run_exp23_primary_model
    run_for_modes "$PROMPT_MODES" "$N_PROMPTS" run_exp21_mode
    run_for_models "$N_PROMPTS" run_bridge_model
    analyze_all "$(models_to_run)"
    synthesize_all
    ;;
  *)
    echo "Unknown phase: $PHASE" >&2
    usage
    exit 2
    ;;
esac

echo "[exp24] phase=${PHASE} complete -> ${RUN_ROOT}"
