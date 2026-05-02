#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${MODE:-smoke}"
MODEL="${MODEL:-llama31_8b}"
PROMPT_MODE="${PROMPT_MODE:-raw_shared}"
EVENT_KINDS="${EVENT_KINDS:-first_diff}"
BOUNDARY_LAYER_OVERRIDE="${BOUNDARY_LAYER_OVERRIDE:-29}"
N_PROMPTS="${N_PROMPTS:-600}"
SMOKE_PROMPTS="${SMOKE_PROMPTS:-20}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
SMOKE_GPU="${SMOKE_GPU:-0}"
N_BOOT="${N_BOOT:-2000}"
SMOKE_N_BOOT="${SMOKE_N_BOOT:-200}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
RUN_NAME="${RUN_NAME:-exp29_llama31_last3_${MODE}_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp29_last3_stack_factorial/${RUN_NAME}}"
PY_RUN="${PY_RUN:-uv run python}"

usage() {
  cat <<EOF
Usage:
  MODE=smoke|full|analyze-only bash scripts/run/run_exp29_last3_stack_factorial_runpod.sh

Environment overrides:
  RUN_NAME, RUN_ROOT, GPU_LIST, SMOKE_GPU, N_PROMPTS, SMOKE_PROMPTS,
  N_BOOT, SMOKE_N_BOOT, DATASET, EXP20_ROOT, EXP20_FALLBACK_ROOT,
  BOUNDARY_LAYER_OVERRIDE
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "smoke" && "$MODE" != "full" && "$MODE" != "analyze-only" ]]; then
  echo "Invalid MODE=${MODE}" >&2
  usage
  exit 2
fi
if [[ "$MODEL" != "llama31_8b" ]]; then
  echo "Exp29 is intentionally Llama-only; got MODEL=${MODEL}" >&2
  exit 2
fi
if [[ "$PROMPT_MODE" != "raw_shared" ]]; then
  echo "Exp29 is intentionally raw_shared-only; got PROMPT_MODE=${PROMPT_MODE}" >&2
  exit 2
fi
if [[ "$EVENT_KINDS" != "first_diff" ]]; then
  echo "Exp29 is intentionally first_diff-only; got EVENT_KINDS=${EVENT_KINDS}" >&2
  exit 2
fi

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

OUT_DIR="${RUN_ROOT}/residual_factorial/${PROMPT_MODE}/${MODEL}"
mkdir -p "$OUT_DIR" "${RUN_ROOT}/logs"

preflight() {
  test -f "$DATASET"
  test -f "${EXP20_ROOT}/${PROMPT_MODE}/${MODEL}/exp20_validation_records.jsonl" || \
    test -f "${EXP20_ROOT}/${PROMPT_MODE}/${MODEL}/exp20_records.jsonl" || \
    test -f "${EXP20_FALLBACK_ROOT}/${PROMPT_MODE}/${MODEL}/exp20_validation_records.jsonl" || \
    test -f "${EXP20_FALLBACK_ROOT}/${PROMPT_MODE}/${MODEL}/exp20_records.jsonl"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  fi
}

collect_worker() {
  local gpu="$1"
  local n_examples="$2"
  local worker_index="$3"
  local n_workers="$4"
  local include_noop="$5"
  local log_name="$6"
  local extra=()
  if [[ "$include_noop" -eq 0 ]]; then
    extra+=(--no-noop-patch)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" $PY_RUN -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --exp20-root "$EXP20_ROOT" \
    --exp20-fallback-root "$EXP20_FALLBACK_ROOT" \
    --out-dir "$OUT_DIR" \
    --device cuda:0 \
    --prompt-mode "$PROMPT_MODE" \
    --n-eval-examples "$n_examples" \
    --worker-index "$worker_index" \
    --n-workers "$n_workers" \
    --event-kinds ${EVENT_KINDS} \
    --boundary-layer-override "$BOUNDARY_LAYER_OVERRIDE" \
    "${extra[@]}" \
    >"${RUN_ROOT}/logs/${log_name}" 2>&1
}

merge_workers() {
  local n_workers="$1"
  $PY_RUN -m src.poc.exp23_midlate_interaction_suite.residual_factorial \
    --model "$MODEL" \
    --out-dir "$OUT_DIR" \
    --merge-only \
    --n-workers "$n_workers" \
    >"${RUN_ROOT}/logs/merge.log" 2>&1
}

analyze() {
  local n_boot="$1"
  $PY_RUN scripts/analysis/analyze_exp23_midlate_interaction_suite.py \
    --run-root "$RUN_ROOT" \
    --models "$MODEL" \
    --prompt-mode "$PROMPT_MODE" \
    --readouts common_it common_pt \
    --n-bootstrap "$n_boot" \
    >"${RUN_ROOT}/logs/analyze.log" 2>&1
}

quick_readout() {
  python - "$RUN_ROOT" <<'PY'
import json, pathlib, sys
run_root = pathlib.Path(sys.argv[1])
summary_path = run_root / "analysis" / "exp23_summary.json"
if not summary_path.exists():
    print(f"[exp29] missing summary: {summary_path}")
    raise SystemExit(0)
summary = json.loads(summary_path.read_text())
res = summary["residual_factorial"]
effects = res["effects"]
for readout in ("common_it", "common_pt"):
    inter = effects[readout]["interaction"]["model_cis"]["llama31_8b"]
    pt = effects[readout]["late_it_given_pt_upstream"]["model_cis"]["llama31_8b"]
    it = effects[readout]["late_it_given_it_upstream"]["model_cis"]["llama31_8b"]
    print(f"[exp29] {readout} interaction={inter['estimate']:.6f} ci=[{inter['ci95_low']:.6f},{inter['ci95_high']:.6f}]")
    print(f"[exp29] {readout} late_from_PT={pt['estimate']:.6f} late_from_IT={it['estimate']:.6f}")
print("[exp29] units", res["n_units_by_model"], "quality", res["quality"])
PY
}

preflight

if [[ "$MODE" == "analyze-only" ]]; then
  analyze "$N_BOOT"
  quick_readout
  exit 0
fi

if [[ "$MODE" == "smoke" ]]; then
  echo "[exp29] smoke root=${RUN_ROOT} gpu=${SMOKE_GPU} n=${SMOKE_PROMPTS}"
  collect_worker "$SMOKE_GPU" "$SMOKE_PROMPTS" 0 1 1 "smoke.log"
  merge_workers 1
  analyze "$SMOKE_N_BOOT"
  quick_readout
  echo "[exp29] smoke complete -> ${RUN_ROOT}"
  exit 0
fi

declare -a gpus=($GPU_LIST)
if [[ "${#gpus[@]}" -lt 1 ]]; then
  echo "GPU_LIST must not be empty" >&2
  exit 2
fi
N_WORKERS="${#gpus[@]}"
echo "[exp29] full root=${RUN_ROOT} n=${N_PROMPTS} workers=${N_WORKERS} gpus=${GPU_LIST}"
status=0
for idx in "${!gpus[@]}"; do
  collect_worker "${gpus[$idx]}" "$N_PROMPTS" "$idx" "$N_WORKERS" 0 "residual_w${idx}of${N_WORKERS}.log" &
done
for _ in "${gpus[@]}"; do
  if ! wait -n; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "[exp29] at least one worker failed; inspect ${RUN_ROOT}/logs" >&2
  exit "$status"
fi
merge_workers "$N_WORKERS"
analyze "$N_BOOT"
quick_readout
echo "[exp29] full complete -> ${RUN_ROOT}"
