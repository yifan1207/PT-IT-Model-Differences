#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

MODE="${MODE:-stage-a}"  # smoke|stage-a|stage-b|analyze-only
RUN_NAME="${RUN_NAME:-exp38_qwen_olmo_final_layer_crosscoder_hardening_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp38_qwen_olmo_final_layer_crosscoder_hardening/${RUN_NAME}}"
GPU_LIST_QWEN="${GPU_LIST_QWEN:-0 1 2 3}"
GPU_LIST_OLMO="${GPU_LIST_OLMO:-4 5 6 7}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
EXP30_RUNNER="${EXP30_RUNNER:-scripts/run/run_exp30_final_readout_crosscoder_mediation.sh}"
GCS_SYNC_DEST="${GCS_SYNC_DEST:-}"

DATASETS_TRAIN="${DATASETS_TRAIN:-data/eval_dataset_v2.jsonl data/exp3_dataset.jsonl data/exp6_dataset.jsonl}"
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
DATASET_EVAL="${DATASET_EVAL:-data/eval_dataset_v2_holdout_0600_1199.jsonl}"
EXP20_ROOT="${EXP20_ROOT:-results/exp20_divergence_token_counterfactual/factorial_validation_holdout_fast_20260425_2009_with_early}"
EXP20_FALLBACK_ROOT="${EXP20_FALLBACK_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"

# Stage A: final-one pilot, disjoint rank/mediate split.
STAGE_A_TOKENS="${STAGE_A_TOKENS:-1000000}"
STAGE_A_STEPS="${STAGE_A_STEPS:-12000}"
STAGE_A_BATCH_TOKENS="${STAGE_A_BATCH_TOKENS:-512}"
STAGE_A_AUXK="${STAGE_A_AUXK:-1024}"
STAGE_A_RANK_PROMPTS="${STAGE_A_RANK_PROMPTS:-80}"
STAGE_A_MEDIATE_SKIP_PROMPTS="${STAGE_A_MEDIATE_SKIP_PROMPTS:-80}"
STAGE_A_DEV_PROMPTS="${STAGE_A_DEV_PROMPTS:-120}"
STAGE_A_K_LIST="${STAGE_A_K_LIST:-25 50 100 200}"
STAGE_A_RANDOM_SEEDS="${STAGE_A_RANDOM_SEEDS:-0 1 2}"
STAGE_A_N_BOOT="${STAGE_A_N_BOOT:-500}"

# Stage B: final-two selected full run.
STAGE_B_TOKENS="${STAGE_B_TOKENS:-2000000}"
STAGE_B_STEPS="${STAGE_B_STEPS:-24000}"
STAGE_B_BATCH_TOKENS="${STAGE_B_BATCH_TOKENS:-512}"
STAGE_B_AUXK="${STAGE_B_AUXK:-1024}"
STAGE_B_RANK_PROMPTS="${STAGE_B_RANK_PROMPTS:-160}"
STAGE_B_MEDIATE_SKIP_PROMPTS="${STAGE_B_MEDIATE_SKIP_PROMPTS:-160}"
STAGE_B_FULL_PROMPTS="${STAGE_B_FULL_PROMPTS:-440}"
STAGE_B_K_LIST="${STAGE_B_K_LIST:-25 50 100 200 500}"
STAGE_B_RANDOM_SEEDS="${STAGE_B_RANDOM_SEEDS:-0 1 2}"
STAGE_B_N_BOOT="${STAGE_B_N_BOOT:-1000}"

QWEN_FINAL2_SELECTED_NAME="${QWEN_FINAL2_SELECTED_NAME:-d196608_k64}"
QWEN_FINAL2_DICT_SIZE="${QWEN_FINAL2_DICT_SIZE:-196608}"
QWEN_FINAL2_K="${QWEN_FINAL2_K:-64}"
OLMO_FINAL2_SELECTED_NAME="${OLMO_FINAL2_SELECTED_NAME:-d262144_k64}"
OLMO_FINAL2_DICT_SIZE="${OLMO_FINAL2_DICT_SIZE:-262144}"
OLMO_FINAL2_K="${OLMO_FINAL2_K:-64}"

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" && "$ROOT" == /workspace/* ]]; then
  export UV_PROJECT_ENVIRONMENT="/tmp/exp38_uv_venv"
fi

mkdir -p "$RUN_ROOT/logs"

run_exp30() {
  local model="$1"
  local root="$2"
  local gpu_list="$3"
  local layers="$4"
  local boundary="$5"
  local mode="$6"
  shift 6
  echo "[exp38] model=${model} mode=${mode} root=${root} layers=${layers} boundary=${boundary} gpus=${gpu_list}"
  MODEL="$model" \
  RUN_ROOT="$root" \
  RUN_NAME="$(basename "$root")" \
  MODE="$mode" \
  DATASETS_TRAIN="$DATASETS_TRAIN" \
  EXCLUDE_DATASETS="$EXCLUDE_DATASETS" \
  DATASET_EVAL="$DATASET_EVAL" \
  EXP20_ROOT="$EXP20_ROOT" \
  EXP20_FALLBACK_ROOT="$EXP20_FALLBACK_ROOT" \
  PROMPT_MODE=raw_shared \
  EVENT_KINDS=first_diff \
  GPU_LIST="$gpu_list" \
  CAUSAL_LAYERS="$layers" \
  BOUNDARY_LAYER_OVERRIDE="$boundary" \
  GCS_SYNC_DEST="$GCS_SYNC_DEST" \
  PY_RUNNER="$PY_RUNNER" \
  env "$@" bash "$EXP30_RUNNER"
}

run_final1_pilot() {
  local model="$1"
  local gpu_list="$2"
  local layer="$3"
  local boundary="$4"
  local configs="$5"
  local root="${RUN_ROOT}/${model}/final1_pilot"
  mkdir -p "$root/logs"
  run_exp30 "$model" "$root" "$gpu_list" "$layer" "$boundary" "pilot-grid" \
    N_TOKENS="$STAGE_A_TOKENS" \
    N_TRAIN_PROMPTS=0 \
    LAYERS="$layer" \
    PILOT_CONFIGS="$configs" \
    >"${RUN_ROOT}/logs/${model}_final1_pilot_train.log" 2>&1
  run_exp30 "$model" "$root" "$gpu_list" "$layer" "$boundary" "score-grid" \
    CAUSAL_RANK_PROMPTS="$STAGE_A_RANK_PROMPTS" \
    CAUSAL_MEDIATE_SKIP_PROMPTS="$STAGE_A_MEDIATE_SKIP_PROMPTS" \
    DEV_PROMPTS="$STAGE_A_DEV_PROMPTS" \
    CAUSAL_K_LIST="$STAGE_A_K_LIST" \
    RANDOM_SEEDS="$STAGE_A_RANDOM_SEEDS" \
    N_BOOT="$STAGE_A_N_BOOT" \
    >"${RUN_ROOT}/logs/${model}_final1_pilot_score.log" 2>&1
}

run_final2_selected() {
  local model="$1"
  local gpu_list="$2"
  local layers="$3"
  local boundary="$4"
  local selected_name="$5"
  local dict_size="$6"
  local kk="$7"
  local root="${RUN_ROOT}/${model}/final2_${selected_name}"
  mkdir -p "$root/logs"
  run_exp30 "$model" "$root" "$gpu_list" "$layers" "$boundary" "full-selected" \
    N_TOKENS="$STAGE_B_TOKENS" \
    N_TRAIN_PROMPTS=0 \
    SELECTED_LAYERS="$layers" \
    SELECTED_NAME="$selected_name" \
    SELECTED_DICT_SIZE="$dict_size" \
    SELECTED_K="$kk" \
    SELECTED_STEPS="$STAGE_B_STEPS" \
    SELECTED_BATCH_TOKENS="$STAGE_B_BATCH_TOKENS" \
    SELECTED_AUXK="$STAGE_B_AUXK" \
    CAUSAL_RANK_PROMPTS="$STAGE_B_RANK_PROMPTS" \
    CAUSAL_MEDIATE_SKIP_PROMPTS="$STAGE_B_MEDIATE_SKIP_PROMPTS" \
    FULL_PROMPTS="$STAGE_B_FULL_PROMPTS" \
    CAUSAL_K_LIST="$STAGE_B_K_LIST" \
    RANDOM_SEEDS="$STAGE_B_RANDOM_SEEDS" \
    N_BOOT="$STAGE_B_N_BOOT" \
    >"${RUN_ROOT}/logs/${model}_final2_${selected_name}.log" 2>&1
}

write_summary() {
  $PY_RUNNER - "$RUN_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
summary_paths = (
    sorted(root.glob("*/*/analysis/summary.json"))
    + sorted(root.glob("*/*/selected_*/analysis/summary.json"))
    + sorted(root.glob("*/final1_pilot/grid/*/analysis/summary.json"))
)
for summary in summary_paths:
    data = json.loads(summary.read_text())
    cfg_path = next((summary.parents[1] / "dictionaries").glob("layer_*/config.json"), None)
    metrics = {}
    layer = None
    if cfg_path is not None:
        cfg = json.loads(cfg_path.read_text())
        metrics = cfg.get("metrics", {})
        layer = cfg.get("layer")
    gates = data.get("success_gates", {})
    rows.append({
        "root": str(summary.parents[1]),
        "layer": layer,
        "ve_pt": metrics.get("heldout_variance_explained_pt"),
        "ve_it": metrics.get("heldout_variance_explained_it"),
        "effective_l0": metrics.get("effective_l0"),
        "alive_fraction": metrics.get("alive_fraction"),
        "causal_top200_interaction_drop": gates.get("causal_top200_interaction_drop"),
        "causal_matched_random200_interaction_drop_mean": gates.get("causal_matched_random200_interaction_drop_mean"),
        "causal_moderate_result": gates.get("causal_moderate_result"),
    })
out_dir = root / "analysis"
out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "exp38_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
with (out_dir / "exp38_summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["root"])
    writer.writeheader()
    writer.writerows(rows)
print(f"[exp38] wrote {out_dir / 'exp38_summary.json'} rows={len(rows)}")
PY
}

case "$MODE" in
  smoke)
    STAGE_A_TOKENS=8000 STAGE_A_STEPS=40 STAGE_A_RANK_PROMPTS=8 STAGE_A_MEDIATE_SKIP_PROMPTS=8 STAGE_A_DEV_PROMPTS=8 \
    run_final1_pilot "qwen3_4b" "${GPU_LIST_QWEN%% *}" "35" "35" "smoke_qwen_l35_d1024_k8,35,1024,8,40,256,32,0"
    write_summary
    ;;
  stage-a)
    qwen_configs="qwen_l35_d81920_k64,35,81920,64,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},0;qwen_l35_d131072_k64,35,131072,64,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},1;qwen_l35_d196608_k64,35,196608,64,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},2;qwen_l35_d196608_k48,35,196608,48,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},3"
    olmo_configs="olmo_l31_d131072_k64,31,131072,64,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},0;olmo_l31_d196608_k64,31,196608,64,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},1;olmo_l31_d262144_k64,31,262144,64,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},2;olmo_l31_d262144_k48,31,262144,48,${STAGE_A_STEPS},${STAGE_A_BATCH_TOKENS},${STAGE_A_AUXK},3"
    run_final1_pilot "qwen3_4b" "$GPU_LIST_QWEN" "35" "35" "$qwen_configs" &
    qwen_pid=$!
    run_final1_pilot "olmo2_7b" "$GPU_LIST_OLMO" "31" "31" "$olmo_configs" &
    olmo_pid=$!
    wait "$qwen_pid"
    wait "$olmo_pid"
    write_summary
    ;;
  stage-b)
    run_final2_selected "qwen3_4b" "$GPU_LIST_QWEN" "34 35" "34" "$QWEN_FINAL2_SELECTED_NAME" "$QWEN_FINAL2_DICT_SIZE" "$QWEN_FINAL2_K" &
    qwen_pid=$!
    run_final2_selected "olmo2_7b" "$GPU_LIST_OLMO" "30 31" "30" "$OLMO_FINAL2_SELECTED_NAME" "$OLMO_FINAL2_DICT_SIZE" "$OLMO_FINAL2_K" &
    olmo_pid=$!
    wait "$qwen_pid"
    wait "$olmo_pid"
    write_summary
    ;;
  analyze-only)
    write_summary
    ;;
  *)
    echo "[exp38] invalid MODE=${MODE}" >&2
    exit 2
    ;;
esac

echo "[exp38] complete mode=${MODE} run_root=${RUN_ROOT}"
