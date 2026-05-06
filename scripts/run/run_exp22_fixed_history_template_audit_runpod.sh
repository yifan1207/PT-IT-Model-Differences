#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

MODE="${MODE:-smoke}"  # smoke|full|analyze-only|sync
RUN_NAME="${RUN_NAME:-exp22_fixed_history_template_audit_$(date -u +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-results/exp22_fixed_history_template_audit/${RUN_NAME}}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b}"
GPU_LIST="${GPU_LIST:-}"
VARIANT_CELLS="${CELLS:-pt_raw it_native it_raw}"
TEACHER_SOURCE="${TEACHER_SOURCE:-it_native}"
WORKERS_PER_MODEL="${WORKERS_PER_MODEL:-1}"
N_EXAMPLES="${N_EXAMPLES:-600}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROBE_FAMILIES="${PROBE_FAMILIES:-raw tuned}"
TUNED_LENS_DIR="${TUNED_LENS_DIR:-/workspace/tuned_lens_probes}"
TOP_K="${TOP_K:-5}"
N_BOOT="${N_BOOT:-2000}"
N_BINS="${N_BINS:-10}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
GCS_SYNC_DEST="${GCS_SYNC_DEST-gs://pt-vs-it-results/results/exp22_fixed_history_template_audit}"
FAIL_ON_QUALITY="${FAIL_ON_QUALITY:-1}"
WRITE_PAPER_SYNTHESIS="${WRITE_PAPER_SYNTHESIS:-}"
export HF_HOME TRANSFORMERS_CACHE HF_HUB_CACHE

if [[ "$MODE" == "smoke" ]]; then
  MODELS="${SMOKE_MODELS:-qwen3_4b}"
  N_EXAMPLES="${SMOKE_EXAMPLES:-8}"
  MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-16}"
  PROBE_FAMILIES="${SMOKE_PROBE_FAMILIES:-raw}"
  N_BOOT="${SMOKE_N_BOOT:-50}"
  FAIL_ON_QUALITY="${SMOKE_FAIL_ON_QUALITY:-0}"
fi
if [[ -z "$WRITE_PAPER_SYNTHESIS" ]]; then
  if [[ "$MODE" == "full" || "$MODE" == "analyze-only" ]]; then
    WRITE_PAPER_SYNTHESIS="1"
  else
    WRITE_PAPER_SYNTHESIS="0"
  fi
fi

if [[ -n "${PY_RUNNER:-}" ]]; then
  read -r -a PYTHON <<< "$PY_RUNNER"
elif [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  PYTHON=(uv run --no-sync python)
else
  PYTHON=(uv run python)
fi

if [[ -z "$GPU_LIST" && "$MODE" != "analyze-only" && "$MODE" != "sync" ]]; then
  GPU_LIST="$("${PYTHON[@]}" - <<'PY'
import torch
print(" ".join(str(i) for i in range(torch.cuda.device_count())))
PY
)"
fi
read -r -a GPUS <<< "${GPU_LIST:-}"
if [[ "$MODE" != "analyze-only" && "$MODE" != "sync" && "${#GPUS[@]}" -lt 1 ]]; then
  echo "[exp22-fixed] GPU mode requires GPU_LIST or visible CUDA GPUs" >&2
  exit 2
fi

LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "$LOG_DIR" "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE"

echo "[exp22-fixed] host $(hostname)"
echo "[exp22-fixed] mode ${MODE}"
echo "[exp22-fixed] run_name ${RUN_NAME}"
echo "[exp22-fixed] run_root ${RUN_ROOT}"
echo "[exp22-fixed] models ${MODELS}"
echo "[exp22-fixed] gpus ${GPU_LIST:-<none>}"
echo "[exp22-fixed] workers_per_model ${WORKERS_PER_MODEL}"
echo "[exp22-fixed] n_examples ${N_EXAMPLES} max_new_tokens ${MAX_NEW_TOKENS}"
echo "[exp22-fixed] probe_families ${PROBE_FAMILIES}"
echo "[exp22-fixed] cells ${VARIANT_CELLS}"
echo "[exp22-fixed] teacher_source ${TEACHER_SOURCE}"
echo "[exp22-fixed] tuned_lens_dir ${TUNED_LENS_DIR}"
echo "[exp22-fixed] hf_home ${HF_HOME}"

sync_outputs() {
  if [[ -z "$GCS_SYNC_DEST" ]]; then
    return
  fi
  if command -v gsutil >/dev/null 2>&1; then
    gsutil -m rsync -r "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}"
  elif [[ -f scripts/infra/gcs_sync_adc.py ]]; then
    "${PYTHON[@]}" scripts/infra/gcs_sync_adc.py upload "$RUN_ROOT" "${GCS_SYNC_DEST%/}/${RUN_NAME}" --workers 24
  else
    echo "[exp22-fixed] no gsutil or ADC sync helper found; skipping sync" >&2
  fi
}

if [[ "$MODE" == "sync" ]]; then
  sync_outputs
  exit 0
fi

if [[ "$MODE" != "analyze-only" && " ${PROBE_FAMILIES} " == *" tuned "* && ! -d "$TUNED_LENS_DIR" ]]; then
  echo "[exp22-fixed] tuned probes requested but TUNED_LENS_DIR missing: ${TUNED_LENS_DIR}" >&2
  exit 3
fi

run_collect_phase() {
  echo "[exp22-fixed] phase collect"
  local pids=()
  local idx=0
  local status=0
  wait_for_batch() {
    local pid
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        status=1
      fi
    done
    pids=()
  }
  for model in $MODELS; do
    for ((w=0; w<WORKERS_PER_MODEL; w++)); do
      local gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
      local model_root="${RUN_ROOT}/${model}"
      local log_path="${LOG_DIR}/collect_${model}_w${w}.log"
      # shellcheck disable=SC2206
      local probe_args=(${PROBE_FAMILIES})
      # shellcheck disable=SC2206
      local cell_args=(${VARIANT_CELLS})
      echo "[exp22-fixed] launch model=${model} worker=${w}/${WORKERS_PER_MODEL} gpu=${gpu}"
      CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON[@]}" -m src.poc.exp22_endpoint_deconfounded_gap.fixed_history \
        --model "$model" \
        --dataset "$DATASET" \
        --out-dir "$model_root" \
        --device cuda:0 \
        --worker-index "$w" \
        --n-workers "$WORKERS_PER_MODEL" \
        --n-eval-examples "$N_EXAMPLES" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --probe-families "${probe_args[@]}" \
        --tuned-lens-dir "$TUNED_LENS_DIR" \
        --top-k "$TOP_K" \
        --cells "${cell_args[@]}" \
        --teacher-source "$TEACHER_SOURCE" \
        >"$log_path" 2>&1 &
      pids+=("$!")
      idx=$((idx + 1))
      if [[ "${#pids[@]}" -ge "${#GPUS[@]}" ]]; then
        wait_for_batch
      fi
    done
  done

  wait_for_batch
  if [[ "$status" -ne 0 ]]; then
    echo "[exp22-fixed] at least one collect worker failed; syncing partial outputs" >&2
    sync_outputs || true
    exit 1
  fi
}

run_merge_phase() {
  echo "[exp22-fixed] phase merge"
  # shellcheck disable=SC2206
  local cell_args=(${VARIANT_CELLS})
  for model in $MODELS; do
    "${PYTHON[@]}" -m src.poc.exp22_endpoint_deconfounded_gap.fixed_history \
      --model "$model" \
      --out-dir "${RUN_ROOT}/${model}" \
      --n-workers "$WORKERS_PER_MODEL" \
      --cells "${cell_args[@]}" \
      --teacher-source "$TEACHER_SOURCE" \
      --merge-only \
      >"${LOG_DIR}/merge_${model}.log" 2>&1
  done
}

run_count_check() {
  echo "[exp22-fixed] phase count-check"
  "${PYTHON[@]}" - <<PY
import gzip, json
from pathlib import Path
root = Path("${RUN_ROOT}")
models = "${MODELS}".split()
cells = "${VARIANT_CELLS}".split()
teacher_source = "${TEACHER_SOURCE}"
expected = int("${N_EXAMPLES}")
summary = {"root": str(root), "teacher_source": teacher_source, "counts": {}, "malformed": {}, "missing": [], "ok": True}
for model in models:
    branches = [(f"teacher_{teacher_source}", root / model / f"teacher_{teacher_source}" / "records.jsonl.gz")]
    branches.extend((cell, root / model / cell / "records.jsonl.gz") for cell in cells)
    for branch, path in branches:
        key = f"{model}/{teacher_source}/{branch}"
        if not path.exists():
            summary["missing"].append(str(path))
            continue
        count = 0
        malformed = 0
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    rec = json.loads(line)
                    malformed += int(bool(rec.get("malformed")))
                except Exception:
                    malformed += 1
        summary["counts"][key] = count
        summary["malformed"][key] = malformed
        if count != expected:
            summary["missing"].append(f"{key}: {count}/{expected}")
        if malformed / max(count, 1) > 0.01:
            summary["missing"].append(f"{key}: malformed {malformed}/{count}")
summary["ok"] = not summary["missing"]
(root / "collect_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
print(json.dumps(summary, indent=2, sort_keys=True))
if not summary["ok"]:
    raise SystemExit(4)
PY
}

run_analysis_phase() {
  echo "[exp22-fixed] phase analyze"
  local quality_args=()
  if [[ "$FAIL_ON_QUALITY" == "1" ]]; then
    quality_args+=(--fail-on-quality)
  fi
  "${PYTHON[@]}" scripts/analysis/analyze_exp22_fixed_history_template_audit.py \
    --root "$RUN_ROOT" \
    --out-dir "${RUN_ROOT}/analysis" \
    --models $MODELS \
    --n-boot "$N_BOOT" \
    --n-bins "$N_BINS" \
    "${quality_args[@]}"

  if [[ "$WRITE_PAPER_SYNTHESIS" == "1" ]]; then
    "${PYTHON[@]}" scripts/analysis/build_exp22_fixed_history_template_audit.py \
      --run-root "$RUN_ROOT" \
      --out-dir results/paper_synthesis
  fi
}

if [[ "$MODE" != "analyze-only" ]]; then
  run_collect_phase
  run_merge_phase
  run_count_check
fi
run_analysis_phase
sync_outputs || true
echo "[exp22-fixed] complete ${RUN_ROOT}"
