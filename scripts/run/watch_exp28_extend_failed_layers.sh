#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:?RUN_NAME is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
MODEL="${MODEL:-llama31_8b}"
PY_RUNNER="${PY_RUNNER:-uv run python}"
LAYERS_TO_CHECK="${LAYERS_TO_CHECK:-19 20 21 22 23 24 25 26}"
MIN_VE="${MIN_VE:-0.80}"
EXTEND_STEPS="${EXTEND_STEPS:-8000}"
DICT_SIZE="${DICT_SIZE:-65536}"
K="${K:-512}"
BATCH_TOKENS="${BATCH_TOKENS:-8192}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
GPU_FREE_MEM_MB="${GPU_FREE_MEM_MB:-1000}"
POLL_SECONDS="${POLL_SECONDS:-60}"

echo "[exp28-watch] start $(date -u)"
echo "[exp28-watch] run_root=${RUN_ROOT}"
echo "[exp28-watch] layers_to_check=${LAYERS_TO_CHECK}"

read -r -a CHECK_LAYERS <<< "$LAYERS_TO_CHECK"
while true; do
  have=0
  for layer in "${CHECK_LAYERS[@]}"; do
    if [[ -f "${RUN_ROOT}/dictionaries/layer_${layer}/config.json" ]]; then
      have=$((have + 1))
    fi
  done
  echo "[exp28-watch] first-wave-configs=${have}/${#CHECK_LAYERS[@]} $(date -u)"
  if [[ "$have" -eq "${#CHECK_LAYERS[@]}" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

FAILED_LAYERS="$($PY_RUNNER - "$RUN_ROOT" "$MIN_VE" "${CHECK_LAYERS[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
min_ve = float(sys.argv[2])
failed = []
for raw_layer in sys.argv[3:]:
    layer = int(raw_layer)
    path = root / "dictionaries" / f"layer_{layer}" / "config.json"
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    ve_pt = float(metrics.get("heldout_variance_explained_pt", -1))
    ve_it = float(metrics.get("heldout_variance_explained_it", -1))
    print(f"[exp28-watch] layer={layer} ve_pt={ve_pt:.4f} ve_it={ve_it:.4f}", file=sys.stderr)
    if ve_pt < min_ve or ve_it < min_ve:
        failed.append(str(layer))
print(" ".join(failed))
PY
)"

echo "[exp28-watch] failed_layers=${FAILED_LAYERS:-none}"
if [[ -z "$FAILED_LAYERS" ]]; then
  echo "[exp28-watch] no extension needed"
  exit 0
fi

while true; do
  FREE_GPUS=""
  while IFS=, read -r idx mem _rest; do
    idx="${idx//[[:space:]]/}"
    mem="${mem//[[:space:]]/}"
    if [[ -n "$idx" && -n "$mem" && "$mem" -lt "$GPU_FREE_MEM_MB" ]]; then
      FREE_GPUS+="${idx} "
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  FREE_GPUS="${FREE_GPUS%" "}"
  echo "[exp28-watch] free_gpus=${FREE_GPUS:-none} $(date -u)"
  if [[ -n "$FREE_GPUS" ]]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[exp28-watch] launching extension layers=${FAILED_LAYERS} gpus=${FREE_GPUS}"
MODE=train \
RUN_NAME="$RUN_NAME" \
RUN_ROOT="$RUN_ROOT" \
MODEL="$MODEL" \
GPU_LIST="$FREE_GPUS" \
LAYERS="$FAILED_LAYERS" \
DICT_SIZE="$DICT_SIZE" \
K="$K" \
STEPS="$EXTEND_STEPS" \
BATCH_TOKENS="$BATCH_TOKENS" \
CHECKPOINT_EVERY="$CHECKPOINT_EVERY" \
RESUME_TRAIN=1 \
MIN_VE="$MIN_VE" \
PY_RUNNER="$PY_RUNNER" \
bash scripts/run/run_exp28_late_mlp_crosscoder_mediation.sh

echo "[exp28-watch] done $(date -u)"
