#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

STEP=""
MODEL="gemma3_4b"
VARIANT="it"
DEVICE="cuda:0"
DTYPE="bfloat16"
DATASET=""
TEXT_FIELD="statement"
LABEL_FIELD="label"
HARMFUL_DATASET=""
HARMLESS_DATASET=""
PROMPT_FIELD="prompt"
ASSISTANT_AXIS_REPO=""
OUT_DIR=""
EXP17_UPSTREAM_ROOT="external/exp17_upstream"

usage() {
  cat <<EOF
Usage:
  bash scripts/run/run_exp17_replications.sh --step STEP [options]

Steps:
  lu
  du-truth
  du-truth-cities
  du-refusal
  du-refusal-upstream
  joint

Common options:
  --model MODEL
  --variant pt|it
  --device DEVICE
  --dtype bfloat16|float16|float32
  --out-dir PATH

Du truthfulness:
  --dataset PATH
  --text-field FIELD
  --label-field FIELD

Du refusal:
  --harmful-dataset PATH
  --harmless-dataset PATH
  --prompt-field FIELD

Lu wrapper:
  --assistant-axis-repo PATH
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --step) STEP="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --text-field) TEXT_FIELD="$2"; shift 2 ;;
    --label-field) LABEL_FIELD="$2"; shift 2 ;;
    --harmful-dataset) HARMFUL_DATASET="$2"; shift 2 ;;
    --harmless-dataset) HARMLESS_DATASET="$2"; shift 2 ;;
    --prompt-field) PROMPT_FIELD="$2"; shift 2 ;;
    --assistant-axis-repo) ASSISTANT_AXIS_REPO="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --upstream-root) EXP17_UPSTREAM_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$STEP" ]]; then
  usage
  exit 1
fi

case "$STEP" in
  lu)
    if [[ -z "$ASSISTANT_AXIS_REPO" ]]; then
      ASSISTANT_AXIS_REPO="${EXP17_UPSTREAM_ROOT}/assistant-axis"
    fi
    cmd=(
      uv run python -m src.poc.exp17_behavioral_direction_replication.lu_pipeline
      --assistant-axis-repo "$ASSISTANT_AXIS_REPO"
      --model "$MODEL"
      --variant "$VARIANT"
    )
    if [[ -n "$OUT_DIR" ]]; then
      cmd+=(--out-dir "$OUT_DIR")
    fi
    "${cmd[@]}"
    ;;
  du-truth)
    if [[ -z "$DATASET" ]]; then
      echo "--dataset is required for step=du-truth" >&2
      exit 1
    fi
    cmd=(
      uv run python -m src.poc.exp17_behavioral_direction_replication.du_truthfulness
      --model "$MODEL"
      --variant "$VARIANT"
      --dataset "$DATASET"
      --text-field "$TEXT_FIELD"
      --label-field "$LABEL_FIELD"
      --device "$DEVICE"
      --dtype "$DTYPE"
    )
    if [[ -n "$OUT_DIR" ]]; then
      cmd+=(--out-dir "$OUT_DIR")
    fi
    "${cmd[@]}"
    ;;
  du-truth-cities)
    DATASET="${EXP17_UPSTREAM_ROOT}/post-training-mechanistic-analysis/Knowledge+Truthfulness/datasets/cities.csv"
    cmd=(
      uv run python -m src.poc.exp17_behavioral_direction_replication.du_truthfulness
      --model "$MODEL"
      --variant "$VARIANT"
      --dataset "$DATASET"
      --text-field "statement"
      --label-field "label"
      --device "$DEVICE"
      --dtype "$DTYPE"
    )
    if [[ -n "$OUT_DIR" ]]; then
      cmd+=(--out-dir "$OUT_DIR")
    fi
    "${cmd[@]}"
    ;;
  du-refusal)
    cmd=(
      uv run python -m src.poc.exp17_behavioral_direction_replication.du_refusal
      --model "$MODEL"
      --variant "$VARIANT"
      --prompt-field "$PROMPT_FIELD"
      --device "$DEVICE"
      --dtype "$DTYPE"
    )
    if [[ -z "$HARMFUL_DATASET" || -z "$HARMLESS_DATASET" ]]; then
      echo "--harmful-dataset and --harmless-dataset are required for step=du-refusal" >&2
      exit 1
    fi
    cmd+=(--harmful-dataset "$HARMFUL_DATASET" --harmless-dataset "$HARMLESS_DATASET")
    if [[ -n "$OUT_DIR" ]]; then
      cmd+=(--out-dir "$OUT_DIR")
    fi
    "${cmd[@]}"
    ;;
  du-refusal-upstream)
    cmd=(
      uv run python -m src.poc.exp17_behavioral_direction_replication.du_refusal
      --model "$MODEL"
      --variant "$VARIANT"
      --use-upstream-du-sources
      --prompt-field "$PROMPT_FIELD"
      --device "$DEVICE"
      --dtype "$DTYPE"
    )
    if [[ -n "$OUT_DIR" ]]; then
      cmd+=(--out-dir "$OUT_DIR")
    fi
    "${cmd[@]}"
    ;;
  joint)
    cmd=(uv run python scripts/analysis/analyze_exp17_joint.py)
    if [[ -n "$OUT_DIR" ]]; then
      cmd+=(--out "$OUT_DIR")
    fi
    "${cmd[@]}"
    ;;
  *)
    echo "Unknown step: $STEP" >&2
    usage
    exit 1
    ;;
esac
