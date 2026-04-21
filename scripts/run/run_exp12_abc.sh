#!/usr/bin/env bash
set -euo pipefail

MODEL=""
GPU="${GPU:-0}"
OUT_DIR=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "usage: bash scripts/run_exp12_abc.sh --model <model_name> [--gpu N] [--out-dir DIR] [extra exp12 args...]"
  exit 1
fi

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="results/exp12_free_running_abc_graft/${MODEL}/abc_raw_eval_v1"
fi

export CUDA_VISIBLE_DEVICES="$GPU"

uv run python -m src.poc.exp12_free_running_abc_graft.run \
  --model "$MODEL" \
  --dataset data/eval_dataset_v2.jsonl \
  --n-prompts 1400 \
  --out-dir "$OUT_DIR" \
  "${EXTRA_ARGS[@]}"
