#!/usr/bin/env bash
set -euo pipefail

# Handoff wrapper for the content/reasoning extension of the v21 headline.
#
# This does not create a new experiment family. It reruns:
#   1. Exp23 residual-state x late-stack factorial on the existing Exp20
#      CONTENT-FACT / CONTENT-REASON / GOV-FORMAT manifest.
#   2. Exp21 productive-opposition/write-out analysis on the same manifest.
#
# Defaults assume a fresh RunPod-style machine where all detected GPUs are usable.
# On shared machines, set GPU_LIST, e.g. GPU_LIST="2 3 4 5 6 7".

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

MODELS="${MODELS:-gemma3_4b qwen3_4b llama31_8b mistral_7b olmo2_7b}"
DATASET="${DATASET:-data/eval_dataset_v2.jsonl}"
EXP20_CONTENT_ROOT="${EXP20_CONTENT_ROOT:-results/exp20_divergence_token_counterfactual/full_runpod_20260423_2148_combined_final}"
N_EXAMPLES="${N_EXAMPLES:-600}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
N_BOOT="${N_BOOT:-2000}"

RUN_EXP23="${RUN_EXP23:-1}"
RUN_EXP21="${RUN_EXP21:-1}"

EXP23_RUN_NAME="${EXP23_RUN_NAME:-exp23_content_reasoning_residual_${TIMESTAMP}}"
EXP21_RUN_NAME="${EXP21_RUN_NAME:-exp21_content_reasoning_${TIMESTAMP}}"

EXP23_RESIDUAL_WORKERS="${EXP23_RESIDUAL_WORKERS:-1}"
EXP21_WORKERS_PER_MODEL="${EXP21_WORKERS_PER_MODEL:-1}"

echo "[content-extension] models=${MODELS}"
echo "[content-extension] dataset=${DATASET}"
echo "[content-extension] exp20_content_root=${EXP20_CONTENT_ROOT}"
echo "[content-extension] n_examples=${N_EXAMPLES}"
echo "[content-extension] gpu_list=${GPU_LIST}"
echo "[content-extension] n_boot=${N_BOOT}"
echo "[content-extension] run_exp23=${RUN_EXP23} run_exp21=${RUN_EXP21}"

if [ "${RUN_EXP23}" = "1" ]; then
  echo "[content-extension] starting Exp23 residual content/reasoning run: ${EXP23_RUN_NAME}"
  bash scripts/run/run_exp23_midlate_interaction_suite.sh \
    --mode full \
    --run-name "${EXP23_RUN_NAME}" \
    --parts residual \
    --models "${MODELS}" \
    --prompt-mode raw_shared \
    --n-prompts "${N_EXAMPLES}" \
    --residual-workers "${EXP23_RESIDUAL_WORKERS}" \
    --gpu-list "${GPU_LIST}" \
    --dataset "${DATASET}" \
    --exp20-root "${EXP20_CONTENT_ROOT}" \
    --exp20-fallback-root "${EXP20_CONTENT_ROOT}" \
    --n-boot "${N_BOOT}"

  uv run python scripts/analysis/analyze_exp23_interaction_subgroups.py \
    --run-root "results/exp23_midlate_interaction_suite/${EXP23_RUN_NAME}" \
    --models ${MODELS} \
    --n-bootstrap "${N_BOOT}"
fi

if [ "${RUN_EXP21}" = "1" ]; then
  echo "[content-extension] starting Exp21 content/reasoning run: ${EXP21_RUN_NAME}"
  RUN_NAME="${EXP21_RUN_NAME}" \
  N_EXAMPLES="${N_EXAMPLES}" \
  DATASET="${DATASET}" \
  EXP20_ROOT="${EXP20_CONTENT_ROOT}" \
  EXP20_FALLBACK_ROOT="${EXP20_CONTENT_ROOT}" \
  PROMPT_MODES="raw_shared" \
  MODELS="${MODELS}" \
  WORKERS_PER_MODEL="${EXP21_WORKERS_PER_MODEL}" \
  GPU_LIST="${GPU_LIST}" \
  N_BOOT="${N_BOOT}" \
    bash scripts/run/run_exp21_productive_opposition_runpod.sh
fi

echo "[content-extension] done"
