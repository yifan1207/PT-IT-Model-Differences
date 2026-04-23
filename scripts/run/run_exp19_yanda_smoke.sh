#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP19_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP19_REMOTE_DIR:-/home/yifan/structral-semantic-features}"
RUN_NAME="${EXP19_RUN_NAME:-exp19_smoke_$(date +%Y%m%d_%H%M%S)}"
MODEL="${EXP19_MODEL:-gemma3_4b}"
GPU="${EXP19_GPU:-0}"
N_PROMPTS="${EXP19_N_PROMPTS:-4}"
MAX_NEW_TOKENS="${EXP19_MAX_NEW_TOKENS:-64}"
BATCH_SIZE="${EXP19_BATCH_SIZE:-4}"
CHUNK_SIZE="${EXP19_CHUNK_SIZE:-4}"
TUNED_LENS_DIR="${EXP19_TUNED_LENS_DIR:-/home/yifan/.cache/exp11_tuned_lens_probes_v3}"

ssh "$HOST" "cd '$REMOTE_DIR' && bash -lc '
set -euo pipefail
echo \"[exp19-smoke] host=\$(hostname) run=${RUN_NAME} model=${MODEL} gpu=${GPU}\"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
RUN_ROOT=\"results/exp19_late_mlp_specificity_controls/${RUN_NAME}\"
mkdir -p \"\$RUN_ROOT/logs\"
CUDA_VISIBLE_DEVICES=${GPU} uv run python -m src.poc.exp19_late_mlp_specificity_controls \
  --model ${MODEL} \
  --dataset data/exp3_dataset.jsonl \
  --n-prompts ${N_PROMPTS} \
  --prompt-seed 0 \
  --teacher-forced \
  --readout-mode both \
  --tuned-lens-dir ${TUNED_LENS_DIR} \
  --max-new-tokens ${MAX_NEW_TOKENS} \
  --chunk-size ${CHUNK_SIZE} \
  --batch-size ${BATCH_SIZE} \
  --out-dir \"\$RUN_ROOT/${MODEL}\" \
  >\"\$RUN_ROOT/logs/${MODEL}_smoke.log\" 2>&1
uv run python scripts/analysis/analyze_exp19.py --run-root \"\$RUN_ROOT\" --out \"\$RUN_ROOT/exp19_specificity_summary.json\"
uv run python scripts/plot/plot_exp19.py --summary \"\$RUN_ROOT/exp19_specificity_summary.json\" --out-dir \"\$RUN_ROOT\"
tail -n 40 \"\$RUN_ROOT/logs/${MODEL}_smoke.log\"
echo \"[exp19-smoke] complete: \$RUN_ROOT\"
'"
