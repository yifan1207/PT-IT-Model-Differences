#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP18_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP18_REMOTE_DIR:-/home/yifan/structral-semantic-features}"
RUN_NAME="${EXP18_RUN_NAME:-smoke_$(date +%Y%m%d_%H%M%S)}"
GPU_PT="${EXP18_GPU_PT:-0}"
GPU_IT="${EXP18_GPU_IT:-1}"
N_SMALL="${EXP18_N_SMALL:-2}"
N_MAIN="${EXP18_N_MAIN:-20}"
TOKENS="${EXP18_TOKENS:-64}"

ssh "$HOST" "cd '$REMOTE_DIR' && bash -lc '
set -euo pipefail
echo \"[exp18-smoke] host: \$(hostname)\"
echo \"[exp18-smoke] git status for Exp18 files\"
git status --short -- src/poc/exp18_midlate_token_handoff scripts/analysis/analyze_exp18_handoff.py scripts/plot/plot_exp18_handoff.py scripts/run/run_exp18_yanda_smoke.sh || true
echo \"[exp18-smoke] nvidia-smi\"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits

echo \"[exp18-smoke] CPU matched-prefix smoke\"
uv run python scripts/analysis/analyze_exp18_handoff.py \
  --models gemma3_4b \
  --allow-missing \
  --max-prompts 5 \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/matched_prefix_smoke
uv run python scripts/plot/plot_exp18_handoff.py \
  --summary results/exp18_midlate_token_handoff/${RUN_NAME}/matched_prefix_smoke/summary.json

echo \"[exp18-smoke] 2-prompt GPU smoke: PT on cuda:${GPU_PT}, IT on cuda:${GPU_IT}\"
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant pt --device cuda:${GPU_PT} \
  --n-eval-examples ${N_SMALL} --max-new-tokens ${TOKENS} \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow/gemma3_4b/pt
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant it --device cuda:${GPU_IT} \
  --n-eval-examples ${N_SMALL} --max-new-tokens ${TOKENS} \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow/gemma3_4b/it
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant pt --merge-only --n-workers 1 \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow/gemma3_4b/pt
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant it --merge-only --n-workers 1 \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow/gemma3_4b/it

echo \"[exp18-smoke] 20-prompt GPU smoke\"
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant pt --device cuda:${GPU_PT} \
  --n-eval-examples ${N_MAIN} --max-new-tokens ${TOKENS} \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow20/gemma3_4b/pt
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant it --device cuda:${GPU_IT} \
  --n-eval-examples ${N_MAIN} --max-new-tokens ${TOKENS} \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow20/gemma3_4b/it
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant pt --merge-only --n-workers 1 \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow20/gemma3_4b/pt
uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
  --model gemma3_4b --variant it --merge-only --n-workers 1 \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow20/gemma3_4b/it

echo \"[exp18-smoke] complete: results/exp18_midlate_token_handoff/${RUN_NAME}\"
'"
