#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP18_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP18_REMOTE_DIR:-/home/yifan/structral-semantic-features}"
RUN_NAME="${EXP18_RUN_NAME:-full_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${EXP18_GPU_LIST:-0,1,3,4,5,6,7}"
N_EXAMPLES="${EXP18_N_EXAMPLES:-600}"

ssh "$HOST" "cd '$REMOTE_DIR' && bash -lc '
set -euo pipefail
IFS=, read -r -a GPUS <<< \"${GPU_LIST}\"
if [[ \${#GPUS[@]} -eq 0 ]]; then
  echo \"No GPUs configured. Set EXP18_GPU_LIST=0,1,3,...\" >&2
  exit 1
fi

echo \"[exp18-full] host: \$(hostname)\"
echo \"[exp18-full] git status for Exp18 files\"
git status --short -- src/poc/exp18_midlate_token_handoff scripts/analysis/analyze_exp18_handoff.py scripts/plot/plot_exp18_handoff.py scripts/run/run_exp18_yanda_full.sh || true
echo \"[exp18-full] nvidia-smi\"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader,nounits
mkdir -p logs

echo \"[exp18-full] matched-prefix analysis\"
uv run python scripts/analysis/analyze_exp18_handoff.py \
  --models all \
  --allow-missing \
  --out-dir results/exp18_midlate_token_handoff/${RUN_NAME}/matched_prefix
uv run python scripts/plot/plot_exp18_handoff.py \
  --summary results/exp18_midlate_token_handoff/${RUN_NAME}/matched_prefix/summary.json

MODELS=(gemma3_4b llama31_8b qwen3_4b mistral_7b olmo2_7b deepseek_v2_lite)
VARIANTS=(pt it)

running=0
slot=0
for model in \"\${MODELS[@]}\"; do
  for variant in \"\${VARIANTS[@]}\"; do
    gpu=\"\${GPUS[\$((slot % \${#GPUS[@]}))]}\"
    slot=\$((slot + 1))
    if [[ \"\$model\" == \"deepseek_v2_lite\" ]]; then
      tokens=64
    else
      tokens=256
    fi
    out=\"results/exp18_midlate_token_handoff/${RUN_NAME}/pure_flow/\${model}/\${variant}\"
    echo \"[exp18-full] launch \$model \$variant on cuda:\$gpu tokens=\$tokens\"
    (
      uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
        --model \"\$model\" --variant \"\$variant\" --device \"cuda:\$gpu\" \
        --n-eval-examples ${N_EXAMPLES} --max-new-tokens \"\$tokens\" \
        --out-dir \"\$out\"
      uv run python -m src.poc.exp18_midlate_token_handoff.collect_pure_flow \
        --model \"\$model\" --variant \"\$variant\" --merge-only --n-workers 1 \
        --out-dir \"\$out\"
    ) > \"logs/exp18_${RUN_NAME}_\${model}_\${variant}.log\" 2>&1 &
    running=\$((running + 1))
    if [[ \$running -ge \${#GPUS[@]} ]]; then
      wait -n
      running=\$((running - 1))
    fi
  done
done
wait

echo \"[exp18-full] complete: results/exp18_midlate_token_handoff/${RUN_NAME}\"
'"
