#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP19_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP19_REMOTE_DIR:-/home/yifan/structral-semantic-features}"
RUN_NAME="${EXP19_RUN_NAME:-exp19_internal_$(date +%Y%m%d_%H%M%S)}"
N_PROMPTS="${EXP19_N_PROMPTS:-600}"
MAX_NEW_TOKENS="${EXP19_MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${EXP19_BATCH_SIZE:-24}"
CHUNK_SIZE="${EXP19_CHUNK_SIZE:-24}"
TUNED_LENS_DIR="${EXP19_TUNED_LENS_DIR:-/home/yifan/.cache/exp11_tuned_lens_probes_v3}"

ssh "$HOST" "cd '$REMOTE_DIR' && bash -lc '
set -euo pipefail
RUN_ROOT=\"results/exp19_late_mlp_specificity_controls/${RUN_NAME}\"
mkdir -p \"\$RUN_ROOT/logs\" \"\$RUN_ROOT/shards\" \"\$RUN_ROOT/merged\"
echo \"[exp19-full] host=\$(hostname) run=${RUN_NAME}\"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

run_single() {
  local gpu=\"\$1\"
  local model=\"\$2\"
  local shard_index=\"\$3\"
  local num_shards=\"\$4\"
  local out_dir=\"\$RUN_ROOT/shards/\${model}__shard\${shard_index}of\${num_shards}\"
  local log_path=\"\$RUN_ROOT/logs/\${model}__shard\${shard_index}of\${num_shards}.log\"
  echo \"[exp19-full] launch model=\$model shard=\$shard_index/\$num_shards gpu=\$gpu\"
  CUDA_VISIBLE_DEVICES=\"\$gpu\" uv run python -m src.poc.exp19_late_mlp_specificity_controls \
    --model \"\$model\" \
    --dataset data/exp3_dataset.jsonl \
    --n-prompts ${N_PROMPTS} \
    --prompt-seed 0 \
    --teacher-forced \
    --readout-mode both \
    --tuned-lens-dir ${TUNED_LENS_DIR} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --chunk-size ${CHUNK_SIZE} \
    --batch-size ${BATCH_SIZE} \
    --num-shards \"\$num_shards\" \
    --shard-index \"\$shard_index\" \
    --out-dir \"\$out_dir\" \
    >\"\$log_path\" 2>&1 &
  LAUNCHED_PID=\$!
}

run_single 0 llama31_8b 0 2; pid_llama0=\"\$LAUNCHED_PID\"
run_single 1 llama31_8b 1 2; pid_llama1=\"\$LAUNCHED_PID\"
run_single 3 mistral_7b 0 2; pid_mistral0=\"\$LAUNCHED_PID\"
run_single 4 mistral_7b 1 2; pid_mistral1=\"\$LAUNCHED_PID\"
run_single 5 olmo2_7b 0 2; pid_olmo0=\"\$LAUNCHED_PID\"
run_single 6 olmo2_7b 1 2; pid_olmo1=\"\$LAUNCHED_PID\"
run_single 7 gemma3_4b 0 1; pid_gemma=\"\$LAUNCHED_PID\"

wait \"\$pid_gemma\"; echo \"[exp19-full] finished gemma3_4b\"
run_single 7 qwen3_4b 0 1; pid_qwen=\"\$LAUNCHED_PID\"

wait \"\$pid_llama0\"; echo \"[exp19-full] finished llama31_8b shard 0/2\"
wait \"\$pid_llama1\"; echo \"[exp19-full] finished llama31_8b shard 1/2\"
wait \"\$pid_mistral0\"; echo \"[exp19-full] finished mistral_7b shard 0/2\"
wait \"\$pid_mistral1\"; echo \"[exp19-full] finished mistral_7b shard 1/2\"
wait \"\$pid_olmo0\"; echo \"[exp19-full] finished olmo2_7b shard 0/2\"
wait \"\$pid_olmo1\"; echo \"[exp19-full] finished olmo2_7b shard 1/2\"
wait \"\$pid_qwen\"; echo \"[exp19-full] finished qwen3_4b\"

uv run python scripts/merge/merge_exp11_shards_local.py --run-root \"\$RUN_ROOT\" --model gemma3_4b --num-shards 1
uv run python scripts/merge/merge_exp11_shards_local.py --run-root \"\$RUN_ROOT\" --model qwen3_4b --num-shards 1
uv run python scripts/merge/merge_exp11_shards_local.py --run-root \"\$RUN_ROOT\" --model llama31_8b --num-shards 2
uv run python scripts/merge/merge_exp11_shards_local.py --run-root \"\$RUN_ROOT\" --model mistral_7b --num-shards 2
uv run python scripts/merge/merge_exp11_shards_local.py --run-root \"\$RUN_ROOT\" --model olmo2_7b --num-shards 2

uv run python scripts/analysis/analyze_exp19.py --run-root \"\$RUN_ROOT\" --out \"\$RUN_ROOT/exp19_specificity_summary.json\"
uv run python scripts/plot/plot_exp19.py --summary \"\$RUN_ROOT/exp19_specificity_summary.json\" --out-dir \"\$RUN_ROOT\"
echo \"[exp19-full] complete: \$RUN_ROOT\"
'"
