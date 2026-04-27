#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP24_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP24_REMOTE_DIR:-/home/yifan/structral-semantic-features}"
REMOTE_SCRIPT="scripts/run/run_exp24_32b_external_validity.sh"
DRY_RUN=0

usage() {
  cat <<EOF
Usage:
  bash scripts/run/run_exp24_32b_external_validity_yanda.sh [--dry-run] [Exp24 options...]

Runs the Exp24 local-on-host launcher on the configured Yanda GPU host.

Defaults:
  EXP24_REMOTE_HOST=${HOST}
  EXP24_REMOTE_DIR=${REMOTE_DIR}

Examples:
  bash scripts/run/run_exp24_32b_external_validity_yanda.sh --dry-run --phase smoke --model qwen25_32b
  bash scripts/run/run_exp24_32b_external_validity_yanda.sh --phase exp20-raw --model qwen25_32b --gpu-group 2,3,4,5
EOF
}

args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) args+=("$1"); shift ;;
  esac
done

remote_cmd=(bash "$REMOTE_SCRIPT" "${args[@]}")
printf -v quoted_cmd "%q " "${remote_cmd[@]}"
printf -v quoted_dir "%q" "$REMOTE_DIR"
ssh_cmd=(ssh "$HOST" "cd ${quoted_dir} && ${quoted_cmd}")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '%q ' "${ssh_cmd[@]}"
  printf '\n'
  exit 0
fi

"${ssh_cmd[@]}"
