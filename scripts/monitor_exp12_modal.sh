#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: bash scripts/monitor_exp12_modal.sh <app_id> <log_path> [checks=8] [sleep_seconds=900]"
  exit 1
fi

APP_ID="$1"
LOG_PATH="$2"
CHECKS="${3:-8}"
SLEEP_SECONDS="${4:-900}"

mkdir -p "$(dirname "$LOG_PATH")"

for i in $(seq 1 "$CHECKS"); do
  {
    echo "===== $(date -u +"%Y-%m-%dT%H:%M:%SZ") check ${i}/${CHECKS} ====="
    uv run modal app list --json
    uv run modal volume ls exp12-results /exp12 --json
    uv run modal app logs "$APP_ID" --since 20m --tail 160 --timestamps || true
    echo
  } >> "$LOG_PATH" 2>&1

  if [[ "$i" -lt "$CHECKS" ]]; then
    sleep "$SLEEP_SECONDS"
  fi
done
