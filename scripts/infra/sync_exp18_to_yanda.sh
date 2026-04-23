#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP18_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP18_REMOTE_DIR:-/home/yifan/structral-semantic-features}"

FILES=(
  "src/poc/exp18_midlate_token_handoff/__init__.py"
  "src/poc/exp18_midlate_token_handoff/metrics.py"
  "src/poc/exp18_midlate_token_handoff/handoff_analysis.py"
  "src/poc/exp18_midlate_token_handoff/collect_pure_flow.py"
  "src/poc/exp18_midlate_token_handoff/tests/test_exp18_metrics.py"
  "src/poc/exp18_midlate_token_handoff/tests/test_handoff_analysis_smoke.py"
  "scripts/analysis/analyze_exp18_handoff.py"
  "scripts/plot/plot_exp18_handoff.py"
  "scripts/run/run_exp18_yanda_smoke.sh"
  "scripts/run/run_exp18_yanda_full.sh"
)

echo "[exp18-sync] preflight remote status"
ssh "$HOST" "cd '$REMOTE_DIR' && git status --short -- src/poc/exp18_midlate_token_handoff scripts/analysis/analyze_exp18_handoff.py scripts/plot/plot_exp18_handoff.py scripts/run/run_exp18_yanda_smoke.sh scripts/run/run_exp18_yanda_full.sh || true"

echo "[exp18-sync] syncing only Exp18 files to $HOST:$REMOTE_DIR"
rsync -av --relative "${FILES[@]}" "$HOST:$REMOTE_DIR/"

echo "[exp18-sync] remote Exp18 status after sync"
ssh "$HOST" "cd '$REMOTE_DIR' && git status --short -- src/poc/exp18_midlate_token_handoff scripts/analysis/analyze_exp18_handoff.py scripts/plot/plot_exp18_handoff.py scripts/run/run_exp18_yanda_smoke.sh scripts/run/run_exp18_yanda_full.sh"
