#!/usr/bin/env bash
set -euo pipefail

HOST="${EXP19_REMOTE_HOST:-yanda-gpu}"
REMOTE_DIR="${EXP19_REMOTE_DIR:-/home/yifan/structral-semantic-features}"

FILES=(
  "src/poc/exp11_matched_prefix_mlp_graft/mlp_graft.py"
  "src/poc/exp11_matched_prefix_mlp_graft/run.py"
  "src/poc/exp19_late_mlp_specificity_controls/__init__.py"
  "src/poc/exp19_late_mlp_specificity_controls/__main__.py"
  "src/poc/exp19_late_mlp_specificity_controls/controls.py"
  "src/poc/exp19_late_mlp_specificity_controls/run.py"
  "src/poc/exp19_late_mlp_specificity_controls/tests/test_controls.py"
  "scripts/analysis/analyze_exp19.py"
  "scripts/plot/plot_exp19.py"
  "scripts/run/run_exp19_yanda_smoke.sh"
  "scripts/run/run_exp19_yanda_internal.sh"
)

echo "[exp19-sync] remote status before sync"
ssh "$HOST" "cd '$REMOTE_DIR' && git status --short -- ${FILES[*]} || true"

echo "[exp19-sync] syncing Exp19 files to $HOST:$REMOTE_DIR"
rsync -av --relative "${FILES[@]}" "$HOST:$REMOTE_DIR/"

echo "[exp19-sync] remote syntax/test check"
ssh "$HOST" "cd '$REMOTE_DIR' && bash -lc '
set -euo pipefail
uv run python -m py_compile \
  src/poc/exp11_matched_prefix_mlp_graft/mlp_graft.py \
  src/poc/exp11_matched_prefix_mlp_graft/run.py \
  src/poc/exp19_late_mlp_specificity_controls/controls.py \
  src/poc/exp19_late_mlp_specificity_controls/run.py \
  scripts/analysis/analyze_exp19.py \
  scripts/plot/plot_exp19.py
uv run pytest -q src/poc/exp19_late_mlp_specificity_controls/tests/test_controls.py
'"

echo "[exp19-sync] remote status after sync"
ssh "$HOST" "cd '$REMOTE_DIR' && git status --short -- ${FILES[*]}"
