#!/usr/bin/env bash
set -euo pipefail

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/reproduce/check_convergence_gap_claims.py "$@"
elif [[ -x .venv/bin/python ]]; then
  .venv/bin/python scripts/reproduce/check_convergence_gap_claims.py "$@"
else
  python3 scripts/reproduce/check_convergence_gap_claims.py "$@"
fi
