#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if command -v uv >/dev/null 2>&1; then
  uv run python scripts/reproduce/check_paper_claims.py "$@"
elif [[ -x ".venv/bin/python" ]]; then
  .venv/bin/python scripts/reproduce/check_paper_claims.py "$@"
else
  python3 scripts/reproduce/check_paper_claims.py "$@"
fi
