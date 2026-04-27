#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

echo "[exp23] run_exp23_midlate_kl_factorial.sh is a compatibility alias." >&2
echo "[exp23] Using canonical combined suite runner: scripts/run/run_exp23_midlate_interaction_suite.sh" >&2
exec bash scripts/run/run_exp23_midlate_interaction_suite.sh "$@"

