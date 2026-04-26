#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SHARD_DIR="${SHARD_DIR:-results/reproducibility/minimal_audit_shard}"
GCS_PREFIX="${GCS_PREFIX:-gs://pt-vs-it-results/reproducibility/minimal_audit_shard/}"

if [[ ! -f "$SHARD_DIR/manifest.json" ]]; then
  cat <<EOF
Missing minimal audit shard at:
  $SHARD_DIR

Fetch the shard with:
  mkdir -p "$SHARD_DIR"
  gsutil -m rsync -r "$GCS_PREFIX" "$SHARD_DIR/"

Then rerun:
  bash scripts/reproduce/reproduce_minimal.sh
EOF
  exit 2
fi

uv run python scripts/reproduce/check_minimal_shard.py --shard-dir "$SHARD_DIR"
