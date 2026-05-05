#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SHARD_DIR="${SHARD_DIR:-results/reproducibility/minimal_audit_shard}"
AUDIT_SHARD_URI="${AUDIT_SHARD_URI:-<anonymous-audit-shard-url>/minimal_audit_shard/}"

if [[ ! -f "$SHARD_DIR/manifest.json" ]]; then
  cat <<EOF
Missing minimal audit shard at:
  $SHARD_DIR

Fetch the anonymized reviewer shard from the supplementary artifact mirror:
  mkdir -p "$SHARD_DIR"
  # Example:
  #   rsync -r "$AUDIT_SHARD_URI" "$SHARD_DIR/"
  # or unpack the submitted artifact archive so that manifest.json appears here.

Then rerun:
  bash scripts/reproduce/reproduce_minimal.sh
EOF
  exit 2
fi

uv run python scripts/reproduce/check_minimal_shard.py --shard-dir "$SHARD_DIR"
