#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SHARD_DIR="${SHARD_DIR:-results/reproducibility/minimal_audit_shard}"
AUDIT_SHARD_URI="${AUDIT_SHARD_URI:-<anonymous-audit-shard-url>/minimal_audit_shard/}"

if [[ ! -f "$SHARD_DIR/manifest.json" ]]; then
  cat <<EOF
Optional minimal audit shard not found at:
  $SHARD_DIR

The submitted supplement is self-contained for CPU claim checking:
  python scripts/reproduce/check_paper_claims.py

For the optional raw-shard smoke test, fetch the anonymized reviewer shard from
the supplementary artifact mirror:
  mkdir -p "$SHARD_DIR"
  # Example:
  #   rsync -r "$AUDIT_SHARD_URI" "$SHARD_DIR/"
  # or unpack the submitted artifact archive so that manifest.json appears here.

Then rerun:
  bash scripts/reproduce/reproduce_minimal.sh
EOF
  if [[ "${REQUIRE_MINIMAL_SHARD:-0}" == "1" ]]; then
    exit 2
  fi
  exit 0
fi

uv run python scripts/reproduce/check_minimal_shard.py --shard-dir "$SHARD_DIR"
