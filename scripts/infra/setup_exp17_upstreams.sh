#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPSTREAM_ROOT="${1:-$ROOT_DIR/external/exp17_upstream}"

ASSISTANT_AXIS_REPO="https://github.com/safety-research/assistant-axis.git"
ASSISTANT_AXIS_COMMIT="a98961956072224eaf244eb289d6c01700b63795"

DU_REPO="https://github.com/HZD01/post-training-mechanistic-analysis.git"
DU_COMMIT="4abef9d2d3d3bd7781dedfb09a48843d63c31f2c"

clone_or_update() {
  local repo_url="$1"
  local commit="$2"
  local dest="$3"

  if [[ ! -d "$dest/.git" ]]; then
    git clone "$repo_url" "$dest"
  fi

  git -C "$dest" fetch --all --tags
  git -C "$dest" checkout "$commit"
}

mkdir -p "$UPSTREAM_ROOT"

clone_or_update "$ASSISTANT_AXIS_REPO" "$ASSISTANT_AXIS_COMMIT" "$UPSTREAM_ROOT/assistant-axis"
clone_or_update "$DU_REPO" "$DU_COMMIT" "$UPSTREAM_ROOT/post-training-mechanistic-analysis"

cat <<EOF
exp17 upstreams ready:
  assistant-axis: $UPSTREAM_ROOT/assistant-axis @ $ASSISTANT_AXIS_COMMIT
  post-training-mechanistic-analysis: $UPSTREAM_ROOT/post-training-mechanistic-analysis @ $DU_COMMIT
EOF
