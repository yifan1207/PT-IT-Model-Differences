#!/usr/bin/env python3
"""Validate the optional 20-prompt raw reproducibility shard.

The shard is intentionally outside git. The manifest should list every raw or
cached file required for the audit, plus the expected summary JSONs produced by
the shard reproduction script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir",
        type=Path,
        default=Path("results/reproducibility/minimal_audit_shard"),
    )
    args = parser.parse_args()
    shard_dir = args.shard_dir
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"Missing shard manifest: {manifest_path}")
        return 2

    manifest = load_json(manifest_path)
    required = manifest.get("required_files", [])
    missing = [rel for rel in required if not (shard_dir / rel).exists()]
    if missing:
        print("Missing files listed in manifest:")
        for rel in missing:
            print(f"  - {rel}")
        return 1

    expected = manifest.get("expected_summary", {})
    if expected:
        summary_rel = expected.get("path", "expected_summary.json")
        summary_path = shard_dir / summary_rel
        if not summary_path.exists():
            print(f"Missing expected summary: {summary_path}")
            return 1
        summary = load_json(summary_path)
        for key, value in expected.get("values", {}).items():
            observed = summary
            for part in key.split("."):
                observed = observed[part]
            if abs(float(observed) - float(value)) > float(
                expected.get("tolerance", 5e-4)
            ):
                print(f"Mismatch for {key}: observed {observed}, expected {value}")
                return 1

    print(f"Minimal audit shard OK: {shard_dir}")
    print(f"family={manifest.get('family')} prompts={manifest.get('n_prompts')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
