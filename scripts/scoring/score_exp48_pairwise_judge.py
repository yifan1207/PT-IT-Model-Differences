#!/usr/bin/env python3
"""Optional pairwise-judge placeholder for Exp48.

Exp48's required bridge is deterministic automatic scoring. This script exists
so the run package has a stable hook for an optional later judge pass without
changing the experiment root layout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or (args.run_root / "analysis" / "optional_pairwise_judge_skipped.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "ok": True,
                "skipped": True,
                "reason": "Exp48 uses deterministic automatic scoring; no LLM judge requested.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"ok": True, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
