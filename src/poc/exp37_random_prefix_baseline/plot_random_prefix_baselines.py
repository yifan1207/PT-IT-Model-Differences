"""Plot Exp37 summary artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.poc.exp37_random_prefix_baseline.analyze_random_prefix_baselines import _plot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _plot(summary, args.out)
    print(f"[exp37] wrote {args.out}")


if __name__ == "__main__":
    main()

