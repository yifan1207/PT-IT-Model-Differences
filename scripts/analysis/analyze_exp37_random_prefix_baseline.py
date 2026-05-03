#!/usr/bin/env python3
"""Analyze Exp37 matched-prefix selection baselines."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp37_random_prefix_baseline.analyze_random_prefix_baselines import main


if __name__ == "__main__":
    main()
