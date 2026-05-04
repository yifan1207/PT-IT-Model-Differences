#!/usr/bin/env python3
"""Wrapper for Exp46 Tulu fixed-support stage-sweep analysis."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp46_tulu_fixed_support_stage_sweep.analyze import main


if __name__ == "__main__":
    main()
