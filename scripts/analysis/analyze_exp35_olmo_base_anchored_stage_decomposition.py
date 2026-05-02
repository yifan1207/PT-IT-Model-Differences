#!/usr/bin/env python3
"""Wrapper for Exp35 OLMo base-anchored stage decomposition analysis."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp35_olmo_base_anchored_stage_decomposition.analyze import main


if __name__ == "__main__":
    main()
