#!/usr/bin/env python
"""Analyze Exp27 natural-rollout residual-opposition records."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp27_natural_rollout_residual_opposition_ntp.analyze import main


if __name__ == "__main__":
    main()
