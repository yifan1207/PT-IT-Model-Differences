#!/usr/bin/env python
"""Thin script wrapper for Exp26 analysis."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp26_residual_opposition_mediation.analyze import main


if __name__ == "__main__":
    main()
