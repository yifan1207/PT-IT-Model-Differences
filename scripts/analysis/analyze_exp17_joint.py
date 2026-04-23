#!/usr/bin/env python3
"""Thin wrapper for the canonical exp17 joint analysis module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp17_behavioral_direction_replication.joint_analysis import main


if __name__ == "__main__":
    main()
