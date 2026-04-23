#!/usr/bin/env python3
"""CLI wrapper for Exp18 matched-prefix handoff analysis."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp18_midlate_token_handoff.handoff_analysis import main


if __name__ == "__main__":
    main()
