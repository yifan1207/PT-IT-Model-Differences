#!/usr/bin/env python3
"""Analyze Exp43 feature-rescue handoff outputs."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp43_feature_rescue_handoff.analyze import add_args, main


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
