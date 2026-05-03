#!/usr/bin/env python
"""Analyze Exp44 middle-to-terminal feature handoff outputs."""

from __future__ import annotations

import argparse

from src.poc.exp44_middle_terminal_feature_handoff.analyze import add_args, main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())

