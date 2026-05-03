#!/usr/bin/env python3
"""Analyze Exp42 terminal feature upstream-conditioning outputs."""

from src.poc.exp42_terminal_feature_upstream_conditioning.analyze import add_args, main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_args(parser)
    main(parser.parse_args())

