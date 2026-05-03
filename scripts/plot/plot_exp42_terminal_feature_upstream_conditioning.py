#!/usr/bin/env python3
"""Re-render Exp42 plots by rerunning the analyzer."""

from src.poc.exp42_terminal_feature_upstream_conditioning.analyze import add_args, main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_args(parser)
    main(parser.parse_args())

