#!/usr/bin/env python
from src.poc.exp45_behavioral_bridge.analyze import add_args, main

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    add_args(parser)
    main(parser.parse_args())

