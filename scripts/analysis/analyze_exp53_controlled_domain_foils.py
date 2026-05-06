#!/usr/bin/env python3
"""Analyze Exp53 controlled non-instruction domain foils."""

from __future__ import annotations

from src.poc.exp53_controlled_domain_finetunes.analyze import parse_args, run


if __name__ == "__main__":
    run(parse_args())

