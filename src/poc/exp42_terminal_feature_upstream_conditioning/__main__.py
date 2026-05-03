"""CLI entrypoint for Exp42."""

from __future__ import annotations

import argparse

from src.poc.exp42_terminal_feature_upstream_conditioning import analyze, collect


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp42 terminal feature upstream-conditioning audit")
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect.add_args(sub.add_parser("collect", help="collect activation gates and causal drops"))
    analyze.add_args(sub.add_parser("analyze", help="analyze Exp42 records"))
    args = parser.parse_args()
    if args.cmd == "collect":
        collect.main(args)
    elif args.cmd == "analyze":
        analyze.main(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()

