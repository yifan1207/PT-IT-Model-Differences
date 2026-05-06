from __future__ import annotations

import argparse

from src.poc.exp51_native_history_crosspatch import analyze, collect


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp51 native-history crosspatch entrypoint.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect", add_help=False)
    sub.add_parser("analyze", add_help=False)
    args, rest = parser.parse_known_args()
    if args.cmd == "collect":
        collect.main_with_args(rest)
    elif args.cmd == "analyze":
        analyze.main_with_args(rest)
    else:
        raise SystemExit(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()
