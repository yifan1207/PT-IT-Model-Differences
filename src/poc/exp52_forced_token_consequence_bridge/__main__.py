from __future__ import annotations

import argparse

from src.poc.exp52_forced_token_consequence_bridge import analyze, collect


def main() -> None:
    parser = argparse.ArgumentParser(description="Exp52 forced-token consequence bridge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    collect.add_args(sub.add_parser("collect"))
    analyze.add_args(sub.add_parser("analyze"))
    args = parser.parse_args()
    if args.cmd == "collect":
        collect.main(args)
    elif args.cmd == "analyze":
        analyze.main(args)
    else:  # pragma: no cover
        parser.error(f"unknown command {args.cmd}")


if __name__ == "__main__":
    main()

