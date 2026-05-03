"""CLI dispatcher for Exp43 feature-rescue handoff experiments."""

from __future__ import annotations

import argparse

from src.poc.exp43_feature_rescue_handoff import analyze, collect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect.add_args(sub.add_parser("collect", help="collect fixed-prefix rescue/handoff records"))
    analyze.add_args(sub.add_parser("analyze", help="analyze fixed-prefix rescue/handoff records"))
    args = parser.parse_args()
    if args.command == "collect":
        collect.main(args)
    elif args.command == "analyze":
        analyze.main(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()

