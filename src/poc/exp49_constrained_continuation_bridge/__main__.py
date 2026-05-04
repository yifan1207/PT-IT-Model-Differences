"""Command dispatcher for Exp49."""

from __future__ import annotations

import argparse

from src.poc.exp49_constrained_continuation_bridge import analyze, collect_candidates, score_sequences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("collect-candidates", help="construct native candidate continuations")
    sub.add_parser("score-sequences", help="score candidate continuations through factorial cells")
    sub.add_parser("analyze", help="analyze scored sequences")
    args, rest = parser.parse_known_args()
    if args.cmd == "collect-candidates":
        collect_candidates.main(rest)
    elif args.cmd == "score-sequences":
        score_sequences.main(rest)
    elif args.cmd == "analyze":
        analyze.main(rest)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()

