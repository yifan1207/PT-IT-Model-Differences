"""Command multiplexer for Exp48."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["sequence", "sequence-suite", "score-sequence", "rescue", "analyze", "adapter-probe"],
    )
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    if args.command == "sequence":
        from src.poc.exp48_static_chimera_sequence_validation.run_sequence_generation import main as run
    elif args.command == "sequence-suite":
        from src.poc.exp48_static_chimera_sequence_validation.run_sequence_suite import main as run
    elif args.command == "score-sequence":
        from src.poc.exp48_static_chimera_sequence_validation.score_sequence_outputs import main as run
    elif args.command == "rescue":
        from src.poc.exp48_static_chimera_sequence_validation.structured_rescue import main as run
    elif args.command == "analyze":
        from src.poc.exp48_static_chimera_sequence_validation.analyze import main as run
    else:
        from src.poc.exp48_static_chimera_sequence_validation.adapter_probe import main as run
    run()


if __name__ == "__main__":
    main()
