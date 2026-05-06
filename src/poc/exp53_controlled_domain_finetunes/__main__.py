"""CLI dispatcher for Exp53 controlled domain fine-tunes."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("train")
    sub.add_parser("merge")
    sub.add_parser("eval")
    sub.add_parser("merge-check")
    sub.add_parser("health")
    sub.add_parser("analyze")
    args, rest = parser.parse_known_args()
    if args.cmd == "prepare":
        from .prepare_data import main as fn
    elif args.cmd == "train":
        from .train_lora import main as fn
    elif args.cmd == "merge":
        from .merge_lora import run, parse_args

        fn = lambda: run(parse_args())
    elif args.cmd == "eval":
        from .eval_domain_nll import run, parse_args

        fn = lambda: run(parse_args())
    elif args.cmd == "merge-check":
        from .check_merge_equivalence import run, parse_args

        fn = lambda: run(parse_args())
    elif args.cmd == "health":
        from .generation_health import run, parse_args

        fn = lambda: run(parse_args())
    else:
        from .analyze import run, parse_args

        fn = lambda: run(parse_args())
    # Reparse in the target module using the untouched process argv. The thin
    # dispatcher exists for discoverability; run scripts invoke modules directly.
    sys.argv = [sys.argv[0], *rest]
    fn()


if __name__ == "__main__":
    main()
