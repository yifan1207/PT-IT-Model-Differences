"""CLI dispatcher for Exp41 bucket steering."""

from __future__ import annotations

import sys

from src.poc.exp41_causal_feature_bucket_steering import analyze, build_bucket_manifest, run_logit_replay


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: python -m src.poc.exp41_causal_feature_bucket_steering {manifest,logit-replay,analyze} [...]")
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    command = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if command == "manifest":
        build_bucket_manifest.main()
    elif command == "logit-replay":
        run_logit_replay.main()
    elif command == "analyze":
        analyze.main()
    else:
        print(f"Unknown Exp41 command: {command}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
