"""CLI dispatcher for Exp28."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["cache", "train", "rank", "causal-rank", "mediate", "analyze"])
    args, rest = parser.parse_known_args()
    if args.phase == "cache":
        from src.poc.exp28_late_mlp_crosscoder_mediation.cache_activations import main as phase_main
    elif args.phase == "train":
        from src.poc.exp28_late_mlp_crosscoder_mediation.train_crosscoders import main as phase_main
    elif args.phase == "rank":
        from src.poc.exp28_late_mlp_crosscoder_mediation.feature_stats import main as phase_main
    elif args.phase == "causal-rank":
        from src.poc.exp28_late_mlp_crosscoder_mediation.causal_feature_rank import main as phase_main
    elif args.phase == "mediate":
        from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import main as phase_main
    else:
        from src.poc.exp28_late_mlp_crosscoder_mediation.analyze import main as phase_main

    import sys

    sys.argv = [sys.argv[0], *rest]
    phase_main()


if __name__ == "__main__":
    main()
