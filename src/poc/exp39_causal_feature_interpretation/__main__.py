"""CLI dispatcher for Exp39 causal feature interpretation."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=[
            "preflight",
            "select",
            "dashboard",
            "merge-dashboards",
            "autointerp",
            "group-labels",
            "validate",
            "analyze",
            "run-all",
        ],
    )
    args, rest = parser.parse_known_args()

    from src.poc.exp39_causal_feature_interpretation import pipeline

    phase_main = {
        "preflight": pipeline.main_preflight,
        "select": pipeline.main_select,
        "dashboard": pipeline.main_dashboard,
        "merge-dashboards": pipeline.main_merge_dashboards,
        "autointerp": pipeline.main_autointerp,
        "group-labels": pipeline.main_group_labels,
        "validate": pipeline.main_validate,
        "analyze": pipeline.main_analyze,
        "run-all": pipeline.main_run_all,
    }[args.phase]

    import sys

    sys.argv = [sys.argv[0], *rest]
    phase_main()


if __name__ == "__main__":
    main()
