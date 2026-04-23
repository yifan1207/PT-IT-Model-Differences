"""Canonical Exp19 runner.

This forwards to the shared exp11/14 matched-prefix engine and enables
``--specificity-controls`` by default so Exp19 stays exactly comparable to the
Exp14 teacher-forced setup.
"""

from __future__ import annotations

import sys

from src.poc.exp11_matched_prefix_mlp_graft.run import main as _main


def main() -> None:
    if "--help" not in sys.argv and "-h" not in sys.argv and "--specificity-controls" not in sys.argv:
        sys.argv.append("--specificity-controls")
    _main()


if __name__ == "__main__":
    main()
