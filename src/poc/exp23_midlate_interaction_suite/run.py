"""Exp23 suite module entrypoint.

The Python entrypoint runs the residual factorial collector. The shell runner
orchestrates the optional Part-A MLP KL factorial alongside it.
"""

from __future__ import annotations

from src.poc.exp23_midlate_interaction_suite.residual_factorial import main


if __name__ == "__main__":
    main()

