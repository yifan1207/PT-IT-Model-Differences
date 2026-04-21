"""Canonical exp14 entrypoint.

The exp14 causal run reuses the matched-prefix graft engine implemented in
``src.poc.exp11_matched_prefix_mlp_graft.run``. The exp14-specific behavior is
activated there through ``--causal-combined``.
"""

from __future__ import annotations

from src.poc.exp11_matched_prefix_mlp_graft.run import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
