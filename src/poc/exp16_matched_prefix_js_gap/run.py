"""Canonical exp16 entrypoint.

The exp16 replay reuses the shared matched-prefix engine implemented in
``src.poc.exp11_matched_prefix_mlp_graft.run``. The exp16-specific behavior is
activated there through ``--causal-combined --js-only`` together with explicit
prompt and teacher-token manifests.
"""

from __future__ import annotations

from src.poc.exp11_matched_prefix_mlp_graft.run import main as _main


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
