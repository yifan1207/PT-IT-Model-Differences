"""Canonical Exp23 entrypoint.

Exp23 reuses the Exp14 matched-prefix causal runner. Pass
``--include-midlate-factorial`` with ``--causal-combined`` to add the mid+late
branches used by the Exp23 analysis.
"""

from __future__ import annotations

from src.poc.exp14_symmetric_matched_prefix_causality.run import main


if __name__ == "__main__":
    main()
