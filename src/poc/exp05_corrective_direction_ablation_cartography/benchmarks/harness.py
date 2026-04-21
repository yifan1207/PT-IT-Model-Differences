from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class HarnessScore:
    benchmark: str
    metric: str
    value: float
    raw: dict[str, Any]


def lm_eval_available() -> bool:
    try:
        __import__("lm_eval")
        return True
    except Exception:
        return False


def run_harness_stub(*_: Any, **__: Any) -> list[HarnessScore]:
    """Thin placeholder integration point for lm-eval-harness.

    Exp5 owns the ablation/intervention surface.  Harness integration remains
    optional and is intentionally shallow: when lm_eval is installed, this module
    is the only place that needs to know how to adapt exp5 generation/scoring to
    task execution.
    """
    if not lm_eval_available():
        raise RuntimeError(
            "lm-evaluation-harness is not installed. Install it separately to use "
            "the exp5 harness backend."
        )
    raise NotImplementedError(
        "Full lm-eval adapter is not implemented in-repo yet. "
        "Exp5 can already run the custom benchmark surface end-to-end."
    )
