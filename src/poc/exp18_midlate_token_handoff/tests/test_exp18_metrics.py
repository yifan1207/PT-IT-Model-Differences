from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp18_midlate_token_handoff.metrics import (
    collapse_category,
    disjoint_windows,
    first_layer_in_topk,
    promote_suppress_transition,
    rank_gain,
    top1_top20_delta,
)


def _row(top1, top20, ranks):
    return {
        "metrics": {
            "top1_token": top1,
            "top20_ids": top20,
            "next_token_rank": ranks,
        }
    }


def test_collapse_category():
    assert collapse_category("STRUCTURAL") == "FORMAT"
    assert collapse_category("DISCOURSE") == "FORMAT"
    assert collapse_category("PUNCTUATION") == "FORMAT"
    assert collapse_category("CONTENT") == "CONTENT"
    assert collapse_category("FUNCTION") == "FUNCTION_OTHER"
    assert collapse_category("OTHER") == "FUNCTION_OTHER"


def test_disjoint_windows_are_clamped_and_ordered():
    windows = disjoint_windows(n_layers=10, phase_boundary=3, corrective_onset=7)
    assert list(windows["early"].layers) == [0, 1, 2]
    assert list(windows["mid_policy"].layers) == [3, 4, 5, 6]
    assert list(windows["late_reconciliation"].layers) == [7, 8, 9]


def test_first_layer_in_topk():
    top20 = [
        [1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10],
        [11, 12, 13, 14, 15],
    ]
    assert first_layer_in_topk(top20, 8, k=5) == 1
    assert first_layer_in_topk(top20, 10, k=3) is None
    assert first_layer_in_topk(top20, 11, k=1) == 2


def test_rank_gain_positive_means_intervention_improves_rank():
    baseline = _row([], [], [50, 40, 30, 20])
    intervention = _row([], [], [50, 35, 10, 5])
    assert rank_gain(baseline, intervention, range(1, 4)) == (5 + 20 + 15) / 3


def test_top1_top20_delta_counts_entries_exits_and_jaccard():
    baseline = _row(
        [1, 1, 2],
        [[1, 2, 3], [1, 2, 3], [2, 3, 4]],
        [10, 10, 10],
    )
    intervention = _row(
        [1, 5, 2],
        [[1, 2, 3], [5, 2, 3], [2, 4, 6]],
        [10, 10, 10],
    )
    out = top1_top20_delta(baseline, intervention, range(0, 3))
    assert out["n_layer_observations"] == 3
    assert out["top1_changed"] == 1
    assert math.isclose(out["top1_change_fraction"], 1 / 3)
    assert out["top20_entries"] == 2
    assert out["top20_exits"] == 2


def test_promote_suppress_sign_convention():
    prev = [0.0, 10.0, 8.0, 7.0]
    curr = [3.0, 8.0, 7.0, 5.0]
    out = promote_suppress_transition(prev, curr, target_id=0, alternative_ids=[1, 2, 3])
    assert out["support_target_delta"] == 3.0
    assert out["repulsion_top10_delta"] == (-2.0 - 1.0 - 2.0) / 3
    assert out["margin_delta"] > 0
