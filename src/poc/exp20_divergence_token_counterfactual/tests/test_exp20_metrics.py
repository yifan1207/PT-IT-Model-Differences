from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.poc.exp20_divergence_token_counterfactual.metrics import (
    classify_assistant_marker,
    find_divergence_events,
    pairwise_agreement,
    summarize_token_clusters,
    window_logit_summary,
)


def _tok(token_id: int, text: str, collapsed: str = "CONTENT") -> dict:
    return {
        "token_id": token_id,
        "token_str": text,
        "token_category": "CONTENT",
        "token_category_collapsed": collapsed,
    }


def test_assistant_marker_is_not_triggered_by_broad_function_words():
    assert classify_assistant_marker("Sure")
    assert classify_assistant_marker(" here's")
    assert not classify_assistant_marker(" of")
    assert not classify_assistant_marker(" to")


def test_find_divergence_events_prefers_first_clean_shared_step():
    pt = [_tok(1, "A"), _tok(2, ","), _tok(3, "cat")]
    it = [_tok(1, "A"), _tok(9, ".", "FORMAT"), _tok(4, "Sure")]
    events = find_divergence_events(pt, it)
    assert events["first_diff"]["step"] == 1
    assert events["first_nonformat_diff"]["step"] == 2
    assert events["first_assistant_marker_diff"]["step"] == 2


def test_pairwise_agreement_tracks_first_divergence_and_length():
    out = pairwise_agreement([1, 2, 3], [1, 9], max_len=4)
    assert out["compared"] == 3
    assert out["same"] == 1
    assert out["first_divergence_step"] == 1
    assert math.isclose(out["agreement_fraction"], 1 / 3)


def test_summarize_token_clusters_marks_unique_and_majority_departure():
    out = summarize_token_clusters(
        {
            "a": [1, 1, 1],
            "b": [1, 2, 2],
            "c": [1, 2, 3],
        },
        max_len=3,
    )
    assert out["leaves_majority_step"]["a"] == 1
    assert out["unique_token_step"]["c"] == 2
    assert out["mean_unique_token_count"] > 1


def test_window_logit_summary_sign_convention():
    out = window_logit_summary([0.0, 1.0, 3.0, 2.0], (1, 3))
    assert out["start_value"] == 1.0
    assert out["end_value"] == 3.0
    assert out["total_delta"] == 3.0
    assert out["mean_step_delta"] == 1.5
