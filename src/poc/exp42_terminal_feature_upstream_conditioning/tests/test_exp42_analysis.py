from __future__ import annotations

from src.poc.exp42_terminal_feature_upstream_conditioning.analyze import (
    _activation_event_rows,
    _causal_minus_control,
    _join_event_rows,
)


def _act(cell: str, feature_set: str, value: float, *, seed=None):
    return {
        "record_type": "activation",
        "model": "toy",
        "prompt_id": "p0",
        "event_kind": "first_diff",
        "feature_set": feature_set,
        "k": 2,
        "control_seed": seed,
        "cell": cell,
        "layer": 0,
        "position_ge_3": True,
        "decoder_weighted_sum": value,
        "decoder_margin_weighted_sum": value * 0.5,
        "sum_activation": value,
        "selected_feature_l0": 1,
        "reconstruction_error_rel": 0.1,
        "feature_l0": 4,
        "selected_union_mass_fraction": 0.2,
    }


def test_activation_gate_uses_it_stack_cells():
    rows = [
        _act("U_IT__L_IT", "causal_top", 5.0),
        _act("U_PT__L_IT", "causal_top", 2.0),
        _act("U_PT__L_PT", "causal_top", 7.0),
        _act("U_IT__L_PT", "causal_top", 6.0),
    ]
    events = _activation_event_rows(rows)
    assert len(events) == 1
    assert events[0]["activation_gate_decoder_weighted"] == 3.0
    assert events[0]["activation_gate_decoder_margin_weighted"] == 1.5
    assert events[0]["pt_branch_activation_gate_decoder_weighted"] == 1.0


def test_causal_minus_random_pairs_by_event_and_k():
    activation = _activation_event_rows(
        [
            _act("U_IT__L_IT", "causal_top", 5.0),
            _act("U_PT__L_IT", "causal_top", 2.0),
            _act("U_IT__L_IT", "causal_matched_random", 4.0, seed=0),
            _act("U_PT__L_IT", "causal_matched_random", 3.0, seed=0),
            _act("U_IT__L_IT", "causal_matched_random", 6.0, seed=1),
            _act("U_PT__L_IT", "causal_matched_random", 5.0, seed=1),
        ]
    )
    joined = _join_event_rows(activation, [])
    diffs = _causal_minus_control(joined, "activation_gate_decoder_weighted", "causal_matched_random")
    assert len(diffs) == 1
    assert diffs[0]["causal_value"] == 3.0
    assert diffs[0]["control_value"] == 1.0
    assert diffs[0]["causal_minus_control"] == 2.0

