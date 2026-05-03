from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import LatentRow
from src.poc.exp44_middle_terminal_feature_handoff.analyze import (
    _control_differences,
    _family_balanced_bootstrap,
)
from src.poc.exp44_middle_terminal_feature_handoff.collect import (
    HandoffSpec,
    _aggregate_activation,
    _build_handoff_specs,
)


def _latent(layer: int, latent_id: int, score: float = 1.0) -> LatentRow:
    return LatentRow(
        layer=layer,
        latent_id=latent_id,
        feature_type="causal_ranked",
        interaction_score=score,
        mean_activation_it=1.0,
        mean_activation_pt=0.0,
        decoder_norm_it=1.0,
        decoder_norm_pt=0.0,
        latent_scaling_ratio=1.0,
        local_margin_attr=score,
    )


def test_handoff_specs_include_all_controls() -> None:
    causal = [_latent(30, i, score=10 - i) for i in range(5)]
    pool = causal + [_latent(30, 100 + i, score=-i) for i in range(20)]
    specs = _build_handoff_specs(
        causal_rows=causal,
        pool_rows=pool,
        k_list=[2],
        random_seeds=[0, 1],
        include_top_active=True,
        include_same_delta_random=True,
    )
    keys = {(spec.feature_set, spec.control_mode, spec.control_seed) for spec in specs}
    assert ("causal_top", "feature_ablate", None) in keys
    assert ("top_active_noncausal", "feature_ablate", None) in keys
    assert ("causal_matched_random", "feature_ablate", 0) in keys
    assert ("causal_same_delta_random", "same_delta_random", 1) in keys


def test_activation_aggregate_uses_exp42_key_shape() -> None:
    spec = HandoffSpec("causal_top", 20, None, "feature_ablate", tuple([_latent(30, 1)]))
    records = [
        {
            "feature_set": "causal_top",
            "k": 20,
            "control_seed": None,
            "sum_activation": 2.0,
            "decoder_margin_weighted_sum": 3.0,
            "decoder_weighted_sum": 4.0,
            "activation_rate": 0.5,
            "selected_feature_l0": 7.0,
            "reconstruction_error_rel": 0.1,
            "selected_union_mass_fraction": 0.2,
        },
        {
            "feature_set": "causal_top",
            "k": 20,
            "control_seed": None,
            "sum_activation": 5.0,
            "decoder_margin_weighted_sum": 6.0,
            "decoder_weighted_sum": 8.0,
            "activation_rate": 0.25,
            "selected_feature_l0": 9.0,
            "reconstruction_error_rel": 0.3,
            "selected_union_mass_fraction": 0.4,
        },
    ]
    agg = _aggregate_activation(records, spec)
    assert agg["sum_activation"] == 7.0
    assert agg["decoder_margin_weighted_sum"] == 9.0
    assert agg["activation_rate"] == 0.375
    assert agg["selected_l0"] == 16.0
    assert agg["reconstruction_error_rel"] == 0.2


def test_control_differences_and_prompt_cluster_bootstrap() -> None:
    rows = [
        {
            "record_type": "handoff",
            "model": "m1",
            "prompt_id": "p1",
            "event_kind": "first_diff",
            "direction": "rescue",
            "window": "mid",
            "feature_set": "causal_top",
            "control_mode": "feature_ablate",
            "k": 200,
            "terminal_mediated_effect": 3.0,
            "terminal_mediated_fraction": 0.6,
            "activation_rescue_decoder_margin_weighted_sum": 2.0,
            "activation_rescue_sum_activation": 5.0,
        },
        {
            "record_type": "handoff",
            "model": "m1",
            "prompt_id": "p1",
            "event_kind": "first_diff",
            "direction": "rescue",
            "window": "mid",
            "feature_set": "causal_matched_random",
            "control_mode": "feature_ablate",
            "k": 200,
            "terminal_mediated_effect": 1.0,
            "terminal_mediated_fraction": 0.2,
            "activation_rescue_decoder_margin_weighted_sum": 0.5,
            "activation_rescue_sum_activation": 2.0,
        },
    ]
    diffs = _control_differences(rows)
    assert len(diffs) == 1
    assert diffs[0]["terminal_mediated_effect_causal_minus_control"] == 2.0
    clustered = []
    for model, values in {"m1": [1.0, 3.0], "m2": [10.0, 14.0]}.items():
        for idx, value in enumerate(values):
            clustered.append({"model": model, "prompt_id": f"p{idx}", "metric": value})
            clustered.append({"model": model, "prompt_id": f"p{idx}", "metric": value})
    fb = _family_balanced_bootstrap(clustered, "metric", n_boot=50, seed=0)
    assert fb["estimate"] == 7.0
    assert fb["n_prompt_clusters"] == 4
