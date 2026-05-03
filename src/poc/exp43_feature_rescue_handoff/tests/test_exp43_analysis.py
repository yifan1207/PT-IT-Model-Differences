from __future__ import annotations

from src.poc.exp28_late_mlp_crosscoder_mediation.run_mediation import LatentRow
from src.poc.exp43_feature_rescue_handoff.analyze import (
    _alpha0_sanity_rows,
    _family_balanced_bootstrap,
    _rescue_control_differences,
)
from src.poc.exp43_feature_rescue_handoff.collect import _build_rescue_specs


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


def test_rescue_specs_include_seeded_same_delta_controls() -> None:
    causal = [_latent(30, i, score=10 - i) for i in range(5)]
    pool = causal + [_latent(30, 100 + i, score=-i) for i in range(20)]
    specs = _build_rescue_specs(
        causal_rows=causal,
        pool_rows=pool,
        k_list=[2],
        random_seeds=[0, 1, 2],
        include_same_delta_random=True,
    )
    same_delta = [s for s in specs if s.feature_set == "causal_same_delta_random"]
    matched = [s for s in specs if s.feature_set == "causal_matched_random"]
    assert len(same_delta) == 3
    assert len(matched) == 3
    assert {s.control_seed for s in same_delta} == {0, 1, 2}


def test_family_balanced_bootstrap_clusters_prompt_rows() -> None:
    rows = []
    for model, values in {"m1": [1.0, 3.0], "m2": [10.0, 14.0]}.items():
        for prompt_idx, value in enumerate(values):
            # Duplicate rows from the same prompt should collapse before the
            # family-balanced estimate, not masquerade as independent events.
            rows.append({"model": model, "prompt_id": f"p{prompt_idx}", "metric": value})
            rows.append({"model": model, "prompt_id": f"p{prompt_idx}", "metric": value})
    result = _family_balanced_bootstrap(rows, "metric", n_boot=50, seed=0)
    assert result["n_families"] == 2
    assert result["n_prompt_clusters"] == 4
    assert result["estimate"] == 7.0
    assert result["ci_low"] is not None
    assert result["ci_high"] is not None


def test_control_differences_and_alpha0_sanity() -> None:
    rows = [
        {
            "record_type": "rescue",
            "model": "m",
            "prompt_id": "p",
            "event_kind": "first_diff",
            "feature_set": "causal_top",
            "control_mode": "feature_delta",
            "k": 20,
            "alpha": 1.0,
            "rescue_gain": 3.0,
            "rescue_fraction": 0.3,
            "interaction_drop_from_rescue": 3.0,
        },
        {
            "record_type": "rescue",
            "model": "m",
            "prompt_id": "p",
            "event_kind": "first_diff",
            "feature_set": "causal_matched_random",
            "control_mode": "feature_delta",
            "k": 20,
            "alpha": 1.0,
            "rescue_gain": 1.0,
            "rescue_fraction": 0.1,
            "interaction_drop_from_rescue": 1.0,
        },
        {
            "record_type": "rescue",
            "model": "m",
            "prompt_id": "p",
            "event_kind": "first_diff",
            "feature_set": "causal_top",
            "control_mode": "feature_delta",
            "k": 20,
            "alpha": 0.0,
            "control_seed": None,
            "rescue_gain": 0.0,
        },
    ]
    diffs = _rescue_control_differences(rows)
    assert len(diffs) == 1
    assert diffs[0]["rescue_gain_causal_minus_control"] == 2.0
    sanity = _alpha0_sanity_rows(rows)
    assert sanity[0]["max_abs_rescue_gain"] == 0.0
