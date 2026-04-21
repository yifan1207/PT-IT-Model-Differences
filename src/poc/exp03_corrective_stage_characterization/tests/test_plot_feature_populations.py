from src.poc.exp03_corrective_stage_characterization.plots.plot_feature_populations import (
    _TOKEN_TYPES,
    _layer_feature_profiles,
    _population_enrichment,
)


def test_layer_feature_profiles_and_baseline_use_feature_events():
    prompt_cache = {
        "p0": {
            "token_types": ["CONTENT", "DISCOURSE"],
            "active_by_layer": {10: [{1}, {2}]},
            "n_steps": 2,
        },
        "p1": {
            "token_types": ["CONTENT", "DISCOURSE"],
            "active_by_layer": {10: [{1}, {2}]},
            "n_steps": 2,
        },
    }
    profiles, baseline = _layer_feature_profiles(prompt_cache, 10)
    assert profiles[1]["CONTENT"] == 1.0
    assert profiles[2]["DISCOURSE"] == 1.0
    assert baseline["CONTENT"] == 0.5
    assert baseline["DISCOURSE"] == 0.5


def test_population_enrichment_reveals_pre_vs_post_difference():
    prompt_cache = {
        "p0": {
            "token_types": ["CONTENT", "DISCOURSE", "CONTENT", "DISCOURSE"],
            "active_by_layer": {10: [{1, 7}, {7}, {1, 7}, {7}], 12: [{8}, {9, 8}, {8}, {9, 8}]},
            "n_steps": 4,
        },
        "p1": {
            "token_types": ["CONTENT", "DISCOURSE", "CONTENT", "DISCOURSE"],
            "active_by_layer": {10: [{1, 7}, {7}, {1, 7}, {7}], 12: [{8}, {9, 8}, {8}, {9, 8}]},
            "n_steps": 4,
        },
    }
    pre_profiles, pre_baseline = _layer_feature_profiles(prompt_cache, 10)
    post_profiles, post_baseline = _layer_feature_profiles(prompt_cache, 12)
    pre_enrichment = _population_enrichment(pre_profiles, pre_baseline, {1})
    post_enrichment = _population_enrichment(post_profiles, post_baseline, {9})
    assert pre_enrichment["CONTENT"] > 0.0
    assert pre_enrichment["DISCOURSE"] < 0.0
    assert post_enrichment["DISCOURSE"] > 0.0
    assert post_enrichment["CONTENT"] < 0.0


def test_population_enrichment_returns_zero_for_empty_population():
    enrichment = _population_enrichment({}, {tok: 0.0 for tok in _TOKEN_TYPES}, set())
    assert enrichment == {tok: 0.0 for tok in _TOKEN_TYPES}
