import numpy as np
import pytest

from src.poc.exp3.plots.plot_e3_9_step_stability import _layer_mean_sem, _phase_means
from src.poc.exp3.plots.plot_e3_10_mind_change import _mind_change_stats
from src.poc.exp3.plots.plot_e3_11_feature_importance import _rank_features
from src.poc.exp3.plots.plot_e3_12_adjacent_layer_kl import _layer_mean_sem as _kl_layer_mean_sem
from src.poc.exp3.plots.plot_e3_12_adjacent_layer_kl import _phase_means as _kl_phase_means
from src.poc.exp3.plots.plot_e3_13_candidate_reshuffling import _jaccard, _layer_overlap


def test_e3_9_layer_and_phase_aggregation():
    results = [{
        "step_to_step_kl": [
            [None, None, None],
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
        ]
    }]
    mean, sem = _layer_mean_sem(results, "step_to_step_kl")
    assert mean[0] == pytest.approx(0.15)
    assert mean[1] == pytest.approx(0.25)
    assert sem[2] > 0.0

    phase = _phase_means(results, "step_to_step_kl")
    assert phase["content (0-11)"] > 0.0


def test_e3_10_mind_change_uses_first_layer_matching_final_token():
    results = [{
        "generated_tokens": [
            {"token_id": 42, "token_str": "FORM"},
            {"token_id": 99, "token_str": "SAFE"},
        ],
        "top1_token_per_layer": [
            [1] * 12 + [42] + [42] * 21,
            [1] * 21 + [50] * 4 + [99] + [99] * 8,
        ],
        "kl_adjacent_layer": [
            [None] + [0.05] * 11 + [0.3] + [0.0] * 21,
            [None] + [0.02] * 24 + [0.4] + [0.0] * 8,
        ],
    }]
    frac, mean_kl, corrected = _mind_change_stats(results)
    assert frac["format"] == pytest.approx(0.5)
    assert frac["corrective"] == pytest.approx(0.5)
    assert mean_kl["format"] == pytest.approx(0.3)
    assert mean_kl["corrective"] == pytest.approx(0.4)
    assert corrected["SAFE"] == 1


def test_e3_11_ranking_uses_activation_sum():
    summary = {
        20: {
            "count": np.array([0, 2, 1], dtype=np.int64),
            "sum": np.array([0.0, 4.0, 10.0], dtype=np.float32),
        },
        21: {
            "count": np.array([0, 5, 0], dtype=np.int64),
            "sum": np.array([0.0, 2.5, 0.0], dtype=np.float32),
        },
    }
    rows = _rank_features(summary, top_k=2)
    assert rows[0][0] == "L20:F2"
    assert rows[0][1] == 10.0
    assert rows[1][0] == "L20:F1"


def test_e3_12_adjacent_layer_aggregation():
    results = [{
        "kl_adjacent_layer": [
            [None, 0.1, 0.2],
            [None, 0.3, 0.4],
        ]
    }]
    mean, sem = _kl_layer_mean_sem(results)
    assert np.isnan(mean[0])
    assert mean[1] == pytest.approx(0.2)
    assert mean[2] == pytest.approx(0.3)
    assert sem[1] > 0.0

    phase = _kl_phase_means(results)
    assert phase["content"] > 0.0


def test_e3_13_jaccard_and_layer_overlap():
    assert _jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)

    results = [{
        "top5_token_ids_per_layer": [
            [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [5, 6, 7, 8, 9],
            ]
        ]
    }]
    mean, sem = _layer_overlap(results)
    assert mean[1] == pytest.approx(1.0)
    assert mean[2] == pytest.approx(1 / 9)
    assert sem[1] == 0.0
