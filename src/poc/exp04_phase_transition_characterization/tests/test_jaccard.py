"""Tests for alignment-based continuity analysis in analysis/jaccard.py."""
import numpy as np
import pytest
import tempfile
import os

from src.poc.exp04_phase_transition_characterization.analysis.jaccard import (
    jaccard,
    jaccard_curve_for_prompt,
    compute_continuity_stats,
    dip_summary,
    load_features_exp4,
)


# ── jaccard() ─────────────────────────────────────────────────────────────────

class TestJaccard:
    def test_identical_sets(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int32)
        assert jaccard(a, b) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([4, 5, 6], dtype=np.int32)
        assert jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([2, 3, 4], dtype=np.int32)
        # intersection = {2,3}, union = {1,2,3,4}
        assert jaccard(a, b) == pytest.approx(2.0 / 4.0)

    def test_empty_both(self):
        a = np.array([], dtype=np.int32)
        b = np.array([], dtype=np.int32)
        assert np.isnan(jaccard(a, b))

    def test_empty_one(self):
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([], dtype=np.int32)
        assert jaccard(a, b) == pytest.approx(0.0)

    def test_single_element_match(self):
        a = np.array([42], dtype=np.int32)
        b = np.array([42], dtype=np.int32)
        assert jaccard(a, b) == pytest.approx(1.0)

    def test_asymmetric_input(self):
        a = np.array([1, 2, 3, 4], dtype=np.int32)
        b = np.array([3, 4, 5], dtype=np.int32)
        # intersection = {3,4}, union = {1,2,3,4,5}
        assert jaccard(a, b) == pytest.approx(2.0 / 5.0)


# ── jaccard_curve_for_prompt() ────────────────────────────────────────────────

class TestJaccardCurveForPrompt:
    def _make_af(self, n_steps: int, n_layers: int,
                 feats_per_step: dict = None) -> np.ndarray:
        """Build object array [n_steps, n_layers] of int32 arrays.

        feats_per_step: optional {(step, layer): [feat_indices]}
        """
        af = np.empty((n_steps, n_layers), dtype=object)
        for s in range(n_steps):
            for l in range(n_layers):
                key = (s, l)
                if feats_per_step and key in feats_per_step:
                    af[s, l] = np.array(feats_per_step[key], dtype=np.int32)
                else:
                    af[s, l] = np.array([], dtype=np.int32)
        return af

    def test_basic_single_step(self):
        # step 0: layer 0 = {0,1,2}, layer 1 = {1,2,3}
        feats = {(0, 0): [0, 1, 2], (0, 1): [1, 2, 3]}
        af = self._make_af(1, 5, feats)
        pairs = [(0, 1)]
        result = jaccard_curve_for_prompt(af, pairs)
        # With a single event, profile-based matching can align all active features.
        assert result[(0, 1)] == [pytest.approx(1.0)]

    def test_two_steps_constant(self):
        feats = {
            (0, 0): [10, 20], (0, 1): [10, 20],
            (1, 0): [10, 20], (1, 1): [10, 20],
        }
        af = self._make_af(2, 3, feats)
        result = jaccard_curve_for_prompt(af, [(0, 1)])
        assert result[(0, 1)] == [pytest.approx(1.0), pytest.approx(1.0)]

    def test_out_of_range_layer(self):
        af = self._make_af(1, 5)
        result = jaccard_curve_for_prompt(af, [(3, 10)])  # layer 10 >= n_layers=5
        assert np.isnan(result[(3, 10)][0])

    def test_multiple_pairs(self):
        # Layer 0 = {0,1}, Layer 1 = {1,2}, Layer 2 = {2,3}
        feats = {(0, 0): [0, 1], (0, 1): [1, 2], (0, 2): [2, 3]}
        af = self._make_af(1, 4, feats)
        result = jaccard_curve_for_prompt(af, [(0, 1), (1, 2)])
        assert result[(0, 1)][0] == pytest.approx(1.0)
        assert result[(1, 2)][0] == pytest.approx(1.0)


# ── compute_jaccard_stats() ───────────────────────────────────────────────────

class TestComputeContinuityStats:
    def _make_features_dict(self, n_prompts: int, n_layers: int,
                            layer_feats: dict) -> dict:
        """Build dict of prompt_id → [1, n_layers] object arrays.

        layer_feats: {layer: [feat_indices]}  — same for all prompts and steps
        """
        out = {}
        for pid in range(n_prompts):
            af = np.empty((1, n_layers), dtype=object)
            for l in range(n_layers):
                feats = layer_feats.get(l, [])
                af[0, l] = np.array(feats, dtype=np.int32)
            out[str(pid)] = af
        return out

    def test_identical_features_all_layers(self):
        # All layers have the same features → all Jaccard = 1
        feats = {l: [0, 1, 2] for l in range(15)}
        fd = self._make_features_dict(3, 15, feats)
        stats = compute_continuity_stats(fd, analysis_start=8, analysis_end=15, dip_layer=11)
        for pair in stats["adjacent_pairs"]:
            assert stats["mean_continuity"][pair] == pytest.approx(1.0)

    def test_disjoint_at_dip(self):
        # Build disjoint prompt-level activation profiles across the dip.
        fd = {}
        for pid in range(5):
            af = np.empty((1, 15), dtype=object)
            for l in range(15):
                af[0, l] = np.array([], dtype=np.int32)
            if pid < 2:
                af[0, 10] = np.array([0, 1, 2], dtype=np.int32)
            else:
                af[0, 11] = np.array([3, 4, 5], dtype=np.int32)
                af[0, 12] = np.array([3, 4, 5], dtype=np.int32)
                af[0, 13] = np.array([3, 4, 5], dtype=np.int32)
            fd[str(pid)] = af
        stats = compute_continuity_stats(fd, analysis_start=8, analysis_end=15, dip_layer=11)
        # C(10, 11) should be 0 — no shared prompt-level activation profiles
        pair = (10, 11)
        assert stats["mean_continuity"][pair] == pytest.approx(0.0)
        # Post-dip adjacent layers still align perfectly.
        assert stats["mean_continuity"][(12, 13)] == pytest.approx(1.0)
        # Cross-dip continuity is also 0.
        cross = stats["cross_dip_pair"]
        assert stats["mean_continuity"][cross] == pytest.approx(0.0)

    def test_feature_death_count(self):
        # 3 features die at layer 8→9: layer 8 has {0,1,2}, layer 9+ is empty
        # feature_death[la] counts |features_at_la - features_at_la+1|
        layer_feats = {8: [0, 1, 2]}  # all other layers empty
        fd = self._make_features_dict(2, 15, layer_feats)
        stats = compute_continuity_stats(fd, analysis_start=8, analysis_end=15, dip_layer=11)
        # At transition 8→9: layer 8 has {0,1,2}, layer 9 has {} → 3 deaths
        assert stats["feature_death"].get(8, 0) == pytest.approx(3.0)

    def test_sem_zero_single_prompt(self):
        # Single prompt → SEM should be 0
        feats = {l: [0, 1] for l in range(15)}
        fd = self._make_features_dict(1, 15, feats)
        stats = compute_continuity_stats(fd, analysis_start=8, analysis_end=15, dip_layer=11)
        for pair in stats["adjacent_pairs"]:
            assert stats["sem_continuity"][pair] == pytest.approx(0.0)

    def test_returns_cross_dip_pair(self):
        feats = {l: [l] for l in range(15)}
        fd = self._make_features_dict(2, 15, feats)
        stats = compute_continuity_stats(fd, analysis_start=8, analysis_end=15, dip_layer=11)
        assert stats["cross_dip_pair"] == (10, 12)
        assert stats["cross_dip_pair"] in stats["mean_continuity"]


# ── dip_summary() ─────────────────────────────────────────────────────────────

class TestDipSummary:
    def _make_stats(self, jaccard_by_pair: dict, dip_layer: int = 11) -> dict:
        """Build a minimal stats dict for dip_summary."""
        adj_pairs = [(l, l + 1) for l in range(8, 14)]
        cross     = (dip_layer - 1, dip_layer + 1)
        all_pairs = adj_pairs + [cross]
        return {
            "adjacent_pairs": adj_pairs,
            "cross_dip_pair": cross,
            "all_pairs":      all_pairs,
            "mean_continuity": {**{p: jaccard_by_pair.get(p, 0.5) for p in all_pairs}},
            "sem_continuity":  {p: 0.0 for p in all_pairs},
            "per_prompt":     {p: [] for p in all_pairs},
            "feature_death":  {p[0]: 1.0 for p in adj_pairs},
            "feature_birth":  {p[0]: 1.0 for p in adj_pairs},
        }

    def test_sharp_dip(self):
        # J drops to 0.1 at the dip, 0.9 elsewhere
        jmap = {(8, 9): 0.9, (9, 10): 0.9, (10, 11): 0.1, (11, 12): 0.1,
                (12, 13): 0.9, (13, 14): 0.9, (10, 12): 0.05}
        stats = self._make_stats(jmap)
        summary = dip_summary(stats, dip_layer=11)
        assert summary["j_across_dip"]    == pytest.approx(0.1)
        assert summary["j_exiting_dip"]   == pytest.approx(0.1)
        assert summary["j_cross_dip"]     == pytest.approx(0.05)
        # Control baseline (below dip) should be ~0.9
        assert summary["j_control_below"] == pytest.approx(0.9)


# ── load_features_exp4() with tmp file ───────────────────────────────────────

class TestLoadFeaturesExp4:
    def test_round_trip(self):
        """Save and reload a features dict through load_features_exp4."""
        # Build synthetic data
        n_layers = 5
        af_arr   = np.empty(n_layers, dtype=object)
        for l in range(n_layers):
            af_arr[l] = np.array([l * 10, l * 10 + 1], dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp_path = f.name
        try:
            np.savez_compressed(tmp_path, prompt_0=af_arr)
            loaded = load_features_exp4(tmp_path)
            assert "prompt_0" in loaded
            arr = loaded["prompt_0"]
            # Should be reshaped to [1, n_layers]
            assert arr.shape == (1, n_layers)
            # Feature content preserved
            np.testing.assert_array_equal(arr[0, 2], np.array([20, 21], dtype=np.int32))
        finally:
            os.unlink(tmp_path)
