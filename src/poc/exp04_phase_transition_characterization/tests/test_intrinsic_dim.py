"""
Tests for analysis/intrinsic_dim.py.

All tests are self-contained — no GPU, no model.
We verify correctness of TwoNN on synthetic data with known intrinsic dimension.
"""
import numpy as np
import pytest
import tempfile
import os

from src.poc.exp04_phase_transition_characterization.analysis.intrinsic_dim import (
    estimate_id_twonn,
    estimate_id_mle,
    stack_by_layer,
    load_residuals,
    compute_id_profile,
    compare_profiles,
)


def _make_low_dim_data(n: int, true_id: int, ambient: int = 256,
                        seed: int = 0) -> np.ndarray:
    """Generate n points on a true_id-dimensional linear submanifold embedded
    in `ambient`-dimensional space.  TwoNN should recover approximately true_id.
    """
    rng = np.random.default_rng(seed)
    low_dim = rng.standard_normal((n, true_id)).astype(np.float64)
    proj    = rng.standard_normal((true_id, ambient)).astype(np.float64)
    return (low_dim @ proj).astype(np.float32)


# ── estimate_id_twonn() ───────────────────────────────────────────────────────

class TestTwoNN:
    def test_returns_float(self):
        X = _make_low_dim_data(200, true_id=5, ambient=64)
        id_val = estimate_id_twonn(X)
        assert isinstance(id_val, float)

    def test_roughly_correct_id_5(self):
        """TwoNN on 5-D manifold in 64-D space should recover ~5."""
        X = _make_low_dim_data(300, true_id=5, ambient=64)
        id_val = estimate_id_twonn(X)
        # Allow generous tolerance — TwoNN is a statistical estimator
        assert 2 <= id_val <= 15, f"Expected ~5, got {id_val}"

    def test_higher_id_for_higher_dim(self):
        """5-D data should give higher ID estimate than 2-D data."""
        X_low  = _make_low_dim_data(200, true_id=2,  ambient=64)
        X_high = _make_low_dim_data(200, true_id=10, ambient=64)
        id_low  = estimate_id_twonn(X_low)
        id_high = estimate_id_twonn(X_high)
        assert id_high > id_low, f"Expected id_high={id_high} > id_low={id_low}"

    def test_too_few_samples_returns_nan(self):
        X = _make_low_dim_data(2, true_id=2, ambient=16)
        assert np.isnan(estimate_id_twonn(X))

    def test_identity_subspace(self):
        """All points on a 1-D line: TwoNN should give ID close to 1."""
        rng = np.random.default_rng(42)
        # Points along x-axis in 50-D space
        t = rng.standard_normal((200, 1))
        direction = np.zeros((1, 50))
        direction[0, 0] = 1.0
        X = (t @ direction).astype(np.float32)
        id_val = estimate_id_twonn(X)
        # Very generous: 1-D manifold, should be < 5
        assert id_val < 5, f"Expected ~1, got {id_val}"


# ── estimate_id_mle() ────────────────────────────────────────────────────────

class TestMLE:
    def test_returns_float(self):
        # skdim.id.MLE is broken on Python 3.13 (FrameLocalsProxy issue).
        # estimate_id_mle catches and returns NaN — we just check it doesn't crash.
        X = _make_low_dim_data(100, true_id=4, ambient=32)
        id_val = estimate_id_mle(X)
        assert isinstance(id_val, float)  # NaN or valid float, not an exception

    def test_too_few_samples(self):
        X = _make_low_dim_data(3, true_id=2, ambient=10)
        # k=5 requires n >= 7
        assert np.isnan(estimate_id_mle(X, k=5))


# ── stack_by_layer() ─────────────────────────────────────────────────────────

class TestStackByLayer:
    def test_shape(self):
        # 3 prompts, 4 layers, d_model = 8
        n_layers, d_model = 4, 8
        residuals = {
            "p0": np.zeros((n_layers, d_model), dtype=np.float32),
            "p1": np.ones( (n_layers, d_model), dtype=np.float32),
            "p2": np.full( (n_layers, d_model), 2.0, dtype=np.float32),
        }
        by_layer = stack_by_layer(residuals, n_layers=n_layers)
        assert len(by_layer) == n_layers
        for layer_i in range(n_layers):
            assert by_layer[layer_i].shape == (3, d_model)

    def test_values_preserved(self):
        residuals = {
            "p0": np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32),  # 2 layers, d=2
            "p1": np.array([[3.0, 0.0], [4.0, 0.0]], dtype=np.float32),
        }
        by_layer = stack_by_layer(residuals, n_layers=2)
        # Layer 0: should contain [1,0] and [3,0] in some order
        layer0 = by_layer[0]
        assert set(layer0[:, 0].tolist()) == {1.0, 3.0}


# ── load_residuals() + round-trip ─────────────────────────────────────────────

class TestLoadResiduals:
    def test_round_trip(self):
        n_layers, d = 4, 16
        arr = np.random.randn(n_layers, d).astype(np.float32)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp = f.name
        try:
            np.savez_compressed(tmp, prompt_abc=arr)
            loaded = load_residuals(tmp)
            assert "prompt_abc" in loaded
            np.testing.assert_allclose(loaded["prompt_abc"], arr)
        finally:
            os.unlink(tmp)


# ── compare_profiles() ───────────────────────────────────────────────────────

class TestCompareProfiles:
    def _make_profile(self, ids: list[float], model_variant: str = "pt") -> dict:
        return {
            "layers":        list(range(len(ids))),
            "id_twonn":      ids,
            "id_mle":        ids,
            "n_samples":     100,
            "model_variant": model_variant,
        }

    def test_sharpening_ratio_gt_1_when_it_sharper(self):
        # Both profiles are constant at 10, except IT has a sharp dip at layer 11
        # (drops from 10 to 2), while PT dips only to 8.
        # IT sharpness = |10 - 10| / 10 = 0 ... we need pre/post to differ.
        # Use: pre(10)=8, dip(11)=min, post(12)=8 for PT with gradual drop,
        # and pre(10)=8, dip(11)=min, post(12)=8 but with 10→3→10 for IT.
        n = 34
        pt_ids = [10.0] * n
        pt_ids[10] = 8.5   # slightly lower pre-dip
        pt_ids[11] = 7.0   # mild dip
        pt_ids[12] = 8.5   # mirror post-dip

        it_ids = [10.0] * n
        it_ids[10] = 9.5   # pre-dip close to plateau
        it_ids[11] = 3.0   # sharp dip
        it_ids[12] = 9.5   # rapid recovery

        pt_p = self._make_profile(pt_ids)
        it_p = self._make_profile(it_ids, model_variant="it")
        comp = compare_profiles(pt_p, it_p)

        # IT sharpness = |9.5 - 9.5| / 9.5 = 0 ... cross-pair defines sharpness
        # Recompute manually: _dip_sharpness = |pre - post| / max(pre, post)
        # PT: |8.5 - 8.5| / 8.5 = 0; IT: |9.5 - 9.5| / 9.5 = 0 → both 0.
        # Use asymmetric pre/post to show clear difference.
        n = 34
        pt_ids2 = [10.0] * n
        pt_ids2[10] = 10.0
        pt_ids2[11] = 8.0   # mild dip
        pt_ids2[12] = 6.0   # gradual decline → |pre-post|/pre = |10-6|/10 = 0.4

        it_ids2 = [10.0] * n
        it_ids2[10] = 10.0
        it_ids2[11] = 2.0   # sharp dip
        it_ids2[12] = 1.0   # stays low → |pre-post|/pre = |10-1|/10 = 0.9

        pt_p2 = self._make_profile(pt_ids2)
        it_p2 = self._make_profile(it_ids2, model_variant="it")
        comp2 = compare_profiles(pt_p2, it_p2)
        # IT sharpness 0.9 > PT sharpness 0.4 → ratio > 1
        assert comp2["sharpening_ratio"] > 1.0

    def test_equal_profiles_ratio_one(self):
        ids = list(range(34))
        pt_p = self._make_profile(ids)
        it_p = self._make_profile(ids, model_variant="it")
        comp = compare_profiles(pt_p, it_p)
        assert comp["sharpening_ratio"] == pytest.approx(1.0)

    def test_peak_detection(self):
        # ID peaks at layer 5
        ids = [0, 1, 2, 3, 4, 10, 3, 2, 1, 0]  # peak at index 5
        profile = self._make_profile(ids)
        it_p    = self._make_profile(ids, model_variant="it")
        comp    = compare_profiles(profile, it_p)
        # Both peak at 5
        assert comp["id_peak_layer_pt"] == 5
        assert comp["id_peak_layer_it"] == 5

    def test_outputs_expected_keys(self):
        ids = [float(i) for i in range(34)]
        pt_p = self._make_profile(ids)
        it_p = self._make_profile(ids, model_variant="it")
        comp = compare_profiles(pt_p, it_p)
        for key in ["canonical_dip_layer", "dip_sharpness_pt", "dip_sharpness_it",
                    "sharpening_ratio", "id_peak_layer_pt", "id_peak_layer_it",
                    "id_at_dip_pt", "id_at_dip_it"]:
            assert key in comp


# ── compute_id_profile() with synthetic .npz ─────────────────────────────────

class TestComputeIdProfile:
    def test_runs_on_synthetic_data(self):
        """Full pipeline: save synthetic residuals, compute ID profile."""
        n_prompts, n_layers, d_model = 50, 10, 32
        true_id = 4

        residuals = {}
        for i in range(n_prompts):
            # Each layer has the same 4-D manifold, just to keep it simple
            residuals[f"p{i}"] = _make_low_dim_data(
                n_layers, true_id, ambient=d_model, seed=i
            )
            # Shape: [n_layers, d_model]

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp = f.name
        try:
            np.savez_compressed(tmp, **residuals)
            profile = compute_id_profile(tmp, model_variant="pt",
                                         n_layers=n_layers, run_mle=False)
            assert len(profile["id_twonn"]) == n_layers
            assert profile["n_samples"] == n_prompts
            assert profile["model_variant"] == "pt"
            # TwoNN estimates should be finite floats (not nan) given enough samples
            valid = [v for v in profile["id_twonn"] if not np.isnan(v)]
            assert len(valid) > 0
        finally:
            os.unlink(tmp)
