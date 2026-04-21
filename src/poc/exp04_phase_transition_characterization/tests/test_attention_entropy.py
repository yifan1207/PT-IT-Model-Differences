"""
Tests for analysis/attention_entropy.py.

All tests are self-contained — no GPU, no model.
"""
import math
import numpy as np
import pytest

from src.poc.exp04_phase_transition_characterization.analysis.attention_entropy import (
    extract_attn_entropy,
    compute_mean_entropy_profile,
    compute_entropy_divergence,
    summarise_dip_region,
    GLOBAL_ATTN_LAYERS,
)
from src.poc.exp04_phase_transition_characterization.collect import _attn_entropy


# ── _attn_entropy() ──────────────────────────────────────────────────────────

class TestAttnEntropy:
    def test_uniform_distribution(self):
        """Uniform attention over T positions → maximum entropy = log(T)."""
        import torch
        T   = 10
        row = torch.ones(T) / T
        H   = _attn_entropy(row)
        assert H == pytest.approx(math.log(T), rel=1e-4)

    def test_peaked_distribution(self):
        """All attention on one token → entropy = 0."""
        import torch
        T   = 10
        row = torch.zeros(T)
        row[3] = 1.0
        H = _attn_entropy(row)
        assert H == pytest.approx(0.0, abs=1e-6)

    def test_zero_entries_dont_crash(self):
        """Zeros in attention (local window) should not cause NaN/inf."""
        import torch
        row = torch.zeros(20)
        row[:5] = 0.2   # 5 positions have equal weight
        H = _attn_entropy(row)
        assert math.isfinite(H)
        assert H == pytest.approx(math.log(5), rel=1e-4)

    def test_negative_fp_noise(self):
        """Tiny negative values from fp arithmetic are clamped to 0."""
        import torch
        row = torch.tensor([0.5, 0.5, -1e-8])
        H = _attn_entropy(row)
        assert math.isfinite(H)


# ── extract_attn_entropy() ───────────────────────────────────────────────────

def _make_results(n_prompts: int, n_layers: int, n_heads: int,
                  entropy_val: float = 1.0,
                  missing_layers: set = None) -> list[dict]:
    """Build synthetic results list with constant attention entropy."""
    records = []
    for _ in range(n_prompts):
        layer_entropies = []
        for l in range(n_layers):
            if missing_layers and l in missing_layers:
                layer_entropies.append(None)
            else:
                layer_entropies.append([entropy_val] * n_heads)
        records.append({"attn_entropy": layer_entropies, "category": "IC"})
    return records


class TestExtractAttnEntropy:
    def test_basic_extraction(self):
        results = _make_results(5, n_layers=10, n_heads=8, entropy_val=2.0)
        data = extract_attn_entropy(results, n_layers=10)
        assert data["n_records"] == 5
        assert data["n_heads"]   == 8
        # Layer 3 should have 5 records
        assert len(data["by_layer"][3]) == 5
        # Each record is [entropy_val] * n_heads
        assert data["by_layer"][3][0] == [pytest.approx(2.0)] * 8

    def test_none_attn_entropy_skipped(self):
        results = [{"attn_entropy": None}, {"attn_entropy": None}]
        data = extract_attn_entropy(results, n_layers=5)
        assert data["n_records"] == 0

    def test_missing_layers_skipped(self):
        results = _make_results(3, n_layers=5, n_heads=4, missing_layers={2})
        data = extract_attn_entropy(results, n_layers=5)
        assert 2 not in data["by_layer"] or len(data["by_layer"][2]) == 0


# ── compute_mean_entropy_profile() ───────────────────────────────────────────

class TestMeanEntropyProfile:
    def test_constant_entropy(self):
        n_layers, n_heads = 8, 4
        results = _make_results(10, n_layers=n_layers, n_heads=n_heads,
                                entropy_val=1.5)
        data    = extract_attn_entropy(results, n_layers=n_layers)
        profile = compute_mean_entropy_profile(data, n_layers=n_layers)

        for l in range(n_layers):
            assert profile["mean_entropy"][l] == pytest.approx(1.5)

    def test_sem_zero_for_constant_data(self):
        results = _make_results(20, n_layers=4, n_heads=2, entropy_val=0.7)
        data    = extract_attn_entropy(results, n_layers=4)
        profile = compute_mean_entropy_profile(data, n_layers=4)
        for l in range(4):
            assert profile["sem_entropy"][l] == pytest.approx(0.0, abs=1e-7)

    def test_missing_layer_gives_nan(self):
        results = _make_results(5, n_layers=6, n_heads=2, missing_layers={3})
        data    = extract_attn_entropy(results, n_layers=6)
        profile = compute_mean_entropy_profile(data, n_layers=6)
        assert np.isnan(profile["mean_entropy"][3])

    def test_global_attn_layer_flagged(self):
        results = _make_results(2, n_layers=34, n_heads=8)
        data    = extract_attn_entropy(results, n_layers=34)
        profile = compute_mean_entropy_profile(data, n_layers=34)
        # Layer 5 is a global attention layer in Gemma 3 4B
        assert profile["is_global"][5] is True
        assert profile["is_global"][0] is False


# ── compute_entropy_divergence() ─────────────────────────────────────────────

class TestEntropyDivergence:
    def test_identical_profiles_zero_diff(self):
        n = 5
        profile_vals = [1.0] * n
        profile = {
            "mean_entropy": profile_vals,
            "sem_entropy":  [0.0] * n,
            "mean_per_head": [[1.0] * 4] * n,
            "is_global": [False] * n,
        }
        div = compute_entropy_divergence(profile, profile, n_layers=n)
        for l in range(n):
            assert div["abs_diff"][l] == pytest.approx(0.0, abs=1e-9)

    def test_divergence_sign(self):
        n = 3
        pt_profile = {
            "mean_entropy":  [1.0, 2.0, 3.0],
            "sem_entropy":   [0.0] * 3,
            "mean_per_head": [[1.0]] * 3,
            "is_global": [False] * 3,
        }
        it_profile = {
            "mean_entropy":  [1.0, 1.5, 4.0],
            "sem_entropy":   [0.0] * 3,
            "mean_per_head": [[1.0]] * 3,
            "is_global": [False] * 3,
        }
        div = compute_entropy_divergence(pt_profile, it_profile, n_layers=n)
        assert div["abs_diff"][0] == pytest.approx(0.0)
        assert div["abs_diff"][1] == pytest.approx(0.5)
        assert div["abs_diff"][2] == pytest.approx(1.0)
        # Layer 1: IT < PT → direction = -1
        assert div["direction"][1] == -1
        # Layer 2: IT > PT → direction = +1
        assert div["direction"][2] == +1


# ── summarise_dip_region() ───────────────────────────────────────────────────

class TestSummariseDipRegion:
    def _make_profile(self, mean_entropy: list) -> dict:
        n = len(mean_entropy)
        return {
            "layers":        list(range(n)),
            "mean_entropy":  mean_entropy,
            "sem_entropy":   [0.0] * n,
            "mean_per_head": [[1.0]] * n,
            "is_global":     [False] * n,
        }

    def test_dip_region_mean(self):
        # Dip region with window=1 around layer 3: layers 2,3,4
        entropies = [1.0, 1.0, 0.5, 0.3, 0.5, 1.0, 1.0]
        profile   = self._make_profile(entropies)
        summary   = summarise_dip_region(profile, dip_layer=3, window=1)
        expected_dip_mean = (0.5 + 0.3 + 0.5) / 3
        assert summary["dip_region_mean"] == pytest.approx(expected_dip_mean)

    def test_entropy_at_dip(self):
        entropies = [2.0, 1.5, 0.8, 0.4, 0.8, 1.5, 2.0]
        profile   = self._make_profile(entropies)
        summary   = summarise_dip_region(profile, dip_layer=3, window=1)
        assert summary["entropy_at_dip"] == pytest.approx(0.4)

    def test_dip_min_layer(self):
        entropies = [1.0, 1.0, 0.5, 0.2, 0.3, 1.0]
        profile   = self._make_profile(entropies)
        # dip_layer=3, window=1 → dip region = layers 2,3,4 → min at layer 3
        summary = summarise_dip_region(profile, dip_layer=3, window=1)
        assert summary["dip_min_layer"] == 3


# ── GLOBAL_ATTN_LAYERS constant ───────────────────────────────────────────────

def test_global_attn_layers_gemma3():
    """Gemma 3 4B has global attention at layers 5,11,17,23,29 (i%6==5)."""
    expected = {5, 11, 17, 23, 29}
    assert GLOBAL_ATTN_LAYERS == expected
