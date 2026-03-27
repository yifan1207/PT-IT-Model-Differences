from __future__ import annotations

import numpy as np
import pytest

from src.poc.exp6.pca_analysis import (
    PCAArtifactsUnavailableError,
    compute_layerwise_pca_stats,
    discover_record_diff_artifact,
    load_record_diffs,
)


def test_compute_layerwise_pca_stats_low_rank_mean_direction_alignment():
    rng = np.random.default_rng(0)
    n_records = 64
    d_model = 16
    base_direction = rng.normal(size=(d_model,))
    base_direction /= np.linalg.norm(base_direction)
    coeffs = rng.normal(loc=0.0, scale=2.0, size=(n_records, 1))
    noise = 0.02 * rng.normal(size=(n_records, d_model))
    arr = coeffs * base_direction[None, :] + noise

    stats = compute_layerwise_pca_stats(
        {25: arr},
        mean_directions={25: base_direction},
    )
    layer_stats = stats[25]
    assert layer_stats.n_records == n_records
    assert layer_stats.pc1_explained_variance > 0.95
    assert layer_stats.n_components_80 == 1
    assert layer_stats.mean_direction_pc1_abs_cosine is not None
    assert layer_stats.mean_direction_pc1_abs_cosine > 0.99


def test_load_record_diffs_supports_paired_it_pt_schema(tmp_path):
    artifact = tmp_path / "results" / "precompute_v2_work" / "record_diff_vectors.npz"
    artifact.parent.mkdir(parents=True)
    record_ids = np.array(["a", "b"], dtype=object)
    it_layer = np.array([[3.0, 1.0], [4.0, 2.0]], dtype=np.float64)
    pt_layer = np.array([[1.0, 0.5], [1.5, 1.0]], dtype=np.float64)
    np.savez(
        artifact,
        record_ids=record_ids,
        it_layer_20=it_layer,
        pt_layer_20=pt_layer,
    )

    found_artifact, ids, per_layer = load_record_diffs(tmp_path, [20], bucket="unused-bucket")
    assert found_artifact == artifact
    assert ids == ["a", "b"]
    expected = it_layer - pt_layer
    np.testing.assert_allclose(per_layer[20], expected)


def test_discover_record_diff_artifact_rejects_aggregate_only_artifacts(tmp_path):
    acts_dir = tmp_path / "results" / "precompute_v2_work" / "acts"
    acts_dir.mkdir(parents=True)
    np.savez(
        acts_dir / "w0.npz",
        it_sum_20=np.array([1.0, 2.0]),
        pt_sum_20=np.array([0.5, 1.0]),
        it_count_20=np.array(10),
        pt_count_20=np.array(10),
    )

    with pytest.raises(PCAArtifactsUnavailableError, match="Exact PCA requires per-record IT–PT difference vectors"):
        discover_record_diff_artifact(tmp_path, bucket="unused-bucket")
