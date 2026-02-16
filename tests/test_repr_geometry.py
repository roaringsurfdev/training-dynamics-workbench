"""Tests for REQ_044: Representational Geometry library and analyzer."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from miscope.analysis.analyzers import AnalyzerRegistry
from miscope.analysis.analyzers.repr_geometry import (
    RepresentationalGeometryAnalyzer,
    _SITES,
    _get_summary_keys,
)
from miscope.analysis.library.geometry import (
    compute_center_spread,
    compute_circularity,
    compute_class_centroids,
    compute_class_dimensionality,
    compute_class_radii,
    compute_fisher_discriminant,
    compute_fourier_alignment,
)

# ── Geometry Library Tests ───────────────────────────────────────────


class TestComputeClassCentroids:
    def test_two_classes_known_centroids(self):
        activations = np.array([
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 1.0],
            [0.0, 3.0],
        ])
        labels = np.array([0, 0, 1, 1])
        centroids = compute_class_centroids(activations, labels, 2)
        np.testing.assert_allclose(centroids[0], [2.0, 0.0])
        np.testing.assert_allclose(centroids[1], [0.0, 2.0])

    def test_single_sample_per_class(self):
        activations = np.array([[1.0, 2.0], [3.0, 4.0]])
        labels = np.array([0, 1])
        centroids = compute_class_centroids(activations, labels, 2)
        np.testing.assert_allclose(centroids[0], [1.0, 2.0])
        np.testing.assert_allclose(centroids[1], [3.0, 4.0])


class TestComputeClassRadii:
    def test_zero_radius_identical_points(self):
        activations = np.array([[1.0, 0.0], [1.0, 0.0]])
        labels = np.array([0, 0])
        centroids = np.array([[1.0, 0.0]])
        radii = compute_class_radii(activations, labels, centroids)
        assert radii[0] == pytest.approx(0.0)

    def test_known_radius(self):
        # Points at distance 1 from centroid in all directions
        centroids = np.array([[0.0, 0.0]])
        activations = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        labels = np.array([0, 0, 0, 0])
        radii = compute_class_radii(activations, labels, centroids)
        assert radii[0] == pytest.approx(1.0)


class TestComputeClassDimensionality:
    def test_one_dimensional_data(self):
        # All variance on one axis
        activations = np.array([
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ])
        labels = np.array([0, 0, 0, 0])
        centroids = np.array([[2.5, 0.0]])
        dims = compute_class_dimensionality(activations, labels, centroids)
        assert dims[0] == pytest.approx(1.0)

    def test_uniform_two_dimensional(self):
        # Equal variance on two axes
        activations = np.array([
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ])
        labels = np.array([0, 0, 0, 0])
        centroids = np.array([[0.0, 0.0]])
        dims = compute_class_dimensionality(activations, labels, centroids)
        assert dims[0] == pytest.approx(2.0)


class TestComputeCenterSpread:
    def test_zero_spread_identical_centroids(self):
        centroids = np.array([[1.0, 1.0], [1.0, 1.0]])
        assert compute_center_spread(centroids) == pytest.approx(0.0)

    def test_known_spread(self):
        centroids = np.array([[1.0, 0.0], [-1.0, 0.0]])
        # Global centroid = [0, 0], distances = [1, 1], RMS = 1
        assert compute_center_spread(centroids) == pytest.approx(1.0)


class TestComputeCircularity:
    def test_perfect_circle(self):
        # Points on a unit circle
        n = 30
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        centroids_2d = np.column_stack([np.cos(angles), np.sin(angles)])
        # Embed in higher dimensions (pad with zeros)
        centroids = np.hstack([centroids_2d, np.zeros((n, 3))])
        score = compute_circularity(centroids)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_random_cloud_low_circularity(self):
        rng = np.random.default_rng(42)
        centroids = rng.standard_normal((30, 10))
        score = compute_circularity(centroids)
        assert score < 0.5

    def test_collinear_points(self):
        # Points on a line — not circular
        centroids = np.column_stack([
            np.linspace(-5, 5, 20),
            np.zeros(20),
            np.zeros(20),
        ])
        score = compute_circularity(centroids)
        assert score < 0.5


class TestComputeFourierAlignment:
    def test_perfect_alignment(self):
        # Centroids arranged as residue r -> angle 2*pi*r/p (k=1)
        p = 17
        angles = 2 * np.pi * np.arange(p) / p
        centroids = np.column_stack([np.cos(angles), np.sin(angles)])
        score = compute_fourier_alignment(centroids, p)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_scrambled_ordering(self):
        # Centroids on a circle but in random order
        p = 17
        rng = np.random.default_rng(42)
        permutation = rng.permutation(p)
        angles = 2 * np.pi * permutation / p
        centroids = np.column_stack([np.cos(angles), np.sin(angles)])
        score = compute_fourier_alignment(centroids, p)
        # Should be low — no frequency k produces good alignment
        assert score < 0.5

    def test_frequency_k_alignment(self):
        # Centroids follow frequency k=3
        p = 17
        k = 3
        angles = 2 * np.pi * k * np.arange(p) / p
        centroids = np.column_stack([np.cos(angles), np.sin(angles)])
        score = compute_fourier_alignment(centroids, p)
        assert score == pytest.approx(1.0, abs=0.01)


class TestComputeFisherDiscriminant:
    def test_well_separated_classes(self):
        # Two classes far apart with tight spread
        activations = np.array([
            [10.0, 0.0],
            [10.1, 0.0],
            [9.9, 0.0],
            [-10.0, 0.0],
            [-10.1, 0.0],
            [-9.9, 0.0],
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        centroids = compute_class_centroids(activations, labels, 2)
        mean_f, min_f = compute_fisher_discriminant(
            activations, labels, centroids
        )
        assert mean_f > 100  # Very well separated
        assert min_f > 100

    def test_overlapping_classes(self):
        rng = np.random.default_rng(42)
        # Two classes centered at same point
        activations = rng.standard_normal((100, 5))
        labels = np.array([0] * 50 + [1] * 50)
        centroids = compute_class_centroids(activations, labels, 2)
        mean_f, min_f = compute_fisher_discriminant(
            activations, labels, centroids
        )
        assert mean_f < 1.0  # Poorly separated


# ── Analyzer Tests ───────────────────────────────────────────────────


class TestRepresentationalGeometryAnalyzer:
    def _make_mock_cache(self, p: int, d_model: int = 8, d_mlp: int = 16):
        """Create a mock activation cache with known activations."""
        n_samples = p * p
        seq_len = 3
        rng = np.random.default_rng(42)

        cache = MagicMock()

        def cache_getitem(key):
            if isinstance(key, tuple):
                if len(key) == 3 and key[2] == "mlp":
                    return torch.tensor(
                        rng.standard_normal((n_samples, seq_len, d_mlp)),
                        dtype=torch.float32,
                    )
                else:
                    return torch.tensor(
                        rng.standard_normal((n_samples, seq_len, d_model)),
                        dtype=torch.float32,
                    )
            return None

        cache.__getitem__ = MagicMock(side_effect=cache_getitem)
        return cache

    def _make_probe(self, p: int):
        """Create a probe tensor for modular addition."""
        pairs = []
        for a in range(p):
            for b in range(p):
                pairs.append([a, b, p])
        return torch.tensor(pairs, dtype=torch.long)

    def test_analyzer_protocol_compliance(self):
        analyzer = RepresentationalGeometryAnalyzer()
        assert analyzer.name == "repr_geometry"
        assert hasattr(analyzer, "analyze")
        assert hasattr(analyzer, "get_summary_keys")
        assert hasattr(analyzer, "compute_summary")

    def test_analyzer_registered(self):
        assert AnalyzerRegistry.is_registered("repr_geometry")

    def test_analyze_returns_expected_keys(self):
        p = 7
        analyzer = RepresentationalGeometryAnalyzer()
        probe = self._make_probe(p)
        cache = self._make_mock_cache(p)
        model = MagicMock()

        result = analyzer.analyze(model, probe, cache, {"params": {"prime": p}})

        # Check all sites have all expected keys
        for site in _SITES:
            assert f"{site}_centroids" in result
            assert f"{site}_radii" in result
            assert f"{site}_dimensionality" in result
            assert f"{site}_mean_radius" in result
            assert f"{site}_mean_dim" in result
            assert f"{site}_center_spread" in result
            assert f"{site}_snr" in result
            assert f"{site}_circularity" in result
            assert f"{site}_fourier_alignment" in result
            assert f"{site}_fisher_mean" in result
            assert f"{site}_fisher_min" in result

    def test_analyze_shapes(self):
        p = 7
        analyzer = RepresentationalGeometryAnalyzer()
        probe = self._make_probe(p)
        cache = self._make_mock_cache(p, d_model=8, d_mlp=16)
        model = MagicMock()

        result = analyzer.analyze(model, probe, cache, {"params": {"prime": p}})

        # Centroid shapes
        assert result["resid_post_centroids"].shape == (p, 8)
        assert result["mlp_out_centroids"].shape == (p, 16)
        # Per-class arrays
        assert result["resid_post_radii"].shape == (p,)
        assert result["resid_post_dimensionality"].shape == (p,)

    def test_summary_keys_match_scalars(self):
        summary_keys = _get_summary_keys()
        assert len(summary_keys) == 4 * 8  # 4 sites × 8 scalar measures

    def test_compute_summary_extracts_scalars(self):
        p = 7
        analyzer = RepresentationalGeometryAnalyzer()
        probe = self._make_probe(p)
        cache = self._make_mock_cache(p)
        model = MagicMock()

        result = analyzer.analyze(model, probe, cache, {"params": {"prime": p}})
        summary = analyzer.compute_summary(result, {})

        # All summary values should be floats
        for key, value in summary.items():
            assert isinstance(value, float), f"{key} is {type(value)}"

        # Summary should not contain centroid arrays
        assert all("centroids" not in k for k in summary)
        assert all("radii" not in k for k in summary)
        assert all("dimensionality" not in k for k in summary)

    def test_labels_computation(self):
        p = 5
        analyzer = RepresentationalGeometryAnalyzer()
        probe = self._make_probe(p)
        labels = analyzer._compute_labels(probe, p)

        # Verify labels are correct: (a + b) mod p
        probe_np = probe.numpy()
        expected = (probe_np[:, 0] + probe_np[:, 1]) % p
        np.testing.assert_array_equal(labels, expected)

        # Each class should have exactly p samples
        for r in range(p):
            assert np.sum(labels == r) == p
