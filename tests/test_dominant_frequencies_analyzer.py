"""Tests for REQ_003_002: Dominant Frequencies Analyzer."""

import os
import tempfile

import numpy as np
import pytest
import torch

from analysis import AnalysisPipeline, Analyzer
from analysis.analyzers import DominantFrequenciesAnalyzer
from FourierEvaluation import get_fourier_bases
from ModuloAdditionSpecification import ModuloAdditionSpecification


class TestDominantFrequenciesAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """DominantFrequenciesAnalyzer implements Analyzer protocol."""
        analyzer = DominantFrequenciesAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_name_property(self):
        """Analyzer has correct name property."""
        analyzer = DominantFrequenciesAnalyzer()
        assert analyzer.name == "dominant_frequencies"

    def test_has_analyze_method(self):
        """Analyzer has analyze method."""
        analyzer = DominantFrequenciesAnalyzer()
        assert callable(analyzer.analyze)


class TestDominantFrequenciesAnalyzerOutput:
    """Tests for analyzer output shape and values."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_spec(self, temp_dir):
        """Create a model spec with minimal training."""
        spec = ModuloAdditionSpecification(
            model_dir=temp_dir,
            prime=17,  # Small prime for fast tests
            device="cpu",
            seed=42,
        )
        spec.train(num_epochs=50, checkpoint_epochs=[0, 25, 49])
        return spec

    @pytest.fixture
    def model_with_cache(self, model_spec):
        """Create model, run forward pass, return model and cache."""
        model_spec.generate_training_data()
        model = model_spec.load_from_file()
        _, cache = model.run_with_cache(model_spec.dataset)
        fourier_basis, _ = get_fourier_bases(model_spec.prime, model_spec.device)
        return model, model_spec.dataset, cache, fourier_basis

    def test_returns_dict(self, model_with_cache):
        """analyze returns a dict."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert isinstance(result, dict)

    def test_returns_coefficients_key(self, model_with_cache):
        """Result contains 'coefficients' key."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert "coefficients" in result

    def test_coefficients_is_numpy_array(self, model_with_cache):
        """Coefficients is a numpy array."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert isinstance(result["coefficients"], np.ndarray)

    def test_coefficients_shape(self, model_with_cache):
        """Coefficients has correct shape (n_fourier_components,)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)

        # For prime=17, fourier_basis has shape (18, 17): 1 constant + 17 sin/cos pairs
        # Actually: (2*(17//2) + 1, 17) = (17, 17) for odd primes
        n_components = fourier_basis.shape[0]
        assert result["coefficients"].shape == (n_components,)

    def test_coefficients_are_non_negative(self, model_with_cache):
        """Coefficient values are non-negative (they are norms)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert np.all(result["coefficients"] >= 0)

    def test_coefficients_on_cpu(self, model_with_cache):
        """Coefficients are on CPU (numpy array, not tensor)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = DominantFrequenciesAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert not isinstance(result["coefficients"], torch.Tensor)


class TestDominantFrequenciesAnalyzerIntegration:
    """Integration tests with AnalysisPipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_spec(self, temp_dir):
        """Create a model spec with minimal training."""
        spec = ModuloAdditionSpecification(
            model_dir=temp_dir,
            prime=17,
            device="cpu",
            seed=42,
        )
        spec.train(num_epochs=50, checkpoint_epochs=[0, 25, 49])
        return spec

    def test_pipeline_creates_artifact(self, model_spec):
        """Pipeline creates dominant_frequencies.npz artifact."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        artifact_path = os.path.join(model_spec.artifacts_dir, "dominant_frequencies.npz")
        assert os.path.exists(artifact_path)

    def test_artifact_contains_epochs(self, model_spec):
        """Artifact contains epochs array."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("dominant_frequencies")
        assert "epochs" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 25, 49])

    def test_artifact_contains_coefficients(self, model_spec):
        """Artifact contains coefficients array."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("dominant_frequencies")
        assert "coefficients" in artifact

    def test_artifact_coefficients_shape(self, model_spec):
        """Artifact coefficients have correct shape (n_epochs, n_components)."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("dominant_frequencies")
        n_epochs = len(model_spec.get_available_checkpoints())
        # For prime=17: n_components = 2*(17//2) + 1 = 17
        n_components = 2 * (model_spec.prime // 2) + 1

        assert artifact["coefficients"].shape == (n_epochs, n_components)

    def test_coefficients_change_across_epochs(self, model_spec):
        """Coefficients differ between epochs (training changes weights)."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("dominant_frequencies")
        coefficients = artifact["coefficients"]

        # First and last epoch should have different coefficients
        assert not np.allclose(coefficients[0], coefficients[-1])
