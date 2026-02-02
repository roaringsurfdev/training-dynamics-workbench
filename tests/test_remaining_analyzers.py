"""Tests for REQ_003_003: Remaining Analyzers."""

import os
import tempfile

import numpy as np
import pytest

from analysis import AnalysisPipeline, Analyzer
from analysis.analyzers import NeuronActivationsAnalyzer, NeuronFreqClustersAnalyzer
from FourierEvaluation import get_fourier_bases
from ModuloAdditionSpecification import ModuloAdditionSpecification


class TestNeuronActivationsAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """NeuronActivationsAnalyzer implements Analyzer protocol."""
        analyzer = NeuronActivationsAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_correct_name(self):
        """Analyzer has correct name property."""
        analyzer = NeuronActivationsAnalyzer()
        assert analyzer.name == "neuron_activations"


class TestNeuronActivationsAnalyzerOutput:
    """Tests for analyzer output."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_spec(self, temp_dir):
        spec = ModuloAdditionSpecification(
            model_dir=temp_dir,
            prime=17,
            device="cpu",
            seed=42,
        )
        spec.train(num_epochs=50, checkpoint_epochs=[0, 25, 49])
        return spec

    @pytest.fixture
    def model_with_cache(self, model_spec):
        model_spec.generate_training_data()
        model = model_spec.load_from_file()
        _, cache = model.run_with_cache(model_spec.dataset)
        fourier_basis, _ = get_fourier_bases(model_spec.prime, model_spec.device)
        return model, model_spec.dataset, cache, fourier_basis

    def test_returns_dict_with_activations(self, model_with_cache):
        """analyze returns dict with 'activations' key."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronActivationsAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert isinstance(result, dict)
        assert "activations" in result

    def test_activations_shape(self, model_with_cache):
        """Activations have shape (d_mlp, p, p)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronActivationsAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)

        p = int(np.sqrt(dataset.shape[0]))
        d_mlp = model.cfg.d_mlp
        assert result["activations"].shape == (d_mlp, p, p)

    def test_activations_is_numpy_array(self, model_with_cache):
        """Activations is a numpy array."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronActivationsAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert isinstance(result["activations"], np.ndarray)


class TestNeuronFreqClustersAnalyzerProtocol:
    """Tests for protocol conformance."""

    def test_conforms_to_analyzer_protocol(self):
        """NeuronFreqClustersAnalyzer implements Analyzer protocol."""
        analyzer = NeuronFreqClustersAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_has_correct_name(self):
        """Analyzer has correct name property."""
        analyzer = NeuronFreqClustersAnalyzer()
        assert analyzer.name == "neuron_freq_norm"


class TestNeuronFreqClustersAnalyzerOutput:
    """Tests for analyzer output."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_spec(self, temp_dir):
        spec = ModuloAdditionSpecification(
            model_dir=temp_dir,
            prime=17,
            device="cpu",
            seed=42,
        )
        spec.train(num_epochs=50, checkpoint_epochs=[0, 25, 49])
        return spec

    @pytest.fixture
    def model_with_cache(self, model_spec):
        model_spec.generate_training_data()
        model = model_spec.load_from_file()
        _, cache = model.run_with_cache(model_spec.dataset)
        fourier_basis, _ = get_fourier_bases(model_spec.prime, model_spec.device)
        return model, model_spec.dataset, cache, fourier_basis

    def test_returns_dict_with_norm_matrix(self, model_with_cache):
        """analyze returns dict with 'norm_matrix' key."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronFreqClustersAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert isinstance(result, dict)
        assert "norm_matrix" in result

    def test_norm_matrix_shape(self, model_with_cache):
        """Norm matrix has shape (n_frequencies, d_mlp)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronFreqClustersAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)

        p = int(np.sqrt(dataset.shape[0]))
        n_frequencies = p // 2
        d_mlp = model.cfg.d_mlp
        assert result["norm_matrix"].shape == (n_frequencies, d_mlp)

    def test_norm_matrix_is_numpy_array(self, model_with_cache):
        """Norm matrix is a numpy array."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronFreqClustersAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert isinstance(result["norm_matrix"], np.ndarray)

    def test_values_are_non_negative(self, model_with_cache):
        """All values are >= 0 (they are fractions)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronFreqClustersAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        assert np.all(result["norm_matrix"] >= 0)

    def test_values_are_bounded(self, model_with_cache):
        """All values are <= 1 (they are fractions)."""
        model, dataset, cache, fourier_basis = model_with_cache
        analyzer = NeuronFreqClustersAnalyzer()
        result = analyzer.analyze(model, dataset, cache, fourier_basis)
        # Allow small numerical tolerance
        assert np.all(result["norm_matrix"] <= 1.01)


class TestRemainingAnalyzersIntegration:
    """Integration tests with AnalysisPipeline."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_spec(self, temp_dir):
        spec = ModuloAdditionSpecification(
            model_dir=temp_dir,
            prime=17,
            device="cpu",
            seed=42,
        )
        spec.train(num_epochs=50, checkpoint_epochs=[0, 25, 49])
        return spec

    def test_neuron_activations_creates_artifact(self, model_spec):
        """Pipeline creates neuron_activations.npz artifact."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(NeuronActivationsAnalyzer())
        pipeline.run()

        artifact_path = os.path.join(model_spec.artifacts_dir, "neuron_activations.npz")
        assert os.path.exists(artifact_path)

    def test_neuron_freq_norm_creates_artifact(self, model_spec):
        """Pipeline creates neuron_freq_norm.npz artifact."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(NeuronFreqClustersAnalyzer())
        pipeline.run()

        artifact_path = os.path.join(model_spec.artifacts_dir, "neuron_freq_norm.npz")
        assert os.path.exists(artifact_path)

    def test_multiple_analyzers_run_together(self, model_spec):
        """Can register and run multiple analyzers."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(NeuronActivationsAnalyzer())
        pipeline.register(NeuronFreqClustersAnalyzer())
        pipeline.run()

        assert os.path.exists(os.path.join(model_spec.artifacts_dir, "neuron_activations.npz"))
        assert os.path.exists(os.path.join(model_spec.artifacts_dir, "neuron_freq_norm.npz"))

    def test_artifact_shapes_are_correct(self, model_spec):
        """Artifact shapes match expectations."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(NeuronActivationsAnalyzer())
        pipeline.register(NeuronFreqClustersAnalyzer())
        pipeline.run()

        n_epochs = len(model_spec.get_available_checkpoints())
        p = model_spec.prime
        d_mlp = 512  # From model config

        activations = pipeline.load_artifact("neuron_activations")
        assert activations["activations"].shape == (n_epochs, d_mlp, p, p)

        freq_norm = pipeline.load_artifact("neuron_freq_norm")
        assert freq_norm["norm_matrix"].shape == (n_epochs, p // 2, d_mlp)
