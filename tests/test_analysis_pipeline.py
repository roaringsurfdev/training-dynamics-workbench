"""Tests for REQ_003_001: Core Infrastructure."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from analysis import AnalysisPipeline, Analyzer
from ModuloAdditionSpecification import ModuloAdditionSpecification


class MockAnalyzer:
    """Mock analyzer for testing pipeline mechanics."""

    def __init__(self, name: str = "mock"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def analyze(self, model, dataset, cache, fourier_basis) -> dict[str, np.ndarray]:
        return {"data": np.ones((10,), dtype=np.float32)}


class TestAnalyzerProtocol:
    """Tests for Analyzer protocol."""

    def test_mock_analyzer_conforms_to_protocol(self):
        """MockAnalyzer implements Analyzer protocol."""
        analyzer = MockAnalyzer()
        assert isinstance(analyzer, Analyzer)

    def test_protocol_requires_name_property(self):
        """Analyzer requires name property."""
        analyzer = MockAnalyzer("test_name")
        assert analyzer.name == "test_name"

    def test_protocol_requires_analyze_method(self):
        """Analyzer requires analyze method."""
        analyzer = MockAnalyzer()
        assert callable(analyzer.analyze)


class TestAnalysisPipelineInstantiation:
    """Tests for AnalysisPipeline instantiation."""

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

    def test_pipeline_instantiation(self, model_spec):
        """Can instantiate AnalysisPipeline with model_spec."""
        pipeline = AnalysisPipeline(model_spec)
        assert pipeline is not None
        assert pipeline.model_spec is model_spec

    def test_pipeline_creates_artifacts_directory(self, model_spec):
        """Pipeline ensures artifacts directory exists."""
        pipeline = AnalysisPipeline(model_spec)
        assert os.path.exists(pipeline.artifacts_dir)

    def test_pipeline_register_returns_self(self, model_spec):
        """register() returns self for chaining."""
        pipeline = AnalysisPipeline(model_spec)
        result = pipeline.register(MockAnalyzer())
        assert result is pipeline

    def test_pipeline_register_chaining(self, model_spec):
        """Can chain multiple register calls."""
        pipeline = AnalysisPipeline(model_spec)
        result = (
            pipeline
            .register(MockAnalyzer("a"))
            .register(MockAnalyzer("b"))
        )
        assert result is pipeline
        assert len(pipeline._analyzers) == 2


class TestAnalysisPipelineRun:
    """Tests for AnalysisPipeline.run()."""

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

    def test_run_with_no_analyzers_does_nothing(self, model_spec):
        """Running with no analyzers doesn't crash."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.run()

    def test_run_creates_manifest(self, model_spec):
        """Running pipeline creates manifest.json."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        manifest_path = os.path.join(model_spec.artifacts_dir, "manifest.json")
        assert os.path.exists(manifest_path)

    def test_run_saves_artifacts(self, model_spec):
        """Running pipeline saves artifact files."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer("test_analyzer"))
        pipeline.run()

        artifact_path = os.path.join(model_spec.artifacts_dir, "test_analyzer.npz")
        assert os.path.exists(artifact_path)

    def test_run_processes_all_checkpoints(self, model_spec):
        """Pipeline processes all available checkpoints."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        completed = pipeline.get_completed_epochs("mock")
        expected = model_spec.get_available_checkpoints()
        assert completed == expected

    def test_run_with_specific_epochs(self, model_spec):
        """Can specify subset of epochs to process."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer())
        pipeline.run(epochs=[0, 25])

        completed = pipeline.get_completed_epochs("mock")
        assert completed == [0, 25]

    def test_manifest_structure(self, model_spec):
        """Manifest has correct structure."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer("test"))
        pipeline.run()

        manifest_path = os.path.join(model_spec.artifacts_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "analyzers" in manifest
        assert "test" in manifest["analyzers"]
        assert "epochs_completed" in manifest["analyzers"]["test"]
        assert "model_config" in manifest
        assert manifest["model_config"]["prime"] == 17


class TestAnalysisPipelineResumability:
    """Tests for pipeline resumability."""

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

    def test_skips_completed_epochs(self, model_spec):
        """Pipeline skips already-completed epochs."""
        pipeline1 = AnalysisPipeline(model_spec)
        pipeline1.register(MockAnalyzer())
        pipeline1.run(epochs=[0, 25])

        pipeline2 = AnalysisPipeline(model_spec)
        pipeline2.register(MockAnalyzer())

        completed_before = pipeline2.get_completed_epochs("mock")
        assert completed_before == [0, 25]

        pipeline2.run()

        completed_after = pipeline2.get_completed_epochs("mock")
        assert completed_after == [0, 25, 49]

    def test_force_recomputes_all(self, model_spec):
        """force=True recomputes even completed epochs."""

        class CountingAnalyzer:
            def __init__(self):
                self.call_count = 0

            @property
            def name(self):
                return "counting"

            def analyze(self, model, dataset, cache, fourier_basis):
                self.call_count += 1
                return {"data": np.ones((5,))}

        pipeline1 = AnalysisPipeline(model_spec)
        analyzer1 = CountingAnalyzer()
        pipeline1.register(analyzer1)
        pipeline1.run()
        first_count = analyzer1.call_count

        pipeline2 = AnalysisPipeline(model_spec)
        analyzer2 = CountingAnalyzer()
        pipeline2.register(analyzer2)
        pipeline2.run(force=True)

        assert analyzer2.call_count == first_count

    def test_manifest_persists_between_sessions(self, model_spec):
        """Manifest is loaded correctly in new pipeline instance."""
        pipeline1 = AnalysisPipeline(model_spec)
        pipeline1.register(MockAnalyzer())
        pipeline1.run(epochs=[0])

        del pipeline1

        pipeline2 = AnalysisPipeline(model_spec)
        completed = pipeline2.get_completed_epochs("mock")
        assert completed == [0]


class TestAnalysisPipelineArtifactLoading:
    """Tests for artifact loading."""

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

    def test_load_artifact_returns_dict(self, model_spec):
        """load_artifact returns dict with epochs and data."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("mock")
        assert isinstance(artifact, dict)
        assert "epochs" in artifact
        assert "data" in artifact

    def test_load_artifact_epochs_array(self, model_spec):
        """Artifact contains correct epochs array."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("mock")
        expected_epochs = np.array([0, 25, 49])
        np.testing.assert_array_equal(artifact["epochs"], expected_epochs)

    def test_load_artifact_data_shape(self, model_spec):
        """Artifact data has correct shape (n_epochs, ...)."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer())
        pipeline.run()

        artifact = pipeline.load_artifact("mock")
        assert artifact["data"].shape[0] == 3
        assert artifact["data"].shape[1] == 10

    def test_load_artifact_not_found_raises(self, model_spec):
        """load_artifact raises FileNotFoundError for missing analyzer."""
        pipeline = AnalysisPipeline(model_spec)

        with pytest.raises(FileNotFoundError):
            pipeline.load_artifact("nonexistent")


class TestAnalysisPipelineMultipleAnalyzers:
    """Tests for multiple analyzers."""

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

    def test_multiple_analyzers_produce_separate_artifacts(self, model_spec):
        """Each analyzer creates its own artifact file."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer("analyzer_a"))
        pipeline.register(MockAnalyzer("analyzer_b"))
        pipeline.run()

        assert os.path.exists(os.path.join(model_spec.artifacts_dir, "analyzer_a.npz"))
        assert os.path.exists(os.path.join(model_spec.artifacts_dir, "analyzer_b.npz"))

    def test_manifest_tracks_all_analyzers(self, model_spec):
        """Manifest tracks completion for all analyzers."""
        pipeline = AnalysisPipeline(model_spec)
        pipeline.register(MockAnalyzer("a"))
        pipeline.register(MockAnalyzer("b"))
        pipeline.run()

        manifest_path = os.path.join(model_spec.artifacts_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "a" in manifest["analyzers"]
        assert "b" in manifest["analyzers"]
