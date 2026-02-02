"""Tests for REQ_003_004: Artifact Loader."""

import json
import os
import tempfile

import numpy as np
import pytest

from analysis import AnalysisPipeline, ArtifactLoader
from analysis.analyzers import DominantFrequenciesAnalyzer
from ModuloAdditionSpecification import ModuloAdditionSpecification


class TestArtifactLoaderInstantiation:
    """Tests for ArtifactLoader instantiation."""

    def test_instantiate_with_directory(self):
        """Can instantiate with artifacts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ArtifactLoader(tmpdir)
            assert loader is not None
            assert loader.artifacts_dir == tmpdir

    def test_instantiate_with_nonexistent_directory(self):
        """Can instantiate even if directory doesn't exist."""
        # This should not raise - directory is checked at load time
        loader = ArtifactLoader("/nonexistent/path")
        assert loader is not None


class TestArtifactLoaderWithEmptyDirectory:
    """Tests with no artifacts."""

    @pytest.fixture
    def empty_artifacts_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    def test_get_available_analyzers_empty(self, empty_artifacts_dir):
        """Returns empty list when no manifest exists."""
        loader = ArtifactLoader(empty_artifacts_dir)
        assert loader.get_available_analyzers() == []

    def test_load_nonexistent_raises(self, empty_artifacts_dir):
        """Raises FileNotFoundError for missing artifact."""
        loader = ArtifactLoader(empty_artifacts_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load("nonexistent")
        assert "nonexistent" in str(exc_info.value)

    def test_get_metadata_raises_for_missing(self, empty_artifacts_dir):
        """Raises KeyError for missing analyzer in manifest."""
        loader = ArtifactLoader(empty_artifacts_dir)
        with pytest.raises(KeyError) as exc_info:
            loader.get_metadata("nonexistent")
        assert "nonexistent" in str(exc_info.value)


class TestArtifactLoaderWithPipeline:
    """Tests with artifacts created by pipeline."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def model_spec_with_artifacts(self, temp_dir):
        """Create model spec, run pipeline, return spec."""
        spec = ModuloAdditionSpecification(
            model_dir=temp_dir,
            prime=17,
            device="cpu",
            seed=42,
        )
        spec.train(num_epochs=50, checkpoint_epochs=[0, 25, 49])

        # Run pipeline to create artifacts
        pipeline = AnalysisPipeline(spec)
        pipeline.register(DominantFrequenciesAnalyzer())
        pipeline.run()

        return spec

    def test_load_artifact(self, model_spec_with_artifacts):
        """Can load artifact using just artifacts directory."""
        loader = ArtifactLoader(model_spec_with_artifacts.artifacts_dir)
        artifact = loader.load("dominant_frequencies")

        assert isinstance(artifact, dict)
        assert "epochs" in artifact
        assert "coefficients" in artifact

    def test_get_available_analyzers(self, model_spec_with_artifacts):
        """Lists available analyzers."""
        loader = ArtifactLoader(model_spec_with_artifacts.artifacts_dir)
        available = loader.get_available_analyzers()

        assert "dominant_frequencies" in available

    def test_get_epochs(self, model_spec_with_artifacts):
        """Gets epochs for analyzer."""
        loader = ArtifactLoader(model_spec_with_artifacts.artifacts_dir)
        epochs = loader.get_epochs("dominant_frequencies")

        assert epochs == [0, 25, 49]

    def test_get_metadata(self, model_spec_with_artifacts):
        """Gets metadata for analyzer."""
        loader = ArtifactLoader(model_spec_with_artifacts.artifacts_dir)
        metadata = loader.get_metadata("dominant_frequencies")

        assert "epochs_completed" in metadata
        assert "shapes" in metadata
        assert "dtypes" in metadata

    def test_get_model_config(self, model_spec_with_artifacts):
        """Gets model config from manifest."""
        loader = ArtifactLoader(model_spec_with_artifacts.artifacts_dir)
        config = loader.get_model_config()

        assert config["prime"] == 17
        assert config["seed"] == 42


class TestArtifactLoaderIndependence:
    """Tests verifying loader works without pipeline."""

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def artifacts_dir_only(self, temp_dir):
        """Create artifacts manually without using pipeline."""
        artifacts_dir = os.path.join(temp_dir, "artifacts")
        os.makedirs(artifacts_dir)

        # Save test artifact
        epochs = np.array([0, 100, 200])
        data = np.random.randn(3, 10)
        np.savez_compressed(
            os.path.join(artifacts_dir, "test_analyzer.npz"),
            epochs=epochs,
            data=data,
        )

        # Save manifest
        manifest = {
            "analyzers": {
                "test_analyzer": {
                    "epochs_completed": [0, 100, 200],
                    "shapes": {"epochs": [3], "data": [3, 10]},
                    "dtypes": {"epochs": "int64", "data": "float64"},
                }
            },
            "model_config": {"prime": 113, "seed": 999},
        }
        with open(os.path.join(artifacts_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

        return artifacts_dir

    def test_load_without_pipeline(self, artifacts_dir_only):
        """Can load artifacts without any pipeline involvement."""
        loader = ArtifactLoader(artifacts_dir_only)
        artifact = loader.load("test_analyzer")

        assert "epochs" in artifact
        assert "data" in artifact
        np.testing.assert_array_equal(artifact["epochs"], [0, 100, 200])

    def test_metadata_without_pipeline(self, artifacts_dir_only):
        """Can read metadata without pipeline."""
        loader = ArtifactLoader(artifacts_dir_only)
        metadata = loader.get_metadata("test_analyzer")

        assert metadata["epochs_completed"] == [0, 100, 200]

    def test_model_config_without_pipeline(self, artifacts_dir_only):
        """Can read model config without pipeline."""
        loader = ArtifactLoader(artifacts_dir_only)
        config = loader.get_model_config()

        assert config["prime"] == 113


class TestArtifactLoaderManifestCaching:
    """Tests for manifest caching."""

    @pytest.fixture
    def artifacts_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = {"analyzers": {"test": {}}, "model_config": {}}
            with open(os.path.join(tmpdir, "manifest.json"), "w") as f:
                json.dump(manifest, f)
            yield tmpdir

    def test_manifest_cached(self, artifacts_dir):
        """Manifest is cached after first access."""
        loader = ArtifactLoader(artifacts_dir)

        # Access manifest twice
        _ = loader.manifest
        _ = loader.manifest

        # Should be the same object
        assert loader._manifest is not None
