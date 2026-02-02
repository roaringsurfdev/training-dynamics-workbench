"""Tests for REQ_004, REQ_005, REQ_006: Visualization Renderers."""
# pyright: reportArgumentType=false
# pyright: reportAttributeAccessIssue=false

import numpy as np
import plotly.graph_objects as go
import pytest


class TestDominantFrequenciesRenderer:
    """Tests for REQ_004: Dominant Frequencies Visualization."""

    @pytest.fixture
    def sample_artifact(self):
        """Create sample dominant frequencies artifact."""
        return {
            "epochs": np.array([0, 50, 100]),
            "coefficients": np.array(
                [
                    [0.5, 1.2, 0.3, 1.5, 0.2, 0.8, 0.1],  # epoch 0
                    [0.6, 1.5, 0.4, 2.0, 0.3, 1.0, 0.2],  # epoch 50
                    [0.7, 1.8, 0.5, 2.5, 0.4, 1.2, 0.3],  # epoch 100
                ]
            ),
        }

    def test_render_returns_figure(self, sample_artifact):
        """Renderer returns a Plotly Figure."""
        from visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(sample_artifact, epoch_idx=0)

        assert isinstance(fig, go.Figure)

    def test_render_with_threshold(self, sample_artifact):
        """Threshold parameter affects visualization."""
        from visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(sample_artifact, epoch_idx=0, threshold=1.0)

        assert isinstance(fig, go.Figure)
        # Figure has horizontal line for threshold
        assert any(
            shape.get("type") == "line"
            for shape in fig.to_dict().get("layout", {}).get("shapes", [])
        )

    def test_render_without_highlight(self, sample_artifact):
        """Can disable dominance highlighting."""
        from visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(sample_artifact, epoch_idx=0, highlight_dominant=False)

        assert isinstance(fig, go.Figure)

    def test_render_custom_title(self, sample_artifact):
        """Can set custom title."""
        from visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(sample_artifact, epoch_idx=0, title="Custom Title")

        assert fig.layout.title.text == "Custom Title"

    def test_render_epoch_in_default_title(self, sample_artifact):
        """Default title includes epoch number."""
        from visualization import render_dominant_frequencies

        fig = render_dominant_frequencies(sample_artifact, epoch_idx=2)

        assert "100" in fig.layout.title.text  # epoch 100

    def test_invalid_epoch_idx_raises(self, sample_artifact):
        """Invalid epoch index raises IndexError."""
        from visualization import render_dominant_frequencies

        with pytest.raises(IndexError):
            render_dominant_frequencies(sample_artifact, epoch_idx=5)

    def test_get_dominant_indices(self, sample_artifact):
        """get_dominant_indices returns indices above threshold."""
        from visualization import get_dominant_indices

        indices = get_dominant_indices(sample_artifact["coefficients"][0], threshold=1.0)

        assert 1 in indices  # 1.2 > 1.0
        assert 3 in indices  # 1.5 > 1.0
        assert 0 not in indices  # 0.5 < 1.0

    def test_get_fourier_basis_names(self):
        """Fourier basis names are correctly generated."""
        from visualization import get_fourier_basis_names

        names = get_fourier_basis_names(7)

        assert names[0] == "Const"
        assert names[1] == "cos 1"
        assert names[2] == "sin 1"
        assert names[3] == "cos 2"

    def test_render_over_time(self, sample_artifact):
        """render_dominant_frequencies_over_time works."""
        from visualization import render_dominant_frequencies_over_time

        fig = render_dominant_frequencies_over_time(sample_artifact, top_k=3)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 traces for top 3 components


class TestNeuronActivationsRenderer:
    """Tests for REQ_005: Neuron Activation Heatmaps."""

    @pytest.fixture
    def sample_artifact(self):
        """Create sample neuron activations artifact."""
        # (n_epochs, d_mlp, p, p)
        np.random.seed(42)
        return {
            "epochs": np.array([0, 50, 100]),
            "activations": np.random.randn(3, 10, 7, 7),  # 10 neurons, p=7
        }

    def test_render_single_heatmap(self, sample_artifact):
        """Single neuron heatmap renders correctly."""
        from visualization import render_neuron_heatmap

        fig = render_neuron_heatmap(sample_artifact, epoch_idx=0, neuron_idx=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_has_correct_shape(self, sample_artifact):
        """Heatmap data has correct dimensions."""
        from visualization import render_neuron_heatmap

        fig = render_neuron_heatmap(sample_artifact, epoch_idx=0, neuron_idx=0)

        # Heatmap z data should be (p, p)
        z_data = fig.data[0].z
        assert z_data.shape == (7, 7)

    def test_heatmap_invalid_epoch(self, sample_artifact):
        """Invalid epoch index raises error."""
        from visualization import render_neuron_heatmap

        with pytest.raises(IndexError):
            render_neuron_heatmap(sample_artifact, epoch_idx=10, neuron_idx=0)

    def test_heatmap_invalid_neuron(self, sample_artifact):
        """Invalid neuron index raises error."""
        from visualization import render_neuron_heatmap

        with pytest.raises(IndexError):
            render_neuron_heatmap(sample_artifact, epoch_idx=0, neuron_idx=100)

    def test_render_neuron_grid(self, sample_artifact):
        """Grid of neurons renders correctly."""
        from visualization import render_neuron_grid

        fig = render_neuron_grid(sample_artifact, epoch_idx=0, neuron_indices=[0, 1, 2, 3, 4])

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5  # 5 neurons

    def test_grid_custom_cols(self, sample_artifact):
        """Grid respects column count."""
        from visualization import render_neuron_grid

        fig = render_neuron_grid(sample_artifact, epoch_idx=0, neuron_indices=[0, 1, 2], cols=2)

        assert isinstance(fig, go.Figure)

    def test_render_neuron_across_epochs(self, sample_artifact):
        """Single neuron across epochs renders correctly."""
        from visualization import render_neuron_across_epochs

        fig = render_neuron_across_epochs(sample_artifact, neuron_idx=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 epochs

    def test_get_most_active_neurons(self, sample_artifact):
        """get_most_active_neurons returns list of indices."""
        from visualization import get_most_active_neurons

        neurons = get_most_active_neurons(sample_artifact, epoch_idx=0, top_k=5)

        assert len(neurons) == 5
        assert all(isinstance(n, int) for n in neurons)
        assert all(0 <= n < 10 for n in neurons)


class TestFreqClustersRenderer:
    """Tests for REQ_006: Neuron Frequency Clusters."""

    @pytest.fixture
    def sample_artifact(self):
        """Create sample frequency clusters artifact."""
        np.random.seed(42)
        # (n_epochs, n_freq, d_mlp)
        # Values should be 0-1 (fraction explained)
        return {
            "epochs": np.array([0, 50, 100]),
            "norm_matrix": np.random.rand(3, 8, 20),  # 8 freqs, 20 neurons
        }

    def test_render_freq_clusters(self, sample_artifact):
        """Frequency clusters heatmap renders correctly."""
        from visualization import render_freq_clusters

        fig = render_freq_clusters(sample_artifact, epoch_idx=0)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        assert isinstance(fig.data[0], go.Heatmap)

    def test_heatmap_has_correct_shape(self, sample_artifact):
        """Heatmap has correct dimensions."""
        from visualization import render_freq_clusters

        fig = render_freq_clusters(sample_artifact, epoch_idx=0)

        z_data = fig.data[0].z
        assert z_data.shape == (8, 20)  # (n_freq, d_mlp)

    def test_sparse_labels(self, sample_artifact):
        """Sparse labels are applied correctly."""
        from visualization import render_freq_clusters

        fig = render_freq_clusters(
            sample_artifact, epoch_idx=0, sparse_labels=True, label_interval=2
        )

        # Check y-axis has sparse tick values
        y_tickvals = fig.layout.yaxis.tickvals
        assert y_tickvals is not None
        assert len(y_tickvals) < 8  # Fewer than all frequencies

    def test_full_labels(self, sample_artifact):
        """Full labels can be shown."""
        from visualization import render_freq_clusters

        fig = render_freq_clusters(sample_artifact, epoch_idx=0, sparse_labels=False)

        y_tickvals = fig.layout.yaxis.tickvals
        assert len(y_tickvals) == 8  # All frequencies

    def test_colorbar_optional(self, sample_artifact):
        """Colorbar can be hidden."""
        from visualization import render_freq_clusters

        fig = render_freq_clusters(sample_artifact, epoch_idx=0, show_colorbar=False)

        assert fig.data[0].showscale is False

    def test_invalid_epoch_raises(self, sample_artifact):
        """Invalid epoch index raises error."""
        from visualization import render_freq_clusters

        with pytest.raises(IndexError):
            render_freq_clusters(sample_artifact, epoch_idx=10)

    def test_render_comparison(self, sample_artifact):
        """Comparison view renders multiple epochs."""
        from visualization import render_freq_clusters_comparison

        fig = render_freq_clusters_comparison(sample_artifact, epoch_indices=[0, 1, 2])

        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # 3 epochs

    def test_get_specialized_neurons(self, sample_artifact):
        """get_specialized_neurons returns correct neurons."""
        from visualization import get_specialized_neurons

        # Set one neuron to be clearly specialized for frequency 1
        sample_artifact["norm_matrix"][0, 0, 5] = 0.95

        neurons = get_specialized_neurons(sample_artifact, epoch_idx=0, frequency=1, threshold=0.85)

        assert 5 in neurons

    def test_get_neuron_specialization(self, sample_artifact):
        """get_neuron_specialization returns dominant frequency."""
        from visualization import get_neuron_specialization

        # Set neuron 3 to be specialized for frequency 5
        sample_artifact["norm_matrix"][0, :, 3] = 0.1
        sample_artifact["norm_matrix"][0, 4, 3] = 0.9

        freq, frac = get_neuron_specialization(sample_artifact, epoch_idx=0, neuron_idx=3)

        assert freq == 5  # 1-indexed (index 4 = freq 5)
        assert frac == pytest.approx(0.9)


class TestVisualizationIntegration:
    """Integration tests using real artifact structure."""

    @pytest.fixture
    def realistic_artifacts(self, tmp_path):
        """Create realistic artifacts similar to pipeline output."""

        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        # Simulate dominant frequencies
        p = 17
        n_epochs = 5
        n_components = 2 * p - 1

        np.savez_compressed(
            artifacts_dir / "dominant_frequencies.npz",
            epochs=np.array([0, 25, 50, 75, 100]),
            coefficients=np.random.rand(n_epochs, n_components),
        )

        # Simulate neuron activations
        d_mlp = 32  # Smaller for tests
        np.savez_compressed(
            artifacts_dir / "neuron_activations.npz",
            epochs=np.array([0, 25, 50, 75, 100]),
            activations=np.random.randn(n_epochs, d_mlp, p, p),
        )

        # Simulate frequency clusters
        np.savez_compressed(
            artifacts_dir / "neuron_freq_norm.npz",
            epochs=np.array([0, 25, 50, 75, 100]),
            norm_matrix=np.random.rand(n_epochs, p // 2, d_mlp),
        )

        return artifacts_dir

    def test_load_and_render_dominant_frequencies(self, realistic_artifacts):
        """Can load and render dominant frequencies."""
        from analysis import ArtifactLoader
        from visualization import render_dominant_frequencies

        loader = ArtifactLoader(str(realistic_artifacts))
        artifact = loader.load("dominant_frequencies")
        fig = render_dominant_frequencies(artifact, epoch_idx=0)

        assert isinstance(fig, go.Figure)

    def test_load_and_render_neuron_heatmap(self, realistic_artifacts):
        """Can load and render neuron heatmap."""
        from analysis import ArtifactLoader
        from visualization import render_neuron_heatmap

        loader = ArtifactLoader(str(realistic_artifacts))
        artifact = loader.load("neuron_activations")
        fig = render_neuron_heatmap(artifact, epoch_idx=0, neuron_idx=0)

        assert isinstance(fig, go.Figure)

    def test_load_and_render_freq_clusters(self, realistic_artifacts):
        """Can load and render frequency clusters."""
        from analysis import ArtifactLoader
        from visualization import render_freq_clusters

        loader = ArtifactLoader(str(realistic_artifacts))
        artifact = loader.load("neuron_freq_norm")
        fig = render_freq_clusters(artifact, epoch_idx=0)

        assert isinstance(fig, go.Figure)

    def test_all_renderers_no_exceptions(self, realistic_artifacts):
        """All renderers work without exceptions on realistic data."""
        from analysis import ArtifactLoader
        from visualization import (
            get_dominant_indices,
            get_most_active_neurons,
            get_neuron_specialization,
            get_specialized_neurons,
            render_dominant_frequencies,
            render_dominant_frequencies_over_time,
            render_freq_clusters,
            render_freq_clusters_comparison,
            render_neuron_across_epochs,
            render_neuron_grid,
            render_neuron_heatmap,
        )

        loader = ArtifactLoader(str(realistic_artifacts))

        # Dominant frequencies
        df_artifact = loader.load("dominant_frequencies")
        render_dominant_frequencies(df_artifact, epoch_idx=0)
        render_dominant_frequencies_over_time(df_artifact)
        get_dominant_indices(df_artifact["coefficients"][0], threshold=0.5)

        # Neuron activations
        na_artifact = loader.load("neuron_activations")
        render_neuron_heatmap(na_artifact, epoch_idx=0, neuron_idx=0)
        render_neuron_grid(na_artifact, epoch_idx=0, neuron_indices=[0, 1, 2])
        render_neuron_across_epochs(na_artifact, neuron_idx=0)
        get_most_active_neurons(na_artifact, epoch_idx=0)

        # Frequency clusters
        fc_artifact = loader.load("neuron_freq_norm")
        render_freq_clusters(fc_artifact, epoch_idx=0)
        render_freq_clusters_comparison(fc_artifact, epoch_indices=[0, 1])
        get_specialized_neurons(fc_artifact, epoch_idx=0, frequency=1, threshold=0.5)
        get_neuron_specialization(fc_artifact, epoch_idx=0, neuron_idx=0)
