"""Visualization package for rendering analysis artifacts.

This package provides Plotly-based renderers for analysis artifacts
created by the analysis pipeline. Each renderer is Gradio-agnostic
and simply returns a plotly.graph_objects.Figure.

Usage:
    from analysis import ArtifactLoader
    from visualization import render_dominant_frequencies

    loader = ArtifactLoader(artifacts_dir)
    artifact = loader.load("dominant_frequencies")
    fig = render_dominant_frequencies(artifact, epoch_idx=0)
    fig.show()  # or pass to Gradio
"""

from visualization.line_plot import line
from visualization.renderers.dominant_frequencies import (
    get_dominant_indices,
    get_fourier_basis_names,
    render_dominant_frequencies,
    render_dominant_frequencies_over_time,
)
from visualization.renderers.neuron_activations import (
    get_most_active_neurons,
    render_neuron_across_epochs,
    render_neuron_grid,
    render_neuron_heatmap,
)
from visualization.renderers.neuron_freq_clusters import (
    get_neuron_specialization,
    get_specialized_neurons,
    render_freq_clusters,
    render_freq_clusters_comparison,
)

__all__ = [
    # REQ_012: Line plot utility (replaces neel-plotly)
    "line",
    # REQ_004: Dominant frequencies
    "render_dominant_frequencies",
    "render_dominant_frequencies_over_time",
    "get_dominant_indices",
    "get_fourier_basis_names",
    # REQ_005: Neuron activations
    "render_neuron_heatmap",
    "render_neuron_grid",
    "render_neuron_across_epochs",
    "get_most_active_neurons",
    # REQ_006: Frequency clusters
    "render_freq_clusters",
    "render_freq_clusters_comparison",
    "get_specialized_neurons",
    "get_neuron_specialization",
]
