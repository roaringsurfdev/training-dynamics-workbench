"""Generic, reusable analysis functions.

This module contains library functions that can be used by any analyzer
and are not family-specific. They provide building blocks for analysis.

Modules:
- fourier: Fourier basis generation and projection for modular arithmetic
- activations: Activation extraction and manipulation utilities
- weights: Weight matrix extraction utilities
- trajectory: PCA projection and velocity computation
"""

from analysis.library.activations import (
    compute_grid_size_from_dataset,
    extract_attention_patterns,
    extract_mlp_activations,
    get_embedding_weights,
    reshape_to_grid,
    run_with_cache,
)
from analysis.library.fourier import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_neuron_coarseness,
    get_dominant_frequency_indices,
    get_fourier_basis,
    project_onto_fourier_basis,
)
from analysis.library.landscape import compute_landscape_flatness
from analysis.library.trajectory import (
    compute_parameter_velocity,
    compute_pca_trajectory,
    flatten_snapshot,
)
from analysis.library.weights import (
    COMPONENT_GROUPS,
    WEIGHT_MATRIX_NAMES,
    compute_participation_ratio,
    compute_weight_singular_values,
    extract_parameter_snapshot,
)

__all__ = [
    # Fourier functions
    "get_fourier_basis",
    "project_onto_fourier_basis",
    "compute_2d_fourier_transform",
    "get_dominant_frequency_indices",
    "compute_frequency_variance_fractions",
    "compute_neuron_coarseness",
    # Activation functions
    "extract_attention_patterns",
    "extract_mlp_activations",
    "reshape_to_grid",
    "get_embedding_weights",
    "run_with_cache",
    "compute_grid_size_from_dataset",
    # Weight matrix functions
    "extract_parameter_snapshot",
    "compute_weight_singular_values",
    "compute_participation_ratio",
    "WEIGHT_MATRIX_NAMES",
    "COMPONENT_GROUPS",
    # Trajectory functions
    "flatten_snapshot",
    "compute_pca_trajectory",
    "compute_parameter_velocity",
    # Landscape flatness functions
    "compute_landscape_flatness",
]
