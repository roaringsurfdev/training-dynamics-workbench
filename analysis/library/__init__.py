"""Generic, reusable analysis functions.

This module contains library functions that can be used by any analyzer
and are not family-specific. They provide building blocks for analysis.

Modules:
- fourier: Fourier basis generation and projection for modular arithmetic
- activations: Activation extraction and manipulation utilities
"""

from analysis.library.activations import (
    compute_grid_size_from_dataset,
    extract_mlp_activations,
    get_embedding_weights,
    reshape_to_grid,
    run_with_cache,
)
from analysis.library.fourier import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    get_dominant_frequency_indices,
    get_fourier_basis,
    project_onto_fourier_basis,
)

__all__ = [
    # Fourier functions
    "get_fourier_basis",
    "project_onto_fourier_basis",
    "compute_2d_fourier_transform",
    "get_dominant_frequency_indices",
    "compute_frequency_variance_fractions",
    # Activation functions
    "extract_mlp_activations",
    "reshape_to_grid",
    "get_embedding_weights",
    "run_with_cache",
    "compute_grid_size_from_dataset",
]
