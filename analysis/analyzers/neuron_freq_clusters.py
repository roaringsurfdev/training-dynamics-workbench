"""Neuron Frequency Clusters Analyzer.

Computes neuron-frequency specialization matrix.
"""

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
    extract_mlp_activations,
    reshape_to_grid,
)


class NeuronFreqClustersAnalyzer:
    """Computes neuron-frequency specialization matrix.

    For each checkpoint, computes what fraction of each neuron's variance
    is explained by each frequency, enabling clustering of neurons by
    their frequency specialization.
    """

    name = "neuron_freq_norm"
    description = "Computes neuron-frequency specialization for clustering"

    def analyze(
        self,
        model: HookedTransformer,
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Compute fraction of variance explained by each frequency for each neuron.

        Args:
            model: The model loaded with checkpoint weights
            probe: Full probe tensor (p^2, 3)
            cache: Activation cache from forward pass
            context: Analysis context containing 'fourier_basis'

        Returns:
            Dict with 'norm_matrix' array of shape (n_frequencies, d_mlp)
            where n_frequencies = p // 2
        """
        fourier_basis = context["fourier_basis"]

        # Get grid size from probe
        p = compute_grid_size_from_dataset(probe)

        # Extract neuron activations at last token position
        neuron_acts = extract_mlp_activations(cache, layer=0, position=-1)

        # Reshape to (d_mlp, p, p)
        reshaped = reshape_to_grid(neuron_acts, p)

        # Compute 2D Fourier transform
        fourier_neuron_acts = compute_2d_fourier_transform(reshaped, fourier_basis)

        # Compute variance fractions by frequency
        neuron_freq_norm = compute_frequency_variance_fractions(fourier_neuron_acts, p)

        return {"norm_matrix": neuron_freq_norm.detach().cpu().numpy()}
