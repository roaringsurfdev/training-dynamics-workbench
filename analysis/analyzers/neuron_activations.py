"""Neuron Activations Analyzer.

Extracts MLP neuron activations reshaped to input space.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from analysis.library import (
    compute_grid_size_from_dataset,
    extract_mlp_activations,
    reshape_to_grid,
)


class NeuronActivationsAnalyzer:
    """Extracts MLP neuron activations reshaped to input space.

    For each checkpoint, extracts activations from the last token position
    and reshapes them to (d_mlp, p, p) for visualization as heatmaps.
    """

    name = "neuron_activations"
    description = "Computes neuron activation heatmaps for (a, b) inputs"

    def analyze(
        self,
        model: HookedTransformer,
        dataset: torch.Tensor,
        cache: ActivationCache,
        fourier_basis: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Extract neuron activations and reshape to (d_mlp, p, p).

        Args:
            model: The model loaded with checkpoint weights
            dataset: Full dataset tensor (p^2, 3)
            cache: Activation cache from forward pass
            fourier_basis: Precomputed Fourier basis (not used by this analyzer)

        Returns:
            Dict with 'activations' array of shape (d_mlp, p, p)
        """
        # Get grid size from dataset
        p = compute_grid_size_from_dataset(dataset)

        # Extract neuron activations at last token position
        neuron_acts = extract_mlp_activations(cache, layer=0, position=-1)

        # Reshape to (d_mlp, p, p)
        activations = reshape_to_grid(neuron_acts, p)

        return {"activations": activations.detach().cpu().numpy()}
