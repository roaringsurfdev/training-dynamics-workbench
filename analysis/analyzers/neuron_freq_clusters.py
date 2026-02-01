"""Neuron Frequency Clusters Analyzer.

Computes neuron-frequency specialization matrix.
"""

import einops
import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


class NeuronFreqClustersAnalyzer:
    """Computes neuron-frequency specialization matrix.

    For each checkpoint, computes what fraction of each neuron's variance
    is explained by each frequency, enabling clustering of neurons by
    their frequency specialization.
    """

    @property
    def name(self) -> str:
        return "neuron_freq_norm"

    def analyze(
        self,
        model: HookedTransformer,
        dataset: torch.Tensor,
        cache: ActivationCache,
        fourier_basis: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Compute fraction of variance explained by each frequency for each neuron.

        Args:
            model: The model loaded with checkpoint weights
            dataset: Full dataset tensor (p^2, 3)
            cache: Activation cache from forward pass
            fourier_basis: Precomputed Fourier basis (n_components, p)

        Returns:
            Dict with 'norm_matrix' array of shape (n_frequencies, d_mlp)
            where n_frequencies = p // 2
        """
        # Get dimensions
        p = int(np.sqrt(dataset.shape[0]))
        d_mlp = model.cfg.d_mlp

        # Extract neuron activations at last token position
        # Shape: (p^2, d_mlp)
        neuron_acts = cache["post", 0, "mlp"][:, -1, :]

        # Reshape to (d_mlp, p, p) and compute 2D Fourier transform
        # fourier_neuron_acts shape: (d_mlp, n_components, n_components)
        reshaped = einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=p, b=p)
        fourier_neuron_acts = fourier_basis @ reshaped @ fourier_basis.T

        # Center by removing DC component
        fourier_neuron_acts[:, 0, 0] = 0.0

        # Compute variance explained by each frequency
        # For frequency k, the relevant indices are:
        #   - (0, 2k-1), (0, 2k): vertical stripes
        #   - (2k-1, 0), (2k, 0): horizontal stripes
        #   - (2k-1, 2k-1), (2k-1, 2k), (2k, 2k-1), (2k, 2k): diagonal patterns
        # Simplified: sum over (0, 2k-1), (0, 2k), (2k-1, 0), (2k, 0) and diagonals
        n_frequencies = p // 2
        neuron_freq_norm = torch.zeros(n_frequencies, d_mlp, device=dataset.device)

        for freq in range(n_frequencies):
            # Indices for sin and cos of this frequency
            # freq=0 corresponds to frequency 1 (indices 1, 2)
            # freq=k corresponds to frequency k+1 (indices 2k+1, 2k+2)
            for x in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
                for y in [0, 2 * (freq + 1) - 1, 2 * (freq + 1)]:
                    if x < fourier_neuron_acts.shape[1] and y < fourier_neuron_acts.shape[2]:
                        neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y] ** 2

        # Normalize by total variance
        total_variance = fourier_neuron_acts.pow(2).sum(dim=[-1, -2])
        # Avoid division by zero
        total_variance = torch.clamp(total_variance, min=1e-10)
        neuron_freq_norm = neuron_freq_norm / total_variance[None, :]

        return {"norm_matrix": neuron_freq_norm.detach().cpu().numpy()}
