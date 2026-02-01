"""Neuron Activations Analyzer.

Extracts MLP neuron activations reshaped to input space.
"""

import einops
import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


class NeuronActivationsAnalyzer:
    """Extracts MLP neuron activations reshaped to input space.

    For each checkpoint, extracts activations from the last token position
    and reshapes them to (d_mlp, p, p) for visualization as heatmaps.
    """

    @property
    def name(self) -> str:
        return "neuron_activations"

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
        # Get prime from dataset shape
        p = int(np.sqrt(dataset.shape[0]))

        # Extract neuron activations at last token position
        # Shape: (p^2, d_mlp)
        neuron_acts = cache["post", 0, "mlp"][:, -1, :]

        # Reshape to (d_mlp, p, p)
        activations = einops.rearrange(
            neuron_acts, "(a b) neuron -> neuron a b", a=p, b=p
        )

        return {"activations": activations.detach().cpu().numpy()}
