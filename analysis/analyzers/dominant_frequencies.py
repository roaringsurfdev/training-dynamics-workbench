"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


class DominantFrequenciesAnalyzer:
    """Computes Fourier coefficient norms for embedding weights.

    For each checkpoint, computes (fourier_basis @ W_E).norm(dim=-1),
    identifying which frequencies dominate the learned embedding representation.
    """

    @property
    def name(self) -> str:
        return "dominant_frequencies"

    def analyze(
        self,
        model: HookedTransformer,
        dataset: torch.Tensor,
        cache: ActivationCache,
        fourier_basis: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Compute Fourier coefficient norms for embedding weights.

        Args:
            model: The model loaded with checkpoint weights
            dataset: Full dataset tensor (not used by this analyzer)
            cache: Activation cache (not used by this analyzer)
            fourier_basis: Precomputed Fourier basis (n_components, p)

        Returns:
            Dict with 'coefficients' array of shape (n_fourier_components,)
        """
        # Get embedding weights, excluding the equals token
        W_E = model.embed.W_E[:-1]

        # Compute norms of embedding projected onto Fourier basis
        # fourier_basis @ W_E gives (n_components, d_model)
        # .norm(dim=-1) gives (n_components,)
        coefficients = (fourier_basis @ W_E).norm(dim=-1)

        return {"coefficients": coefficients.detach().cpu().numpy()}
