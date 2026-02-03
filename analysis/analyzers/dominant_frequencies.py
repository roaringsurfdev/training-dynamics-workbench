"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from analysis.library import get_embedding_weights, project_onto_fourier_basis


class DominantFrequenciesAnalyzer:
    """Computes Fourier coefficient norms for embedding weights.

    For each checkpoint, computes (fourier_basis @ W_E).norm(dim=-1),
    identifying which frequencies dominate the learned embedding representation.
    """

    name = "dominant_frequencies"
    description = "Identifies dominant frequencies in learned embeddings"

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
        W_E = get_embedding_weights(model, exclude_special_tokens=1)

        # Compute norms of embedding projected onto Fourier basis
        coefficients = project_onto_fourier_basis(W_E, fourier_basis)

        return {"coefficients": coefficients.detach().cpu().numpy()}
