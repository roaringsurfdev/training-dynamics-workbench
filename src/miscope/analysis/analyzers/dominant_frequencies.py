"""Dominant Frequencies Analyzer.

Computes Fourier coefficient norms for embedding weights.
"""

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from miscope.analysis.library import get_embedding_weights, project_onto_fourier_basis


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
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Compute Fourier coefficient norms for embedding weights.

        Args:
            model: The model loaded with checkpoint weights
            probe: The analysis dataset (not used by this analyzer)
            cache: Activation cache (not used by this analyzer)
            context: Analysis context containing 'fourier_basis'

        Returns:
            Dict with 'coefficients' array of shape (n_fourier_components,)
        """
        fourier_basis = context["fourier_basis"]

        # Get embedding weights, excluding the equals token
        W_E = get_embedding_weights(model, exclude_special_tokens=1)

        # Compute norms of embedding projected onto Fourier basis
        coefficients = project_onto_fourier_basis(W_E, fourier_basis)

        return {"coefficients": coefficients.detach().cpu().numpy()}
