"""Protocol definitions for analysis modules."""

from typing import Protocol, runtime_checkable

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


@runtime_checkable
class Analyzer(Protocol):
    """Protocol defining the interface for all analyzers.

    Analyzers compute analysis on a single checkpoint and return
    artifact-ready numpy arrays.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer (used in artifact naming)."""
        ...

    def analyze(
        self,
        model: HookedTransformer,
        dataset: torch.Tensor,
        cache: ActivationCache,
        fourier_basis: torch.Tensor,
    ) -> dict[str, np.ndarray]:
        """
        Run analysis on a single checkpoint.

        Args:
            model: The model loaded with checkpoint weights
            dataset: Full dataset tensor (p^2, 3) of [a, b, =] inputs
            cache: Activation cache from forward pass
            fourier_basis: Precomputed Fourier basis (n_components, p)

        Returns:
            Dict mapping artifact keys to numpy arrays.
            Keys become field names in the saved .npz file.
        """
        ...
