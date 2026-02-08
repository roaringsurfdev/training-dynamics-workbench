"""Protocol definitions for analysis modules."""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache


@dataclass
class AnalysisRunConfig:
    """Configuration for an analysis run.

    Specifies what work the pipeline should perform. This is variant-agnostic;
    the same config can be applied to multiple variants.

    Attributes:
        analyzers: Which analyzers to run (by name). If empty, uses all
            analyzers registered for the variant's family.
        checkpoints: Which checkpoints to analyze. None means all available.
    """

    analyzers: list[str] = field(default_factory=list)
    """Which analyzers to run (by name). Empty list means all family analyzers."""

    checkpoints: list[int] | None = None
    """Which checkpoints to analyze. None means all available."""


@runtime_checkable
class Analyzer(Protocol):
    """Protocol defining the interface for all analyzers.

    Analyzers compute analysis on a single checkpoint and return
    artifact-ready numpy arrays.

    Analyzers receive a context dict prepared by the ModelFamily, which
    contains domain parameters and any precomputed values (e.g., fourier_basis).
    This allows the pipeline to be family-agnostic while analyzers can access
    the domain-specific values they need.

    Optional Summary Statistics (REQ_022):
        Analyzers may optionally implement two additional methods to produce
        summary statistics — small per-epoch values (scalars or small arrays)
        that are accumulated across checkpoints and saved as a single file.

        - get_summary_keys() -> list[str]:
            Declare the summary statistic keys this analyzer produces.
        - compute_summary(result, context) -> dict[str, float | np.ndarray]:
            Compute summary statistics from this epoch's analysis result.

        These methods are NOT part of the required protocol. The pipeline
        detects them via hasattr() to maintain backward compatibility with
        analyzers that only produce per-epoch artifacts.
    """

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer (used in artifact naming)."""
        ...

    def analyze(
        self,
        model: HookedTransformer,
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """
        Run analysis on a single checkpoint.

        Args:
            model: The model loaded with checkpoint weights
            probe: The analysis dataset tensor (e.g., full (a, b) grid)
            cache: Activation cache from forward pass
            context: Family-provided analysis context containing:
                - 'params': Domain parameter values (e.g., {'prime': 113, 'seed': 42})
                - Family-specific precomputed values (e.g., 'fourier_basis')

        Returns:
            Dict mapping artifact keys to numpy arrays.
            Keys become field names in the saved .npz file.
        """
        ...
