"""Parameter Snapshot Analyzer.

Extracts and stores all trainable weight matrices per checkpoint for
parameter trajectory projection, velocity analysis, and downstream
geometric analyses (REQ_029).
"""

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from analysis.library import extract_parameter_snapshot


class ParameterSnapshotAnalyzer:
    """Stores per-epoch weight matrix snapshots for trajectory analysis.

    Unlike other analyzers, this does not use the forward pass, probe,
    or activation cache. It extracts weights directly from the model.
    The probe and cache arguments are accepted (to conform to the
    Analyzer protocol) but ignored.
    """

    name = "parameter_snapshot"
    description = "Stores weight matrix snapshots for trajectory analysis"

    def analyze(
        self,
        model: HookedTransformer,
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Extract all trainable weight matrices from the model.

        Args:
            model: The model loaded with checkpoint weights
            probe: Unused (protocol conformance)
            cache: Unused (protocol conformance)
            context: Unused (protocol conformance)

        Returns:
            Dict mapping weight matrix names to numpy arrays in
            their original shapes (e.g., W_E, W_Q, W_in, etc.)
        """
        return extract_parameter_snapshot(model)
