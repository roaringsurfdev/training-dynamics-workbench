"""State management for the dashboard."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DashboardState:
    """Centralized state for dashboard synchronization.

    This state is passed through Gradio's gr.State() mechanism
    to persist data across interactions.
    """

    # Model selection
    selected_model_path: str | None = None

    # Available epochs from loaded artifacts
    available_epochs: list[int] = field(default_factory=list)
    current_epoch_idx: int = 0

    # Cached artifacts for fast slider updates
    dominant_freq_artifact: dict[str, np.ndarray] | None = None
    neuron_activations_artifact: dict[str, np.ndarray] | None = None
    freq_clusters_artifact: dict[str, np.ndarray] | None = None

    # Loss data
    train_losses: list[float] | None = None
    test_losses: list[float] | None = None

    # Model config
    model_config: dict[str, Any] | None = None

    # UI parameters
    selected_neuron: int = 0
    n_neurons: int = 512

    def get_current_epoch(self) -> int:
        """Get actual epoch number at current index."""
        if self.available_epochs and 0 <= self.current_epoch_idx < len(self.available_epochs):
            return self.available_epochs[self.current_epoch_idx]
        return 0

    def clear_artifacts(self) -> None:
        """Clear all cached artifacts."""
        self.dominant_freq_artifact = None
        self.neuron_activations_artifact = None
        self.freq_clusters_artifact = None
        self.train_losses = None
        self.test_losses = None
        self.available_epochs = []
        self.current_epoch_idx = 0
        self.model_config = None
