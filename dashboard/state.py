"""State management for the dashboard."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class DashboardState:
    """Centralized state for dashboard synchronization.

    This state is passed through Gradio's gr.State() mechanism
    to persist data across interactions.

    Artifact data is NOT cached in state. Instead, artifacts_dir is stored
    and per-epoch data is loaded on demand via ArtifactLoader.
    """

    # Family/Variant selection (REQ_021d)
    selected_family_name: str | None = None
    selected_variant_name: str | None = None

    # Model selection (legacy, kept for compatibility)
    selected_model_path: str | None = None

    # Artifacts directory for on-demand loading (REQ_021f)
    artifacts_dir: str | None = None

    # Available epochs from artifact filesystem
    available_epochs: list[int] = field(default_factory=list)
    current_epoch_idx: int = 0

    # Available analyzers for this variant
    available_analyzers: list[str] = field(default_factory=list)

    # Loss data
    train_losses: list[float] | None = None
    test_losses: list[float] | None = None

    # Model config
    model_config: dict[str, Any] | None = None

    # UI parameters
    selected_neuron: int = 0
    n_neurons: int = 512

    # Attention pattern position pair (REQ_025)
    selected_to_position: int = 2  # = token
    selected_from_position: int = 0  # a token

    # Parameter trajectory component group (REQ_029)
    selected_trajectory_group: str = "all"

    # Effective dimensionality (REQ_030)
    selected_sv_matrix: str = "W_in"
    selected_sv_head: int = 0

    # Cached parameter snapshots for trajectory rendering (REQ_029)
    # Unlike per-epoch artifacts, trajectory needs all epochs loaded at once.
    # Cached on first access per variant to avoid reloading on every slider change.
    _trajectory_snapshots: list[dict[str, Any]] | None = field(
        default=None, repr=False
    )
    _trajectory_epochs: list[int] | None = field(default=None, repr=False)

    def get_current_epoch(self) -> int:
        """Get actual epoch number at current index."""
        if self.available_epochs and 0 <= self.current_epoch_idx < len(self.available_epochs):
            return self.available_epochs[self.current_epoch_idx]
        return 0

    def clear_artifacts(self) -> None:
        """Clear artifact references and related state."""
        self.artifacts_dir = None
        self.available_analyzers = []
        self.train_losses = None
        self.test_losses = None
        self.available_epochs = []
        self.current_epoch_idx = 0
        self.model_config = None
        self._trajectory_snapshots = None
        self._trajectory_epochs = None

    def get_trajectory_data(
        self,
    ) -> tuple[list[dict[str, np.ndarray]], list[int]] | None:
        """Get cached trajectory snapshots, loading on first access.

        Returns:
            Tuple of (snapshots, epochs) or None if not available.
        """
        if self._trajectory_snapshots is not None:
            return self._trajectory_snapshots, self._trajectory_epochs  # type: ignore[return-value]

        if (
            not self.artifacts_dir
            or "parameter_snapshot" not in self.available_analyzers
        ):
            return None

        from analysis import ArtifactLoader

        loader = ArtifactLoader(self.artifacts_dir)
        epochs = loader.get_epochs("parameter_snapshot")
        if not epochs:
            return None

        snapshots = [loader.load_epoch("parameter_snapshot", e) for e in epochs]
        self._trajectory_snapshots = snapshots
        self._trajectory_epochs = epochs
        return snapshots, epochs

    def clear_selection(self) -> None:
        """Clear family/variant selection and all artifacts."""
        self.selected_family_name = None
        self.selected_variant_name = None
        self.selected_model_path = None
        self.clear_artifacts()
