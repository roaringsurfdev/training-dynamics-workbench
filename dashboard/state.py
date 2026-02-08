"""State management for the dashboard."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


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

    # Coarseness summary data (loaded once per variant, small file)
    coarseness_summary: dict[str, Any] | None = None

    # UI parameters
    selected_neuron: int = 0
    n_neurons: int = 512

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
        self.coarseness_summary = None

    def clear_selection(self) -> None:
        """Clear family/variant selection and all artifacts."""
        self.selected_family_name = None
        self.selected_variant_name = None
        self.selected_model_path = None
        self.clear_artifacts()
