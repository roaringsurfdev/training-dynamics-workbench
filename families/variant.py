"""Variant class representing a specific trained model within a family."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from safetensors.torch import load_file

from families.types import VariantState

if TYPE_CHECKING:
    from transformer_lens import HookedTransformer

    from families.protocols import ModelFamily


class Variant:
    """A specific trained model within a family.

    Variants share the same architecture and analysis logic but differ
    in domain-specific parameters (e.g., modulus, seed).

    State is derived from filesystem presence:
    - UNTRAINED: No checkpoints directory or empty
    - TRAINED: Checkpoints exist but no artifacts
    - ANALYZED: Both checkpoints and artifacts exist
    """

    def __init__(
        self,
        family: ModelFamily,
        params: dict[str, Any],
        results_dir: Path,
    ):
        """Initialize a Variant.

        Args:
            family: The ModelFamily this variant belongs to
            params: Domain parameter values (e.g., {"prime": 113, "seed": 42})
            results_dir: Root results directory (typically "results/")
        """
        self._family = family
        self._params = params
        self._results_dir = Path(results_dir)

    @property
    def family(self) -> ModelFamily:
        """The ModelFamily this variant belongs to."""
        return self._family

    @property
    def params(self) -> dict[str, Any]:
        """Domain parameter values for this variant."""
        return self._params.copy()

    @property
    def name(self) -> str:
        """Variant directory name derived from family pattern."""
        return self._family.get_variant_directory_name(self._params)

    @property
    def variant_dir(self) -> Path:
        """Path to this variant's directory.

        Structure: results/{family.name}/{variant_name}/
        """
        return self._results_dir / self._family.name / self.name

    @property
    def checkpoints_dir(self) -> Path:
        """Path to checkpoints subdirectory."""
        return self.variant_dir / "checkpoints"

    @property
    def artifacts_dir(self) -> Path:
        """Path to analysis artifacts subdirectory."""
        return self.variant_dir / "artifacts"

    @property
    def metadata_path(self) -> Path:
        """Path to training metadata JSON file."""
        return self.variant_dir / "metadata.json"

    @property
    def config_path(self) -> Path:
        """Path to model config JSON file."""
        return self.variant_dir / "config.json"

    @property
    def state(self) -> VariantState:
        """Current state based on filesystem presence."""
        has_checkpoints = self._has_checkpoints()
        has_artifacts = self._has_artifacts()

        if has_checkpoints and has_artifacts:
            return VariantState.ANALYZED
        elif has_checkpoints:
            return VariantState.TRAINED
        else:
            return VariantState.UNTRAINED

    def _has_checkpoints(self) -> bool:
        """Check if any checkpoint files exist."""
        if not self.checkpoints_dir.exists():
            return False
        # Look for .safetensors files
        return any(self.checkpoints_dir.glob("checkpoint_epoch_*.safetensors"))

    def _has_artifacts(self) -> bool:
        """Check if any artifact files exist."""
        if not self.artifacts_dir.exists():
            return False
        # Look for any .npz files (analysis artifacts)
        return any(self.artifacts_dir.glob("**/*.npz"))

    def get_available_checkpoints(self) -> list[int]:
        """List checkpoint epochs available on disk.

        Returns:
            Sorted list of epoch numbers with available checkpoints
        """
        epochs = []
        if self.checkpoints_dir.exists():
            for path in self.checkpoints_dir.glob("checkpoint_epoch_*.safetensors"):
                # Extract epoch from filename: checkpoint_epoch_00100.safetensors
                epoch_str = path.stem.replace("checkpoint_epoch_", "")
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    continue
        return sorted(epochs)

    def load_checkpoint(self, epoch: int) -> dict[str, Any]:
        """Load model state_dict from a specific checkpoint.

        Args:
            epoch: The epoch number of the checkpoint to load

        Returns:
            Model state_dict

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = self.checkpoints_dir / f"checkpoint_epoch_{epoch:05d}.safetensors"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found for epoch {epoch} at {checkpoint_path}")
        return load_file(checkpoint_path)

    def load_model_at_checkpoint(self, epoch: int) -> HookedTransformer:
        """Load a HookedTransformer with weights from a specific checkpoint.

        Args:
            epoch: The epoch number of the checkpoint to load

        Returns:
            HookedTransformer with checkpoint weights loaded
        """
        model = self._family.create_model(self._params)
        state_dict = self.load_checkpoint(epoch)
        model.load_state_dict(state_dict)
        return model

    def ensure_directories(self) -> None:
        """Create variant directories if they don't exist."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"Variant(family={self._family.name!r}, name={self.name!r}, state={self.state.value})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Variant):
            return NotImplemented
        return self._family.name == other._family.name and self._params == other._params

    def __hash__(self) -> int:
        return hash((self._family.name, tuple(sorted(self._params.items()))))
