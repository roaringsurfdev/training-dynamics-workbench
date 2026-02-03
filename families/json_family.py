"""JSON-based ModelFamily implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from families.types import AnalysisDatasetSpec, ArchitectureSpec, ParameterSpec


class JsonModelFamily:
    """ModelFamily implementation loaded from a family.json file.

    This is a data-only implementation that stores configuration.
    Actual model creation and dataset generation are delegated to
    family-specific implementations (see modulo_addition_1layer).

    For families that need custom logic, subclass this or implement
    the ModelFamily protocol directly.
    """

    def __init__(self, config: dict[str, Any], config_path: Path | None = None):
        """Initialize from config dict.

        Args:
            config: Parsed family.json content
            config_path: Path to the family.json file (for error messages)
        """
        self._config = config
        self._config_path = config_path
        self._validate_config()

    @classmethod
    def from_json(cls, path: Path | str) -> JsonModelFamily:
        """Load a JsonModelFamily from a family.json file.

        Args:
            path: Path to family.json

        Returns:
            JsonModelFamily instance
        """
        path = Path(path)
        with open(path) as f:
            config = json.load(f)
        return cls(config, config_path=path)

    def _validate_config(self) -> None:
        """Validate required fields are present."""
        required_fields = [
            "name",
            "display_name",
            "description",
            "architecture",
            "domain_parameters",
            "analyzers",
            "visualizations",
            "variant_pattern",
        ]
        missing = [f for f in required_fields if f not in self._config]
        if missing:
            location = f" in {self._config_path}" if self._config_path else ""
            raise KeyError(f"Missing required fields{location}: {missing}")

    @property
    def name(self) -> str:
        """Unique identifier, used as directory key."""
        return self._config["name"]

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        return self._config["display_name"]

    @property
    def description(self) -> str:
        """Brief description of the family."""
        return self._config["description"]

    @property
    def architecture(self) -> ArchitectureSpec:
        """Architectural properties."""
        return self._config["architecture"]

    @property
    def domain_parameters(self) -> dict[str, ParameterSpec]:
        """Parameters that vary across variants."""
        return self._config["domain_parameters"]

    @property
    def analyzers(self) -> list[str]:
        """Analyzer identifiers valid for this family."""
        return self._config["analyzers"]

    @property
    def visualizations(self) -> list[str]:
        """Visualization identifiers valid for this family."""
        return self._config["visualizations"]

    @property
    def analysis_dataset(self) -> AnalysisDatasetSpec:
        """Specification for the analysis dataset."""
        return self._config.get("analysis_dataset", {})

    @property
    def variant_pattern(self) -> str:
        """Pattern for variant directory names."""
        return self._config["variant_pattern"]

    def get_variant_directory_name(self, params: dict[str, Any]) -> str:
        """Generate variant directory name from parameters.

        Args:
            params: Domain parameter values

        Returns:
            Directory name with parameters substituted
        """
        return self.variant_pattern.format(**params)

    def create_model(self, params: dict[str, Any]) -> Any:
        """Create a model instance.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"create_model() not implemented for {self.name}. "
            "Use a family-specific implementation."
        )

    def generate_analysis_dataset(self, params: dict[str, Any]) -> torch.Tensor:
        """Generate the analysis dataset.

        Note: This base implementation raises NotImplementedError.
        Family-specific subclasses should override this method.

        Args:
            params: Domain parameter values

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError(
            f"generate_analysis_dataset() not implemented for {self.name}. "
            "Use a family-specific implementation."
        )

    def get_default_params(self) -> dict[str, Any]:
        """Get default parameter values from domain_parameters.

        Returns:
            Dict of parameter name to default value
        """
        return {
            name: spec.get("default")
            for name, spec in self.domain_parameters.items()
            if "default" in spec
        }

    def __repr__(self) -> str:
        return f"JsonModelFamily(name={self.name!r})"
