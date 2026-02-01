"""Standalone artifact loader for visualization components."""

import json
import os
from typing import Any

import numpy as np


class ArtifactLoader:
    """Loads analysis artifacts for visualization components.

    Provides a standalone interface for loading artifacts without
    requiring the full pipeline or model specification.
    """

    def __init__(self, artifacts_dir: str):
        """
        Initialize the artifact loader.

        Args:
            artifacts_dir: Path to the artifacts directory
        """
        self.artifacts_dir = artifacts_dir
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest(self) -> dict[str, Any]:
        """Load and cache the manifest."""
        if self._manifest is None:
            self._manifest = self._load_manifest()
        return self._manifest

    def load(self, analyzer_name: str) -> dict[str, np.ndarray]:
        """
        Load aggregated artifact for an analyzer.

        Args:
            analyzer_name: Name of the analyzer (e.g., "dominant_frequencies")

        Returns:
            Dict with 'epochs' array and data arrays

        Raises:
            FileNotFoundError: If artifact doesn't exist
        """
        artifact_path = os.path.join(self.artifacts_dir, f"{analyzer_name}.npz")

        if not os.path.exists(artifact_path):
            raise FileNotFoundError(
                f"No artifact found for '{analyzer_name}'. "
                f"Expected file: {artifact_path}"
            )

        return dict(np.load(artifact_path))

    def get_available_analyzers(self) -> list[str]:
        """
        List available analyzers from manifest.

        Returns:
            List of analyzer names with saved artifacts
        """
        return list(self.manifest.get("analyzers", {}).keys())

    def get_epochs(self, analyzer_name: str) -> list[int]:
        """
        Get list of epochs available for an analyzer.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Sorted list of epoch numbers
        """
        analyzer_info = self.manifest.get("analyzers", {}).get(analyzer_name, {})
        return sorted(analyzer_info.get("epochs_completed", []))

    def get_metadata(self, analyzer_name: str) -> dict[str, Any]:
        """
        Get metadata for an analyzer.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dict with shapes, dtypes, updated_at, etc.

        Raises:
            KeyError: If analyzer not found in manifest
        """
        analyzers = self.manifest.get("analyzers", {})

        if analyzer_name not in analyzers:
            available = list(analyzers.keys())
            raise KeyError(
                f"Analyzer '{analyzer_name}' not found in manifest. "
                f"Available: {available}"
            )

        return analyzers[analyzer_name]

    def get_model_config(self) -> dict[str, Any]:
        """
        Get model configuration from manifest.

        Returns:
            Dict with model config (prime, seed, etc.)
        """
        return self.manifest.get("model_config", {})

    def _load_manifest(self) -> dict[str, Any]:
        """Load manifest from disk."""
        manifest_path = os.path.join(self.artifacts_dir, "manifest.json")

        if not os.path.exists(manifest_path):
            return {}

        with open(manifest_path, "r") as f:
            return json.load(f)
