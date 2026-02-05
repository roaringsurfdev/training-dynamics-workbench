"""Analysis pipeline orchestrating analysis across checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm.auto as tqdm

from analysis.protocols import AnalysisRunConfig, Analyzer

if TYPE_CHECKING:
    from families import Variant


@dataclass
class AnalysisResult:
    """Container for single-epoch analysis results."""

    epoch: int
    analyzer_name: str
    artifacts: dict[str, np.ndarray]


class AnalysisPipeline:
    """Orchestrates analysis across checkpoints.

    Enforces the scientific invariant: same Variant + same Probe across
    all analyzed checkpoints. The only variable is the checkpoint (training moment).
    """

    def __init__(
        self,
        variant: Variant,
        config: AnalysisRunConfig | None = None,
    ):
        """
        Initialize the analysis pipeline.

        The pipeline retrieves what it needs from the variant:
        - variant.family → analyzers, model creation, probe generation, analysis context
        - variant → checkpoints, checkpoint loading, artifacts_dir
        - config → which (analyzers × checkpoints) to compute

        Args:
            variant: The Variant to analyze
            config: Analysis configuration. If None, uses defaults (all analyzers,
                    all checkpoints).
        """
        self.variant = variant
        self.config = config or AnalysisRunConfig()
        self.artifacts_dir = str(variant.artifacts_dir)

        self._analyzers: list[Analyzer] = []
        self._manifest: dict[str, Any] = {}
        self._results: dict[str, dict[int, dict[str, np.ndarray]]] = {}

        # Determine device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._manifest = self._load_manifest()

    def register(self, analyzer: Analyzer) -> AnalysisPipeline:
        """
        Register an analyzer with the pipeline.

        Args:
            analyzer: Analyzer instance conforming to Analyzer protocol

        Returns:
            Self for method chaining
        """
        self._analyzers.append(analyzer)
        return self

    def run(
        self,
        force: bool = False,
        save_every: int = 10,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """
        Execute analysis pipeline across checkpoints.

        Args:
            force: If True, recompute even if artifacts exist
            save_every: Save artifacts to disk every N epochs
            progress_callback: Optional callback(progress, description) for UI updates.
                               Progress is a float from 0.0 to 1.0.
        """
        if not self._analyzers:
            return

        # Determine target epochs from config or all available
        available_epochs = self.variant.get_available_checkpoints()
        if self.config.checkpoints is not None:
            target_epochs = [e for e in self.config.checkpoints if e in available_epochs]
        else:
            target_epochs = available_epochs

        if not target_epochs:
            return

        self._init_results_buffers()

        work_queue = self._build_work_queue(target_epochs, force)
        if not work_queue:
            return

        all_epochs_needed = sorted(set(e for _, needed in work_queue for e in needed))

        # Generate probe from family (enforces scientific invariant)
        probe = self.variant.family.generate_analysis_dataset(
            self.variant.params, device=self._device
        )

        # Prepare analysis context from family (contains params + precomputed values)
        context = self.variant.family.prepare_analysis_context(
            self.variant.params, self._device
        )

        total_epochs = len(all_epochs_needed)
        epochs_processed = 0
        for epoch in tqdm.tqdm(all_epochs_needed, desc="Analyzing checkpoints"):
            if progress_callback:
                progress = epochs_processed / total_epochs
                progress_callback(
                    progress,
                    f"Analyzing checkpoint {epoch} ({epochs_processed + 1}/{total_epochs})",
                )

            self._run_single_epoch(epoch, work_queue, probe, context)
            epochs_processed += 1

            if epochs_processed % save_every == 0:
                self._save_artifacts()

        if progress_callback:
            progress_callback(0.95, "Saving artifacts...")

        self._save_artifacts()

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

    def get_completed_epochs(self, analyzer_name: str) -> list[int]:
        """
        Return list of epochs with completed analysis for given analyzer.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Sorted list of completed epoch numbers
        """
        analyzer_info = self._manifest.get("analyzers", {}).get(analyzer_name, {})
        return sorted(analyzer_info.get("epochs_completed", []))

    def load_artifact(self, analyzer_name: str) -> dict[str, np.ndarray]:
        """
        Load aggregated artifact for an analyzer.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Dict with 'epochs' array and data arrays
        """
        artifact_path = os.path.join(self.artifacts_dir, f"{analyzer_name}.npz")
        if not os.path.exists(artifact_path):
            raise FileNotFoundError(f"No artifact found for {analyzer_name}")

        return dict(np.load(artifact_path))

    def _init_results_buffers(self) -> None:
        """Initialize in-memory buffers for results."""
        for analyzer in self._analyzers:
            if analyzer.name not in self._results:
                existing = self._load_existing_results(analyzer.name)
                self._results[analyzer.name] = existing

    def _load_existing_results(self, analyzer_name: str) -> dict[int, dict[str, np.ndarray]]:
        """Load existing results from disk if available."""
        results: dict[int, dict[str, np.ndarray]] = {}
        artifact_path = os.path.join(self.artifacts_dir, f"{analyzer_name}.npz")

        if os.path.exists(artifact_path):
            data = np.load(artifact_path)
            epochs = data["epochs"]
            keys = [k for k in data.keys() if k != "epochs"]

            for i, epoch in enumerate(epochs):
                results[int(epoch)] = {k: data[k][i] for k in keys}

        return results

    def _build_work_queue(
        self, target_epochs: list[int], force: bool
    ) -> list[tuple[Analyzer, list[int]]]:
        """Build work queue of (analyzer, epochs_needed) tuples."""
        work_queue = []

        for analyzer in self._analyzers:
            if force:
                missing = target_epochs
            else:
                completed = set(self.get_completed_epochs(analyzer.name))
                missing = [e for e in target_epochs if e not in completed]

            if missing:
                work_queue.append((analyzer, missing))

        return work_queue

    def _run_single_epoch(
        self,
        epoch: int,
        work_queue: list[tuple[Analyzer, list[int]]],
        probe: torch.Tensor,
        context: dict[str, Any],
    ) -> None:
        """Run all relevant analyzers on a single checkpoint."""
        # Load checkpoint and create model via variant/family
        state_dict = self.variant.load_checkpoint(epoch)
        model = self.variant.family.create_model(self.variant.params, device=self._device)
        model.load_state_dict(state_dict)

        with torch.inference_mode():
            _, cache = model.run_with_cache(probe)

        for analyzer, needed_epochs in work_queue:
            if epoch in needed_epochs:
                result = analyzer.analyze(model, probe, cache, context)
                self._store_result(analyzer.name, epoch, result)

    def _store_result(self, analyzer_name: str, epoch: int, result: dict[str, np.ndarray]) -> None:
        """Store analysis result in memory buffer."""
        self._results[analyzer_name][epoch] = result

    def _save_artifacts(self) -> None:
        """Persist all artifacts and manifest to disk."""
        for analyzer_name, epoch_results in self._results.items():
            if not epoch_results:
                continue

            self._save_analyzer_artifact(analyzer_name, epoch_results)

        self._save_manifest()

    def _save_analyzer_artifact(
        self, analyzer_name: str, epoch_results: dict[int, dict[str, np.ndarray]]
    ) -> None:
        """Save a single analyzer's artifacts to disk."""
        epochs = sorted(epoch_results.keys())
        if not epochs:
            return

        first_result = epoch_results[epochs[0]]
        keys = list(first_result.keys())

        save_dict = {"epochs": np.array(epochs)}
        for key in keys:
            stacked = np.stack([epoch_results[e][key] for e in epochs], axis=0)
            save_dict[key] = stacked

        artifact_path = os.path.join(self.artifacts_dir, f"{analyzer_name}.npz")
        # np.savez_compressed adds .npz extension, so use base name for temp file
        temp_base = os.path.join(self.artifacts_dir, f".{analyzer_name}_tmp")
        np.savez_compressed(temp_base, **save_dict)  # type: ignore[arg-type]
        temp_path = temp_base + ".npz"
        os.replace(temp_path, artifact_path)

        self._update_manifest_for_analyzer(analyzer_name, epochs, save_dict)

    def _update_manifest_for_analyzer(
        self, analyzer_name: str, epochs: list[int], save_dict: dict
    ) -> None:
        """Update manifest entry for an analyzer."""
        if "analyzers" not in self._manifest:
            self._manifest["analyzers"] = {}

        keys = [k for k in save_dict.keys() if k != "epochs"]
        shapes = {k: list(save_dict[k].shape) for k in keys}
        dtypes = {k: str(save_dict[k].dtype) for k in keys}

        self._manifest["analyzers"][analyzer_name] = {
            "epochs_completed": epochs,
            "shapes": shapes,
            "dtypes": dtypes,
            "updated_at": datetime.now(UTC).isoformat(),
        }

    def _save_manifest(self) -> None:
        """Save manifest to disk atomically."""
        # Store variant params generically instead of hardcoded prime/seed
        self._manifest["variant_params"] = self.variant.params
        self._manifest["family_name"] = self.variant.family.name

        manifest_path = os.path.join(self.artifacts_dir, "manifest.json")
        temp_path = manifest_path + ".tmp"

        with open(temp_path, "w") as f:
            json.dump(self._manifest, f, indent=2)

        os.replace(temp_path, manifest_path)

    def _load_manifest(self) -> dict:
        """Load manifest from disk, or return empty dict."""
        manifest_path = os.path.join(self.artifacts_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                return json.load(f)

        return {}
