"""Analysis pipeline orchestrating analysis across checkpoints."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm.auto as tqdm

from analysis.protocols import AnalysisRunConfig, Analyzer

if TYPE_CHECKING:
    from families import Variant


class AnalysisPipeline:
    """Orchestrates analysis across checkpoints.

    Enforces the scientific invariant: same Variant + same Probe across
    all analyzed checkpoints. The only variable is the checkpoint (training moment).

    Artifacts are stored as one file per (analyzer, epoch):
        artifacts/{analyzer_name}/epoch_{NNNNN}.npz

    This mirrors the checkpoint storage pattern and enables:
    - Constant memory usage (no in-memory buffer)
    - Incremental computation (resume by checking file existence)
    - Parallel computation (independent files per epoch)
    - On-demand loading (load one epoch at a time for visualization)
    """

    def __init__(
        self,
        variant: Variant,
        config: AnalysisRunConfig | None = None,
    ):
        """Initialize the analysis pipeline.

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

        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.artifacts_dir, exist_ok=True)
        self._manifest = self._load_manifest()

    def register(self, analyzer: Analyzer) -> AnalysisPipeline:
        """Register an analyzer with the pipeline.

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
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        """Execute analysis pipeline across checkpoints.

        Each analyzer result is saved immediately to disk as a per-epoch file.
        No in-memory buffer is maintained across epochs.

        Args:
            force: If True, recompute even if artifacts exist
            progress_callback: Optional callback(progress, description) for UI updates.
                               Progress is a float from 0.0 to 1.0.
        """
        if not self._analyzers:
            return

        available_epochs = self.variant.get_available_checkpoints()
        if self.config.checkpoints is not None:
            target_epochs = [e for e in self.config.checkpoints if e in available_epochs]
        else:
            target_epochs = available_epochs

        if not target_epochs:
            return

        work_queue = self._build_work_queue(target_epochs, force)
        if not work_queue:
            return

        all_epochs_needed = sorted(set(e for _, needed in work_queue for e in needed))

        # Generate probe from family (enforces scientific invariant)
        probe = self.variant.family.generate_analysis_dataset(
            self.variant.params, device=self._device
        )

        # Prepare analysis context from family (contains params + precomputed values)
        context = self.variant.family.prepare_analysis_context(self.variant.params, self._device)

        total_epochs = len(all_epochs_needed)
        for i, epoch in enumerate(tqdm.tqdm(all_epochs_needed, desc="Analyzing checkpoints")):
            if progress_callback:
                progress_callback(
                    i / total_epochs,
                    f"Analyzing checkpoint {epoch} ({i + 1}/{total_epochs})",
                )

            self._run_single_epoch(epoch, work_queue, probe, context)

        # Save manifest with metadata at end of run
        self._update_manifest(work_queue)
        self._save_manifest()

        if progress_callback:
            progress_callback(1.0, "Analysis complete")

    def get_completed_epochs(self, analyzer_name: str) -> list[int]:
        """Return list of epochs with completed analysis for given analyzer.

        Determines completion from file existence on disk.

        Args:
            analyzer_name: Name of the analyzer

        Returns:
            Sorted list of completed epoch numbers
        """
        analyzer_dir = os.path.join(self.artifacts_dir, analyzer_name)
        if not os.path.isdir(analyzer_dir):
            return []

        epochs = []
        for filename in os.listdir(analyzer_dir):
            if filename.startswith("epoch_") and filename.endswith(".npz"):
                epoch_str = filename[len("epoch_") : -len(".npz")]
                try:
                    epochs.append(int(epoch_str))
                except ValueError:
                    continue
        return sorted(epochs)

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
        """Run all relevant analyzers on a single checkpoint.

        Saves each result immediately to disk, then cleans up GPU memory.
        """
        state_dict = self.variant.load_checkpoint(epoch)
        model = self.variant.family.create_model(self.variant.params, device=self._device)
        model.load_state_dict(state_dict)

        with torch.inference_mode():
            _, cache = model.run_with_cache(probe)

        for analyzer, needed_epochs in work_queue:
            if epoch in needed_epochs:
                result = analyzer.analyze(model, probe, cache, context)
                self._save_epoch_artifact(analyzer.name, epoch, result)

        # Explicit cleanup to prevent GPU memory accumulation
        del model, cache, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_epoch_artifact(
        self, analyzer_name: str, epoch: int, result: dict[str, np.ndarray]
    ) -> None:
        """Save a single epoch's analysis result to disk.

        Writes to: artifacts/{analyzer_name}/epoch_{NNNNN}.npz

        Args:
            analyzer_name: Name of the analyzer
            epoch: Epoch number
            result: Dict of numpy arrays from the analyzer
        """
        analyzer_dir = os.path.join(self.artifacts_dir, analyzer_name)
        os.makedirs(analyzer_dir, exist_ok=True)

        artifact_path = os.path.join(analyzer_dir, f"epoch_{epoch:05d}.npz")
        temp_base = os.path.join(analyzer_dir, f".epoch_{epoch:05d}_tmp")
        np.savez_compressed(temp_base, **result)  # type: ignore[arg-type]
        temp_path = temp_base + ".npz"
        os.replace(temp_path, artifact_path)

    def _update_manifest(self, work_queue: list[tuple[Analyzer, list[int]]]) -> None:
        """Update manifest with metadata for all analyzers that ran."""
        if "analyzers" not in self._manifest:
            self._manifest["analyzers"] = {}

        for analyzer, _ in work_queue:
            completed = self.get_completed_epochs(analyzer.name)
            if not completed:
                continue

            # Load one epoch to get shapes and dtypes
            sample_path = os.path.join(
                self.artifacts_dir, analyzer.name, f"epoch_{completed[0]:05d}.npz"
            )
            sample = dict(np.load(sample_path))
            shapes = {k: list(v.shape) for k, v in sample.items()}
            dtypes = {k: str(v.dtype) for k, v in sample.items()}

            self._manifest["analyzers"][analyzer.name] = {
                "epochs_completed": completed,
                "shapes": shapes,
                "dtypes": dtypes,
                "updated_at": datetime.now(UTC).isoformat(),
            }

    def _save_manifest(self) -> None:
        """Save manifest to disk atomically."""
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
