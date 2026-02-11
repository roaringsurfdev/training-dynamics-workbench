# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-02-10

### Added

- **Parameter Space Trajectory Projections** (REQ_029)
  - PCA-based trajectory visualization of model weight evolution through training
  - 2D trajectory (PC1 vs PC2) with epoch-colored points and current epoch highlight
  - Explained variance scree plot (individual + cumulative)
  - Parameter velocity plot (L2 norm of change per epoch, normalized by epoch gap)
  - Per-component group velocity comparison (Embedding, Attention, MLP)
  - `compute_pca_trajectory()` and `compute_parameter_velocity()` library functions

- **Weight Matrix Effective Dimensionality** (REQ_030)
  - New `EffectiveDimensionalityAnalyzer` computes SVD-based participation ratio per checkpoint
  - Dimensionality trajectory renderer (one line per weight matrix across epochs)
  - Singular value spectrum renderer (per-epoch bar chart)
  - Summary statistics: participation ratio per matrix

- **Loss Landscape Flatness** (REQ_031)
  - New `LandscapeFlatnessAnalyzer` measures sensitivity to random perturbations
  - `compute_landscape_flatness()` library function with configurable perturbation scale/samples
  - Flatness trajectory renderer (selectable metric: mean/max/std delta loss)
  - Perturbation distribution renderer (per-epoch histogram of delta losses)
  - Summary statistics: mean, max, std delta loss per epoch

- **Parameter Trajectory PC3 Visualization** (REQ_032)
  - 3D interactive PCA trajectory (PC1 vs PC2 vs PC3) with rotation/zoom/pan
  - PC1 vs PC3 and PC2 vs PC3 2D projection panels
  - Shared `_render_trajectory_2d()` helper eliminates rendering code duplication
  - All projections respond to component group selector and epoch slider
  - Confirms trajectory "dip" structure is genuine 3D geometry, not a projection artifact

- **Visualization Export** (REQ_033)
  - `export_figure()`: Static export of any Plotly Figure to PNG, SVG, PDF, or HTML
  - `export_animation()`: Animated GIF from per-epoch renderers
  - `export_cross_epoch_animation()`: Animated GIF sweeping current_epoch across cross-epoch renderers
  - `export_variant_visualization()`: Name-based convenience function (25 visualizations mapped)
  - Works without a running dashboard — designed for notebook/CLI/Claude programmatic use
  - Dependencies added: kaleido, Pillow

### Changed

- Parameter velocity now normalized by epoch gap for non-uniform checkpoint schedules

### References

- Archived requirements: `requirements/archive/v0.3.1-trajectory-export/`
- Milestone summary: `requirements/archive/v0.3.1-trajectory-export/MILESTONE_SUMMARY.md`

## [0.3.0] - 2026-02-08

### Added

- **Attention Head Pattern Visualization** (REQ_025)
  - New `AttentionPatternsAnalyzer` extracts full attention pattern tensor per checkpoint
  - 2x2 grid renderer showing all 4 attention heads as heatmaps
  - Position pair dropdown: view different attention relationships (e.g., `= attending to a`)
  - `extract_attention_patterns()` library function

- **Attention Head Frequency Specialization** (REQ_026)
  - New `AttentionFreqAnalyzer` computes Fourier frequency decomposition of attention patterns
  - Per-epoch frequency heatmap (frequencies x heads)
  - Cross-epoch specialization trajectory (one line per head)
  - Cross-epoch dominant frequency step plot
  - Summary statistics: dominant frequency, max fraction, mean specialization per head

- **Neuron Frequency Specialization Summary Statistics** (REQ_027)
  - Extended `NeuronFreqClustersAnalyzer` with `get_summary_keys()` and `compute_summary()`
  - Tracks specialized neuron counts per frequency with configurable threshold (default 0.9)
  - Low/mid/high frequency range bucket counts
  - Cross-epoch specialization trajectory renderer (total + low/mid/high lines)
  - Per-frequency specialization heatmap renderer (frequencies x epochs)

### Fixed

- **Variant dropdown UX** (REQ_028)
  - Variants now sorted alphabetically in dropdown and table
  - Default selection is `None` instead of first variant, so users can select the first item directly
- **Attention plot clipping** — full-width layout for attention head visualization

### References

- Archived requirements: `requirements/archive/v0.3.0-attention-specialization/`
- Milestone summary: `requirements/archive/v0.3.0-attention-specialization/MILESTONE_SUMMARY.md`

## [0.2.1] - 2026-02-08

### Added

- **Family-Specific Summary Statistics** (REQ_022)
  - Optional `get_summary_keys()` and `compute_summary()` methods on Analyzer protocol
  - Pipeline collects summary statistics inline and writes single `summary.npz` per analyzer
  - ArtifactLoader `load_summary()` and `has_summary()` for cross-epoch data access
  - Gap-filling support for incremental summary updates

- **Coarseness Analyzer** (REQ_023)
  - New `CoarsenessAnalyzer` quantifies blob vs plaid neuron patterns
  - Per-epoch artifact: per-neuron coarseness values `(d_mlp,)`
  - Summary statistics: mean, std, median, percentiles, blob count, histogram
  - Library function: `compute_neuron_coarseness()` in `analysis/library/fourier.py`
  - Registered for Modulo Addition 1-Layer family

- **Coarseness Visualizations** (REQ_024)
  - Coarseness trajectory line plot with percentile band and epoch indicator
  - Per-epoch coarseness distribution histogram with blob/plaid/transitional coloring
  - Blob count trajectory renderer (notebook-focused)
  - Per-neuron coarseness bar chart (notebook-focused)
  - Conditional dashboard panels: appear only when coarseness artifacts exist
  - All eight trained variants analyzed with coarseness data

### References

- Archived requirements: `requirements/archive/v0.2.1-coarseness/`
- Milestone summary: `requirements/archive/v0.2.1-coarseness/MILESTONE_SUMMARY.md`

## [0.2.0] - 2026-02-06

### Added

- **Model Family Abstraction** (REQ_021)
  - `ModelFamily` protocol with `Variant`, `FamilyRegistry`, `ArchitectureSpec`
  - JSON-driven family definitions in `model_families/`
  - Generic analysis library (`analysis/library/`) + family-bound analyzers (`analysis/analyzers/`)
  - Modulo Addition 1-Layer family implementation with five trained variants

- **Per-Epoch Artifact Storage** (REQ_021f)
  - Artifacts stored as `epoch_{NNNNN}.npz` per (analyzer, epoch) — eliminates memory exhaustion
  - `ArtifactLoader` with on-demand `load_epoch()`, stacked `load_epochs()`, and discovery methods
  - Dashboard loads single epochs per slider interaction

- **Dashboard Integration** (REQ_021d, REQ_021e)
  - Family-aware Analysis tab with variant selection
  - Training tab with family selection and domain parameter inputs
  - Dynamic visualization tabs generated from family configuration

- **Checkpoint Epoch-Index Display** (REQ_020)
  - Checkpoint markers on loss curve show "Epoch: X (Index: Y)" in tooltip
  - Current Epoch display shows "Epoch X (Index Y)" near slider

### Architecture

```
model_families/          # Family definitions (family.json)
families/                # Protocols, registry, implementations
analysis/
  library/               # Generic analysis functions (Fourier, activations)
  analyzers/             # Family-bound analyzers
  pipeline.py            # Per-epoch artifact persistence
  artifact_loader.py     # On-demand loading
results/                 # Per-variant checkpoints + artifacts
```

### References

- Archived requirements: `requirements/archive/v0.2.0-foundations/`
- Milestone summary: `requirements/archive/v0.2.0-foundations/MILESTONE_SUMMARY.md`

## [0.1.3] - 2026-02-03

### Added

- **Checkpoint Epoch-Index Display** (REQ_020)
  - Checkpoint markers on loss curve show "Epoch: X (Index: Y)" in tooltip
  - Current Epoch display shows "Epoch X (Index Y)" near slider
  - Enables quick navigation to specific epochs

### Fixed

- **Slider None Guard** (BUG_003)
  - Fixed TypeError when clearing slider text input before typing new value
  - Gracefully handles None values by falling back to current state

## [0.1.2] - 2026-02-01

### Changed

- **Python 3.11+ Compatibility** (REQ_013)
  - Broadened support from Python 3.13-only to 3.11, 3.12, and 3.13
  - Relaxed numpy constraint (>=1.26) for TransformerLens compatibility
  - CI now tests against all three Python versions

- **Removed neel-plotly Dependency** (REQ_012)
  - Replaced git dependency with native Plotly visualization utilities
  - New `visualization.line()` function provides equivalent functionality
  - Zero external git dependencies improves installation reliability

- **Code Quality Enforcement** (REQ_011)
  - Ruff linting and formatting checks now blocking in CI
  - Pyright type checking now blocking in CI
  - Fixed all linting violations and type errors across codebase

### References

- Archived requirements: `requirements/archive/v0.1.2-quality/`
- Milestone summary: `requirements/archive/v0.1.2-quality/MILESTONE_SUMMARY.md`

## [0.1.1] - 2026-02-01

### Fixed

- Dashboard now uses GPU for training and analysis when CUDA is available (REQ_016, REQ_018)
  - Training previously required `CUDA_VISIBLE_DEVICES` env var to detect GPU
  - Analysis was hardcoded to CPU only

### References

- Archived requirements: `requirements/archive/v0.1.1-cuda/`
- Milestone summary: `requirements/archive/v0.1.1-cuda/MILESTONE_SUMMARY.md`

## [0.1.0] - 2026-02-01

**MVP Release** - First functional version of the Training Dynamics Workbench.

### Added

- **Training Runner** (REQ_001, REQ_002)
  - Configurable checkpoint epochs with optimized default schedule
  - Safetensors persistence with metadata.json and config.json
  - Backward-compatible loading for legacy pickle format

- **Analysis Pipeline** (REQ_003)
  - `AnalysisPipeline` orchestrator with resumable artifact generation
  - `ArtifactLoader` for visualization-layer independence
  - Progress callbacks for UI integration
  - Three analyzers: dominant frequencies, neuron activations, frequency clusters

- **Visualizations** (REQ_004, REQ_005, REQ_006)
  - Dominant embedding frequencies bar plot with threshold highlighting
  - Neuron activation heatmaps (single neuron + grid views)
  - Neuron-frequency cluster heatmap

- **Dashboard** (REQ_007, REQ_008, REQ_009, REQ_010)
  - Gradio web interface with Training and Analysis tabs
  - Synchronized epoch slider across all visualizations
  - Loss curves with epoch indicator line
  - Application versioning displayed in header

### Architecture

```
Training (ModuloAdditionSpecification)
    ↓ checkpoints/*.safetensors
Analysis (AnalysisPipeline + Analyzers)
    ↓ artifacts/*.npz
Visualization (Renderers)
    ↓ Plotly figures
Dashboard (Gradio)
```

### Technical Notes

- Python 3.13 with PyTorch + CUDA support
- TransformerLens for model architecture
- Artifact-based design separates expensive computation from cheap visualization
- 156 tests across 8 test files

### References

- Archived requirements: `requirements/archive/v0.1.0-mvp/`
- Milestone summary: `requirements/archive/v0.1.0-mvp/MILESTONE_SUMMARY.md`
