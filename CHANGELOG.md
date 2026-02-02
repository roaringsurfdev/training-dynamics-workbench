# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
