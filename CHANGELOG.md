# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
