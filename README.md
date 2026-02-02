# Training Dynamics Workbench

A mechanistic interpretability research tool for analyzing how neural networks learn during training. Built to streamline the analysis of emergent model behaviors through interactive visualizations of training dynamics.

## Purpose

Train small experimental Transformer models, capture snapshots during training, and analyze how computational structures emerge over time. Move beyond static post-training analysis to understand **when** and **how** models develop their learned algorithms.

## Current Status

**Version: 0.1.2** | [Changelog](CHANGELOG.md)

The MVP is complete and functional. The workbench supports end-to-end training, analysis, and visualization of modulo addition grokking experiments.

## Features

### Training Runner
- Train 1-layer Transformers on modulo addition task
- Configurable checkpoint epochs (densify around grokking phase)
- Safetensors persistence with immediate disk writes
- Automatic CUDA detection for GPU acceleration

### Analysis Pipeline
- Forward pass computation with activation caching
- Fourier transform analysis of embeddings and activations
- Neuron frequency cluster analysis
- Persistent artifacts (NumPy `.npz` files)
- Resumable analysis (skip existing artifacts)

### Interactive Dashboard
- Gradio web interface for training and analysis
- **Synchronized checkpoint slider** across all visualizations
- Loss curves with epoch indicator
- Dominant Fourier frequencies bar plot
- Neuron activation heatmaps (single + grid views)
- Neuron-frequency cluster specialization heatmap

## Technology Stack

- **Python 3.11+** - Supports 3.11, 3.12, and 3.13
- **PyTorch + CUDA** - Training and model execution
- **TransformerLens** - Model architecture (HookedTransformer)
- **Safetensors** - Modern checkpoint persistence
- **Plotly** - Interactive visualizations
- **Gradio** - Dashboard UI
- **NumPy** - Analysis artifact storage
- **pytest** - Testing (156 tests)

## Getting Started

### Prerequisites
```bash
# Python 3.11 or later
python --version  # Should be 3.11, 3.12, or 3.13

# CUDA support (optional but recommended)
nvidia-smi  # Check GPU availability
```

### Installation
```bash
# Clone the repository
git clone https://github.com/roaringsurfdev/training-dynamics-workbench.git
cd training-dynamics-workbench

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Running the Dashboard
```bash
uv run python -m dashboard.app
```

The dashboard will open in your browser at `http://localhost:7860`.

### Running Tests
```bash
# All CPU tests
uv run pytest -m "not gpu" -v

# Include GPU tests (requires CUDA)
uv run pytest -v
```

## Architecture

```
Training (ModuloAdditionSpecification)
    ↓ checkpoints/*.safetensors
Analysis (AnalysisPipeline + Analyzers)
    ↓ artifacts/*.npz
Visualization (Renderers)
    ↓ Plotly figures
Dashboard (Gradio)
```

This artifact-based design separates expensive computation from visualization, enabling fast iteration without re-running forward passes.

## Project Structure

```
├── dashboard/               # Gradio web interface
│   ├── app.py              # Main dashboard application
│   └── version.py          # Version management
├── analysis/               # Analysis pipeline and analyzers
├── visualization/          # Plotly rendering utilities
├── training/               # Training runner
├── tests/                  # pytest test suite
├── requirements/           # Structured requirements
│   ├── active/            # Current requirements
│   └── archive/           # Completed requirements by version
└── results/               # Training outputs (not in repo)
    └── model_p{prime}_seed{seed}/
        ├── checkpoints/   # Model checkpoints (.safetensors)
        ├── artifacts/     # Analysis artifacts (.npz)
        ├── metadata.json  # Training metrics
        └── config.json    # Model configuration
```

## Research Context

This workbench is built around the "grokking" phenomenon in neural networks - delayed generalization that occurs well after training loss has converged. The synchronized visualization slider enables exploration of:

- When dominant Fourier frequencies emerge in embedding space
- How neuron activation patterns evolve during training
- Correlation between frequency emergence and neuron specialization

Based on research from Neel Nanda's "A Mechanistic Interpretability Analysis of Grokking" and the TransformerLens library.

## Roadmap

The project is in active development. Planned directions include:

- Additional visualization types for deeper analysis
- Support for more model architectures beyond 1-layer Transformers
- Backend improvements for larger-scale experiments
- Enhanced checkpoint scheduling based on training dynamics

See [requirements/active/](requirements/active/) for current work items.

## Documentation

- [CHANGELOG.md](CHANGELOG.md) - Release history
- [PROJECT.md](PROJECT.md) - Detailed project scope and architecture
- [Claude.md](Claude.md) - Development collaboration framework

## License

MIT

## Acknowledgments

- Neel Nanda's mechanistic interpretability research on grokking
- TransformerLens library for interpretability tooling
- The broader mechanistic interpretability research community
