# Training Dynamics Workbench

A mechanistic interpretability research tool for studying how neural networks learn during training. The workbench systematizes training, analysis, and visualization so that researchers can focus on exploring emergent behaviors rather than managing infrastructure.

## Purpose

The workbench exists to answer: **"How does behavior X change across training?"**

Train small Transformer models, capture checkpoints at key moments, and analyze how computational structures emerge over time. The platform enforces a scientific invariant — same model variant, same probe, only the checkpoint varies — so that visualizations are directly comparable and scientifically meaningful.

## Current Status

**v0.2.0 — First Foundational Release**

The MVP (v0.1.x) proved viability as a prototype. This release establishes the architecture for sustained research: Model Families, Variants, per-epoch artifact storage, and a family-aware dashboard.

Five trained Modulo Addition variants across different primes are actively being used for research.

## Workflow

```
Define Family → Train Variants → Analyze Checkpoints → Explore Visualizations
```

1. **Define** a Model Family with architecture, analyzers, and probes (`model_families/`)
2. **Train** Variants with different domain parameters (e.g., different primes or seeds)
3. **Analyze** each variant's checkpoints to generate per-epoch analysis artifacts
4. **Explore** training dynamics through synchronized interactive visualizations

## Key Features

### Model Families and Variants
- Model Family protocol for grouping structurally similar models
- Variants differ only in domain parameters (e.g., modulus, seed)
- Family defines architecture, analyzers, probes, and visualizations
- FamilyRegistry discovers and manages families from `model_families/`

### Analysis Engine
- Per-epoch artifact storage: one file per (analyzer, epoch) for constant memory usage
- Resumable analysis: skips already-computed epochs automatically
- Three built-in analyzers: Dominant Frequencies, Neuron Activations, Frequency Clusters
- Extensible Analyzer protocol for adding new analysis functions

### Workbench Dashboard
- Family and variant selection with state-aware UI
- Training and analysis from the dashboard
- **Synchronized checkpoint slider** across all visualizations
- On-demand per-epoch loading (no full dataset in memory)
- Interactive Plotly visualizations with hover details

## Technology Stack

- **Python 3.11+** - Supports 3.11, 3.12, and 3.13
- **PyTorch + CUDA** - Training and model execution
- **TransformerLens** - Model architecture (HookedTransformer)
- **Safetensors** - Checkpoint persistence
- **Plotly** - Interactive visualizations
- **Gradio** - Dashboard UI
- **NumPy** - Analysis artifact storage (per-epoch `.npz` files)
- **pytest** - Testing framework

## Project Structure

```
/
├── analysis/                     # Analysis engine
│   ├── analyzers/               # Analyzer implementations
│   ├── library/                 # Shared analysis utilities (Fourier basis, etc.)
│   ├── pipeline.py              # AnalysisPipeline orchestration
│   └── artifact_loader.py       # Per-epoch artifact loading
├── dashboard/                    # Gradio web dashboard
│   ├── components/              # UI components (family selector, loss curves)
│   ├── app.py                   # Main application
│   └── state.py                 # Dashboard state management
├── families/                     # Model Family framework
│   ├── implementations/         # Concrete family implementations
│   └── protocols.py             # ModelFamily protocol, Variant, FamilyRegistry
├── model_families/               # Family definitions (family.json files)
│   └── modulo_addition_1layer/  # Modulo Addition 1-Layer family
├── visualization/                # Visualization renderers
│   └── renderers/               # Per-epoch and cross-epoch renderers
├── requirements/                 # Structured requirements
├── tests/                        # Test suite
└── results/                      # Training outputs (not in repo)
    └── {family}/
        └── {variant}/
            ├── checkpoints/     # Model checkpoints (.safetensors)
            ├── artifacts/       # Per-epoch analysis artifacts
            │   └── {analyzer}/  # epoch_00000.npz, epoch_00100.npz, ...
            ├── metadata.json    # Training metrics
            └── config.json      # Variant configuration
```

## Getting Started

### Prerequisites
```bash
python --version  # Python 3.13
nvidia-smi        # CUDA support (optional but recommended)
```

### Installation
```bash
git clone <repository-url>
cd training-dynamics-workbench

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Running the Dashboard
```bash
python main.py
```

### Running Tests
```bash
pytest
```

## Architecture

### Three-Layer Design

1. **Model Families** — Define model architecture, training data, probes, and analyzers
2. **Analysis Engine** — Load checkpoints, run forward passes, generate per-epoch artifacts
3. **Workbench Dashboard** — Family-aware UI for training, analysis, and visualization

### Per-Epoch Artifact Storage

```
Training → Checkpoints (safetensors) → Analysis (per-epoch) → Artifacts (disk) → Visualizations (on-demand)
```

Analysis artifacts are stored one file per (analyzer, epoch), eliminating memory exhaustion during analysis and enabling the dashboard to load only the data it needs for the current view.

## Documentation

- [**PROJECT.md**](PROJECT.md) - Project scope, architecture, and current status
- [**DOMAIN_MODEL.md**](DOMAIN_MODEL.md) - Core domain objects and relationships
- [**requirements/**](requirements/) - Structured requirements

## Future Directions

- Notebook-based exploratory visualization design
- Analysis Reports (web-based visualization components per family)
- Gap-filling pattern for incremental analysis
- Additional model families beyond Modulo Addition
- Automatic grokking phase detection
- Side-by-side variant comparison
- AWS deployment for larger models

## Acknowledgments

Based on research from:
- Neel Nanda's "A Mechanistic Interpretability Analysis of Grokking"
- TransformerLens library for interpretability research

---
**Last Updated:** 2026-02-06
