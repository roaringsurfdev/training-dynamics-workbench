# Training Dynamics Workbench

A mechanistic interpretability research tool for analyzing how neural networks learn during training. Built to streamline the analysis of emergent model behaviors through interactive visualizations of training dynamics.

## Purpose

Train small experimental Transformer models, capture snapshots during training, and analyze how computational structures emerge over time. Move beyond static post-training analysis to understand **when** and **how** models develop their learned algorithms.

## Current Status

**Phase: Requirements Definition Complete ✓**

- [x] Project structure and collaboration framework established
- [x] Baseline modulo addition model implementation (from Neel Nanda's grokking experiments)
- [x] Fourier analysis utilities implemented
- [x] 8 structured requirements defined for MVP
- [ ] Implementation in progress

## MVP Goal

**End-to-end workflow for modulo addition grokking analysis:**

1. **Train** a 1-layer Transformer on modulo addition task
2. **Capture** checkpoints at configurable epochs (densify around grokking phase)
3. **Analyze** checkpoints to generate visualization artifacts
4. **Explore** training dynamics through synchronized interactive visualizations:
   - Dominant Fourier frequencies in embedding space
   - Neuron activation patterns over input space
   - Neuron-frequency cluster specialization

**Key Innovation:** Synchronized checkpoint slider across all visualizations to observe correlations - "When frequencies emerge at epoch X, what happens to neuron clusters?"

## Technology Stack

- **Python 3.13** - Core language
- **PyTorch + CUDA** - Training and model execution
- **TransformerLens** - Model architecture (HookedTransformer)
- **Safetensors** - Modern checkpoint persistence
- **Plotly** - Interactive visualizations
- **Gradio** - Dashboard UI (ML-focused)
- **NumPy** - Analysis artifact storage
- **pytest** - Testing framework

## Architecture

### Three-Layer Design

1. **Training Runner** - Execute training runs with configurable checkpoints
2. **Analysis Engine** - Process checkpoints to generate analysis artifacts (cached computations)
3. **Workbench** - Gradio dashboard for training control and interactive visualization

### Artifact-Based Analysis

```
Training → Checkpoints (safetensors) → Analysis (compute once) → Artifacts (disk) → Visualizations (iterate)
```

This separation enables fast iteration on visualizations without re-running expensive forward passes.

## Project Structure

```
/
├── Claude.md                      # Collaboration framework and coding guidelines
├── PROJECT.md                     # Project scope, MVP definition, tech decisions
├── README.md                      # This file
├── requirements/                  # Structured requirements (8 files)
│   ├── README.md                 # Requirements overview
├── policies/                      # Development policies
│   ├── debugging/                # Structured debugging process
│   └── requirements/             # Requirements workflow templates
├── notes/                        # Observations and ideas
└── results/                      # Training outputs (not in repo)
    └── model_p{prime}_seed{seed}/
        ├── checkpoints/          # Model checkpoints (.safetensors)
        ├── artifacts/            # Analysis artifacts (.npz)
        ├── metadata.json         # Training metrics
        └── config.json           # Model configuration
```

## Key Features (MVP)

### Training Runner
- Configurable checkpoint epochs (integer list - densify around grokking)
- Safetensors persistence with immediate disk writes (constant memory)
- Parameterized by modulus (p), seeds, training fraction

### Analysis Engine
- Forward pass computation with activation caching
- Fourier transform analysis of embeddings and activations
- Neuron frequency cluster analysis
- Persistent artifacts (NumPy `.npz` files)
- Resumable analysis (skip existing artifacts)

### Workbench Dashboard
- Training run configuration and execution
- Analysis triggering with progress indication
- **Synchronized checkpoint slider** across all visualizations
- Interactive Plotly visualizations with hover details
- Two-stage neuron exploration (browse trained → select for history)

## Research Hypothesis

**Frequency Emergence During Grokking:**
- Early training shows noise in Fourier space
- Dominant frequencies emerge at specific training phases
- Frequencies may shift before settling into final computational structure
- Neurons specialize to detect specific frequency combinations

The synchronized slider enables manual exploration of these dynamics.

## Getting Started

### Prerequisites
```bash
# Python 3.13 with virtual environment
python --version  # Should be 3.13

# CUDA support for PyTorch (optional but recommended)
nvidia-smi  # Check GPU availability
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd training-dynamics-workbench

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (once requirements.txt is created)
pip install -r requirements.txt
```

### Running the Baseline
```bash
# Currently available: baseline modulo addition training
python ModuloAdditionRefactored.py
```

## Documentation

- [**PROJECT.md**](PROJECT.md) - Detailed project scope, architecture, MVP definition
- [**Claude.md**](Claude.md) - Collaboration framework and coding principles
- [**requirements/**](requirements/README.md) - Structured requirements for MVP implementation
- [**policies/**](policies/) - Development policies (debugging, requirements workflow)

## Collaboration

This project uses structured requirements and development policies for AI-assisted development. See [Claude.md](Claude.md) for collaboration framework.

## Future Enhancements (Post-MVP)

- Multiple model architectures beyond 1-layer Transformers
- Configurable visualization dashboard
- Automatic grokking phase detection
- Intelligent checkpoint frequency based on loss dynamics
- Side-by-side comparison of training runs
- Independent visualization sliders with link/unlink
- AWS deployment for larger models
- Export capabilities for visualizations

## License

[Specify license]

## Acknowledgments

Based on research from:
- Neel Nanda's "A Mechanistic Interpretability Analysis of Grokking"
- TransformerLens library for interpretability research

---

**Status:** Requirements complete, ready for implementation.
**Last Updated:** 2026-01-30
