# REQ_021b: Analysis Library Architecture

**Status:** Active
**Priority:** High
**Parent:** [REQ_021](REQ_021_model_families.md)
**Dependencies:** None (can be worked in parallel with REQ_021a)
**Last Updated:** 2026-02-03

## Problem Statement

Current analysis code is tightly coupled to `ModuloAdditionSpecification`. To support multiple model families, we need:
1. A library of generic, reusable analysis functions
2. Family-bound analyzers that compose library functions
3. Clear separation so new families can reuse existing analysis logic

## Scope

This sub-requirement covers:
1. Creating `analysis/library/` with generic functions
2. Creating `analysis/analyzers/` with composable analyzers
3. Migrating existing analysis code to this structure
4. Defining the Analyzer protocol

## Proposed Design

### Directory Structure

```
analysis/
  __init__.py
  library/                   # Generic, reusable functions
    __init__.py
    fourier.py              # Modular Fourier basis, FFT utilities
    activations.py          # Gradient magnitude, coarseness metrics
    attention.py            # Attention pattern analysis
  analyzers/                 # Family-bound analyzers
    __init__.py
    base.py                 # Analyzer protocol
    dominant_frequencies.py
    neuron_activations.py
    neuron_frequency_clusters.py
```

### Library Functions

**fourier.py** - Modular Fourier analysis utilities:
```python
def get_modular_fourier_basis(p: int, device: torch.device) -> torch.Tensor:
    """Generate Fourier basis for modular arithmetic with period p."""
    ...

def compute_fft_2d(tensor: torch.Tensor) -> torch.Tensor:
    """Compute 2D FFT for activation analysis."""
    ...

def project_onto_fourier_basis(
    weights: torch.Tensor,
    basis: torch.Tensor
) -> torch.Tensor:
    """Project weight matrix onto Fourier basis."""
    ...
```

**activations.py** - Activation analysis utilities:
```python
def compute_neuron_activations(
    model: HookedTransformer,
    inputs: torch.Tensor,
    layer: int
) -> torch.Tensor:
    """Extract MLP neuron activations for given inputs."""
    ...

def compute_activation_grid(
    activations: torch.Tensor,
    grid_shape: tuple[int, int]
) -> torch.Tensor:
    """Reshape activations into (a, b) grid format."""
    ...
```

### Analyzer Protocol

```python
from typing import Protocol, Any
from pathlib import Path
import torch
from transformer_lens import HookedTransformer

class AnalysisResult(TypedDict):
    """Standard structure for analyzer output."""
    data: Any  # The computed analysis data
    metadata: dict[str, Any]  # Computation parameters, timestamps, etc.

class Analyzer(Protocol):
    """Protocol for family-bound analyzers."""

    @property
    def name(self) -> str:
        """Unique identifier for this analyzer."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description."""
        ...

    def analyze(
        self,
        model: HookedTransformer,
        analysis_dataset: torch.Tensor,
        params: dict[str, Any]
    ) -> AnalysisResult:
        """Run analysis and return results."""
        ...

    def save_artifact(
        self,
        result: AnalysisResult,
        artifacts_dir: Path,
        checkpoint_epoch: int
    ) -> Path:
        """Persist analysis result to disk."""
        ...

    def load_artifact(
        self,
        artifacts_dir: Path,
        checkpoint_epoch: int
    ) -> AnalysisResult:
        """Load previously computed result."""
        ...
```

### Analyzer Implementations

**dominant_frequencies.py:**
```python
class DominantFrequenciesAnalyzer:
    """Analyzes which Fourier frequencies dominate the embedding space."""

    name = "dominant_frequencies"
    description = "Identifies dominant frequencies in learned embeddings"

    def analyze(self, model, analysis_dataset, params):
        # Uses library/fourier.py functions
        basis = get_modular_fourier_basis(params["prime"], model.device)
        embedding = model.W_E
        coefficients = project_onto_fourier_basis(embedding, basis)
        norms = coefficients.norm(dim=-1)
        ...
```

**neuron_activations.py:**
```python
class NeuronActivationsAnalyzer:
    """Analyzes MLP neuron activation patterns over input grid."""

    name = "neuron_activations"
    description = "Computes neuron activation heatmaps for (a, b) inputs"

    def analyze(self, model, analysis_dataset, params):
        # Uses library/activations.py functions
        activations = compute_neuron_activations(model, analysis_dataset, layer=0)
        grid = compute_activation_grid(activations, (params["prime"], params["prime"]))
        ...
```

### AnalyzerRegistry

```python
class AnalyzerRegistry:
    """Registry of available analyzers."""

    _analyzers: dict[str, type[Analyzer]] = {}

    @classmethod
    def register(cls, analyzer_class: type[Analyzer]) -> type[Analyzer]:
        """Decorator to register an analyzer."""
        cls._analyzers[analyzer_class.name] = analyzer_class
        return analyzer_class

    @classmethod
    def get(cls, name: str) -> Analyzer:
        """Get analyzer instance by name."""
        return cls._analyzers[name]()

    @classmethod
    def get_for_family(cls, family: ModelFamily) -> list[Analyzer]:
        """Get all analyzers valid for a family."""
        return [cls.get(name) for name in family.analyzers]
```

## Conditions of Satisfaction

- [ ] `analysis/library/` contains extracted generic functions
- [ ] `analysis/analyzers/` contains family-bound analyzers
- [ ] `Analyzer` protocol defined with analyze/save/load methods
- [ ] `AnalyzerRegistry` can discover and instantiate analyzers
- [ ] Existing Fourier analysis code migrated to `library/fourier.py`
- [ ] Existing activation analysis code migrated to `library/activations.py`
- [ ] At least 3 analyzers implemented (dominant_frequencies, neuron_activations, neuron_frequency_clusters)
- [ ] Analyzers can be instantiated and run independently of dashboard
- [ ] Unit tests for library functions

## Constraints

**Must have:**
- Clear separation: library functions have no family knowledge
- Analyzers compose library functions, don't duplicate logic
- Artifact format supports incremental loading (don't load all checkpoints to view one)

**Must avoid:**
- Library functions depending on specific family implementations
- Analyzers with hardcoded model architecture assumptions

**Flexible:**
- Artifact storage format (numpy, torch, pickle, etc.)
- Whether analyzers are classes or functions with protocol wrapper

## Migration Notes

Code to extract from existing implementation:
- `FourierEvaluation.py` → `library/fourier.py`
- Activation extraction from `ModuloAdditionRefactored.py` → `library/activations.py`
- Visualization-specific analysis → respective analyzer modules

The goal is extraction and reorganization, not rewriting. Preserve working logic while establishing clean boundaries.
