"""Analysis pipeline for training dynamics workbench.

This package provides:
- library/: Generic, reusable analysis functions (fourier, activations)
- analyzers/: Family-bound analyzers that compose library functions
- AnalysisPipeline: Orchestrates analysis across checkpoints
- AnalyzerRegistry: Discovers and instantiates analyzers
"""

from analysis.analyzers import AnalyzerRegistry
from analysis.artifact_loader import ArtifactLoader
from analysis.pipeline import AnalysisPipeline
from analysis.protocols import Analyzer

__all__ = [
    "Analyzer",
    "AnalyzerRegistry",
    "AnalysisPipeline",
    "ArtifactLoader",
]
