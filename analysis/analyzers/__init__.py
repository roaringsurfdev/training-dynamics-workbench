"""Individual analysis modules.

Analyzers compute analysis on model checkpoints and produce artifacts.
Each analyzer implements the Analyzer protocol and can be registered
with the AnalyzerRegistry for discovery.
"""

from analysis.analyzers.dominant_frequencies import DominantFrequenciesAnalyzer
from analysis.analyzers.neuron_activations import NeuronActivationsAnalyzer
from analysis.analyzers.neuron_freq_clusters import NeuronFreqClustersAnalyzer
from analysis.analyzers.registry import AnalyzerRegistry

__all__ = [
    "AnalyzerRegistry",
    "DominantFrequenciesAnalyzer",
    "NeuronActivationsAnalyzer",
    "NeuronFreqClustersAnalyzer",
]
