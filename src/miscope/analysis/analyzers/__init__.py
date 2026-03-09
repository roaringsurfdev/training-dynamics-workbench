"""Individual analysis modules.

Analyzers compute analysis on model checkpoints and produce artifacts.
Each analyzer implements the Analyzer protocol and can be registered
with the AnalyzerRegistry for discovery.
"""

# Analyzer Registry
# Per-epoch Analyzers with no dependences
from miscope.analysis.analyzers.attention_freq import AttentionFreqAnalyzer
from miscope.analysis.analyzers.attention_patterns import AttentionPatternsAnalyzer
from miscope.analysis.analyzers.dominant_frequencies import DominantFrequenciesAnalyzer
from miscope.analysis.analyzers.effective_dimensionality import EffectiveDimensionalityAnalyzer
from miscope.analysis.analyzers.fourier_frequency_quality import FourierFrequencyQualityAnalyzer
from miscope.analysis.analyzers.landscape_flatness import LandscapeFlatnessAnalyzer
from miscope.analysis.analyzers.neuron_activations import NeuronActivationsAnalyzer
from miscope.analysis.analyzers.neuron_dynamics import NeuronDynamicsAnalyzer
from miscope.analysis.analyzers.attention_fourier import AttentionFourierAnalyzer
from miscope.analysis.analyzers.parameter_snapshot import ParameterSnapshotAnalyzer

# Secondary Analyzers
from miscope.analysis.analyzers.neuron_fourier import NeuronFourierAnalyzer
from miscope.analysis.analyzers.neuron_freq_clusters import NeuronFreqClustersAnalyzer

# Cross-epoch Analyzers
from miscope.analysis.analyzers.parameter_trajectory_pca import ParameterTrajectoryPCA
from miscope.analysis.analyzers.registry import AnalyzerRegistry
from miscope.analysis.analyzers.repr_geometry import RepresentationalGeometryAnalyzer
from miscope.analysis.analyzers.global_centroid_pca import GlobalCentroidPCA
from miscope.analysis.analyzers.centroid_dmd import CentroidDMD

__all__ = [
    "AnalyzerRegistry",
    "AttentionFourierAnalyzer",
    "AttentionFreqAnalyzer",
    "FourierFrequencyQualityAnalyzer",
    "NeuronFourierAnalyzer",
    "AttentionPatternsAnalyzer",
    "DominantFrequenciesAnalyzer",
    "EffectiveDimensionalityAnalyzer",
    "LandscapeFlatnessAnalyzer",
    "NeuronActivationsAnalyzer",
    "NeuronDynamicsAnalyzer",
    "NeuronFreqClustersAnalyzer",
    "ParameterSnapshotAnalyzer",
    "ParameterTrajectoryPCA",
    "RepresentationalGeometryAnalyzer",
    "GlobalCentroidPCA",
    "CentroidDMD"
]
