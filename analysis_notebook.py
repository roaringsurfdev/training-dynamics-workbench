# %% Import
import os
import numpy as np
from pathlib import Path

from analysis import AnalysisPipeline, ArtifactLoader
from analysis.analyzers import (
    DominantFrequenciesAnalyzer,
    NeuronActivationsAnalyzer,
    NeuronFreqClustersAnalyzer,
    CoarsenessAnalyzer,
)
from families import FamilyRegistry, TrainingResult

import plotly.express as px
import pandas as pd

# %% Compute Ideal Fourier Spectrum

def compute_ideal_fourier_spectrum(p):
    """
    Compute the theoretical Fourier spectrum for f(a,b) = (a+b) mod p
    """
    # Create the addition table
    addition_table = np.zeros((p, p))
    for a in range(p):
        for b in range(p):
            addition_table[a, b] = (a + b) % p
    
    # Compute 2D FFT
    fourier = np.fft.fft2(addition_table)
    power_spectrum = np.abs(fourier)**2
    
    # Normalize
    power_spectrum = power_spectrum / power_spectrum.sum()
    
    return power_spectrum

# For your primes:
for p in [97, 101, 109, 113]:
    spectrum = compute_ideal_fourier_spectrum(p)
    # Find dominant components
    threshold = 0.01  # 1% of total power
    dominant_components = np.argwhere(spectrum > threshold * spectrum.max())
    print(f"Prime {p}: Dominant (k,ℓ) components:")
    for k, l in dominant_components[:10]:  # Show top 10
        print(f"  ({k}, {l}): {spectrum[k,l]:.4f}")
# %% Analysis Pipeline Methods
_registry: FamilyRegistry | None = None


def get_registry() -> FamilyRegistry:
    """Get or create the global FamilyRegistry instance."""
    global _registry
    if _registry is None:
        _registry = FamilyRegistry(
            model_families_dir=Path("model_families"),
            results_dir=Path("results"),
        )
    return _registry

def run_analysis_for_variant(
    variant_name: str,
    family_name: str,
):
    """Run analysis pipeline on the selected variant."""
    # Use family_name from dropdown (fixes issue where state.selected_family_name is None on page load)

    try:
        print("Initializing analysis...")

        registry = get_registry()
        family = registry.get_family(family_name)
        variants = registry.get_variants(family)

        # Find the selected variant
        variant = None
        for v in variants:
            if v.name == variant_name:
                variant = v
                break

        if variant is None:
            print("Variant not found")
            return

        print("Starting analysis pipeline...")

        def pipeline_progress(pct: float, desc: str):
            ui_progress = 0.1 + (pct * 0.9)
            print(f"{ui_progress}, desc={desc}")

        # Pipeline now takes Variant directly (no adapter needed)
        pipeline = AnalysisPipeline(variant)
        pipeline.register(CoarsenessAnalyzer())
        pipeline.run(progress_callback=pipeline_progress)

        print(f"Analysis complete! Artifacts saved to {variant.artifacts_dir}")

    except Exception as e:
        import traceback

        print(f"Analysis failed: {e}\n\n{traceback.format_exc()}")

# %% Test CoarsenessAnalyzer
model_family_name = "modulo_addition_1layer"
model_variant_p_values = [97, 101, 109, 113]
model_variant_seed_values = [485, 999]
run_analysis = False
analyzer_name = "coarseness"

for p in model_variant_p_values:
    for seed in model_variant_seed_values:
        model_variant_name = f"{model_family_name}_p{p}_seed{seed}"
        if run_analysis:
            run_analysis_for_variant(family_name=model_family_name, variant_name=model_variant_name)

        summary_data_path = os.path.join(os.getcwd(), f"results/{model_family_name}/{model_variant_name}/artifacts/{analyzer_name}/summary.npz")
        with np.load(summary_data_path, allow_pickle=False) as data_from_file:
            data = {
                'epoch': data_from_file['epochs'],
                'mean_coarseness': data_from_file['mean_coarseness'],
                'std_coarseness': data_from_file['std_coarseness'],
                'p25_coarseness': data_from_file['p25_coarseness'],
            }
            df = pd.DataFrame(data)

        fig = px.line(df, x='epoch', y='mean_coarseness', title=f"Mean Coarseness Across Epochs<br>{model_variant_name}")
        fig.show()

        fig = px.line(df, x='epoch', y='std_coarseness', title=f"std Coarseness Across Epochs<br>{model_variant_name}")
        fig.show()

        fig = px.line(df, x='epoch', y='p25_coarseness', title=f"p25 Coarseness Across Epochs<br>{model_variant_name}")
        fig.show()

# %%
