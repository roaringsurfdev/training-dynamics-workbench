# %% imports
import sys
import os

import numpy as np

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from miscope import load_family
from miscope import visualization
from miscope.visualization.export import export_animation

# %% create animation method
def create_animation(prime, seed, analyzer_name, render_fn, render_fn_name):
    family = load_family("modulo_addition_1layer")
    variant = family.get_variant(prime=prime, seed=seed)
    checkpoints = variant.get_available_checkpoints()
    artifacts_dir = variant.artifacts_dir
    analyzer_name = analyzer_name
    output_path = os.path.join("animations", f"{variant.name}_{render_fn_name}.gif")
    animation_path = export_animation(
        render_fn=render_fn, 
        artifacts_dir=artifacts_dir, 
        analyzer_name=analyzer_name,
        output_path=output_path,
        fps=2)

# %% load model and list variants
render_fn = visualization.render_centroid_pca
render_fn_name = "centroid_pca"
analyzer_name = "repr_geometry"
create_animation(109, 485, analyzer_name, render_fn, render_fn_name)
# %%
