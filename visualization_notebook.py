# %% imports

from visualization.export import export_variant_visualization

# %% export visualization examples

# Export a specific visualization
export_variant_visualization(
    "results/modulo_addition_1layer/modulo_addition_1layer_p113_seed999",
    "parameter_trajectory",
    output_dir="/tmp/analysis",
)

# Export interactive 3D trajectory as HTML
export_variant_visualization(
    "results/modulo_addition_1layer/modulo_addition_1layer_p101_seed999",
    "trajectory_3d",
    format="html",
)

# %%
