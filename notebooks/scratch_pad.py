# %% imports
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from tdw import load_family

# %% load model and list variants
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)
#model = variant.load_model_at_checkpoint(9000)

# %% analyze probe
probe = variant.make_probe([[3, 29]])
logits, cache = variant.run_with_cache(probe, 9000)
# %%