# %% imports
import sys
import os

parent_dir = os.path.dirname(os.path.dirname(__file__))
sys.path.append(parent_dir)

from tdw import load_family

# %% load model and list variants
family = load_family("modulo_addition_1layer")
variant = family.get_variant(prime=113, seed=999)
model = variant.load_model_at_checkpoint(9000)

# %% analyze probe
probe = variant.make_probe([[3, 29]])
logits, cache = model.run_with_cache(probe)
for param_name, param in cache.items():
    print(param_name, param.shape)
# %%
W_E = model.embed.W_E[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)

# %%
