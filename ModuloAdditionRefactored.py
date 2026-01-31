# Sections of this baseline is taken from the Grokking_demo.ipynb colab at 
# https://colab.research.google.com/github/TransformerLensOrg/TransformerLens/blob/main/demos/Grokking_Demo.ipynb

# %% imports
import os
from pathlib import Path

import numpy as np
import einops

import torch
import transformer_lens.utils as utils

import plotly.io as pio
import plotly.express as px
from neel_plotly.plot import line
from neel_plotly.plot import line as line2

import FourierEvaluation
from ModuloAdditionSpecification import ModuloAdditionSpecification

# %% initialize environment and model
p = 113
device = "cuda" if torch.cuda.is_available() else "cpu"
# Define the location to save the model, using a relative path
MODEL_PATH = os.path.join(os.getcwd(), "results", "grokking_demo_refactored.pth")

# Create the directory if it does not exist
os.makedirs(Path(MODEL_PATH).parent, exist_ok=True)

model_spec = ModuloAdditionSpecification(MODEL_PATH, p, device)

TRAIN_MODEL = False

# %% define the data

train_data, train_labels, test_data, test_labels, train_indices, test_indices = model_spec.generate_training_data()
model = model_spec.create_model()

# %% baseline (pre-training)
pre_train_logits = model(train_data)
pre_train_loss = model_spec.loss_function(pre_train_logits, train_labels)
print(f"Pre-training loss on training data: {pre_train_loss}")
pre_train_test_logits = model(test_data)
pre_train_test_loss = model_spec.loss_function(pre_train_test_logits, test_labels)
print(f"Pre-training loss on test data: {pre_train_test_loss}")
print(f"Uniform loss: {np.log(p)}") # Uniform cross-entropy -log(1/p)


# %% model training
model = model_spec.create_model()

if TRAIN_MODEL:
    model = model_spec.train()
else:
    model = model_spec.load_from_file()

# %% Show model training
pio.renderers.default = "vscode"
pio.templates['plotly'].layout.xaxis.title.font.size = 20
pio.templates['plotly'].layout.yaxis.title.font.size = 20
pio.templates['plotly'].layout.title.font.size = 30


line([model_spec.train_losses[::100], model_spec.test_losses[::100]], x=np.arange(0, len(model_spec.train_losses), 100), xaxis="Epoch", yaxis="Loss", log_y=True, title="Training Curve for Modular Addition", line_labels=['train', 'test'], toggle_x=True, toggle_y=True)

# %% Analyis - helper functions
def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x":xaxis, "y":yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x":xaxis, "y":yaxis, "color":caxis}, **kwargs).show(renderer)

# %% Analysis - run model with dataset
probe_logits, probe_cache = model.run_with_cache(model_spec.dataset)
print(probe_logits.numel())

# %% Analysis - get key weight matrices
W_E = model.embed.W_E[:-1]
print("W_E", W_E.shape)
W_neur = W_E @ model.blocks[0].attn.W_V @ model.blocks[0].attn.W_O @ model.blocks[0].mlp.W_in
print("W_neur", W_neur.shape)
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)

# %% Analysis - Looking at activations
neuron_acts = probe_cache["post", 0, "mlp"][:, -1, :]

# %% Analysis - Show parameter shapes
for param_name, param in probe_cache.items():
    print(param_name, param.shape)

# %% Analysis - Average Attention Pattern per Head
imshow(probe_cache["pattern", 0].mean(dim=0)[:, -1, :], title="Average Attention Pattern per Head", xaxis="Source", yaxis="Head", x=['a', 'b', '='])
# %% Analysis - Average Attention Pattern per Head
imshow(probe_cache["pattern", 0][5][:, -1, :], title="Average Attention Pattern per Head", xaxis="Source", yaxis="Head", x=['a', 'b', '='])
# %% Analysis - Attention for Head 0 from a b -> =
imshow(probe_cache["pattern", 0][:, 0, -1, 0].reshape(p, p), title="Attention for Head 0 from a -> =", xaxis="b", yaxis="a")
# %% Analysis - Attention for Head 0 from a -> =
imshow(
    einops.rearrange(probe_cache["pattern", 0][:, :, -1, 0], "(a b) head -> head a b", a=p, b=p), 
    title="Attention for Head 0 from a b -> =", xaxis="b", yaxis="a", facet_col=0)
# %% Analysis - Neuron activations
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p), 
    title="First 5 neuron activations", xaxis="b", yaxis="a", facet_col=0)

# %% Analysis - Single Value Decomposition
U, S, Vh = torch.svd(W_E)
line(S, title="Singular Values")
imshow(U, title="Principal Components on the Input")

# %% Analysis - Single Value Decomposition with Control: Random Gaussian matrix
U, S, Vh = torch.svd(torch.randn_like(W_E))
line(S, title="Singular Values - Random Control")
imshow(U, title="Principal Components - Random Control")

# %% Explaining the Algorithm - The Embedding is a lookup table
U, S, Vh = torch.svd(W_E)
line(U[:, :8].T, title="Principal Components of the embedding", xaxis="Input Vocabulary")

# %% Explaining the Algorithm - Fourier basis setup
fourier_basis, fourier_basis_names = FourierEvaluation.get_fourier_bases(p,device)
dominant_indices = FourierEvaluation.get_dominant_bases_indices(fourier_basis, W_E)
dominant_bases = FourierEvaluation.get_dominant_bases(fourier_basis, W_E)
print(f"Dominant Bases: {dominant_bases}")
print(f"Dominant Bases (indices): {dominant_indices}")

imshow(fourier_basis, xaxis="Input", yaxis="Component", y=fourier_basis_names)

line2(fourier_basis[:8], xaxis="Input", line_labels=fourier_basis_names[:8], title="First 8 Fourier Components")
line2(fourier_basis[25:29], xaxis="Input", line_labels=fourier_basis_names[25:29], title="Middle Fourier Components")
imshow(fourier_basis @ fourier_basis.T, title="All Fourier Vectors are Orthogonal")

# %% Explaining the Algorithm - Norms of Embedding in Fourier Basis
line2((fourier_basis @ W_E).norm(dim=-1), xaxis="Fourier Component", x=fourier_basis_names, title="Norms of Embedding in Fourier Basis")

# %% Explaining the Algorithm - Key frequencies
key_freq_indices = dominant_indices
key_freqs = dominant_indices[1::2]
print(f"New key_freqs:\n{key_freqs}")

fourier_embed = fourier_basis @ W_E
key_fourier_embed = fourier_embed[key_freq_indices]
print("key_fourier_embed", key_fourier_embed.shape)
imshow(key_fourier_embed @ key_fourier_embed.T, title="Dot Product of embedding of key Fourier Terms")

# %% Explaining the Algorithm - Key frequencies - Cos
key_freq_indices_cos = dominant_indices[1::2]
print(f"Refactored cos frequency indices: {key_freq_indices_cos}")
line2(fourier_basis[key_freq_indices_cos], title="Cos of key freqs", line_labels=key_freq_indices_cos)

# %% Explaining the Algorithm - Constructive Interference
line2(fourier_basis[key_freq_indices_cos].mean(0), title="Constructive Interference")

# %% Analyse Neuron Activations
# Visualization: First 5 neuron activations
imshow(
    einops.rearrange(neuron_acts[:, :5], "(a b) neuron -> neuron a b", a=p, b=p), 
    title="First 5 neuron activations", xaxis="b", yaxis="a", facet_col=0)
# Visualization: First Neuron Activation
imshow(
    einops.rearrange(neuron_acts[:, 0], "(a b) -> a b", a=p, b=p), 
    title="First neuron act", xaxis="b", yaxis="a",)

# %% Analyse plain frequency heat maps
# Visualization: Cos a * Cos b
for i in range(len(key_freqs)):
    fourier_basis_index = key_freq_indices_cos[i]
    fourer_basis_frequency = key_freqs[i]//2
    imshow(fourier_basis[fourier_basis_index][None, :] * fourier_basis[fourier_basis_index][:, None], title=f"Cos {fourer_basis_frequency}a * cos {fourer_basis_frequency}b")
# Visualization: Cos a * Constant
for i in range(len(key_freqs)):
    fourier_basis_index = key_freq_indices_cos[i]
    fourer_basis_frequency = key_freqs[i]//2
    imshow(fourier_basis[fourier_basis_index][None, :] * fourier_basis[0][:, None], title=f"Cos {fourer_basis_frequency}a * const")

# %% Analysis - 2D Fourier Transform of neurons 0-4
for i in range(5):
    imshow(fourier_basis @ neuron_acts[:, i].reshape(p, p) @ fourier_basis.T, title=f"2D Fourier Transformer of neuron {i}", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)
# %% Analysis - 2D Fourier Transform of random control
imshow(fourier_basis @ torch.randn_like(neuron_acts[:, 0]).reshape(p, p) @ fourier_basis.T, title="2D Fourier Transformer of RANDOM", xaxis="b", yaxis="a", x=fourier_basis_names, y=fourier_basis_names)

# %% Analysis - Neuron clusters
fourier_neuron_acts = fourier_basis @ einops.rearrange(neuron_acts, "(a b) neuron -> neuron a b", a=p, b=p) @ fourier_basis.T
# Center these by removing the mean - doesn't matter!
fourier_neuron_acts[:, 0, 0] = 0.
print("fourier_neuron_acts", fourier_neuron_acts.shape)
neuron_freq_norm = torch.zeros(p//2, model.cfg.d_mlp).to(device)
for freq in range(0, p//2):
    for x in [0, 2*(freq+1) - 1, 2*(freq+1)]:
        for y in [0, 2*(freq+1) - 1, 2*(freq+1)]:
            neuron_freq_norm[freq] += fourier_neuron_acts[:, x, y]**2

neuron_freq_norm = neuron_freq_norm / fourier_neuron_acts.pow(2).sum(dim=[-1, -2])[None, :]
# Visualization: Neuron Frac Explained by Freq
imshow(neuron_freq_norm, xaxis="Neuron", yaxis="Freq", y=torch.arange(1, p//2+1), title="Neuron Frac Explained by Freq")

# Visualization: Max Neuron Frac Explained over Freqs
line2(neuron_freq_norm.max(dim=0).values.sort().values, xaxis="Neuron", title="Max Neuron Frac Explained over Freqs")
# %% Analysis - Read Off the Neuron-Logit Weights to Interpret
W_logit = model.blocks[0].mlp.W_out @ model.unembed.W_U
print("W_logit", W_logit.shape)
line2((W_logit @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title="W_logit in the Fourier Basis")
# %%
for i in range(len(key_freqs)):
    frequency = key_freqs[i]//2
    neurons = neuron_freq_norm[frequency-1]>0.85
    line2((W_logit[neurons] @ fourier_basis.T).norm(dim=0), x=fourier_basis_names, title=f"W_logit for freq {frequency} neurons in the Fourier Basis")
# %%
for i in range(len(key_freqs)):
    frequency = key_freqs[i]
    sin_frequency = frequency - 1
    W_logit_fourier = W_logit @ fourier_basis
    neurons_sin_i = W_logit_fourier[:, sin_frequency]
    line2(neurons_sin_i, title=f"W_logit @ fourier_basis for Frequency {frequency//2}")
# %% Analysis - Fourier heatmap over inputs for sin9c
freq = 55
neurons_sin_9 = W_logit_fourier[:, 2*freq-1]
inputs_sin_9c = neuron_acts @ neurons_sin_9
imshow(fourier_basis @ inputs_sin_9c.reshape(p, p) @ fourier_basis.T, title="Fourier Heatmap over inputs for sin17c", x=fourier_basis_names, y=fourier_basis_names)


# %%
