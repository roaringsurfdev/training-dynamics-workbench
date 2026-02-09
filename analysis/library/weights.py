"""Weight matrix extraction utilities.

Provides functions for extracting and analyzing model weight matrices
across training checkpoints. Used by parameter trajectory and
effective dimensionality analyzers.
"""

import numpy as np
from transformer_lens import HookedTransformer


# Weight matrices to extract, in consistent order
WEIGHT_MATRIX_NAMES = [
    "W_E",
    "W_pos",
    "W_Q",
    "W_K",
    "W_V",
    "W_O",
    "W_in",
    "W_out",
    "W_U",
]

# Predefined component groups for trajectory/velocity analysis
COMPONENT_GROUPS = {
    "embedding": ["W_E", "W_pos", "W_U"],
    "attention": ["W_Q", "W_K", "W_V", "W_O"],
    "mlp": ["W_in", "W_out"],
}


def extract_parameter_snapshot(
    model: HookedTransformer,
) -> dict[str, np.ndarray]:
    """Extract all trainable weight matrices from a model.

    Returns dict mapping weight matrix names to numpy arrays
    in their original shapes. Only includes parameters with
    requires_grad=True (excludes frozen biases).

    Args:
        model: HookedTransformer model at a specific checkpoint.

    Returns:
        Dict with keys from WEIGHT_MATRIX_NAMES, values are numpy arrays.
    """
    snapshot = {}

    snapshot["W_E"] = _to_numpy(model.embed.W_E)
    snapshot["W_pos"] = _to_numpy(model.pos_embed.W_pos)

    block = model.blocks[0]
    snapshot["W_Q"] = _to_numpy(block.attn.W_Q)
    snapshot["W_K"] = _to_numpy(block.attn.W_K)
    snapshot["W_V"] = _to_numpy(block.attn.W_V)
    snapshot["W_O"] = _to_numpy(block.attn.W_O)
    snapshot["W_in"] = _to_numpy(block.mlp.W_in)
    snapshot["W_out"] = _to_numpy(block.mlp.W_out)

    snapshot["W_U"] = _to_numpy(model.unembed.W_U)

    return snapshot


def _to_numpy(tensor) -> np.ndarray:
    """Convert a parameter tensor to numpy, detaching from graph."""
    return tensor.detach().cpu().numpy()
