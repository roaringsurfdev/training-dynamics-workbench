"""Modulo Addition 1-Layer family implementation.

This module provides the concrete implementation for the modulo addition
single-layer transformer family, based on Neel Nanda's grokking experiment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import einops
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

from families.json_family import JsonModelFamily


class ModuloAddition1LayerFamily(JsonModelFamily):
    """Implementation of ModelFamily for 1-layer modular addition transformer.

    This family represents single-layer transformers trained on the modular
    addition task: given inputs (a, b), predict (a + b) mod p.

    The model architecture matches Neel Nanda's grokking experiment:
    - 1 layer, 4 heads, d_model=128, d_mlp=512
    - ReLU activation, no layer norm
    - Vocabulary size = p + 1 (0 to p-1 for numbers, p for equals token)

    Domain parameters:
    - prime: The modulus p for the addition task
    - seed: Random seed for model initialization
    """

    def create_model(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> HookedTransformer:
        """Create a HookedTransformer for modular addition.

        Args:
            params: Domain parameters containing 'prime' and optionally 'seed'
            device: Device to place the model on (default: None, uses default device)

        Returns:
            HookedTransformer configured for modular addition
        """
        p = params["prime"]
        seed = params.get("seed", self.get_default_params().get("seed", 999))

        arch = self.architecture

        cfg = HookedTransformerConfig(
            n_layers=arch.get("n_layers", 1),
            n_heads=arch.get("n_heads", 4),
            d_model=arch.get("d_model", 128),
            d_head=arch.get("d_head", 32),
            d_mlp=arch.get("d_mlp", 512),
            act_fn=arch.get("act_fn", "relu"),
            normalization_type=arch.get("normalization_type"),
            d_vocab=p + 1,  # 0 to p-1 for numbers, p for equals token
            d_vocab_out=p,  # Output is 0 to p-1
            n_ctx=arch.get("n_ctx", 3),  # a, b, =
            init_weights=True,
            device=device,
            seed=seed,
        )

        model = HookedTransformer(cfg)

        # Disable biases (matches original experiment)
        for name, param in model.named_parameters():
            if "b_" in name:
                param.requires_grad = False

        return model

    def generate_analysis_dataset(
        self,
        params: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Generate full (a, b) input grid for analysis.

        Creates all p^2 input combinations for modular addition analysis.

        Args:
            params: Domain parameters containing 'prime'
            device: Device to place the dataset on

        Returns:
            Tensor of shape (p^2, 3) containing [a, b, equals_token] rows
        """
        p = params["prime"]

        # Create all (a, b) pairs
        a_vector = einops.repeat(torch.arange(p), "i -> (i j)", j=p)
        b_vector = einops.repeat(torch.arange(p), "j -> (i j)", i=p)
        equals_vector = einops.repeat(
            torch.tensor(p), " -> (i j)", i=p, j=p
        )

        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1)

        if device:
            dataset = dataset.to(device)

        return dataset

    def get_labels(self, params: dict[str, Any]) -> torch.Tensor:
        """Get ground truth labels for the analysis dataset.

        Args:
            params: Domain parameters containing 'prime'

        Returns:
            Tensor of shape (p^2,) containing (a + b) mod p
        """
        p = params["prime"]
        dataset = self.generate_analysis_dataset(params)
        return (dataset[:, 0] + dataset[:, 1]) % p


def load_modulo_addition_1layer_family(
    model_families_dir: Path | str = "model_families",
) -> ModuloAddition1LayerFamily:
    """Load the modulo addition 1-layer family from the standard location.

    Args:
        model_families_dir: Path to model_families directory

    Returns:
        ModuloAddition1LayerFamily instance
    """
    family_json = Path(model_families_dir) / "modulo_addition_1layer" / "family.json"
    return ModuloAddition1LayerFamily.from_json(family_json)
