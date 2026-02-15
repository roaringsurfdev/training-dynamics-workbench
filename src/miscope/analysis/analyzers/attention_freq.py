"""Attention Head Frequency Specialization Analyzer.

Computes Fourier frequency decomposition of attention patterns per head,
analogous to neuron_freq_clusters for MLP neurons.
"""

from typing import Any

import einops
import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from miscope.analysis.library import (
    compute_2d_fourier_transform,
    compute_frequency_variance_fractions,
    compute_grid_size_from_dataset,
    extract_attention_patterns,
)


class AttentionFreqAnalyzer:
    """Computes frequency decomposition of attention patterns per head.

    For each checkpoint, extracts the attention pattern for the = → a
    position pair, reshapes to a (n_heads, p, p) grid, applies the modular
    Fourier transform, and computes per-frequency variance fractions.

    Output mirrors neuron_freq_norm: freq_matrix of shape (n_freq, n_heads).
    """

    name = "attention_freq"
    description = "Frequency decomposition of attention patterns per head"

    def __init__(
        self,
        to_position: int = 2,
        from_position: int = 0,
    ):
        self.to_position = to_position
        self.from_position = from_position

    def analyze(
        self,
        model: HookedTransformer,  # noqa: ARG002
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute frequency variance fractions for each attention head.

        Args:
            model: The model loaded with checkpoint weights
            probe: Full probe tensor (p^2, 3)
            cache: Activation cache from forward pass
            context: Analysis context containing 'fourier_basis'

        Returns:
            Dict with 'freq_matrix' array of shape (n_freq, n_heads)
        """
        fourier_basis = context["fourier_basis"]
        p = compute_grid_size_from_dataset(probe)

        # Extract attention patterns: (p*p, n_heads, n_pos, n_pos)
        attn = extract_attention_patterns(cache, layer=0)

        # Select position pair, e.g. = → a: (p*p, n_heads)
        attn_pair = attn[:, :, self.to_position, self.from_position]

        # Reshape batch to (p, p) grid: (n_heads, p, p)
        attn_grid = einops.rearrange(attn_pair, "(a b) h -> h a b", a=p, b=p)

        # 2D Fourier transform: (n_heads, n_comp, n_comp)
        fourier_attn = compute_2d_fourier_transform(attn_grid, fourier_basis)

        # Frequency variance fractions: (n_freq, n_heads)
        freq_matrix = compute_frequency_variance_fractions(fourier_attn, p)

        return {"freq_matrix": freq_matrix.detach().cpu().numpy()}

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys."""
        return [
            "dominant_freq_per_head",
            "max_frac_per_head",
            "mean_specialization",
        ]

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, float | np.ndarray]:
        """Compute summary statistics from this epoch's frequency result.

        Args:
            result: Dict with 'freq_matrix' array of shape (n_freq, n_heads)
            context: Analysis context (unused)

        Returns:
            Dict with per-head dominant frequency, max fraction, and mean specialization
        """
        freq_matrix = result["freq_matrix"]  # (n_freq, n_heads)
        max_frac_per_head = freq_matrix.max(axis=0)  # (n_heads,)
        dominant_freq_per_head = freq_matrix.argmax(axis=0)  # (n_heads,)

        return {
            "dominant_freq_per_head": dominant_freq_per_head.astype(np.float64),
            "max_frac_per_head": max_frac_per_head.astype(np.float64),
            "mean_specialization": float(max_frac_per_head.mean()),
        }
