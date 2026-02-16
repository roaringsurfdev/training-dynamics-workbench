"""Representational Geometry Analyzer.

Computes geometric properties of class manifolds in activation space
at multiple sites in the network. Tracks how representational structure
evolves during training.
"""

from typing import Any

import numpy as np
import torch
from transformer_lens import HookedTransformer
from transformer_lens.ActivationCache import ActivationCache

from miscope.analysis.library import (
    compute_grid_size_from_dataset,
    extract_mlp_activations,
    extract_residual_stream,
)
from miscope.analysis.library.geometry import (
    compute_center_spread,
    compute_circularity,
    compute_class_centroids,
    compute_class_dimensionality,
    compute_class_radii,
    compute_fisher_discriminant,
    compute_fourier_alignment,
)

# Activation sites to probe, with extraction config
_SITES = {
    "resid_pre": {"extractor": "residual", "location": "resid_pre"},
    "attn_out": {"extractor": "residual", "location": "attn_out"},
    "mlp_out": {"extractor": "mlp"},
    "resid_post": {"extractor": "residual", "location": "resid_post"},
}

# Summary keys: 8 scalar measures per site
_SCALAR_KEYS = [
    "mean_radius",
    "mean_dim",
    "center_spread",
    "snr",
    "circularity",
    "fourier_alignment",
    "fisher_mean",
    "fisher_min",
]


def _get_summary_keys() -> list[str]:
    """Build the full list of summary stat keys across all sites."""
    return [f"{site}_{key}" for site in _SITES for key in _SCALAR_KEYS]


class RepresentationalGeometryAnalyzer:
    """Computes geometric properties of class manifolds in activation space.

    For each checkpoint, extracts activations at 4 network sites, groups
    them by output class, and computes per-class and global geometric
    measures including centroids, radii, dimensionality, SNR, circularity,
    Fourier alignment, and Fisher discriminant ratios.
    """

    name = "repr_geometry"
    description = "Tracks representational geometry evolution across training"

    def analyze(
        self,
        model: HookedTransformer,  # noqa: ARG002
        probe: torch.Tensor,
        cache: ActivationCache,
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute geometric measures at all activation sites.

        Args:
            model: The model (unused — activations come from cache)
            probe: Full probe tensor (p^2, 3)
            cache: Activation cache from forward pass
            context: Analysis context with 'params' containing 'prime'

        Returns:
            Dict with site-prefixed keys for centroids, radii,
            dimensionality, and global scalar measures.
        """
        p = compute_grid_size_from_dataset(probe)
        labels = self._compute_labels(probe, p)

        result: dict[str, np.ndarray] = {}
        for site_name, site_config in _SITES.items():
            activations = self._extract_site(cache, site_config)
            site_result = self._compute_site_measures(activations, labels, p)
            for key, value in site_result.items():
                result[f"{site_name}_{key}"] = value

        return result

    def get_summary_keys(self) -> list[str]:
        """Declare summary statistic keys (scalars only)."""
        return _get_summary_keys()

    def compute_summary(
        self,
        result: dict[str, np.ndarray],
        context: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, float | np.ndarray]:
        """Extract scalar summary stats from epoch result.

        Simply picks out the scalar keys — they're already computed
        by analyze().
        """
        summary_keys = _get_summary_keys()
        return {key: float(result[key]) for key in summary_keys}

    def _compute_labels(self, probe: torch.Tensor, p: int) -> np.ndarray:
        """Compute output class labels: (a + b) mod p."""
        probe_np = probe.cpu().numpy()
        a = probe_np[:, 0].astype(int)
        b = probe_np[:, 1].astype(int)
        return (a + b) % p

    def _extract_site(
        self,
        cache: ActivationCache,
        site_config: dict,
    ) -> np.ndarray:
        """Extract activations from cache for a given site."""
        if site_config["extractor"] == "mlp":
            acts = extract_mlp_activations(cache, layer=0, position=-1)
        else:
            acts = extract_residual_stream(
                cache, layer=0, position=-1, location=site_config["location"]
            )
        return acts.detach().cpu().numpy()

    def _compute_site_measures(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        p: int,
    ) -> dict[str, np.ndarray]:
        """Compute all geometric measures for one activation site."""
        centroids = compute_class_centroids(activations, labels, p)
        radii = compute_class_radii(activations, labels, centroids)
        dimensionality = compute_class_dimensionality(
            activations, labels, centroids
        )

        mean_radius = np.mean(radii)
        mean_dim = np.mean(dimensionality)
        center_spread = compute_center_spread(centroids)
        snr = (center_spread**2 / mean_radius**2) if mean_radius > 0 else 0.0
        circularity = compute_circularity(centroids)
        fourier_align = compute_fourier_alignment(centroids, p)
        fisher_mean, fisher_min = compute_fisher_discriminant(
            activations, labels, centroids
        )

        return {
            "centroids": centroids,
            "radii": radii,
            "dimensionality": dimensionality,
            "mean_radius": np.float64(mean_radius),
            "mean_dim": np.float64(mean_dim),
            "center_spread": np.float64(center_spread),
            "snr": np.float64(snr),
            "circularity": np.float64(circularity),
            "fourier_alignment": np.float64(fourier_align),
            "fisher_mean": np.float64(fisher_mean),
            "fisher_min": np.float64(fisher_min),
        }
