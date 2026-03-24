"""Neuron Group PCA Analyzer.

Cross-epoch analyzer that measures within-frequency-group coordination
in weight space. Neurons are grouped by dominant frequency at the final
checkpoint. For each group, tracks PC1 variance explained (alignment)
and mean within-group spread (dispersion) across all training epochs.

High PC1 variance explained → neurons in the group point in similar
directions in weight space (coordinated unit).
Low PC1 variance explained → neurons with the same nominal frequency
are spread across multiple directions (diffuse).
"""

from typing import Any

import numpy as np

from miscope.analysis.artifact_loader import ArtifactLoader


class NeuronGroupPCAAnalyzer:
    """Measures within-frequency-group coordination in weight space.

    Groups neurons by dominant frequency at the final checkpoint, then
    tracks group alignment (PC1 var explained) and dispersion (mean L2
    spread) over all training epochs.

    Cross-epoch artifact keys:
        group_freqs   int32   (n_groups,)           frequency index per group
        group_sizes   int32   (n_groups,)           neuron count per group
        pc1_var       float32 (n_epochs, n_groups)  PC1 variance explained
        mean_spread   float32 (n_epochs, n_groups)  mean L2 distance from centroid
        epochs        int32   (n_epochs,)
    """

    name = "neuron_group_pca"
    requires = ["neuron_freq_norm", "parameter_snapshot"]

    def analyze_across_epochs(
        self,
        artifacts_dir: str,
        epochs: list[int],
        context: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        """Compute group coordination metrics across all checkpoints."""
        loader = ArtifactLoader(artifacts_dir)
        sorted_epochs = sorted(epochs)

        group_freqs, group_members = _assign_groups(loader, sorted_epochs[-1])

        if not group_freqs:
            return _empty_result(sorted_epochs)

        n_groups = len(group_freqs)
        n_epochs = len(sorted_epochs)
        pc1_var = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)
        mean_spread = np.full((n_epochs, n_groups), np.nan, dtype=np.float32)

        for ep_idx, epoch in enumerate(sorted_epochs):
            snap = loader.load_epoch("parameter_snapshot", epoch)
            W_in = snap["W_in"]  # (d_model, d_mlp)
            for g_idx, members in enumerate(group_members):
                pc1, spread = _group_pca_stats(W_in[:, members])
                pc1_var[ep_idx, g_idx] = pc1
                mean_spread[ep_idx, g_idx] = spread

        return {
            "group_freqs": np.array(group_freqs, dtype=np.int32),
            "group_sizes": np.array([len(m) for m in group_members], dtype=np.int32),
            "pc1_var": pc1_var,
            "mean_spread": mean_spread,
            "epochs": np.array(sorted_epochs, dtype=np.int32),
        }


def _assign_groups(
    loader: ArtifactLoader,
    reference_epoch: int,
) -> tuple[list[int], list[np.ndarray]]:
    """Assign neurons to frequency groups using the reference epoch.

    Returns only groups with at least 2 neurons (PCA requires >= 2).
    Group assignment is by argmax of norm_matrix — no threshold applied,
    so all neurons are assigned to exactly one group.

    Returns:
        (group_freqs, group_members): parallel lists of frequency index
        and member neuron indices for each group.
    """
    norm = loader.load_epoch("neuron_freq_norm", reference_epoch)
    norm_matrix = norm["norm_matrix"]  # (n_freq, d_mlp)
    dominant_freq = np.argmax(norm_matrix, axis=0)  # (d_mlp,)

    group_freqs = []
    group_members = []
    for f in range(norm_matrix.shape[0]):
        members = np.where(dominant_freq == f)[0]
        if len(members) >= 2:
            group_freqs.append(f)
            group_members.append(members)

    return group_freqs, group_members


def _group_pca_stats(group_W: np.ndarray) -> tuple[float, float]:
    """Compute PC1 variance explained and mean spread for a neuron group.

    Args:
        group_W: (d_model, n_group) weight vectors for neurons in the group

    Returns:
        (pc1_var_explained, mean_L2_spread)
    """
    centroid = group_W.mean(axis=1, keepdims=True)  # (d_model, 1)
    centered = group_W - centroid  # (d_model, n_group)

    spread = float(np.linalg.norm(centered, axis=0).mean())

    _, s, _ = np.linalg.svd(centered, full_matrices=False)
    total_var = float((s**2).sum())
    pc1_var = float(s[0] ** 2 / total_var) if total_var > 1e-10 else 1.0

    return pc1_var, spread


def _empty_result(epochs: list[int]) -> dict[str, np.ndarray]:
    n = len(epochs)
    return {
        "group_freqs": np.array([], dtype=np.int32),
        "group_sizes": np.array([], dtype=np.int32),
        "pc1_var": np.empty((n, 0), dtype=np.float32),
        "mean_spread": np.empty((n, 0), dtype=np.float32),
        "epochs": np.array(epochs, dtype=np.int32),
    }
