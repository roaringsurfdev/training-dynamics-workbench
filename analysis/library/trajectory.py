"""Parameter trajectory analysis utilities.

Provides PCA projection and velocity computation for parameter space
trajectory visualization across training checkpoints.
"""

import numpy as np
from sklearn.decomposition import PCA

from analysis.library.weights import WEIGHT_MATRIX_NAMES


def flatten_snapshot(
    snapshot: dict[str, np.ndarray],
    components: list[str] | None = None,
) -> np.ndarray:
    """Flatten selected weight matrices into a single parameter vector.

    Args:
        snapshot: Per-epoch artifact dict from ParameterSnapshotAnalyzer.
        components: Weight matrix names to include. None = all.

    Returns:
        1D array of concatenated, flattened parameters.
    """
    if components is None:
        components = [k for k in WEIGHT_MATRIX_NAMES if k in snapshot]

    parts = [snapshot[k].flatten() for k in components if k in snapshot]
    return np.concatenate(parts)


def compute_pca_trajectory(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
    n_components: int = 3,
) -> dict[str, np.ndarray]:
    """Compute PCA projection of parameter trajectory.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        components: Weight matrix names to include. None = all.
        n_components: Number of principal components.

    Returns:
        Dict with:
          "projections": (n_epochs, n_components) in PC space
          "explained_variance_ratio": (n_components,) fraction per PC
          "explained_variance": (n_components,) eigenvalues
    """
    vectors = np.array([flatten_snapshot(s, components) for s in snapshots])

    n_components = min(n_components, len(snapshots), vectors.shape[1])
    pca = PCA(n_components=n_components)
    projections = pca.fit_transform(vectors)

    return {
        "projections": projections,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance": pca.explained_variance_,
    }


def compute_parameter_velocity(
    snapshots: list[dict[str, np.ndarray]],
    components: list[str] | None = None,
) -> np.ndarray:
    """Compute L2 norm of parameter change between consecutive epochs.

    Args:
        snapshots: List of per-epoch snapshot dicts, ordered by epoch.
        components: Weight matrix names to include. None = all.

    Returns:
        1D array of length (n_epochs - 1). velocity[i] = ||theta_{i+1} - theta_i||
    """
    vectors = [flatten_snapshot(s, components) for s in snapshots]

    velocities = []
    for i in range(len(vectors) - 1):
        delta = vectors[i + 1] - vectors[i]
        velocities.append(float(np.linalg.norm(delta)))

    return np.array(velocities)
