"""Representational geometry computation functions.

Pure numpy functions for computing geometric properties of class manifolds
in activation space. No torch dependency — all inputs are numpy arrays.

Functions:
- compute_class_centroids: Mean activation vector per output class
- compute_class_radii: RMS distance from centroid per class
- compute_class_dimensionality: Effective dimensionality (participation ratio) per class
- compute_center_spread: RMS distance of centroids from global centroid
- compute_circularity: How well centroids lie on a circle (Kåsa circle fit)
- compute_fourier_alignment: Whether angular ordering matches residue class ordering
- compute_fisher_discriminant: Pairwise Fisher discriminant ratio statistics
"""

import numpy as np


def compute_class_centroids(
    activations: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Compute mean activation vector per output class.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        n_classes: Number of distinct classes (p)

    Returns:
        Centroid matrix, shape (n_classes, d)
    """
    d = activations.shape[1]
    centroids = np.zeros((n_classes, d))
    for r in range(n_classes):
        mask = labels == r
        centroids[r] = activations[mask].mean(axis=0)
    return centroids


def compute_class_radii(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute RMS distance from centroid for each class.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Radii array, shape (n_classes,)
    """
    n_classes = centroids.shape[0]
    radii = np.zeros(n_classes)
    for r in range(n_classes):
        mask = labels == r
        diffs = activations[mask] - centroids[r]
        radii[r] = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
    return radii


def compute_class_dimensionality(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Compute effective dimensionality (participation ratio) per class.

    For each class, runs PCA on centered activations and computes:
        D_eff = (sum(eigenvalues))^2 / sum(eigenvalues^2)

    This equals 1 if all variance is on one axis, and d if variance
    is uniform across d dimensions.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Effective dimensionality array, shape (n_classes,)
    """
    n_classes = centroids.shape[0]
    dims = np.zeros(n_classes)
    for r in range(n_classes):
        mask = labels == r
        centered = activations[mask] - centroids[r]
        cov = centered.T @ centered / centered.shape[0]
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        sum_ev = eigenvalues.sum()
        sum_ev_sq = (eigenvalues**2).sum()
        if sum_ev_sq > 0:
            dims[r] = sum_ev**2 / sum_ev_sq
        else:
            dims[r] = 0.0
    return dims


def compute_center_spread(centroids: np.ndarray) -> float:
    """Compute RMS distance of centroids from their global mean.

    Args:
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Center spread (scalar)
    """
    global_centroid = centroids.mean(axis=0)
    diffs = centroids - global_centroid
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))


def compute_circularity(centroids: np.ndarray) -> float:
    """Compute how well centroids lie on a circle in their top-2 PCA subspace.

    Projects centroids into top-2 PCs, fits a circle using the algebraic
    Kåsa method, and returns a score weighted by how much variance the
    top-2 PCs capture:

        raw_score = 1 - (mean_squared_residual / variance_in_2d_plane)
        score = raw_score * variance_explained_ratio

    The weighting ensures that data which is essentially 1D (collinear)
    or high-dimensional (random cloud) scores low even if the 2D
    projection happens to look circular.

    Score of 1.0 means perfect circle, 0.0 means no circular structure.
    Clamped to [0, 1].

    Args:
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Circularity score in [0, 1]
    """
    projected, var_explained = _pca_project_2d(centroids)
    cx, cy, radius = _kasa_circle_fit(projected)
    distances = np.sqrt((projected[:, 0] - cx) ** 2 + (projected[:, 1] - cy) ** 2)
    residuals = distances - radius
    msr = np.mean(residuals**2)
    variance = np.var(projected[:, 0]) + np.var(projected[:, 1])
    if variance < 1e-12:
        return 0.0
    raw_score = 1.0 - msr / variance
    score = raw_score * var_explained
    return float(np.clip(score, 0.0, 1.0))


def compute_fourier_alignment(centroids: np.ndarray, p: int) -> float:
    """Compute whether angular ordering of centroids matches residue class ordering.

    Projects centroids to top-2 PCs, computes angles, then finds the
    frequency k that best explains the angular positions as theta_r = 2*pi*k*r/p.
    Returns R^2 of the best fit.

    Args:
        centroids: Centroid matrix, shape (p, d) where rows are ordered by class
        p: Prime (number of classes, should equal centroids.shape[0])

    Returns:
        Fourier alignment R^2 in [0, 1]
    """
    projected, _ = _pca_project_2d(centroids)
    cx, cy, _ = _kasa_circle_fit(projected)
    angles = np.arctan2(projected[:, 1] - cy, projected[:, 0] - cx)

    residue_indices = np.arange(p)
    best_r2 = 0.0

    for k in range(1, p):
        expected_angles = 2 * np.pi * k * residue_indices / p
        # Fit: angles ≈ expected_angles + offset (mod 2pi)
        # Use circular correlation via complex exponentials
        z_observed = np.exp(1j * angles)
        z_expected = np.exp(1j * expected_angles)
        correlation = np.abs(np.mean(z_observed * np.conj(z_expected))) ** 2
        if correlation > best_r2:
            best_r2 = correlation

    return float(best_r2)


def compute_fisher_discriminant(
    activations: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
) -> tuple[float, float]:
    """Compute pairwise Fisher discriminant ratio statistics.

    For each pair of classes (r, s):
        J(r, s) = ||mu_r - mu_s||^2 / (sigma_r^2 + sigma_s^2)

    where sigma_r^2 is the mean within-class variance for class r.

    Args:
        activations: Activation matrix, shape (n_samples, d)
        labels: Integer class labels, shape (n_samples,)
        centroids: Centroid matrix, shape (n_classes, d)

    Returns:
        Tuple of (mean_fisher, min_fisher) across all class pairs
    """
    n_classes = centroids.shape[0]
    # Compute within-class variance for each class
    variances = np.zeros(n_classes)
    for r in range(n_classes):
        mask = labels == r
        diffs = activations[mask] - centroids[r]
        variances[r] = np.mean(np.sum(diffs**2, axis=1))

    fisher_values = []
    for r in range(n_classes):
        for s in range(r + 1, n_classes):
            between = np.sum((centroids[r] - centroids[s]) ** 2)
            within = variances[r] + variances[s]
            if within > 0:
                fisher_values.append(between / within)
            else:
                fisher_values.append(float("inf"))

    fisher_arr = np.array(fisher_values)
    # Replace inf with finite max for mean computation
    finite_mask = np.isfinite(fisher_arr)
    if not finite_mask.any():
        return 0.0, 0.0
    mean_fisher = float(np.mean(fisher_arr[finite_mask]))
    min_fisher = float(np.min(fisher_arr[finite_mask]))
    return mean_fisher, min_fisher


# --- Private helpers ---


def _pca_project_2d(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Project points into their top-2 principal components.

    Returns a quality score that combines two factors:
    - How much variance the top-2 PCs capture (low for high-D random data)
    - How balanced the two PCs are (low for 1D/collinear data)

    This ensures circular structure is only reported when the data
    genuinely lives in a 2D subspace.

    Args:
        points: Matrix of shape (n, d)

    Returns:
        Tuple of (projected points of shape (n, 2),
                  2D quality score in [0, 1])
    """
    centered = points - points.mean(axis=0)
    cov = centered.T @ centered / centered.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order; take last 2
    top2_vecs = eigenvectors[:, -2:][:, ::-1]
    total_var = eigenvalues.sum()
    if total_var < 1e-12:
        return centered @ top2_vecs, 0.0
    top2_var = eigenvalues[-2:].sum()
    var_explained = top2_var / total_var
    # Balance: ratio of PC2/PC1 variance (1.0 for circle, 0.0 for line)
    ev1 = eigenvalues[-1]
    ev2 = eigenvalues[-2]
    balance = float(ev2 / ev1) if ev1 > 1e-12 else 0.0
    quality = float(var_explained * balance)
    return centered @ top2_vecs, quality


def _kasa_circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to 2D points using the algebraic Kåsa method.

    Solves the least-squares system for a, b, c in:
        x^2 + y^2 + ax + by + c = 0

    Args:
        points: 2D points of shape (n, 2)

    Returns:
        Tuple of (center_x, center_y, radius)
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coeff, c = result
    cx = -a / 2
    cy = -b_coeff / 2
    radius_sq = cx**2 + cy**2 - c
    radius = np.sqrt(max(radius_sq, 0.0))
    return float(cx), float(cy), float(radius)
