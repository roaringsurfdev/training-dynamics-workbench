"""Shape characterization primitives.

Per REQ_106 / REQ_109 layering rule: characterizations are *measures*, not
*derivations*. Each function takes an already-projected ndarray (typically
a 2D PCA projection or a curve in some coordinate system) and returns a
typed result describing the shape. Projection / coordinate-frame
materialization is the caller's responsibility — never re-derived inside
a measure.

Point-cloud shape characterizations (caller passes a 2D/3D PCA projection):
- characterize_circularity: how well projected points lie on a circle.
- characterize_fourier_alignment: angular ordering matches residue-class ordering.
- characterize_surface: quadratic-surface fit — flat / bowl / saddle classification.

Curve-shape characterizations (caller passes a 2D/3D curve in coordinates):
- compute_arc_length: cumulative arc-length along a curve.
- detect_self_intersection: locate the lemniscate node — the closest self-approach.
- compute_signed_loop_area: signed area enclosed by a loop segment.
- compute_curvature_profile: signed curvature κ(s) along a 2D curve.

Trajectory-dynamics characterizations (caller passes a time-ordered curve):
- characterize_jerk: third time-derivative; spikes at regime transitions.

Periodic-trajectory shape characterizations (caller passes a 2D trajectory
parameterized over a periodic domain):
- characterize_lissajous: dominant per-axis Fourier mode + joint structural
  metrics (frequency match, amplitude ratio, phase offset, residual).
"""

from typing import NamedTuple

import numpy as np

from miscope.analysis.library.fourier_basis import (
    get_fourier_basis,
    project_onto_fourier_basis,
)


class SurfaceParameters(NamedTuple):
    """Quadratic-surface fit result describing whether a 3D point cloud is
    flat, bowl-shaped, or saddle-shaped.

    Fields:
        r2_linear: R² of a planar fit ``z = d·x + e·y + f`` — captures tilt only.
        r2_quadratic: R² of the full quadratic fit ``z = a·x² + b·y² + c·x·y + d·x + e·y + f``.
        r2_curvature: ``r2_quadratic − r2_linear`` clamped to [0, 1]. Variance
            explained by the quadratic terms net of any planar tilt.
        a: x² coefficient.
        b: y² coefficient.
        c: x·y coefficient.
        shape: ``"flat/blob"``, ``"bowl"``, or ``"saddle"`` per the Hessian
            determinant ``4ab − c²``.
    """

    r2_linear: float
    r2_quadratic: float
    r2_curvature: float
    a: float
    b: float
    c: float
    shape: str


def characterize_circularity(projection_2d: np.ndarray, var_explained: float) -> float:
    """Score how well 2D-projected points lie on a circle.

    Fits a circle via the algebraic Kåsa method, then weights the residual-
    based fit score by two factors:

    - ``var_explained`` (caller-supplied): how much of the underlying total
      variance the 2D projection captures. Discounts cases where the data
      is high-dimensional and the 2D projection is just an arbitrary slice.
    - *balance* (computed from the projection): how comparable the variances
      along the two PCs are. Discounts cases where the projection is
      essentially 1D (collinear) — for a true circle the two PCs have
      similar variance; for a line, the second is near zero.

    Score of 1.0 = perfect circle; 0.0 = no circular structure. Clamped to [0, 1].

    Args:
        projection_2d: ``(n, 2)`` data projected onto top-2 principal components.
        var_explained: Fraction of total variance captured by the 2D projection
            (typically ``sum(explained_variance_ratio[:2])``).

    Returns:
        Circularity score in ``[0, 1]``.
    """
    var_x = float(np.var(projection_2d[:, 0]))
    var_y = float(np.var(projection_2d[:, 1]))
    variance = var_x + var_y
    if variance < 1e-12 or var_x < 1e-12:
        return 0.0
    balance = var_y / var_x

    cx, cy, radius = _kasa_circle_fit(projection_2d)
    distances = np.sqrt((projection_2d[:, 0] - cx) ** 2 + (projection_2d[:, 1] - cy) ** 2)
    residuals = distances - radius
    msr = float(np.mean(residuals**2))
    raw_score = 1.0 - msr / variance
    score = raw_score * var_explained * balance
    return float(np.clip(score, 0.0, 1.0))


def characterize_fourier_alignment(projection_2d: np.ndarray, p: int) -> float:
    """Score whether angular ordering of 2D-projected points matches residues.

    Computes angles around the projection's fitted-circle center, then
    finds the frequency ``k`` that best explains the angular positions
    as ``theta_r = 2*pi*k*r/p``. Returns ``R^2`` of the best fit across
    all candidate frequencies in ``1..p-1``.

    Args:
        projection_2d: ``(p, 2)`` data projected onto top-2 principal
            components, ordered by class index.
        p: Number of classes (prime). Must equal ``projection_2d.shape[0]``.

    Returns:
        Fourier alignment ``R^2`` in ``[0, 1]``.
    """
    cx, cy, _ = _kasa_circle_fit(projection_2d)
    angles = np.arctan2(projection_2d[:, 1] - cy, projection_2d[:, 0] - cx)

    z_observed = np.exp(1j * angles)
    k_values = np.arange(1, p)
    residue_indices = np.arange(p)
    expected = 2 * np.pi * k_values[:, np.newaxis] * residue_indices[np.newaxis, :] / p
    z_expected = np.exp(1j * expected)
    correlations = np.abs(np.mean(z_observed[np.newaxis, :] * np.conj(z_expected), axis=1)) ** 2
    return float(np.max(correlations))


def _kasa_circle_fit(points: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to 2D points via the algebraic Kåsa method.

    Solves the least-squares system for ``a, b, c`` in
    ``x^2 + y^2 + a*x + b*y + c = 0``.

    Args:
        points: ``(n, 2)`` array of 2D coordinates.

    Returns:
        Tuple ``(center_x, center_y, radius)``.
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


# ---------------------------------------------------------------------------
# Curve-shape characterizations (lemniscate analysis)
# ---------------------------------------------------------------------------


def compute_arc_length(curve: np.ndarray) -> np.ndarray:
    """Cumulative arc-length along a 2D or 3D curve.

    Args:
        curve: ``(n_points, d)`` array of coordinates.

    Returns:
        ``(n_points,)`` cumulative arc-length, starting at 0.
    """
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def detect_self_intersection(
    curve: np.ndarray,
    min_arc_sep_fraction: float = 0.15,
) -> dict:
    """Find the closest self-approach in a 2D curve — the lemniscate node.

    Scans all pairs of non-adjacent points separated by at least
    ``min_arc_sep_fraction * total_arc_length`` of arc, returning the pair
    with minimum Euclidean distance. This localizes the node where the curve
    crosses itself (or approaches itself most closely).

    Args:
        curve: ``(n_points, 2)`` array of 2D coordinates.
        min_arc_sep_fraction: Minimum arc-length gap between candidate pairs
            as a fraction of total arc length. Filters out adjacent segments.
            Default 0.15 (15% of total arc).

    Returns:
        Dict with:
            ``node_position``: ``(2,)`` midpoint of the closest-approach pair.
            ``idx_pair``: ``(i, j)`` index pair of the closest approach.
            ``min_distance``: float — Euclidean distance at closest approach.
            ``arc_length``: ``(n_points,)`` cumulative arc-length array.
    """
    arc = compute_arc_length(curve)
    total_arc = arc[-1]
    min_sep = min_arc_sep_fraction * total_arc

    n = len(curve)
    diff = curve[:, np.newaxis, :] - curve[np.newaxis, :, :]  # (n, n, 2)
    dist_sq = np.sum(diff**2, axis=2)  # (n, n)

    arc_sep = np.abs(arc[:, np.newaxis] - arc[np.newaxis, :])  # (n, n)
    valid = (arc_sep >= min_sep) & np.triu(np.ones((n, n), dtype=bool), k=1)

    if not valid.any():
        mid = n // 2
        return {
            "node_position": curve[mid].copy(),
            "idx_pair": (0, n - 1),
            "min_distance": float("inf"),
            "arc_length": arc,
        }

    dist_sq_masked = np.where(valid, dist_sq, np.inf)
    flat_idx = int(np.argmin(dist_sq_masked))
    i, j = divmod(flat_idx, n)

    return {
        "node_position": (curve[i] + curve[j]) / 2.0,
        "idx_pair": (int(i), int(j)),
        "min_distance": float(np.sqrt(dist_sq[i, j])),
        "arc_length": arc,
    }


def compute_signed_loop_area(
    curve: np.ndarray,
    idx_pair: tuple[int, int],
) -> float:
    """Signed area of the loop enclosed between two self-intersection indices.

    Extracts ``curve[i:j+1]`` and applies the shoelace formula. Sign reflects
    traversal direction: positive = CCW, negative = CW. Overshooters produce
    a larger-magnitude area; never-arrives produce ~0 (loop not completed).

    Args:
        curve: ``(n_points, 2)`` array of 2D coordinates.
        idx_pair: ``(i, j)`` indices from :func:`detect_self_intersection`.

    Returns:
        Signed area of the enclosed loop. Near zero if no loop is completed.
    """
    i, j = idx_pair
    loop = curve[i : j + 1]
    if len(loop) < 3:
        return 0.0
    x, y = loop[:, 0], loop[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def compute_curvature_profile(
    curve: np.ndarray,
    n_norm_points: int = 100,
) -> dict:
    """Signed curvature κ as a function of normalized arc-length.

    Uses the discrete Frenet-Serret formula::

        κ = (x' · y'' − y' · x'') / (x'² + y'²)^(3/2)

    Derivatives are computed via :func:`numpy.gradient` (central differences)
    using the arc-length parameterization so the result is intrinsic to curve
    shape rather than epoch spacing. The profile is then resampled onto a
    uniform normalized arc-length grid ``[0, 1]`` for cross-variant comparison.

    Args:
        curve: ``(n_points, 2)`` array of 2D coordinates.
        n_norm_points: Number of output samples on the normalized grid.

    Returns:
        Dict with:
            ``s_norm``: ``(n_norm_points,)`` normalized arc-length in ``[0, 1]``.
            ``kappa``: ``(n_norm_points,)`` curvature at each ``s_norm`` sample.
            ``s_raw``: ``(n_points,)`` raw cumulative arc-length.
            ``kappa_raw``: ``(n_points,)`` curvature at each original point.
    """
    arc = compute_arc_length(curve)
    total_arc = arc[-1]

    x, y = curve[:, 0], curve[:, 1]

    dx = np.gradient(x, arc)
    dy = np.gradient(y, arc)
    d2x = np.gradient(dx, arc)
    d2y = np.gradient(dy, arc)

    denom = (dx**2 + dy**2) ** 1.5
    kappa_raw = np.where(denom > 1e-12, (dx * d2y - dy * d2x) / denom, 0.0)

    s_norm_raw = arc / total_arc if total_arc > 1e-12 else np.linspace(0.0, 1.0, len(arc))
    s_norm = np.linspace(0.0, 1.0, n_norm_points)
    kappa = np.interp(s_norm, s_norm_raw, kappa_raw)

    return {
        "s_norm": s_norm,
        "kappa": kappa,
        "s_raw": arc,
        "kappa_raw": kappa_raw,
    }


# ---------------------------------------------------------------------------
# Trajectory-dynamics characterizations
# ---------------------------------------------------------------------------


def characterize_jerk(
    trajectory: np.ndarray,
    time_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Per-timestep jerk: the third time-derivative of position.

    Jerk is the rate of change of acceleration. A trajectory accelerating
    smoothly along a fixed curve has near-zero jerk; trajectories that
    *suddenly* change their acceleration (regime transitions, kinks, tube
    switches) show spikes in the jerk magnitude. Useful as a higher-order
    regime-change detector — picks up transitions that velocity and
    acceleration have not yet bent through.

    Computed via :func:`numpy.gradient` chained three times. Central
    differences with arc-length / epoch-gap parameterization handle
    non-uniform timestep spacing correctly.

    Args:
        trajectory: ``(n_timesteps, d)`` time-ordered curve in some
            coordinate frame.
        time_axis: ``(n_timesteps,)`` per-step time labels (e.g. epoch
            numbers). When omitted, uniform unit spacing is assumed.

    Returns:
        ``(n_timesteps, d)`` jerk vector at each timestep. Caller takes
        ``np.linalg.norm(jerk, axis=1)`` for a scalar magnitude per
        timestep if a single channel for spike detection is wanted.
    """
    if trajectory.ndim != 2:
        raise ValueError(f"characterize_jerk expects 2D input, got shape {trajectory.shape}")
    if trajectory.shape[0] < 4:
        raise ValueError(
            f"characterize_jerk needs at least 4 timesteps to compute a third "
            f"derivative; got {trajectory.shape[0]}"
        )
    # edge_order=2 gives second-order accurate one-sided differences at the
    # boundaries — exact on quadratics, far less boundary contamination than
    # the default first-order behavior when chained three times.
    if time_axis is None:
        velocity = np.gradient(trajectory, axis=0, edge_order=2)
        accel = np.gradient(velocity, axis=0, edge_order=2)
        jerk = np.gradient(accel, axis=0, edge_order=2)
    else:
        if time_axis.ndim != 1 or time_axis.shape[0] != trajectory.shape[0]:
            raise ValueError(
                f"time_axis must be 1D with length n_timesteps={trajectory.shape[0]}; "
                f"got shape {time_axis.shape}"
            )
        velocity = np.gradient(trajectory, time_axis, axis=0, edge_order=2)
        accel = np.gradient(velocity, time_axis, axis=0, edge_order=2)
        jerk = np.gradient(accel, time_axis, axis=0, edge_order=2)
    return jerk


# ---------------------------------------------------------------------------
# Periodic-trajectory shape characterization (2D Lissajous structure)
# ---------------------------------------------------------------------------


class LissajousParameters(NamedTuple):
    """Multi-axis Fourier structure of a 2D periodic trajectory.

    Quantifies the dominant Fourier mode on each of two axes and the
    relationship between them — frequency match, amplitude ratio, phase
    offset, and joint fit quality — without committing to a particular
    theoretical generator (e.g. CR3BP saddle-center-center linearization).
    Use as a building block for theory-laden analyzers (halo classification,
    Lyapunov-family fits, etc.).

    Reconstruction convention: each axis's dominant mode is
    ``signal ≈ amplitude · cos(2π · frequency · t / period − phase)``.
    Phase comes from :func:`numpy.arctan2` ``(sin_coeff, cos_coeff)`` and
    lies in ``(−π, π]``.

    Fields:
        period: Length of one full period of the trajectory.
        frequency_x: Dominant integer frequency on axis 0
            (``1 ≤ k ≤ (period−1)//2``).
        frequency_y: Dominant integer frequency on axis 1.
        amplitude_x: Natural amplitude (signal-space) at the dominant
            frequency on axis 0.
        amplitude_y: Natural amplitude on axis 1.
        phase_x: Phase of the dominant mode on axis 0, in radians.
        phase_y: Phase of the dominant mode on axis 1, in radians.
        r2_x: Fraction of axis-0 (zero-mean) variance explained by its
            dominant single-frequency mode.
        r2_y: Fraction of axis-1 variance explained.
        same_frequency: ``True`` iff ``frequency_x == frequency_y``.
        amplitude_ratio: ``amplitude_y / amplitude_x``. Zero when
            ``amplitude_x`` is zero.
        phase_offset: ``phase_y − phase_x`` wrapped to ``(−π, π]``.
            Interpretation requires ``same_frequency`` (otherwise the
            two axes oscillate at different rates and the offset mixes
            phases of different modes).
        joint_r2: Combined fit quality across both axes, computed as
            ``(power_x_dominant + power_y_dominant) /
              (total_power_x + total_power_y)``. Approaches 1.0 for a
            clean Lissajous (each axis is a single dominant mode).
    """

    period: int
    frequency_x: int
    frequency_y: int
    amplitude_x: float
    amplitude_y: float
    phase_x: float
    phase_y: float
    r2_x: float
    r2_y: float
    same_frequency: bool
    amplitude_ratio: float
    phase_offset: float
    joint_r2: float


def characterize_lissajous(
    trajectory: np.ndarray,
    period_axis: int = 0,
) -> LissajousParameters:
    """Characterize the dominant Fourier structure of a 2D periodic trajectory.

    For each axis, finds the integer frequency whose Fourier component
    carries the most variance, then computes joint structural metrics —
    same-frequency match, amplitude ratio, phase offset between the
    dominant modes, and combined fit quality across both axes.

    Theory-agnostic: the primitive measures observable Lissajous structure
    without committing to a particular generator. CR3BP-specific quantities
    like the κ-ratio for saddle-center-center modes, halo classification,
    and out-of-plane mode detection are downstream compositions of this
    primitive plus additional Fourier projections.

    Natural amplitude convention: ``A = magnitude · sqrt(2 / period)``,
    where ``magnitude`` is the unit-normalized projection magnitude
    returned by :class:`PeriodicFourierBasis`. This recovers the
    amplitude of the original signal in the form
    ``A · cos(2π · k · t / N − φ)``.

    Args:
        trajectory: ``(period, 2)`` array when ``period_axis=0``, or
            ``(2, period)`` when ``period_axis=1``.
        period_axis: Which axis carries the periodic dimension. Default 0.

    Returns:
        :class:`LissajousParameters`.
    """
    if trajectory.ndim != 2:
        raise ValueError(
            f"characterize_lissajous expects 2D input, got shape {trajectory.shape}"
        )
    if period_axis == 1:
        trajectory = trajectory.T
    elif period_axis != 0:
        raise ValueError(f"period_axis must be 0 or 1, got {period_axis}")
    period, n_axes = trajectory.shape
    if n_axes != 2:
        raise ValueError(
            f"characterize_lissajous expects two axes; got {n_axes} on the non-period dimension"
        )

    basis = get_fourier_basis(period)
    fourier = project_onto_fourier_basis(trajectory, basis, period_axis=0)
    # All frequency-axis arrays here have shape (n_frequencies, 2).

    freq_x = int(fourier.dominant_frequency[0])
    freq_y = int(fourier.dominant_frequency[1])
    idx_x = freq_x - 1  # frequency labels are 1-indexed
    idx_y = freq_y - 1

    natural_scale = float(np.sqrt(2.0 / period))
    amp_x = float(fourier.magnitudes[idx_x, 0]) * natural_scale
    amp_y = float(fourier.magnitudes[idx_y, 1]) * natural_scale
    phase_x = float(fourier.phases[idx_x, 0])
    phase_y = float(fourier.phases[idx_y, 1])
    r2_x = float(fourier.fractional_power[idx_x, 0])
    r2_y = float(fourier.fractional_power[idx_y, 1])

    same_frequency = freq_x == freq_y
    amplitude_ratio = amp_y / amp_x if amp_x > 0 else 0.0
    phase_offset = float((phase_y - phase_x + np.pi) % (2 * np.pi) - np.pi)

    fitted_power = float(fourier.power[idx_x, 0] + fourier.power[idx_y, 1])
    total_power = float(fourier.power.sum())
    joint_r2 = fitted_power / total_power if total_power > 0 else 0.0

    return LissajousParameters(
        period=period,
        frequency_x=freq_x,
        frequency_y=freq_y,
        amplitude_x=amp_x,
        amplitude_y=amp_y,
        phase_x=phase_x,
        phase_y=phase_y,
        r2_x=r2_x,
        r2_y=r2_y,
        same_frequency=same_frequency,
        amplitude_ratio=amplitude_ratio,
        phase_offset=phase_offset,
        joint_r2=joint_r2,
    )


# ---------------------------------------------------------------------------
# Point-cloud surface characterization (quadratic fit: flat / bowl / saddle)
# ---------------------------------------------------------------------------


_SURFACE_FLAT_THRESHOLD = 0.05
_SURFACE_MIN_POINTS = 7  # minimum to fit 6 quadratic parameters

_SHAPE_TO_INT = {"flat/blob": 0, "bowl": 1, "saddle": 2}
_INT_TO_SHAPE = {v: k for k, v in _SHAPE_TO_INT.items()}


def characterize_surface(point_cloud_3d: np.ndarray) -> SurfaceParameters:
    """Fit a quadratic surface to a 3D point cloud and classify the shape.

    Treats the first two columns as predictors and the third as response,
    then runs a two-stage regression to isolate curvature from planar tilt:

        Stage 1 (linear):    z = d·x + e·y + f
        Stage 2 (quadratic): z = a·x² + b·y² + c·x·y + d·x + e·y + f

    ``r2_curvature = r2_quadratic − r2_linear`` captures only the variance
    explained by the quadratic terms net of any planar tilt — robust even
    when the projection is not perfectly centered.

    Shape classification uses the Hessian determinant ``4ab − c²``, which
    is rotation-invariant: a saddle rotated away from the coordinate axes
    is still detected even when ``a`` and ``b`` share the same sign.

    Args:
        point_cloud_3d: ``(n_points, 3)`` array. Typically the top-3 PCA
            projection of a neuron group's weight vectors. Must have at
            least 7 rows for a meaningful quadratic fit; smaller inputs
            return NaN parameters and ``shape="flat/blob"``.

    Returns:
        :class:`SurfaceParameters`.
    """
    if point_cloud_3d.shape[0] < _SURFACE_MIN_POINTS:
        return _surface_nan_result()

    x = point_cloud_3d[:, 0]
    y = point_cloud_3d[:, 1]
    z = point_cloud_3d[:, 2]

    r2_linear, _ = _fit_surface_linear(x, y, z)
    r2_quadratic, (a, b, c) = _fit_surface_quadratic(x, y, z)

    r2_curvature = float(np.clip(r2_quadratic - r2_linear, 0.0, 1.0))
    shape = _classify_surface_shape(r2_curvature, a, b, c)

    return SurfaceParameters(
        r2_linear=float(r2_linear),
        r2_quadratic=float(r2_quadratic),
        r2_curvature=r2_curvature,
        a=float(a),
        b=float(b),
        c=float(c),
        shape=shape,
    )


def decode_shapes(shape_int: np.ndarray) -> list[str]:
    """Convert integer shape labels back to human-readable strings."""
    return [_INT_TO_SHAPE.get(int(v), "flat/blob") for v in shape_int]


def _fit_surface_linear(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit ``z = d·x + e·y + f`` via least squares."""
    X = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    return _surface_r2(z, X @ coeffs), coeffs


def _fit_surface_quadratic(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[float, tuple[float, float, float]]:
    """Fit ``z = a·x² + b·y² + c·x·y + d·x + e·y + f`` via least squares."""
    X = np.column_stack([x**2, y**2, x * y, x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
    return _surface_r2(z, X @ coeffs), (a, b, c)


def _surface_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Coefficient of determination for a regression fit."""
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return float(np.clip(1.0 - ss_res / ss_tot, 0.0, 1.0))


def _classify_surface_shape(r2_curvature: float, a: float, b: float, c: float) -> str:
    """Classify surface from curvature R² and quadratic coefficients."""
    if r2_curvature < _SURFACE_FLAT_THRESHOLD:
        return "flat/blob"
    if 4 * a * b - c**2 > 0:
        return "bowl"
    return "saddle"


def _surface_nan_result() -> SurfaceParameters:
    """NaN result for point clouds too small for a meaningful fit."""
    return SurfaceParameters(
        r2_linear=float("nan"),
        r2_quadratic=float("nan"),
        r2_curvature=float("nan"),
        a=float("nan"),
        b=float("nan"),
        c=float("nan"),
        shape="flat/blob",
    )
