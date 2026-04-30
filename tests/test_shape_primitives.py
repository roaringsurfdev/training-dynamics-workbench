"""Unit tests for shape primitives in library/shape.py (REQ_109 phase 2)."""

import numpy as np
import pytest

from miscope.analysis.library.shape import (
    characterize_jerk,
    compute_arc_length,
    compute_curvature_profile,
    compute_signed_loop_area,
    detect_self_intersection,
)


def _unit_circle(n: int = 100) -> np.ndarray:
    """Sampled unit circle, n points, evenly spaced in angle."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def _lemniscate(n: int = 200, a: float = 1.0) -> np.ndarray:
    """Lemniscate of Gerono parametrized by t in [0, 2π).

    x = a * sin(t), y = a * sin(t) * cos(t). Self-intersects at the origin
    when traversing both lobes.
    """
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([a * np.sin(t), a * np.sin(t) * np.cos(t)])


class TestComputeArcLength:
    def test_starts_at_zero(self):
        curve = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
        arc = compute_arc_length(curve)
        assert arc[0] == 0.0

    def test_cumulative_length_known_curve(self):
        # 3-4-5 triangle then up 4: total should be 5 + 4 = 9
        curve = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 8.0]])
        arc = compute_arc_length(curve)
        np.testing.assert_allclose(arc, [0.0, 5.0, 9.0])

    def test_unit_circle_perimeter(self):
        curve = _unit_circle(n=1000)
        # Closed-loop approximation requires connecting back to start
        closed = np.vstack([curve, curve[:1]])
        arc = compute_arc_length(closed)
        np.testing.assert_allclose(arc[-1], 2 * np.pi, atol=1e-3)

    def test_3d_curve(self):
        # Helix segment: x = cos(t), y = sin(t), z = t/(2π) for t ∈ [0, 2π]
        # Arc length = sqrt(2) * 2π
        n = 500
        t = np.linspace(0, 2 * np.pi, n)
        curve = np.column_stack([np.cos(t), np.sin(t), t / (2 * np.pi)])
        arc = compute_arc_length(curve)
        expected = np.sqrt(1 + (1 / (2 * np.pi)) ** 2) * 2 * np.pi
        np.testing.assert_allclose(arc[-1], expected, rtol=1e-3)

    def test_output_shape(self):
        curve = np.zeros((10, 2))
        arc = compute_arc_length(curve)
        assert arc.shape == (10,)


class TestDetectSelfIntersection:
    def test_lemniscate_node_near_origin(self):
        curve = _lemniscate(n=400)
        result = detect_self_intersection(curve)
        # The lemniscate of Gerono passes through (0, 0) twice — the node
        # should land near the origin.
        node = result["node_position"]
        assert abs(node[0]) < 0.05
        assert abs(node[1]) < 0.05

    def test_lemniscate_close_approach(self):
        curve = _lemniscate(n=400)
        result = detect_self_intersection(curve)
        # The two passes through the origin should be very close.
        assert result["min_distance"] < 0.05

    def test_returns_required_keys(self):
        curve = _lemniscate(n=200)
        result = detect_self_intersection(curve)
        assert {"node_position", "idx_pair", "min_distance", "arc_length"} <= result.keys()

    def test_idx_pair_separated(self):
        # Indices i, j must be far enough apart on the arc that they aren't neighbors
        curve = _lemniscate(n=200)
        result = detect_self_intersection(curve, min_arc_sep_fraction=0.25)
        i, j = result["idx_pair"]
        assert j > i
        assert (j - i) > 10  # Well-separated along the curve

    def test_non_intersecting_curve(self):
        # Open arc that never approaches itself
        n = 100
        t = np.linspace(0, 1.5 * np.pi, n)
        curve = np.column_stack([np.cos(t), np.sin(t)])
        result = detect_self_intersection(curve, min_arc_sep_fraction=0.5)
        # Distance between far-apart points on an arc is non-zero; assert > 0
        assert result["min_distance"] > 0.0


class TestComputeSignedLoopArea:
    def test_unit_circle_loop_area(self):
        # Closed unit circle traversed CCW
        curve = _unit_circle(n=1000)
        area = compute_signed_loop_area(curve, idx_pair=(0, len(curve) - 1))
        np.testing.assert_allclose(area, np.pi, atol=1e-2)

    def test_traversal_direction_sign(self):
        ccw = _unit_circle(n=1000)
        cw = ccw[::-1]
        area_ccw = compute_signed_loop_area(ccw, idx_pair=(0, len(ccw) - 1))
        area_cw = compute_signed_loop_area(cw, idx_pair=(0, len(cw) - 1))
        assert area_ccw > 0
        assert area_cw < 0
        np.testing.assert_allclose(area_ccw, -area_cw)

    def test_short_loop_returns_zero(self):
        curve = np.array([[0.0, 0.0], [1.0, 0.0]])
        assert compute_signed_loop_area(curve, idx_pair=(0, 1)) == 0.0

    def test_lemniscate_loop_area(self):
        # First lobe of lemniscate of Gerono — closed loop with non-zero area
        curve = _lemniscate(n=400)
        node = detect_self_intersection(curve)
        area = compute_signed_loop_area(curve, node["idx_pair"])
        assert abs(area) > 0.1


class TestComputeCurvatureProfile:
    def test_returns_required_keys(self):
        curve = _unit_circle(n=200)
        result = compute_curvature_profile(curve)
        assert {"s_norm", "kappa", "s_raw", "kappa_raw"} <= result.keys()

    def test_n_norm_points_respected(self):
        curve = _unit_circle(n=200)
        result = compute_curvature_profile(curve, n_norm_points=50)
        assert result["s_norm"].shape == (50,)
        assert result["kappa"].shape == (50,)

    def test_s_norm_in_unit_interval(self):
        curve = _unit_circle(n=200)
        result = compute_curvature_profile(curve)
        assert result["s_norm"][0] == pytest.approx(0.0)
        assert result["s_norm"][-1] == pytest.approx(1.0)

    def test_unit_circle_constant_curvature(self):
        # Curvature of unit circle is 1 everywhere; closed loop sampling needs
        # the wrap-around point so np.gradient sees a smooth derivative
        curve = _unit_circle(n=500)
        closed = np.vstack([curve, curve[:1]])
        result = compute_curvature_profile(closed)
        # Skip endpoints where finite-difference is least accurate
        interior = result["kappa"][5:-5]
        np.testing.assert_allclose(np.abs(interior), 1.0, atol=0.05)

    def test_straight_line_zero_curvature(self):
        n = 100
        curve = np.column_stack([np.linspace(0, 10, n), np.zeros(n)])
        result = compute_curvature_profile(curve)
        # Curvature of a straight line is 0; allow numerical noise
        np.testing.assert_allclose(result["kappa"], 0.0, atol=1e-6)


class TestCharacterizeJerk:
    def test_output_shape_matches_input(self):
        # Vector form: same shape as input
        rng = np.random.default_rng(0)
        trajectory = rng.normal(size=(50, 3))
        jerk = characterize_jerk(trajectory)
        assert jerk.shape == trajectory.shape

    def test_constant_velocity_zero_jerk(self):
        # x(t) = v*t — first derivative is constant, all higher derivatives zero
        n = 50
        velocity = np.array([1.0, 2.0, -0.5])
        t = np.arange(n, dtype=float)
        trajectory = t[:, np.newaxis] * velocity
        jerk = characterize_jerk(trajectory)
        np.testing.assert_allclose(jerk, 0.0, atol=1e-10)

    def test_constant_acceleration_zero_jerk(self):
        # x(t) = 0.5 * a * t² — second derivative is constant, third is zero
        n = 50
        accel = np.array([1.0, -2.0])
        t = np.arange(n, dtype=float)
        trajectory = 0.5 * t[:, np.newaxis] ** 2 * accel
        jerk = characterize_jerk(trajectory)
        # np.gradient is exact on quadratic interiors; endpoints may have noise
        np.testing.assert_allclose(jerk[2:-2], 0.0, atol=1e-9)

    def test_constant_jerk_for_cubic(self):
        # x(t) = (1/6) * j * t³ — third derivative is the constant j
        n = 50
        jerk_const = np.array([1.0, -2.0, 0.5])
        t = np.arange(n, dtype=float)
        trajectory = (t[:, np.newaxis] ** 3 / 6.0) * jerk_const
        jerk = characterize_jerk(trajectory)
        # np.gradient is exact on cubics for interior points; check well inside.
        # Broadcast-subtract to compare against the per-component constant.
        np.testing.assert_allclose(jerk[5:-5] - jerk_const, 0.0, atol=1e-8)

    def test_time_axis_changes_scale(self):
        # Same trajectory sampled with epoch gap = 10 should have jerk
        # scaled by 1/10³ relative to unit-spacing baseline
        rng = np.random.default_rng(0)
        trajectory = rng.normal(size=(40, 2))
        jerk_unit = characterize_jerk(trajectory)
        epochs_dense = np.arange(40, dtype=float) * 10.0
        jerk_scaled = characterize_jerk(trajectory, time_axis=epochs_dense)
        np.testing.assert_allclose(jerk_scaled, jerk_unit / (10.0**3), atol=1e-9)

    def test_non_uniform_time_axis(self):
        # Mix dense + sparse spacing — characterize_jerk must respect the gaps,
        # so a smooth cubic still gives near-constant jerk inside each uniform
        # segment. Boundary cells (curve start, segment transition, curve end)
        # are contaminated by the second-order one-sided differences propagating
        # inward through three gradient passes; only interior cells away from
        # those three boundaries are clean.
        sparse = np.arange(0, 100, 10, dtype=float)  # 10 points at gap 10 (indices 0–9)
        dense = np.arange(100, 110, 1, dtype=float)  # 10 points at gap 1 (indices 10–19)
        epochs = np.concatenate([sparse, dense])
        trajectory = epochs[:, np.newaxis] ** 3 / 6.0
        jerk = characterize_jerk(trajectory, time_axis=epochs)
        # Clean interior of sparse segment: indices 3–7 (away from start + transition)
        np.testing.assert_allclose(jerk[3:8, 0], 1.0, atol=1e-6)
        # Clean interior of dense segment: indices 13–16 (away from transition + end)
        np.testing.assert_allclose(jerk[13:17, 0], 1.0, atol=1e-6)

    def test_rejects_short_trajectory(self):
        with pytest.raises(ValueError, match="at least 4 timesteps"):
            characterize_jerk(np.zeros((3, 2)))

    def test_rejects_1d_input(self):
        with pytest.raises(ValueError, match="2D input"):
            characterize_jerk(np.zeros(10))

    def test_rejects_mismatched_time_axis(self):
        trajectory = np.zeros((10, 2))
        with pytest.raises(ValueError, match="length n_timesteps"):
            characterize_jerk(trajectory, time_axis=np.arange(5, dtype=float))
